"""Tests for the per-coin re-entry cap preflight (T4.3 port, upstream
6181322895ab).

Three layers:

1. risk_gates.reentry_cap_reason — the pure helper, fully hermetic:
   cap<=0/None/bad-string never blocks (disabled); n >= cap blocks with the
   exact reason string (so with cap=3 the 3rd+ entry is cut); n < cap passes.

2. memory.count_entries_since — the trade-record reader, tested against a
   fresh AgentMemory with flush() stubbed (no network, no live config, no
   disk): counts only this coin's executed entries with executed_at >=
   since_ms, ignores missing executed_at, returns 0 for an unknown coin, and
   respects `limit` (the recent-trade scan window).

3. wiring smoke (mirrors tests/test_history_floor.py's AST-extraction
   harness): the fresh-candidate branch of _process_coin_run must short-
   circuit with a REENTRY_CAP ta_skip BEFORE the paid research call when the
   per-coin entry count is at/over the cap — and the code default (no
   `reentry_cap` config) must be a no-op (no counter read, research runs).
"""
import ast
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Test isolation (mirror tests/conftest.py) — MUST run before hermes imports
_tmpdir = tempfile.mkdtemp(prefix="hermes-test-reentrycap-")
os.environ["HERMES_AGENT_MEMORY_FILE"] = os.path.join(_tmpdir, ".agent-memory.json")
os.environ["HERMES_AGENT_CONFIG_FILE"] = os.path.join(_tmpdir, ".agent-config.json")
os.environ["HERMES_DSL_STATE_FILE"] = os.path.join(_tmpdir, ".dsl-state.json")
os.environ["HERMES_LEDGER_FILE"] = os.path.join(_tmpdir, "trades.jsonl")
os.environ["HERMES_DUEL_FILE"] = os.path.join(_tmpdir, ".hermes-trader-duel.jsonl")
os.environ.pop("LLM_DUEL_MODEL", None)

from hermes_trader.agents.risk_gates import reentry_cap_reason  # noqa: E402
from hermes_trader.agents.memory import AgentMemory  # noqa: E402
from hermes_trader.agents.executor import momentum_reentry_allowed  # noqa: E402


# ---------------------------------------------------------------------------
# 1. the pure helper
# ---------------------------------------------------------------------------

def test_disabled_zero_never_blocks():
    assert reentry_cap_reason("A", 5, 0) == ""


def test_disabled_none_never_blocks():
    assert reentry_cap_reason("A", 5, None) == ""


def test_disabled_negative_never_blocks():
    assert reentry_cap_reason("A", 5, -1) == ""


def test_disabled_bad_string_never_blocks():
    assert reentry_cap_reason("A", 5, "x") == ""


def test_count_below_cap_passes():
    assert reentry_cap_reason("A", 2, 3) == ""


def test_count_zero_passes():
    assert reentry_cap_reason("A", 0, 3) == ""


def test_count_none_passes():
    assert reentry_cap_reason("A", None, 3) == ""


def test_count_equal_cap_blocks():
    """Strict `>=`: with cap=3 the 3rd entry (n==cap) is already blocked."""
    assert reentry_cap_reason("A", 3, 3) == "reentry_cap (3 entries in window >= cap 3)"


def test_count_above_cap_blocks():
    assert reentry_cap_reason("A", 5, 3) == "reentry_cap (5 entries in window >= cap 3)"


def test_reason_string_shape():
    """Pin the exact operator-facing string (downstream dashboards key on it)."""
    r = reentry_cap_reason("BTC", 4, 3)
    assert r == "reentry_cap (4 entries in window >= cap 3)"


# ---------------------------------------------------------------------------
# 2. memory.count_entries_since (real AgentMemory, flush stubbed)
# ---------------------------------------------------------------------------

def _mem():
    m = AgentMemory()
    m.flush = lambda: None  # never write to disk in tests
    return m


def _trade(coin, ts):
    t = {"coin": coin, "side": "long", "entry_px": 1.0, "size_usd": 10.0,
         "order_id": f"OID-{coin}-{ts}", "executed_at": ts}
    return t


def test_counts_only_in_window_entries():
    """3 A-entries in-window + 2 A-entries out-of-window + 1 other coin -> 3."""
    m = _mem()
    since = 10_000
    # A in-window (executed_at >= since)
    for ts in (10_000, 20_000, 30_000):
        m.record_trade(_trade("A", ts))
    # A out-of-window (executed_at < since)
    for ts in (5_000, 9_000):
        m.record_trade(_trade("A", ts))
    # another coin
    m.record_trade(_trade("B", 50_000))
    assert m.count_entries_since("A", since) == 3
    assert m.count_entries_since("B", since) == 1


def test_ignores_missing_executed_at():
    m = _mem()
    since = 10_000
    m.record_trade(_trade("A", 20_000))
    m.record_trade(_trade("A", 30_000))
    bad = {"coin": "A", "side": "long", "entry_px": 1.0, "size_usd": 10.0,
           "order_id": "OID-A-missing"}  # no executed_at
    m.record_trade(bad)
    assert m.count_entries_since("A", since) == 2


def test_unknown_coin_returns_zero():
    m = _mem()
    m.record_trade(_trade("A", 20_000))
    assert m.count_entries_since("ZZZ", 0) == 0


def test_respects_limit_window():
    """limit bounds the recent-trade scan window (get_recent_trades(limit))."""
    m = _mem()
    for i in range(5):  # 5 A-entries, all in-window
        m.record_trade(_trade("A", 1000 + i * 1000))
    assert m.count_entries_since("A", 0, limit=100) == 5  # full window
    assert m.count_entries_since("A", 0, limit=2) == 2     # only the newest 2
    assert m.count_entries_since("A", 0, limit=1) == 1


# ---------------------------------------------------------------------------
# 3. wiring smoke: the fresh-candidate pre-research path
# ---------------------------------------------------------------------------

LOOP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "trading_loop.py")

_loop_src = open(LOOP_PATH).read()
_tree = ast.parse(_loop_src)
_FUNCS = {n.name: n for n in _tree.body if isinstance(n, ast.FunctionDef)}


def _extract(name):
    assert name in _FUNCS, f"{name} not found at module level in trading_loop.py"
    return ast.get_source_segment(_loop_src, _FUNCS[name])


class _Logger:
    def info(self, *a, **k): pass

    def warning(self, *a, **k): pass

    def error(self, *a, **k): pass


class _FakeMemory:
    """Fake memory: records ta_skip events and drives count_entries_since with
    a preset count (also recording each call's since_ms for assertions)."""

    def __init__(self, entry_count=0):
        self.entry_count = entry_count
        self.events = []
        self.count_calls = []

    def record_perception(self, p):
        self.events.append(("perception", p.get("coin")))

    def loss_cooldown_remaining_min(self, coin):
        return 0.0

    def last_close_for(self, coin):
        return {}

    def count_entries_since(self, coin, since_ms, limit=100):
        self.count_calls.append((coin, since_ms, limit))
        return self.entry_count


def _ta_skip_signals(memory):
    return [e[1].get("signal") for e in memory.events
            if e[0] == "log_event" and e[1].get("event") == "ta_skip"]


def _new_env(memory):
    """Namespace for the real _process_coin_run body with every external
    touch point faked; analyze_perception confirms so the fresh coin always
    reaches the pre-research gates and (if not blocked) the paid research."""
    research_calls = []

    def fake_research(coin, perception):
        research_calls.append(coin)
        return {"id": "fake-analysis", "coin": coin, "verdict": "PASS",
                "confidence": 0.5, "reasoning": "stub",
                "entry_px": 0.0, "stop_px": 0.0, "tp_px": 0.0}

    ns = {
        "time": time,
        "memory": memory,
        "logger": _Logger(),
        "log_event": lambda e: memory.events.append(("log_event", e)),
        "analyze_perception": lambda p: {"signal": "CONFIRMED", "score": 55.0},
        "_burst_fired": lambda p: False,
        "_remaining_minutes": lambda ms: max(1, int(ms / 60_000)),
        "research": fake_research,
        "route_verdict": lambda analysis, **kw: {
            "action": "none", "verdict": analysis.get("verdict"), "result": {}},
        "momentum_reentry_allowed": momentum_reentry_allowed,
        "fetch_hl_candles": lambda coin, interval="5m", count=100, fresh=False: None,
        "_research_lock": threading.Lock(),
        "_last_research_by_coin": {},
        "_last_progress_ts": 0.0,
    }
    exec(compile(_extract("_process_coin_run"), LOOP_PATH, "exec"), ns)
    return ns, research_calls


def _run_coin(ns, cfg_cd, coin="NEWCOIN"):
    perception = {"coin": coin, "composite_score": 40.0, "mid": 1.0,
                  "triggers": [{"name": "momentumBurst", "fired": True}],
                  "whale_signal": None}
    ctx = {
        "now_ms": int(time.time() * 1000),
        "held_coins": set(),
        "held_research_ms": 3 * 60_000,
        "cooldown_ms": 30 * 60_000,
        "recent_trades_by_coin": {},
        "blocklist": set(),
        "cfg_cd": cfg_cd,
    }
    ns["_process_coin_run"](perception, ctx)


def test_wiring_default_disabled_is_noop():
    """Code default (reentry_cap absent): the cap is a no-op — the counter is
    never read and the paid research still runs."""
    mem = _FakeMemory(entry_count=99)  # even a huge count must not matter
    ns, research_calls = _new_env(mem)
    _run_coin(ns, cfg_cd={})
    assert research_calls == ["NEWCOIN"], research_calls
    assert mem.count_calls == [], "disabled cap must not read the counter"
    assert "REENTRY_CAP" not in _ta_skip_signals(mem)


def test_wiring_enabled_blocks_at_cap_before_paid_research():
    """reentry_cap enabled, count == cap -> short-circuit with a REENTRY_CAP
    ta_skip, NO paid research, and exactly one counter read inside the window."""
    mem = _FakeMemory(entry_count=3)
    ns, research_calls = _new_env(mem)
    _run_coin(ns, cfg_cd={"reentry_cap": {"enabled": True,
                                          "max_per_coin": 3,
                                          "window_hours": 24}})
    assert research_calls == [], f"research was paid: {research_calls}"
    assert "REENTRY_CAP" in _ta_skip_signals(mem), _ta_skip_signals(mem)
    assert len(mem.count_calls) == 1, mem.count_calls
    coin, since_ms, limit = mem.count_calls[0]
    assert coin == "NEWCOIN"
    # default 24h window: since_ms = now_ms - 24h
    assert abs((time.time() * 1000 - since_ms) - 24 * 3_600_000) < 5_000
    assert limit == 100


def test_wiring_enabled_blocks_above_cap():
    """count > cap also blocks (the n >= cap rule)."""
    mem = _FakeMemory(entry_count=7)
    ns, research_calls = _new_env(mem)
    _run_coin(ns, cfg_cd={"reentry_cap": {"enabled": True,
                                          "max_per_coin": 3,
                                          "window_hours": 24}})
    assert research_calls == []
    assert "REENTRY_CAP" in _ta_skip_signals(mem)


def test_wiring_enabled_below_cap_researches_normally():
    """count < cap -> the gate passes and the paid research still runs (the
    cap only trims the 3rd+ re-entry/coin/window)."""
    mem = _FakeMemory(entry_count=2)
    ns, research_calls = _new_env(mem)
    _run_coin(ns, cfg_cd={"reentry_cap": {"enabled": True,
                                          "max_per_coin": 3,
                                          "window_hours": 24}})
    assert research_calls == ["NEWCOIN"], research_calls
    assert mem.count_calls, "enabled cap must read the counter"
    assert "REENTRY_CAP" not in _ta_skip_signals(mem)


def test_wiring_custom_window_hours():
    """window_hours is honored: a 1h window -> since_ms = now_ms - 1h."""
    mem = _FakeMemory(entry_count=3)
    ns, research_calls = _new_env(mem)
    _run_coin(ns, cfg_cd={"reentry_cap": {"enabled": True,
                                          "max_per_coin": 2,
                                          "window_hours": 1}})
    assert research_calls == []
    assert "REENTRY_CAP" in _ta_skip_signals(mem)
    _coin, since_ms, _limit = mem.count_calls[0]
    assert abs((time.time() * 1000 - since_ms) - 1 * 3_600_000) < 5_000


def test_source_wiring_order():
    """The skip sits in the fresh-candidate branch (after the T4.2 history
    floor, before the TA filter) and is a per-branch risk_gates import."""
    i_hist = _loop_src.index("T4.2 port (upstream cd6eaeec677f)")
    i_reentry = _loop_src.index("T4.3 port (upstream 6181322895ab)")
    i_ta = _loop_src.index("TA filter — cheap statistical gate")
    assert i_hist < i_reentry < i_ta
    assert "from hermes_trader.agents.risk_gates import reentry_cap_reason" in _loop_src
