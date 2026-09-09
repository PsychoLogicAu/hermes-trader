"""Tests for the min_history_bars preflight gate (T4.2 port, upstream
cd6eaeec677f).

Two layers:

1. risk_gates.history_floor_reason — the pure helper, fully hermetic
   (stubbed `fetch_daily`, no network, no live config):
   young coin blocks; mature / exactly-min / empty / None / fetch-error
   all fail-OPEN; disabled (0/None) never fetches.

2. wiring smoke (mirrors tests/test_cooldown_research_skip.py's
   AST-extraction harness): the fresh-candidate branch of
   _process_coin_run must short-circuit with a HISTORY_FLOOR ta_skip
   BEFORE the paid research call when the coin has a young daily
   history — and the default (min_history_bars absent) must never
   fetch daily candles.
"""
import ast
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Test isolation (mirror tests/conftest.py) — MUST run before hermes imports
_tmpdir = tempfile.mkdtemp(prefix="hermes-test-histfloor-")
os.environ["HERMES_AGENT_MEMORY_FILE"] = os.path.join(_tmpdir, ".agent-memory.json")
os.environ["HERMES_AGENT_CONFIG_FILE"] = os.path.join(_tmpdir, ".agent-config.json")
os.environ["HERMES_DSL_STATE_FILE"] = os.path.join(_tmpdir, ".dsl-state.json")
os.environ["HERMES_LEDGER_FILE"] = os.path.join(_tmpdir, "trades.jsonl")
os.environ["HERMES_DUEL_FILE"] = os.path.join(_tmpdir, ".hermes-trader-duel.jsonl")
os.environ.pop("LLM_DUEL_MODEL", None)

from hermes_trader.agents.risk_gates import history_floor_reason  # noqa: E402
from hermes_trader.agents.executor import momentum_reentry_allowed  # noqa: E402
from hermes_trader.agents import dsl_exit  # noqa: E402

# Other tests in the same process may leave DSL trackers behind; a leftover
# <COIN>_long/<COIN>_short tracker would change the held branch's behavior.
dsl_exit._active_positions.clear()


class _Recorder:
    """fetch_daily stub that records calls and returns a fixed payload."""

    def __init__(self, payload=None, raise_exc=None):
        self.payload = payload
        self.raise_exc = raise_exc
        self.calls = []

    def __call__(self, coin, n):
        self.calls.append((coin, n))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.payload


# ---------------------------------------------------------------------------
# the pure helper
# ---------------------------------------------------------------------------

def test_young_coin_blocks():
    """6 daily bars, min 60 -> blocked with the preflight reason string."""
    r = history_floor_reason("FRESHCOIN", 60, _Recorder(payload=[1] * 6))
    assert r == "history_floor_preflight (6d < 60d history)"
    assert r.startswith("history_floor_preflight")


def test_mature_coin_passes():
    stub = _Recorder(payload=[1] * 60)
    assert history_floor_reason("OLD", 60, stub) == ""
    assert stub.calls[0][0] == "OLD"
    # fetches min_hist + 5 (bounded overshoot for partial-bar slack)
    assert stub.calls[0][1] == 65


def test_exactly_min_passes():
    """Strict `<`: exactly min_hist bars is NOT young."""
    assert history_floor_reason("EXACT", 60, _Recorder(payload=[1] * 60)) == ""


def test_empty_read_fails_open():
    """A 429 can surface as an empty list — never treat that as young."""
    assert history_floor_reason("EMPTY", 60, _Recorder(payload=[])) == ""


def test_none_fails_open():
    assert history_floor_reason("NONE", 60, _Recorder(payload=None)) == ""


def test_fetch_error_fails_open():
    """Simulated 429/timeout: a transient failure must not block."""
    stub = _Recorder(raise_exc=RuntimeError("429 too many requests"))
    assert history_floor_reason("DOWN", 60, stub) == ""
    assert stub.calls == [("DOWN", 65)]


def test_disabled_zero_never_fetches():
    stub = _Recorder(payload=[1] * 6)
    assert history_floor_reason("FRESH", 0, stub) == ""
    assert stub.calls == [], "disabled gate must not fetch"


def test_disabled_none_never_fetches():
    stub = _Recorder(payload=[1] * 6)
    assert history_floor_reason("FRESH", None, stub) == ""
    assert stub.calls == []


def test_disabled_negative_never_fetches():
    stub = _Recorder(payload=[1] * 6)
    assert history_floor_reason("FRESH", -5, stub) == ""
    assert stub.calls == []


# ---------------------------------------------------------------------------
# wiring smoke: the fresh-candidate pre-research path
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
    def __init__(self):
        self.cooldown_remaining = {}
        self.last_closes = {}
        self.events = []

    def record_perception(self, p):
        self.events.append(("perception", p.get("coin")))

    def loss_cooldown_remaining_min(self, coin):
        return self.cooldown_remaining.get(coin, 0.0)

    def last_close_for(self, coin):
        return self.last_closes.get(coin) or {}


def _ta_skip_signals(memory):
    return [e[1].get("signal") for e in memory.events
            if e[0] == "log_event" and e[1].get("event") == "ta_skip"]


def _new_env(memory, daily_bars, cfg_cd):
    """Namespace for the real _process_coin_run body with every external
    touch point faked; `fetch_hl_candles` returns `daily_bars` and records
    calls."""
    research_calls = []
    fetch_calls = []

    def fake_research(coin, perception):
        research_calls.append(coin)
        return {"id": "fake-analysis", "coin": coin, "verdict": "PASS",
                "confidence": 0.5, "reasoning": "stub",
                "entry_px": 0.0, "stop_px": 0.0, "tp_px": 0.0}

    def fake_fetch(coin, interval="5m", count=100, fresh=False):
        fetch_calls.append((coin, interval, count))
        return daily_bars

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
        "fetch_hl_candles": fake_fetch,
        "_research_lock": threading.Lock(),
        "_last_research_by_coin": {},
        "_last_progress_ts": 0.0,
    }
    exec(compile(_extract("_process_coin_run"), LOOP_PATH, "exec"), ns)
    return ns, research_calls, fetch_calls


def _run_coin(ns, perception, cfg_cd):
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


def _perception(coin="NEWCOIN", score=40.0):
    return {"coin": coin, "composite_score": score, "mid": 1.0,
            "triggers": [{"name": "momentumBurst", "fired": True}],
            "whale_signal": None}


def test_wiring_young_fresh_coin_skips_before_paid_research():
    """A fresh (not-held, no cooldowns) coin with only 6 daily bars and
    min_history_bars=60 must short-circuit with a HISTORY_FLOOR ta_skip —
    NO paid research, one 1d fetch."""
    mem = _FakeMemory()
    ns, research_calls, fetch_calls = _new_env(
        mem, daily_bars=[0] * 6, cfg_cd={"min_history_bars": 60})
    _run_coin(ns, _perception(), {"min_history_bars": 60})
    assert research_calls == [], f"research was paid: {research_calls}"
    assert "HISTORY_FLOOR" in _ta_skip_signals(mem), _ta_skip_signals(mem)
    assert fetch_calls == [("NEWCOIN", "1d", 65)], fetch_calls


def test_wiring_mature_fresh_coin_researches_normally():
    """60+ daily bars + min_history_bars=60 -> the gate passes and the paid
    research still runs (the gate only trims young coins)."""
    mem = _FakeMemory()
    ns, research_calls, fetch_calls = _new_env(
        mem, daily_bars=[0] * 60, cfg_cd={"min_history_bars": 60})
    _run_coin(ns, _perception(), {"min_history_bars": 60})
    assert research_calls == ["NEWCOIN"], research_calls
    assert fetch_calls == [("NEWCOIN", "1d", 65)], fetch_calls


def test_wiring_disabled_default_never_fetches():
    """Code default (key absent -> 0): no daily fetch at all, research runs
    — the gate is a no-op until the owner enables it in .agent-config.json."""
    mem = _FakeMemory()
    ns, research_calls, fetch_calls = _new_env(
        mem, daily_bars=[0] * 6, cfg_cd={})
    _run_coin(ns, _perception(), {})
    assert research_calls == ["NEWCOIN"], research_calls
    assert fetch_calls == [], f"default config must not fetch: {fetch_calls}"


def test_wiring_fetch_failure_fails_open_and_researches():
    """A transient daily-candle fetch error must not block a fresh coin."""
    mem = _FakeMemory()
    ns, research_calls, fetch_calls = _new_env(
        mem, daily_bars=None, cfg_cd={"min_history_bars": 60})
    ns["fetch_hl_candles"] = (
        lambda coin, interval="5m", count=100, fresh=False:
        (_ for _ in ()).throw(RuntimeError("429")))
    _run_coin(ns, _perception(), {"min_history_bars": 60})
    assert research_calls == ["NEWCOIN"], research_calls
    assert "HISTORY_FLOOR" not in _ta_skip_signals(mem)


def test_source_wiring_order():
    """The skip sits in the fresh-candidate branch (after loss-cooldown,
    before the TA filter) and fetch_hl_candles is a top-level import."""
    assert ("from hermes_trader.client.hl_client import (fetch_account_state,"
            in _loop_src and "fetch_hl_candles," in _loop_src)
    i_losscd = _loop_src.index("pre-research loss-cooldown")
    i_hist = _loop_src.index("T4.2 port (upstream cd6eaeec677f)")
    i_ta = _loop_src.index("TA filter — cheap statistical gate")
    assert i_losscd < i_hist < i_ta
