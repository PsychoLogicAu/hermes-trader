"""Cooldown-aware research gating + forced held-position re-evaluation.

Two fixes from the 2026-08-26 ZEC post-mortem:

1. LOSS-COOLDOWN RESEARCH SKIP (trading_loop._process_coin_run)
   The pre-research skip only checked the 30min standard cooldown, so a coin
   inside the 180min loss cooldown kept burning paid LLM researches that the
   executor's loss-cooldown gate would refuse anyway (321 paid researches
   across 08-25/26; ZEC's 07:39 LONG verdict was the proof it was a genuine
   re-entry attempt, blocked at execution). Now the NOT-held branch also
   skips when the loss cooldown is active, PRESERVING the momentum-reentry
   bypass (same rule as executor.maybe_execute).

2. FORCED HELD RE-EVAL (trading_loop._forced_held_reeval)
   The AI close-check only fired when a held coin showed up in scan results
   AND passed the TA filter — a held coin that stopped triggering (ZEC short,
   ~8h, zero close-checks) was never re-evaluated. Now each held coin that
   didn't trigger this scan and has been quiet for >= held_reval_min_age_min
   gets a synthetic perception routed through the same _process_coin path.

Testing approach: scripts/trading_loop.py is NOT importable (the main
`while True` loop is module-level and would start trading), so — mirroring
tests/test_parallel_research_race.py — we extract the functions by AST and
exec them in a stubbed namespace. NO network, NO exchange calls, NO live
state files (env isolation below + tests/conftest.py).
"""
import ast
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Test isolation (mirror tests/conftest.py) — MUST run before hermes imports
_tmpdir = tempfile.mkdtemp(prefix="hermes-test-coldskip-")
os.environ["HERMES_AGENT_MEMORY_FILE"] = os.path.join(_tmpdir, ".agent-memory.json")
os.environ["HERMES_AGENT_CONFIG_FILE"] = os.path.join(_tmpdir, ".agent-config.json")
os.environ["HERMES_DSL_STATE_FILE"] = os.path.join(_tmpdir, ".dsl-state.json")
os.environ["HERMES_LEDGER_FILE"] = os.path.join(_tmpdir, "trades.jsonl")
os.environ["HERMES_DUEL_FILE"] = os.path.join(_tmpdir, ".hermes-trader-duel.jsonl")
os.environ.pop("LLM_DUEL_MODEL", None)

from hermes_trader.agents.executor import momentum_reentry_allowed  # noqa: E402
from hermes_trader.agents import dsl_exit  # noqa: E402

# Other tests in the same process may leave DSL trackers behind; a leftover
# <COIN>_long/<COIN>_short tracker would change the infancy branch's behavior.
dsl_exit._active_positions.clear()

LOOP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "trading_loop.py")
EXEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hermes_trader", "agents", "executor.py")

_loop_src = open(LOOP_PATH).read()
_exec_src = open(EXEC_PATH).read()
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


def _new_env(memory):
    """Namespace for the real _process_coin_run body, with every external
    touch point faked. Returns (ns, research_calls)."""
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
        "_research_lock": threading.Lock(),
        "_last_research_by_coin": {},
        "_last_progress_ts": 0.0,
    }
    exec(compile(_extract("_process_coin_run"), LOOP_PATH, "exec"), ns)
    return ns, research_calls


def _run_coin(ns, perception, cfg_cd, held=None):
    ctx = {
        "now_ms": int(time.time() * 1000),
        "held_coins": set(held or []),
        "held_research_ms": 3 * 60_000,
        "cooldown_ms": 30 * 60_000,
        "recent_trades_by_coin": {},
        "blocklist": set(),
        "cfg_cd": cfg_cd,
    }
    ns["_process_coin_run"](perception, ctx)


CFG_MOMENTUM_OFF = {
    "cooldown_min": 30, "loss_cooldown_min": 180,
    "momentum_reentry": {"enabled": False},
    "held_research_interval_min": 3, "min_ai_close_hold_min": 25,
}
CFG_MOMENTUM_ON = {
    "cooldown_min": 30, "loss_cooldown_min": 180,
    "momentum_reentry": {"enabled": True, "reclaim_pct": 1.0,
                         "min_composite": 30},
}


def _forced(perceptions, cfg_cd, held_coins, last_research=None,
            mids=None, raise_mids=False):
    def _get_mids(**kw):
        if raise_mids:
            raise RuntimeError("mid fetch down")
        return mids or {}

    ns = {
        "time": time,
        "logger": _Logger(),
        "_research_lock": threading.Lock(),
        "_last_research_by_coin": dict(last_research or {}),
        "get_all_hl_mids": _get_mids,
    }
    exec(compile(_extract("_forced_held_reeval"), LOOP_PATH, "exec"), ns)
    return ns["_forced_held_reeval"](perceptions, cfg_cd, held_coins)


# ── Phase 1: pre-research loss-cooldown skip (the ZEC fix) ────────────────────

def test_loss_cooldown_not_held_no_paid_research():
    """Not held + 180min loss cooldown active (standard 30min long expired —
    the exact ZEC 07:38 shape, 141min after the arm) → skip, NO research."""
    mem = _FakeMemory()
    mem.cooldown_remaining["ZEC"] = 39.0
    ns, calls = _new_env(mem)
    _run_coin(ns, {"coin": "ZEC", "composite_score": 0, "mid": 790.0,
                   "triggers": [{"name": "bandSnapback", "fired": True}],
                   "whale_signal": None}, CFG_MOMENTUM_OFF)
    assert calls == [], f"research was paid: {calls}"
    assert "LOSS_COOLDOWN" in _ta_skip_signals(mem), _ta_skip_signals(mem)


def test_loss_cooldown_momentum_bypass_preserved():
    """Same, but momentum re-entry ELIGIBLE (price reclaimed above the stop,
    strong composite, last close was a long) → the designed bypass must NOT
    be starved: research runs."""
    mem = _FakeMemory()
    mem.cooldown_remaining["SPCX"] = 100.0
    mem.last_closes["SPCX"] = {"exit_px": 100.0, "side": "long"}
    ns, calls = _new_env(mem)
    _run_coin(ns, {"coin": "SPCX", "composite_score": 60, "mid": 102.0,
                   "triggers": [{"name": "momentumBurst", "fired": True}],
                   "whale_signal": None}, CFG_MOMENTUM_ON)
    assert calls == ["SPCX"], f"momentum bypass was starved: {calls}"


def test_no_cooldowns_research_runs_normally():
    mem = _FakeMemory()
    ns, calls = _new_env(mem)
    _run_coin(ns, {"coin": "FRESH", "composite_score": 40, "mid": 1.0,
                   "triggers": [{"name": "momentumBurst", "fired": True}],
                   "whale_signal": None}, CFG_MOMENTUM_OFF)
    assert calls == ["FRESH"], f"calls={calls}"


def test_held_coin_loss_cooldown_does_not_starve_close_path():
    """A HELD coin with an active loss cooldown must NOT be skipped by the
    new gate (it lives in the not-held branch) — the AI keeps its CLOSE path."""
    mem = _FakeMemory()
    mem.cooldown_remaining["HOLDX"] = 60.0
    ns, calls = _new_env(mem)
    # Past the 3min held-research interval so the HELD_THROTTLE lets it through.
    ns["_last_research_by_coin"]["HOLDX"] = int(time.time() * 1000) - 600_000
    _run_coin(ns, {"coin": "HOLDX", "composite_score": 0, "mid": 5.0,
                   "triggers": [{"name": "bandSnapback", "fired": True}],
                   "whale_signal": None}, CFG_MOMENTUM_OFF, held=["HOLDX"])
    assert calls == ["HOLDX"], f"AI CLOSE path starved: {calls}"
    assert "LOSS_COOLDOWN" not in _ta_skip_signals(mem), _ta_skip_signals(mem)


# ── Phase 2: forced held re-eval ──────────────────────────────────────────────

def test_forced_reval_no_held_coins_no_network():
    assert _forced([], {}, set(), raise_mids=True) == []


def test_forced_reval_triggered_held_coin_not_forced():
    """A held coin that DID trigger this scan is already in the worklist —
    no double pay."""
    assert _forced([{"coin": "ZEC"}], {}, {"ZEC"}, raise_mids=True) == []


def test_forced_reval_quiet_held_coin_forced():
    NOW = int(time.time() * 1000)
    got = _forced([{"coin": "OTHER"}], {}, {"ZEC"},
                  last_research={"ZEC": NOW - 150 * 60_000},
                  mids={"ZEC": 788.5})
    assert len(got) == 1, f"got={got}"
    g = got[0]
    assert g["coin"] == "ZEC" and g["mid"] == 788.5, g
    assert any(t["name"] == "heldReeval" and t.get("reason")
               for t in g["triggers"]), g["triggers"]
    assert g["composite_score"] == 0
    assert g["type"] == "perp"


def test_forced_reval_recently_researched_not_forced():
    NOW = int(time.time() * 1000)
    assert _forced([], {}, {"ZEC"},
                   last_research={"ZEC": NOW - 30 * 60_000},
                   mids={"ZEC": 788.5}) == []


def test_forced_reval_disabled_via_config():
    assert _forced([], {"held_reval_min_age_min": 0}, {"ZEC"},
                   raise_mids=True) == []


def test_forced_reval_mid_failure_swallowed():
    assert _forced([], {}, {"ZEC"},
                   last_research={}, raise_mids=True) == []


# ── Phase 3: source wiring (integration invariants) ───────────────────────────

def test_loop_wiring():
    assert ("from hermes_trader.agents.executor import close_position_market, "
            "maybe_execute, monitor_exits, momentum_reentry_allowed, "
            "route_verdict") in _loop_src
    assert "_forced = _forced_held_reeval(results, _cfg_cd, held_coins)" in _loop_src
    assert "_worklist = list(results) + _forced" in _loop_src
    assert "for perception in _worklist:" in _loop_src
    # The new skip sits AFTER the held branch (held coins keep their CLOSE
    # path) and BEFORE the TA filter.
    i_held = _loop_src.index("if coin in held_coins:")
    i_losscd = _loop_src.index("pre-research loss-cooldown")
    i_ta = _loop_src.index("TA filter — cheap statistical gate")
    assert i_held < i_losscd < i_ta


def test_executor_gate_order():
    """The loss-cooldown gate must fire BEFORE the runner gate: when both
    would block, the trade result says WHY the rule fired, not the quality
    filter that happened to run first (2026-08-26 ZEC: the log read
    runner_gate_blocked while the 180min cooldown was armed — it looked
    broken, it wasn't, just masked)."""
    import re
    m = re.search(r"^def maybe_execute\(", _exec_src, re.M)
    assert m, "maybe_execute not found"
    rest = _exec_src[m.end():]
    m2 = re.search(r"^def ", rest, re.M)
    body = rest[:m2.start()] if m2 else rest
    i_lc = body.index("_lc_remaining = memory.loss_cooldown_remaining_min")
    i_rg = body.index("_runner_entry_block_reason(analysis, config)")
    assert i_lc < i_rg, "loss-cooldown gate must precede the runner gate"