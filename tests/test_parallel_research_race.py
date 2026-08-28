"""Race test for the parallel-research execution lock.

Proves the financial-safety invariant behind research_max_workers > 1: two
concurrent maybe_execute() calls must NOT both open a position when only one
capital slot is free.

Two phases:
  1. LOCK-FREE demonstration — the underlying function (unwrap via
     __wrapped__, set by functools.wraps) is called concurrently with the
     max_concurrent gate seeing an empty book; both calls place an order.
     This is the bug the lock exists to fix.
  2. LOCKED path — the same concurrent call through the public (serialized)
     maybe_execute places exactly ONE order: the first call registers its DSL
     position, and the second call's gate sees it via the DSL re-entry backstop
     merge and is blocked by max_concurrent.

Also covers the research_max_workers clamp (compute_research_workers) and the
trading_loop integration (helper import + call site).

NO network, NO exchange calls — every external touch point is monkeypatched.
Runs standalone (no pytest): python3 tests/test_parallel_research_race.py
pytest-compatible too (CI runs the tests/ suite).
"""
import os
import sys
import tempfile
import threading
import time
import uuid

# ── Test isolation (mirror tests/conftest.py) — MUST run before hermes imports
_tmp = tempfile.mkdtemp(prefix="hermes-test-parallel-")
os.environ["HERMES_AGENT_MEMORY_FILE"] = os.path.join(_tmp, ".agent-memory.json")
os.environ["HERMES_AGENT_CONFIG_FILE"] = os.path.join(_tmp, ".agent-config.json")
os.environ["HERMES_DSL_STATE_FILE"] = os.path.join(_tmp, ".dsl-state.json")
os.environ["SESSION_LOG_PATH"] = os.path.join(_tmp, "session.jsonl")
os.environ.pop("HYPERLIQUID_PRIVATE_KEY", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hermes_trader.agents.executor as ex  # noqa: E402
from hermes_trader.agents import dsl_exit  # noqa: E402
from hermes_trader.agents import market_regime as _mr  # noqa: E402
from hermes_trader.agents.risk_gates import eval_all_gates  # noqa: E402
from hermes_trader.agents.research_concurrency import compute_research_workers  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


# ── Mocks ─────────────────────────────────────────────────────────────────────
orders_placed = []
order_lock = threading.Lock()


def _cfg():
    return {
        "mode": "LIVE",
        "enable_crypto": True,
        "enable_hip3": False,
        "max_concurrent": 1,            # one capital slot — the race window
        "min_ai_confidence": 0.8,
        "max_trade_notional_usd": 200,
        "max_daily_loss_usd": -100,
        "daily_giveback_halt_pct": 0.0,
        "min_market_volume_usd": 5_000_000,
        "min_short_volume_usd": 0,
        "coin_allowlist": [],
        "coin_blocklist": [],
        "max_crypto_long_correlated": 2,
        "max_total_notional_pct": 1.0,
        "counter_regime_min_conf": 0.7,
        "block_counter_trend_bypass": False,
        "crowded_with_min_conf": 0.0,
        "cooldown_min": 60,
        "runner_entry_gate": {},        # disabled → _runner_entry_block_reason early-exit
        "capital_rotation": {},         # disabled
        "leverage": 3,
        "equity_fraction_per_trade": 0.01,
        "conviction_sizing": True,
        "min_available_margin_pct": 0.10,
        "atr_risk_sizing": {"enabled": False},
        "sl_atr_mult": 1.5,
        "tp_scale_fraction": 0.5,
        "shadow_signals": {},
        "whale_force_execute": False,
        "whale_regime_bypass": False,
    }


def _install_mocks():
    """Point every external touch point at in-memory fakes. Idempotent."""
    # Config — read_agent_config() is called fresh inside maybe_execute per call
    # (this is also what makes the research_max_workers knob hot).
    ex.read_agent_config = lambda: _cfg()
    # Gate inputs — no network
    ex._runner_entry_block_reason = lambda a, c: ""
    ex.memory.loss_cooldown_remaining_min = lambda coin: 0.0
    ex.memory.last_close_for = lambda coin: {}
    ex.memory.get_recent_trades = lambda n: []
    ex.memory.latest_trade_ts_by_coin = lambda n: {}
    ex.memory.peak_daily_pnl = lambda: 0.0
    ex.resolve_user_address = lambda: "0xTEST"
    # Account state: EMPTY book, $1000 equity — the shared pre-trade view both
    # racing threads see. The lock must make the second thread see the DSL
    # backstop instead.
    ex.fetch_account_state = lambda user, **kw: {
        "equity": 1000.0, "available": 900.0, "spot_usdc": 0.0,
        "asset_positions": [], "total_ntl": 0.0,
        "dex_equity": {"": 1000.0}, "dex_available": {"": 900.0},
    }
    ex._get_market_volume_24h = lambda coin: 1e8
    ex.get_max_leverage = lambda coin: 10
    ex.get_hl_price = lambda coin: 100.0
    ex.get_hl_atr = lambda *a, **k: 0.5
    ex.min_entry_notional_usd = lambda coin, px: 10.0
    ex.entry_size_for_notional = lambda coin, notional, px: round(notional / px, 4)
    ex.set_leverage = lambda coin, lev: None
    # Orders — the thing under test. The sleep widens the race window: every
    # caller that PASSES the gate sits here before register_position() runs, so
    # a lock-free caller still can't see the other thread's DSL registration.
    def _place(is_buy, size, mid, coin, **kw):
        time.sleep(0.2)
        with order_lock:
            orders_placed.append(coin)
        return {"ok": True, "order_id": f"oid-{len(orders_placed)}",
                "avg_px": mid, "total_sz": size}
    ex.place_hl_order = _place
    ex.place_hl_trigger_order = lambda *a, **k: {"ok": True}
    ex.cancel_open_orders_for_coin = lambda coin: None
    # Ledger — must not append test rows to the live trades.jsonl
    ex.record_open = lambda **kw: None
    ex.record_close = lambda **kw: None
    # Regime — neutral, no funding crowd
    _mr.detect_regime = lambda coin, **kw: "neutral"
    ex._attach_chronos_to_result = (
        lambda result, coin, side: result.update(
            chronos_median_pct=None, chronos_aligned=None, chronos_error="test")
    )
    # Gate-side Chronos read: maybe_execute calls get_chronos_signal_sync
    # (warm-cache / bounded compute). Stub it — the test must not fetch
    # candles or load the model. Error signal = "no usable forecast", so
    # both chronos shadow gates pass with no opinion.
    class _NoopSig:
        median_pct = None
        q10_path_pct = None
        q90_path_pct = None
        error = "test"
    ex.get_chronos_signal_sync = lambda coin, side: _NoopSig()
    os.environ["HYPERLIQUID_PRIVATE_KEY"] = "0xTEST"


def _fresh_registry():
    dsl_exit._active_positions.clear()
    dsl_exit._loaded_from_disk = False
    orders_placed.clear()


def _analysis(coin):
    return {
        "id": str(uuid.uuid4()),
        "coin": coin, "verdict": "LONG", "side": "long",
        "confidence": 0.9, "composite_score": 60.0,
        "entry_px": 100.0, "stop_px": 99.0, "tp_px": 105.0,
        "reasoning": "test setup", "news_risk": "none", "news_context": "",
    }


def _run_concurrent(fn):
    """Fire maybe_execute concurrently on two coins; return the results."""
    _fresh_registry()
    results = [None, None]
    barrier = threading.Barrier(2)

    def _worker(i, coin):
        barrier.wait()  # release both threads at the same instant
        results[i] = fn(_analysis(coin))

    t1 = threading.Thread(target=_worker, args=(0, "TESTA"))
    t2 = threading.Thread(target=_worker, args=(1, "TESTB"))
    t1.start(); t2.start()
    t1.join(timeout=30); t2.join(timeout=30)
    assert not t1.is_alive() and not t2.is_alive(), "worker thread hung (deadlock?)"
    return results


_install_mocks()

print("== Phase 1: WITHOUT the execution lock (the race the lock fixes) ==")
r1 = _run_concurrent(ex.maybe_execute.__wrapped__)
n1 = len(orders_placed)
check("lock-free: both concurrent calls open (demonstrates the TOCTOU)",
      n1 == 2, f"orders={orders_placed} (expected 2 — the double-fire bug)")
check("lock-free: neither call reports blocked",
      all(r and "blocked_by" not in r for r in r1), f"results={r1}")

print("== Phase 2: WITH the execution lock (public maybe_execute) ==")
r2 = _run_concurrent(ex.maybe_execute)
n2 = len(orders_placed)
check("locked: exactly ONE order placed", n2 == 1, f"orders={orders_placed}")
blocked = [r for r in r2 if r and "blocked_by" in r]
check("locked: the loser is blocked by max_concurrent (not dropped)",
      len(blocked) == 1
      and any("max positions reached" in s for s in blocked[0]["blocked_by"]),
      f"results={[ (r.get('blocked_by') or r.get('reason')) for r in r2]}")
check("locked: winner executed cleanly",
      any(r and r.get("executed") for r in r2), f"results={r2}")
check("locked: winner registered a DSL position",
      len(dsl_exit._active_positions) == 1,
      f"registry={list(dsl_exit._active_positions)}")
print(f"   -> loser saw the DSL backstop: {blocked[0]['blocked_by'] if blocked else 'n/a'}")

print("== Phase 3: research_max_workers knob clamping ==")
check("absent -> 1 (sequential default)", compute_research_workers({}, 37) == 1)
check("4 with 37 triggers -> 4", compute_research_workers({"research_max_workers": 4}, 37) == 4)
check("4 with 2 triggers -> 2 (can't exceed coin count)",
      compute_research_workers({"research_max_workers": 4}, 2) == 2)
check("0 -> 1 (floor)", compute_research_workers({"research_max_workers": 0}, 37) == 1)
check("-3 -> 1 (floor)", compute_research_workers({"research_max_workers": -3}, 37) == 1)
check("null -> 1", compute_research_workers({"research_max_workers": None}, 37) == 1)
check("string '4' -> 4", compute_research_workers({"research_max_workers": "4"}, 37) == 4)
check("garbage 'abc' -> 1 (no exception)",
      compute_research_workers({"research_max_workers": "abc"}, 37) == 1)
check("float 4.9 -> 4 (int truncation)",
      compute_research_workers({"research_max_workers": 4.9}, 37) == 4)
check("0 triggers -> 1 (safe floor)", compute_research_workers({"research_max_workers": 4}, 0) == 1)

print("== Phase 4: trading_loop integration ==")
loop_src = open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "trading_loop.py")).read()
check("loop imports compute_research_workers",
      "from hermes_trader.agents.research_concurrency import compute_research_workers"
      in loop_src)
check("loop call site uses the helper with the hot-read config dict",
      "compute_research_workers(_cfg_cd, _n_research)" in loop_src)
check("loop no longer inlines the int() clamp (helper owns it)",
      'int(_cfg_cd.get("research_max_workers"' not in loop_src)
check("loop keeps the sequential fast-path for workers==1",
      "if _workers == 1:" in loop_src and "_process_coin(perception, _ctx)" in loop_src)
check("loop uses ThreadPoolExecutor + as_completed",
      "ThreadPoolExecutor(" in loop_src and "as_completed(" in loop_src)
check("loop keeps the watchdog bump after the pool",
      "_last_progress_ts = time.time()  # watchdog: a full cycle completed" in loop_src)

# gate-level sanity: max_concurrent is the gate that does the blocking, and the
# DSL backstop feed (active_position_coins) is what makes the second caller see
# the first caller's position.
from hermes_trader.agents.risk_gates import GateContext, max_concurrent_positions_gate
ctx_full = GateContext(confidence=0.9,
                       current_positions=[{"coin": "TESTA", "side": "long", "size_usd": 0}],
                       trade_notional_usd=30, daily_pnl=0,
                       market_volume_24h_usd=1e8, coin="TESTB", trade_side="long",
                       has_binary_news_risk=False, equity=1000, total_open_notional=0)
check("gate: 1 tracked position with max_concurrent=1 blocks a second open",
      max_concurrent_positions_gate(ctx_full, 1)["pass"] is False)
check("gate: empty book with max_concurrent=1 allows an open",
      max_concurrent_positions_gate(GateContext(
          confidence=0.9, current_positions=[], trade_notional_usd=30,
          daily_pnl=0, market_volume_24h_usd=1e8, coin="TESTB",
          trade_side="long", has_binary_news_risk=False, equity=1000,
          total_open_notional=0), 1)["pass"] is True)

print()
print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL PASS")