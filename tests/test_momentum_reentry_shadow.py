"""P6: momentum_reentry.shadow_mode — the loss-cooldown bypass becomes
log-only when the flag is on (code default false = current bypass behavior,
a no-op merge). The condition keeps COMPUTING in both states; the accrual
line `[gate][SHADOW] momentum_reentry WOULD BYPASS` is the counterfactual
record. Both call sites (executor cooldown bypass + trading_loop
pre-research skip) consume only the (allow, reason) tuple, so the
cooldown binds at both when shadow is on. Hermetic: pure function + one
maybe_execute-level pin (patched I/O surface, no live LLM/candles)."""

import logging

from hermes_trader.agents.executor import momentum_reentry_allowed

_SHADOW_LINE = "[gate][SHADOW] momentum_reentry WOULD BYPASS"

# Fires the re-entry condition: LONG stop-out at 100, mid +2% above it,
# composite 50 >= 30.
_ON = {"momentum_reentry": {"enabled": True, "reclaim_pct": 1.0,
                            "min_composite": 30}}
_ON_SHADOW = {"momentum_reentry": {"enabled": True, "reclaim_pct": 1.0,
                                   "min_composite": 30, "shadow_mode": True}}
_NO_SHADOW_KEY = {"momentum_reentry": {"enabled": True, "reclaim_pct": 1.0,
                                       "min_composite": 30}}  # == _ON; the
# absence of the shadow_mode key IS the no-op guarantee under test.
_DISABLED = {"momentum_reentry": {"enabled": False, "shadow_mode": True}}


# ── (a) fire + shadow OFF -> (True, reason) ────────────────────────────────

def test_fire_shadow_off_allows_bypass():
    ok, why = momentum_reentry_allowed(100.0, "long", 102.0, 50, _ON)
    assert ok is True
    assert "reclaimed" in why


# ── (b) fire + shadow ON -> (False, ...) + exactly one accrual line ────────

def test_fire_shadow_on_does_not_bypass_and_accrues(caplog):
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        ok, why = momentum_reentry_allowed(100.0, "long", 102.0, 50,
                                           _ON_SHADOW, coin="SPCX")
    assert ok is False
    assert "reclaimed" in why  # reason still carried for future debug
    shadow_lines = [r for r in caplog.records
                    if _SHADOW_LINE in r.getMessage()]
    assert len(shadow_lines) == 1, caplog.text
    assert "SPCX" in shadow_lines[0].getMessage()
    assert "cooldown still binds" in shadow_lines[0].getMessage()


# ── (c) no-fire: (False, "") in BOTH shadow states, no accrual line ────────

def test_no_fire_shadow_on_is_silent(caplog):
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        assert momentum_reentry_allowed(100.0, "long", 98.0, 50, _ON_SHADOW,
                                        coin="ZEC") == (False, "")
        assert momentum_reentry_allowed(100.0, "long", 100.5, 50, _ON_SHADOW,
                                        coin="ZEC") == (False, "")
    assert not [r for r in caplog.records if _SHADOW_LINE in r.getMessage()], \
        caplog.text


def test_no_fire_shadow_off():
    assert momentum_reentry_allowed(100.0, "long", 98.0, 50, _ON) == (False, "")
    assert momentum_reentry_allowed(100.0, "long", 100.5, 50, _ON) == (False, "")
    assert momentum_reentry_allowed(100.0, "long", 102.0, 20, _ON) == (False, "")


# ── (d) shadow key ABSENT -> byte-identical to pre-change behavior ─────────

def test_shadow_key_absent_is_noop():
    # The pre-change function: same call, same expected tuple. shadow_mode
    # absent from config must behave exactly as before (bypass allowed).
    assert _NO_SHADOW_KEY.get("momentum_reentry", {}).get("shadow_mode", False) is False
    ok, why = momentum_reentry_allowed(100.0, "long", 102.0, 50, _NO_SHADOW_KEY)
    assert (ok, why) == (True, "reclaimed +2.0% above stop 100, composite 50")


# ── (e) enabled=false + shadow ON -> (False, ""), shadow never evaluated ───

def test_disabled_short_circuits_shadow(caplog):
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        assert momentum_reentry_allowed(100.0, "long", 102.0, 50, _DISABLED,
                                        coin="TON") == (False, "")
    assert not [r for r in caplog.records if _SHADOW_LINE in r.getMessage()], \
        caplog.text


# ── (f) EXECUTION-LEVEL: the cooldown STILL BINDS at maybe_execute ─────────
# The TODO's requirement that shadow mode leaves the cooldown in force: a
# coin in active loss-cooldown whose conditions WOULD fire the re-entry,
# with shadow_mode=true, must come back with the loss_cooldown block reason
# (not executed, not bypassed) AND accrue the shadow line.

def _exec_baseline(monkeypatch, cfg_overrides=None, state_overrides=None):
    """Patch executor's I/O surface with sane defaults (same pattern as
    tests/test_cleanup.py::_exec_baseline); return (executor, captured)."""
    from hermes_trader.agents import executor
    cfg = {
        "mode": "LIVE", "enable_crypto": True, "enable_hip3": True,
        "equity_fraction_per_trade": 0.10, "leverage": 10,
        "max_trade_notional_usd": 100000, "max_concurrent": 18,
        "max_total_notional_pct": 40.0,
        "min_available_margin_pct": 0.10, "cooldown_min": 60,
        "min_ai_confidence": 0.30, "counter_regime_min_conf": 0.65,
        "max_crypto_long_correlated": 5, "min_market_volume_usd": 5_000_000,
        "min_hip3_volume_usd": 500_000, "conviction_sizing": True,
        "dsl_exit": {"max_loss_pct": 2.0, "max_loss_roe_pct": 30.0,
                     "protect_pct": 0.5, "retrace_threshold": 0.3,
                     "hard_timeout_minutes": 180.0},
    }
    cfg.update(cfg_overrides or {})
    state = {"equity": 1000.0, "available": 500.0, "total_ntl": 0.0,
             "asset_positions": []}
    state.update(state_overrides or {})
    captured = {}

    monkeypatch.setattr(executor, "read_agent_config", lambda: cfg)
    monkeypatch.setattr(executor, "resolve_user_address", lambda: "0xMASTER")
    monkeypatch.setattr(executor, "fetch_account_state", lambda u, **kw: state)
    monkeypatch.setattr(executor, "get_hl_price", lambda c: 100.0)
    monkeypatch.setattr(executor, "get_hl_atr", lambda *a, **k: 2.0)
    monkeypatch.setattr(executor, "get_max_leverage", lambda c: 40)
    monkeypatch.setattr(executor, "min_entry_notional_usd", lambda c, mid: 10.5)
    monkeypatch.setattr(executor, "entry_size_for_notional", lambda c, n, mid: n / mid)
    monkeypatch.setattr(executor, "set_leverage", lambda c, l: {"ok": True})
    monkeypatch.setattr(executor, "place_hl_trigger_order", lambda *a, **k: {"ok": True})
    monkeypatch.setattr("hermes_trader.client.hl_client._http_post",
                        lambda p, pl: {"marginSummary": {"accountValue": "500"}})
    monkeypatch.setattr("hermes_trader.agents.market_regime.detect_regime", lambda c: "neutral")
    monkeypatch.setattr("hermes_trader.agents.hyperfeed.market_get_funding_regime",
                        lambda: {"regime": "NEUTRAL", "regimes_by_class": {}})

    def _place(is_buy, size, mid, coin):
        captured["is_buy"] = is_buy
        captured["size"] = size
        captured["coin"] = coin
        return {"ok": True, "order_id": "OID1", "avg_px": mid}
    monkeypatch.setattr(executor, "place_hl_order", _place)
    monkeypatch.setattr(executor, "register_position", lambda *a, **k: None)
    monkeypatch.setattr(executor.memory, "track_daily_pnl", lambda *a, **k: None)
    monkeypatch.setattr(executor.memory, "get_daily_pnl", lambda: 0.0)
    monkeypatch.setattr(executor.memory, "get_recent_trades", lambda n=10: [])
    monkeypatch.setattr(executor.memory, "record_trade", lambda t: None)
    monkeypatch.setenv("HYPERLIQUID_PRIVATE_KEY", "0xabc")
    return executor, captured, cfg


def _analysis(**kw):
    base = {"id": "a1", "coin": "TON", "verdict": "LONG", "side": "long",
            "confidence": 0.70, "composite_score": 50, "mid": 102.0,
            "entry_px": 100, "stop_px": 95, "tp_px": 110,
            "news_context": "no news"}
    base.update(kw)
    return base


def _arm_cooldown_and_last_close(monkeypatch, executor, coin, remaining_min,
                                 exit_px, side):
    """Stub the cooldown READ (not the singleton's internal dicts) so the
    test is immune to suite-order pollution: test_parallel_research_race
    permanently overwrites the shared singleton's methods (bare attribute
    assignment, never restored), so poking `_cooldowns`/`_closes` directly
    is unreliable at this suite position. The established pattern for
    exactly this hazard is test_daily_kill_and_stop_cooldown."""
    monkeypatch.setattr(executor.memory, "loss_cooldown_remaining_min",
                        lambda c: float(remaining_min) if c == coin else 0.0)
    monkeypatch.setattr(executor.memory, "last_close_for",
                        lambda c: ({"exit_px": exit_px, "side": side}
                                   if c == coin else None))


def test_maybe_execute_shadow_on_cooldown_still_binds(monkeypatch, caplog):
    """Shadow ON + firing re-entry condition -> NOT executed, loss_cooldown
    reason, and the accrual line fires at the execution call site."""
    ex, captured, _ = _exec_baseline(
        monkeypatch,
        cfg_overrides={"momentum_reentry": {
            "enabled": True, "reclaim_pct": 1.0, "min_composite": 30,
            "shadow_mode": True}})
    _arm_cooldown_and_last_close(monkeypatch, ex, "TON", 180.0, 100.0, "long")
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        res = ex.maybe_execute(_analysis())
    assert res["executed"] is False, res
    assert "loss_cooldown" in res["reason"], res
    assert not captured  # no order was placed — the cooldown bound
    shadow_lines = [r for r in caplog.records
                    if _SHADOW_LINE in r.getMessage()]
    assert len(shadow_lines) == 1, caplog.text
    assert "TON" in shadow_lines[0].getMessage()


def test_maybe_execute_shadow_off_cooldown_bypassed(monkeypatch):
    """Shadow OFF (default) + same firing condition -> the pre-change bypass
    stands: the trade proceeds past the cooldown gate all the way to order
    placement (mocked) — no loss_cooldown block."""
    ex, captured, _ = _exec_baseline(
        monkeypatch,
        cfg_overrides={"momentum_reentry": {
            "enabled": True, "reclaim_pct": 1.0, "min_composite": 30}})
    _arm_cooldown_and_last_close(monkeypatch, ex, "TON", 180.0, 100.0, "long")
    res = ex.maybe_execute(_analysis())
    assert res["executed"] is True, res
    assert captured.get("coin") == "TON", (res, captured)


def test_maybe_execute_call_site_passes_coin_kwarg(monkeypatch):
    """Call-site pin (executor cooldown branch): the _mr_ok=False path the
    shadow return produces leads to the loss_cooldown dict — with the coin
    name propagated into the accrual line, proving coin=analysis['coin']
    is wired at the call site."""
    from hermes_trader.agents import executor
    from unittest.mock import patch as _patch
    ex, _, _ = _exec_baseline(
        monkeypatch,
        cfg_overrides={"momentum_reentry": {
            "enabled": True, "reclaim_pct": 1.0, "min_composite": 30,
            "shadow_mode": True}})
    _arm_cooldown_and_last_close(monkeypatch, ex, "TON", 180.0, 100.0, "long")
    with _patch.object(executor, "momentum_reentry_allowed",
                       wraps=executor.momentum_reentry_allowed) as spied:
        res = ex.maybe_execute(_analysis())
    assert res["executed"] is False and "loss_cooldown" in res["reason"]
    assert spied.call_count == 1
    _, kw = spied.call_args
    assert kw.get("coin") == "TON", f"call site did not pass coin: {spied.call_args}"


# ── (g) PRE-RESEARCH-SKIP level: function-level pin ───────────────────────
# The trading_loop skip branch is `if not _mr_ok: ... return` — with shadow
# on the function returns (False, ...), so the skip branch applies (the paid
# LLM research happens, then is blocked at execution). Full integration test
# not required; the skip branch consumes only the tuple.

def test_pre_research_skip_sees_false_when_shadow_on():
    """Same firing inputs the trading_loop pre-research call site would pass
    (perception mid + composite vs last close), shadow ON -> (False, ...)
    which is exactly what the skip branch's `if not _mr_ok` keys on."""
    # trading_loop.py: _mr_ok, _mr_why = momentum_reentry_allowed(
    #     _last_close.get("exit_px"), _last_close.get("side"),
    #     perception.get("mid"), score, _cfg_cd, coin=coin)
    _last_close = {"exit_px": 100.0, "side": "long"}
    perception = {"mid": 102.0}
    score = 50
    ok, why = momentum_reentry_allowed(
        _last_close.get("exit_px"), _last_close.get("side"),
        perception.get("mid"), score, _ON_SHADOW, coin="SPCX")
    assert not ok  # `if not _mr_ok:` -> skip branch applies
    assert "reclaimed" in why  # reason carried, unused by the skip branch
    # and with shadow off the same inputs return ok=True (no skip)
    ok_off, _ = momentum_reentry_allowed(
        _last_close.get("exit_px"), _last_close.get("side"),
        perception.get("mid"), score, _ON, coin="SPCX")
    assert ok_off
