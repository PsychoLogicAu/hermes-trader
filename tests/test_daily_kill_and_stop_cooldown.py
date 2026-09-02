"""Equity-relative daily kill + halt timer, and stop-specific loss cooldown."""
import time

import pytest

from hermes_trader.agents.risk_gates import (
    GateContext, daily_loss_kill_switch, effective_daily_kill_usd,
    flatten_daily_kill_usd)


def _ctx(daily_pnl=0.0, equity=100.0):
    return GateContext(
        confidence=0.9, current_positions=[], trade_notional_usd=30,
        daily_pnl=daily_pnl, market_volume_24h_usd=1e8, coin="ETH",
        trade_side="long", has_binary_news_risk=False, equity=equity,
        total_open_notional=0.0)


# ── effective_daily_kill_usd ────────────────────────────────────────────────

def test_pct_of_equity_basic():
    cfg = {"daily_kill_pct_of_equity": 0.10, "daily_kill_cap_usd": 100,
           "daily_kill_min_usd": 8}
    assert effective_daily_kill_usd(cfg, 90.0) == pytest.approx(9.0)
    assert effective_daily_kill_usd(cfg, 500.0) == pytest.approx(50.0)


def test_cap_binds_on_big_equity():
    cfg = {"daily_kill_pct_of_equity": 0.10, "daily_kill_cap_usd": 100,
           "daily_kill_min_usd": 8}
    # 10% of $1M = $100k but the cap holds it at $100
    assert effective_daily_kill_usd(cfg, 1_000_000.0) == 100.0


def test_floor_binds_on_small_equity():
    cfg = {"daily_kill_pct_of_equity": 0.10, "daily_kill_cap_usd": 100,
           "daily_kill_min_usd": 8}
    # 10% of $30 = $3 but the floor keeps the brake at $8 (noise guard)
    assert effective_daily_kill_usd(cfg, 30.0) == 8.0


def test_disabled_when_pct_unset_or_zero():
    assert effective_daily_kill_usd({"daily_kill_pct_of_equity": 0}, 500.0) == 0.0
    assert effective_daily_kill_usd({}, 500.0) == 0.0  # pct unset → disabled
    # gate passes outright when the kill is disabled, even deep in the red
    r = daily_loss_kill_switch(_ctx(daily_pnl=-25.0), 0.0)
    assert r["pass"] is True


# ── flatten_daily_kill_usd (hard flatten = mult × halt threshold) ──────────

def test_flatten_is_125x_the_halt_threshold_by_default():
    base = {"daily_kill_pct_of_equity": 0.20}
    assert effective_daily_kill_usd(base, 100.0) == 20.0
    assert flatten_daily_kill_usd(base, 100.0) == 25.0  # 1.25 × 20


def test_flatten_mult_overridable():
    cfg = {"daily_kill_pct_of_equity": 0.20, "daily_kill_flatten_mult": 1.5}
    assert flatten_daily_kill_usd(cfg, 100.0) == 30.0


def test_flatten_mult_one_restores_flat_behaviour():
    cfg = {"daily_kill_pct_of_equity": 0.20, "daily_kill_flatten_mult": 1.0}
    assert flatten_daily_kill_usd(cfg, 100.0) == effective_daily_kill_usd(cfg, 100.0)


def test_flatten_disabled_with_kill_switch():
    assert flatten_daily_kill_usd({}, 500.0) == 0.0
    assert flatten_daily_kill_usd({"daily_kill_pct_of_equity": 0}, 500.0) == 0.0


def test_flatten_respects_cap_and_floor():
    # base clamps to the $100 cap on a big account; flatten = 1.25 × capped
    cfg = {"daily_kill_pct_of_equity": 0.20, "daily_kill_cap_usd": 100}
    assert flatten_daily_kill_usd(cfg, 10_000.0) == 125.0
    # floor binds on a tiny account; flatten = 1.25 × floored
    cfg = {"daily_kill_pct_of_equity": 0.20, "daily_kill_min_usd": 8}
    assert flatten_daily_kill_usd(cfg, 10.0) == 10.0  # 8 × 1.25


# ── daily_loss_kill_switch ──────────────────────────────────────────────────

def test_equity_relative_positive_shape_blocks():
    # threshold +9 means block when day PnL <= -9
    assert daily_loss_kill_switch(_ctx(daily_pnl=-9.5), 9.0)["pass"] is False
    assert daily_loss_kill_switch(_ctx(daily_pnl=-8.9), 9.0)["pass"] is True


def test_halt_timer_blocks_even_after_partial_recovery():
    # day recovered to -4 (above the -9 block line) but the halt timer set
    # at the breach still runs -> entries stay blocked
    r = daily_loss_kill_switch(_ctx(daily_pnl=-4.0), 9.0, halt_remaining_min=120)
    assert r["pass"] is False
    assert "halt active" in r["reason"]


# ── halt timer arm / clear / expiry (memory) ───────────────────────────────

def test_halt_timer_arm_extend_clear(monkeypatch):
    from hermes_trader.agents.memory import AgentMemory
    m = AgentMemory.__new__(AgentMemory)
    m._daily_halt_until = 0
    monkeypatch.setattr(m, "flush", lambda: None)
    assert m.daily_halt_remaining_min() == 0.0
    m.arm_daily_halt(360)
    assert m.daily_halt_remaining_min() == pytest.approx(360, abs=1)
    # extend-or-set never shortens
    m.arm_daily_halt(60)
    assert m.daily_halt_remaining_min() > 300
    m.clear_daily_halt()
    assert m.daily_halt_remaining_min() == 0.0
    # expired timers read as 0
    m._daily_halt_until = int(time.time() * 1000) - 1000
    assert m.daily_halt_remaining_min() == 0.0


# ── stop-specific cooldown arming (close path) ─────────────────────────────

def test_max_loss_arms_longer_cooldown(monkeypatch):
    """close_position_market-equivalent arm logic: max_loss close picks
    max(loss_cooldown_min, stop_loss_cooldown_min); other exits keep the plain."""
    from hermes_trader.agents import executor as ex
    cfg = {"loss_cooldown_min": 180, "stop_loss_cooldown_min": 360}
    for reason, want in (("max_loss (5.2% spot / 15.6% ROE)", 360.0),
                         ("floor_breach", 180.0),
                         ("stale_flat_timeout", 180.0)):
        lc = float(cfg["loss_cooldown_min"])
        if ex._exit_type(reason) == "max_loss":
            lc = max(lc, float(cfg["stop_loss_cooldown_min"]))
        assert lc == want


def test_stop_cooldown_blocks_reentry(monkeypatch):
    """An armed stop cooldown blocks re-entry the same way the plain one does.

    Stubs the cooldown READ (not the singleton dict) so the test is immune to
    the suite-order memory.load() pollution that makes the TON twin flaky."""
    from hermes_trader.agents import executor as ex
    monkeypatch.setattr(
        ex, "read_agent_config",
        lambda: {"mode": "LIVE", "enable_crypto": True, "loss_cooldown_min": 180,
                 "stop_loss_cooldown_min": 360})
    monkeypatch.setattr(ex.memory, "flush", lambda: None)
    monkeypatch.setattr(ex.memory, "loss_cooldown_remaining_min",
                        lambda coin: 300.0 if coin == "PURR" else 0.0)
    monkeypatch.setattr(ex.memory, "last_close_for", lambda coin: {})
    res = ex.maybe_execute({"id": "t9", "coin": "PURR", "verdict": "LONG",
                            "side": "long", "confidence": 0.9})
    assert res["executed"] is False
    assert "loss_cooldown" in res["reason"]
    assert "300min" in res["reason"]  # remaining time reported


# ── close-armed standard cooldown (cooldown_floor_ts_by_coin) ───────────────

def _floor_mem():
    from hermes_trader.agents.memory import AgentMemory
    m = AgentMemory.__new__(AgentMemory)
    m._trades = []
    m._closes = []
    return m


def test_floor_map_prefers_close_when_later():
    m = _floor_mem()
    m._trades = [{"coin": "ETH", "executed_at": 1_000_000}]
    m._closes = [{"coin": "ETH", "closed_at": 5_000_000}]
    assert m.cooldown_floor_ts_by_coin()["ETH"] == 5_000_000


def test_floor_map_prefers_open_fill_when_position_still_hot():
    m = _floor_mem()
    m._trades = [{"coin": "ETH", "executed_at": 9_000_000}]
    m._closes = [{"coin": "ETH", "closed_at": 5_000_000}]  # older close
    assert m.cooldown_floor_ts_by_coin()["ETH"] == 9_000_000


def test_floor_map_includes_open_only_and_close_only_coins():
    m = _floor_mem()
    m._trades = [{"coin": "OPENX", "executed_at": 3_000_000}]
    m._closes = [{"coin": "CLOSEX", "closed_at": 4_000_000}]
    out = m.cooldown_floor_ts_by_coin()
    assert out["OPENX"] == 3_000_000 and out["CLOSEX"] == 4_000_000


# ── halt survives the UTC roll; early-release only on the armed day ─────────

def test_halt_day_recorded_and_cleared(monkeypatch):
    from hermes_trader.agents.memory import AgentMemory
    m = AgentMemory.__new__(AgentMemory)
    m._daily_halt_until = 0
    m._daily_halt_day = ""
    monkeypatch.setattr(m, "flush", lambda: None)
    m.arm_daily_halt(360, utc_day="2026-09-02")
    assert m.daily_halt_armed_day() == "2026-09-02"
    m.clear_daily_halt()
    assert m.daily_halt_armed_day() == ""


def test_expired_halt_has_no_armed_day(monkeypatch):
    from hermes_trader.agents.memory import AgentMemory
    m = AgentMemory.__new__(AgentMemory)
    m._daily_halt_until = int(time.time() * 1000) - 60_000
    m._daily_halt_day = "2026-09-01"
    assert m.daily_halt_armed_day() == ""


def test_utc_roll_no_longer_clears_halt_until():
    """The 23:55-breach case: a halt must run to its own expiry, NOT be
    laundered by the UTC day rolling. track_daily_pnl's roll branch must
    leave _daily_halt_until untouched."""
    import datetime as _d
    from hermes_trader.agents.memory import AgentMemory
    m = AgentMemory.__new__(AgentMemory)
    m._daily_halt_until = int(time.time() * 1000) + 3600_000
    m._daily_halt_day = "2026-09-01"
    m._last_eq_reading = 0.0
    m._last_eq_reading_ts = 0.0
    m._start_of_day_equity = 100.0
    m._day_start_ts = 0            # ancient → forces the roll branch
    m._daily_pnl = -99.0
    m._peak_daily_pnl = 0.0
    m._equity = 0.0
    m.flush = lambda: None
    m.track_daily_pnl(91.0)        # new UTC day begins
    assert m._daily_halt_until > int(time.time() * 1000)  # halt SURVIVED
    assert m._daily_pnl == 0.0                            # day pnl did reset
