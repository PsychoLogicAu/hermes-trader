"""T2.1/T2.2 — maintenance-aware liquidation bound on the server-side backup SL.

Hermetic (pure math + in-memory, no network). Pins the liq invariant:
  true isolated liq distance (frac of entry)  = 1/lev - maint
  maint                                      = 1/(2 * coin_max_leverage)
  a stop of width `stop` is SAFE iff         stop <= liq_safety_frac * (1/lev - maint)

The DSL primary floor (max_loss spot) is the binding NORMAL exit and is inside
liq for every live coin class; the WIDER 1.5x-ATR backup SL is the only stop
that can sit OUTSIDE liq (high-ATR / low-maxLeverage coin). `liq_safe_leverage`
caps leverage to keep that SL reachable. These tests pin the audit conclusion
and the cap behavior so a regression (e.g. a stop silently placed outside liq)
fails loudly.
"""
import pytest

from hermes_trader.agents.executor import (
    _DEFAULT_LIQ_SAFETY_FRAC,
    _DEFAULT_SL_ATR_MULT,
    liq_safe_leverage,
)

# Live geometry baseline (2026-09-08): config leverage 5 (clamped per-coin by
# min(5, maxLev)); DSL floor = min(max_loss_pct=5.0, max_loss_roe_pct/lev) =
# 5% spot for every lev in [2,5]; ATR stop disabled. backup SL = 1.5x ATR.
LIV_LEV_CFG = 5
DSL_SPOT_FLOOR_PCT = 5.0
ROE_PCT = 30.0
SAFETY = _DEFAULT_LIQ_SAFETY_FRAC  # 0.85

# (label, coin_max_leverage) — representative classes from the audit.
COIN_CLASSES = [
    ("high maxLev 20", 20),
    ("high maxLev 50", 50),
    ("mid maxLev 10", 10),
    ("low maxLev 5", 5),
    ("very low maxLev 3", 3),
    ("very low maxLev 2", 2),
]


def _maint(max_lev):
    return 1.0 / (2.0 * max_lev)


def _true_liq(lev, max_lev):
    """Maintenance-aware isolated liq distance (fraction of entry)."""
    return 1.0 / lev - _maint(max_lev)


def _live_lev(max_lev):
    return min(LIV_LEV_CFG, max_lev)


def _dsl_floor_frac(lev):
    """Live DSL primary floor as a fraction of entry (ATR stop disabled)."""
    return min(DSL_SPOT_FLOOR_PCT, ROE_PCT / lev) / 100.0


# ── the maintenance-aware liq distance is real (naive 1/lev overstates it) ──
def test_naive_one_over_lev_overstates_liq_on_low_maxlev_coin():
    # maxLev-5 coin at 5x: naive 1/lev = 20%, true liq = 1/5 - 1/10 = 10%.
    assert _true_liq(5, 5) == pytest.approx(0.10)
    assert _true_liq(5, 5) < 1.0 / 5.0
    # On a high-maxLev coin the two nearly agree (maint negligible).
    assert _true_liq(5, 50) == pytest.approx(0.20 - 0.01)


# ── AUDIT CONCLUSION A: the DSL floor is inside liq for EVERY class ─────────
def test_dsl_floor_inside_liq_for_every_coin_class():
    for _label, max_lev in COIN_CLASSES:
        lev = _live_lev(max_lev)
        safe_bound = SAFETY * _true_liq(lev, max_lev)
        dsl = _dsl_floor_frac(lev)
        assert dsl <= safe_bound, (
            f"DSL floor {dsl:.3f} outside safety bound {safe_bound:.3f} "
            f"for maxLev={max_lev} lev={lev}"
        )


# ── AUDIT CONCLUSION B: the 1.5x ATR backup SL CAN sit outside liq ─────────
def test_backup_sl_sits_outside_liq_on_high_atr_low_maxlev_coin():
    # 1.5x ATR with ATR = 10% of price => 15% spot SL.
    sl_frac = _DEFAULT_SL_ATR_MULT * 0.10
    assert sl_frac == pytest.approx(0.15)
    # At the low maxLeverage classes the true liq (safety-scaled) is BELOW 15%,
    # so the SL is OUTSIDE liq — the naive 1/lev bound would (wrongly) call it safe.
    for _label, max_lev in [("low 5", 5), ("mid 10", 10), ("high 20", 20)]:
        lev = _live_lev(max_lev)
        assert sl_frac > SAFETY * _true_liq(lev, max_lev), f"maxLev={max_lev}"
        # and the naive bound overstates reachability:
        assert 1.0 / lev > sl_frac or max_lev >= 20  # naive says "plenty of room"
    # ...yet the cap must still reduce leverage to keep the SL reachable.
    assert liq_safe_leverage(5, 0.15, 5) < 5
    assert liq_safe_leverage(5, 0.15, 10) < 5


# ── THE INVARIANT: the cap keeps the stop strictly inside liq ───────────────
@pytest.mark.parametrize("max_lev", [2, 3, 5, 10, 20, 50])
@pytest.mark.parametrize("stop_pct", [5.0, 8.0, 12.0, 15.0, 20.0])
@pytest.mark.parametrize("lev", [1, 2, 3, 4, 5, 10])
def test_liq_invariant_stop_inside_liquidation(lev, stop_pct, max_lev):
    """For every class / stop / leverage combo, the returned leverage keeps the
    stop inside the safety-scaled maintenance-aware liq distance."""
    out = liq_safe_leverage(lev, stop_pct / 100.0, max_lev, SAFETY)
    assert 1 <= out <= max(1, lev)
    if out >= 1 and (stop_pct / 100.0) > 0:
        safe_bound = SAFETY * _true_liq(out, max_lev)
        assert stop_pct / 100.0 <= safe_bound + 1e-9, (
            f"stop {stop_pct}% still outside liq at lev={out} maxLev={max_lev}"
        )


# ── THE CAP: it only tightens, is load-bearing where maint dominates ────────
def test_cap_only_tightens_never_raises_leverage():
    for max_lev in [2, 3, 5, 10, 20, 50]:
        for stop in (0.05, 0.10, 0.15, 0.20):
            for lev in (1, 2, 3, 4, 5, 10):
                out = liq_safe_leverage(lev, stop, max_lev, SAFETY)
                assert out <= lev
                assert out >= 1


def test_cap_is_load_bearing_on_low_maxlev_but_inert_on_high_maxlev():
    # Low maxLeverage + wide stop: the cap must fire (maint dominates liq).
    assert liq_safe_leverage(5, 0.15, 5, SAFETY) < 5
    # Same stop/lev on a high-maxLeverage coin: maint negligible, no cap.
    assert liq_safe_leverage(5, 0.15, 50, SAFETY) == 5
    # A narrow stop always fits (the normal-case no-op the guard must not break).
    assert liq_safe_leverage(5, 0.03, 5, SAFETY) == 5


def test_cap_fires_later_for_higher_maxlev_than_for_low():
    # The cap point (where the SL just stops fitting) moves to lower leverage as
    # maxLev shrinks — maintenance margin dominates the liq distance there.
    assert liq_safe_leverage(5, 0.15, 20, SAFETY) > liq_safe_leverage(5, 0.15, 5, SAFETY)


# ── GATED / degenerate inputs ───────────────────────────────────────────────
def test_zero_safety_or_stop_disables_bound():
    # liq_safety_frac=0 => disabled, returns the request unchanged.
    assert liq_safe_leverage(5, 0.15, 5, 0.0) == 5
    # no stop requested => nothing to honor.
    assert liq_safe_leverage(5, 0.0, 5, SAFETY) == 5
    # lev already 1 => nothing to cap.
    assert liq_safe_leverage(1, 0.15, 5, SAFETY) == 1


def test_unknown_max_leverage_uses_naive_bound_not_1x():
    # coin_max_leverage=0 (missing meta) must NOT read as maint=inf (that would
    # force lev 1). With no maint info the bound degrades to the naive 1/lev,
    # which at least never under-caps to a hair-trigger.
    assert liq_safe_leverage(5, 0.15, 0, SAFETY) >= 2


# ── WIRING: maybe_execute actually applies the cap before the order ─────────
def test_maybe_execute_caps_leverage_to_honor_backup_sl(monkeypatch):
    """A high-ATR, low-maxLeverage coin: the backup SL (1.5x ATR) sits outside
    liq at the requested leverage, so the executor must set the CAPPED leverage
    on the exchange and size the notional at it (no margin shortfall)."""
    from test_cleanup import _exec_baseline, _analysis
    # ATR 10% of mid (100 => atr 10) => 1.5x SL = 15% spot; maxLev 5 at lev 5
    # liquidates at 10% (8.5% safety-scaled) -> SL outside liq -> cap to 3x.
    ex, captured, _ = _exec_baseline(
        monkeypatch,
        cfg_overrides={
            "leverage": 5,
            "equity_fraction_per_trade": 0.10,
            "max_trade_notional_usd": 100000,
        },
    )
    monkeypatch.setattr(ex, "get_max_leverage", lambda c: 5)
    monkeypatch.setattr(ex, "get_hl_atr", lambda *a, **k: 10.0)
    set_lev = []
    monkeypatch.setattr(ex, "set_leverage", lambda c, l: set_lev.append(l) or {"ok": True})

    r = ex.maybe_execute(_analysis())
    assert r["executed"] is True, r
    assert set_lev == [3], f"expected leverage capped to 3x, got {set_lev}"
    # Notional sized at the CAPPED lev (1000 x 0.10 x 1.0 conv x 3 = $300) / 100 = 3.0 coins.
    assert abs(captured["size"] - 3.0) < 1e-6, f"size should reflect 3x lev: {captured}"


def test_maybe_execute_no_cap_when_backup_sl_inside_liq(monkeypatch):
    """Same harness, but a low-ATR / high-maxLeverage coin: the SL is inside
    liq, so the requested leverage is preserved (the guard is a no-op)."""
    from test_cleanup import _exec_baseline, _analysis
    ex, captured, _ = _exec_baseline(
        monkeypatch,
        cfg_overrides={
            "leverage": 5,
            "equity_fraction_per_trade": 0.10,
            "max_trade_notional_usd": 100000,
        },
    )
    # baseline get_hl_atr=2.0 => 3% SL; maxLev 40 at lev 5 liquidates at 17.5%.
    monkeypatch.setattr(ex, "get_max_leverage", lambda c: 40)
    set_lev = []
    monkeypatch.setattr(ex, "set_leverage", lambda c, l: set_lev.append(l) or {"ok": True})

    r = ex.maybe_execute(_analysis())
    assert r["executed"] is True, r
    assert set_lev == [5], f"expected 5x preserved (SL inside liq), got {set_lev}"
