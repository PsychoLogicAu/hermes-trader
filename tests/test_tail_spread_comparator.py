"""Tests for the tail-vs-spread shadow comparator (E2c follow-up, plan D4).

The armed chronos_tail_trigger_gate keys off the q10/q90 PATH tail. The 2026-09
model sweep (scratch/e2c_followups.py) showed the tail's stop-out AUC is nearly
matched by the forecast BAND WIDTH (spread_pct) alone — a scalar that needs no
path at all. This comparator runs alongside the real gate and records, per
evaluated entry, what each rule WOULD have done, so accrual can decide whether
the path-based veto earns its complexity over a width-only veto.

Contract these tests pin:
  * the comparator NEVER blocks — pass is always True regardless of trips;
  * disabled (default) → no marker at all ({'pass': True});
  * enabled → marker carries tail value, spread, and both trip flags computed
    side-aware (long: min q10 <= -x / spread >= s; short: max q90 >= x);
  * missing data (no path / no spread) → trips False, never raises.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import GateContext, tail_spread_comparator  # noqa: E402

CMP_CFG = {"enabled": True, "window_steps": 6, "min_adv_path_pct": 2.5, "spread_pct_threshold": 4.0}


def _ctx(side="long", q10=None, q90=None, spread=None):
    return GateContext(
        confidence=0.75,
        current_positions=[],
        trade_notional_usd=50.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin="TEST",
        trade_side=side,
        has_binary_news_risk=False,
        equity=100.0,
        total_open_notional=0.0,
        chronos_spread_pct=spread,
        chronos_q10_path_pct=q10,
        chronos_q90_path_pct=q90,
    )


def test_disabled_by_default_no_marker():
    res = tail_spread_comparator(_ctx(q10=[-3.0] * 6, spread=5.0), {})
    assert res == {"pass": True}
    res2 = tail_spread_comparator(_ctx(), None)
    assert res2 == {"pass": True}


def test_enabled_returns_pass_true_even_when_both_trip():
    ctx = _ctx(q10=[-1.0, -2.0, -3.1, -2.5, -1.0, -0.5], spread=4.5)
    res = tail_spread_comparator(ctx, CMP_CFG)
    assert res["pass"] is True  # NEVER blocks
    cmp = res["cmp"]
    assert cmp["tail_trip"] is True and cmp["spread_trip"] is True
    assert abs(cmp["tail_pct"] - (-3.1)) < 1e-9


def test_tail_trips_spread_does_not():
    ctx = _ctx(q10=[-1.0, -2.6, -1.5, -1.0, -0.5, -0.2], spread=2.0)
    cmp = tail_spread_comparator(ctx, CMP_CFG)["cmp"]
    assert cmp["tail_trip"] is True and cmp["spread_trip"] is False


def test_spread_trips_tail_does_not():
    # wide band but shallow early tail — the divergence case accrual cares about
    ctx = _ctx(q10=[-0.5, -0.8, -1.2, -1.0, -0.6, -0.3], spread=5.5)
    cmp = tail_spread_comparator(ctx, CMP_CFG)["cmp"]
    assert cmp["tail_trip"] is False and cmp["spread_trip"] is True


def test_short_side_uses_q90():
    ctx = _ctx(side="short", q90=[1.0, 2.7, 1.2, 1.0, 0.5, 0.2], spread=3.0)
    cmp = tail_spread_comparator(ctx, CMP_CFG)["cmp"]
    assert cmp["tail_trip"] is True and cmp["spread_trip"] is False
    assert abs(cmp["tail_pct"] - 2.7) < 1e-9


def test_missing_data_fails_safe_no_trips():
    ctx = _ctx(q10=None, spread=None)
    res = tail_spread_comparator(ctx, CMP_CFG)
    assert res["pass"] is True
    cmp = res["cmp"]
    assert cmp["tail_trip"] is False and cmp["spread_trip"] is False
    # short path with no q90 either
    ctx2 = _ctx(side="short", q90=[-1.0, 0.5], spread=None)  # shorter than window
    res2 = tail_spread_comparator(ctx2, CMP_CFG)
    assert res2["pass"] is True and res2["cmp"]["tail_trip"] is False
