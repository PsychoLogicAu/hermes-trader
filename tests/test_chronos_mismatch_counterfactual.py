"""Tests for the chronos_mismatch ratio-aware deadband COUNTERFACTUAL.

The live gate keeps its fixed `min_abs_median_pct` deadband untouched. On the
block path it now also records — via `_chronos_ratio_deadband_rescue` and the
`counterfactual_rescue` marker — whether a ratio-aware deadband
`max(fixed, min_conf_ratio * spread)` would have RESCUED the block (the HEMI
replay, 2026-08-30: a median inside the model's own p10-p90 band is noise, not
a directional claim). Because the ratio-aware deadband is always >= the fixed
one, it can only rescue, never add — so the counterfactual is a pure
over-block sample and MUST NOT change the gate's pass/fail.

These tests pin that invariant: identical pass/fail with and without the
spread, rescue marker present exactly when |med| < ratio_deadband, and inert
when the spread is unavailable.
"""

import pytest
from typing import Optional

from hermes_trader.agents.risk_gates import (
    GateContext,
    _chronos_ratio_deadband_rescue,
    chronos_mismatch_gate,
)


def _ctx(side="long", conf=0.82, composite=45.0, med=-1.70, spread: Optional[float] = 8.34, coin="HEMI"):
    return GateContext(
        confidence=conf,
        current_positions=[],
        trade_notional_usd=50.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin=coin,
        trade_side=side,
        has_binary_news_risk=False,
        equity=1000.0,
        total_open_notional=50.0,
        composite_score=composite,
        chronos_median_pct=med,
        chronos_spread_pct=spread,
    )


GATE = {
    "enabled": True,
    "shadow_mode": False,  # live (hard-block) mode — the real config
    "min_conf": 0.90,
    "min_composite": 60.0,
    "min_abs_median_pct": 0.5,
    "min_conf_ratio": 0.25,
}


# ── The HEMI 19:00 case: a low-confidence fade inside its own band ──────────
@pytest.mark.parametrize(
    "med,spread,expect_rescue",
    [
        # HEMI 19:00 long: |−1.70| >= 0.5 (fixed blocks) but < 0.25×8.34=2.085
        (-1.70, 8.34, True),
        # Confident fade: |−3.0| >= 0.5 AND >= 0.25×4.0=1.0 → no rescue
        (-3.0, 4.0, False),
        # Tiny median inside the fixed deadband → gate passes, never reaches
        # the block path → no rescue computed (abs(med) < 0.5)
        (-0.08, 4.98, False),
        # Spread unavailable (None) → counterfactual inert
        (-1.70, None, False),
        # Spread zero → inert
        (-1.70, 0.0, False),
        # Exactly at the ratio floor: |med| == ratio_deadband → not <, no rescue
        (-1.00, 4.0, False),
        # Short side mirror: short vs +1.70% inside a 8.34% band
        (1.70, 8.34, True),
        # Aligned forecast (long vs +median) → no mismatch → pass, no rescue
        (2.5, 3.0, False),
    ],
)
def test_rescue_helper(med, spread, expect_rescue):
    ctx = _ctx(med=med, spread=spread)
    got = _chronos_ratio_deadband_rescue(ctx, 0.5, GATE)
    if expect_rescue:
        assert got is not None
        assert got["would_pass"] is True
        assert got["fixed_deadband_pct"] == 0.5
        assert got["min_conf_ratio"] == 0.25
        # ratio deadband must be wider than the fixed one (the safety property)
        assert got["ratio_deadband_pct"] > got["fixed_deadband_pct"]
        # and must actually cover |med| (that's what a rescue means)
        assert abs(med) < got["ratio_deadband_pct"]
    else:
        assert got is None


def test_rescue_values_hemi_1900():
    ctx = _ctx(med=-1.70, spread=8.34)
    got = _chronos_ratio_deadband_rescue(ctx, 0.5, GATE)
    assert got is not None
    assert got["ratio_deadband_pct"] == round(max(0.5, 0.25 * 8.34), 4)
    assert got["median_pct"] == -1.70
    assert got["spread_pct"] == 8.34


# ── Gate integration: rescue is logged, pass/fail is UNCHANGED ──────────────
def test_gate_blocks_and_flags_rescue_live():
    # HEMI 19:00 in LIVE mode: the fixed rule blocks (low conf), and the
    # counterfactual says the ratio rule would have rescued it.
    ctx = _ctx(med=-1.70, spread=8.34)
    res = chronos_mismatch_gate(ctx, GATE)
    assert res["pass"] is False  # live rule still blocks — behavior unchanged
    assert "chronos_mismatch" in res["reason"]
    cf = res.get("counterfactual_rescue")
    assert cf is not None
    assert cf["would_pass"] is True
    assert "shadow_would_block" not in res  # live mode, not shadow


def test_gate_pass_fail_invariant_without_spread():
    # Same entry, but no spread (error signal): the gate must produce an
    # IDENTICAL block with no rescue marker.
    with_spread = _ctx(med=-1.70, spread=8.34)
    no_spread = _ctx(med=-1.70, spread=None)
    r1 = chronos_mismatch_gate(with_spread, GATE)
    r2 = chronos_mismatch_gate(no_spread, GATE)
    assert r1["pass"] == r2["pass"] == False
    assert r1["reason"] == r2["reason"]  # reason text is spread-independent
    assert "counterfactual_rescue" in r1
    assert "counterfactual_rescue" not in r2


def test_gate_confident_fade_blocks_no_rescue():
    # A genuinely confident, wide-band-escaping fade: blocked, no rescue.
    ctx = _ctx(med=-3.0, spread=4.0)
    res = chronos_mismatch_gate(ctx, GATE)
    assert res["pass"] is False
    assert "counterfactual_rescue" not in res


def test_gate_shadow_mode_also_flags_rescue():
    # Shadow mode: pass=True + shadow_would_block, AND the counterfactual is
    # still recorded (the sample accrues in either mode).
    shadow_cfg = dict(GATE, shadow_mode=True)
    ctx = _ctx(med=-1.70, spread=8.34)
    res = chronos_mismatch_gate(ctx, shadow_cfg)
    assert res["pass"] is True
    assert res["shadow_would_block"] is True
    assert res.get("counterfactual_rescue") is not None


def test_gate_no_opinion_passes_no_rescue():
    # Median inside the fixed deadband → pass, counterfactual never computed.
    ctx = _ctx(med=-0.08, spread=4.98)
    res = chronos_mismatch_gate(ctx, GATE)
    assert res["pass"] is True
    assert "counterfactual_rescue" not in res
    assert "reason" not in res


def test_rescue_respects_configured_ratio():
    # A tighter ratio floor narrows the rescue zone: at 0.50, |−1.70| is NOT
    # < 0.50×8.34=4.17 → still a rescue; at 0.10 it's < 0.834? no, 1.70>0.834
    # → NOT a rescue. Confirm the knob actually moves the boundary.
    ctx = _ctx(med=-1.70, spread=8.34)
    assert _chronos_ratio_deadband_rescue(ctx, 0.5, dict(GATE, min_conf_ratio=0.50)) is not None
    assert _chronos_ratio_deadband_rescue(ctx, 0.5, dict(GATE, min_conf_ratio=0.10)) is None
    # And a custom fixed deadband is honored in the marker.
    got = _chronos_ratio_deadband_rescue(_ctx(med=-2.0, spread=10.0), 1.0, GATE)
    assert got is not None
    assert got["fixed_deadband_pct"] == 1.0
    assert got["ratio_deadband_pct"] == round(max(1.0, 0.25 * 10.0), 4)