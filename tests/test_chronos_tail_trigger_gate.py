"""Tests for the chronos tail-trigger conviction gate.

Shape-based counter-forecast veto, validated by the 2026-08-28 60-flag replay
(adverse-quantile early TAIL > path-mean > endpoint for loss avoidance).

The gate is a pure function of GateContext: it reads the cached per-step
Chronos-2 quantile paths (``ctx.chronos_q10_path_pct`` for longs,
``ctx.chronos_q90_path_pct`` for shorts) — no candle fetch, no model call.
When the ADVERSE quantile's early tail (min q10 / max q90 over the first
``window_steps`` 5m steps) breaches ``-min_adv_path_pct`` and the entry does
not carry elevated conviction, the gate vetoes (or, in shadow mode, only
marks ``shadow_would_block``).

The semantic the replay locked in, and which these tests pin:
  * a LOSER with a shallow MEAN but a DEEP early tail (TRUMP: mean ~-1.05%,
    q10 tail ~-5.7%) is caught by the tail trigger even though the path-mean
    scalar is near the deadband;
  * a WINNER with a mean that would trip the path-mean gate but a SHALLOW
    early tail (VVV: mean ~-0.8%, q10 tail ~-2.4%) is RELEASED by the tail
    trigger — that selectivity is the entire edge over chronos_mismatch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    chronos_tail_trigger_gate,
)

# The live .agent-config.json chronos_tail_trigger_gate block as armed 2026-08-28.
TAIL_CFG = {
    "enabled": True,
    "shadow_mode": True,
    "window_steps": 6,
    "min_adv_path_pct": 3.0,
    "min_conf": 0.90,
    "min_composite": 60.0,
}


def _ctx(
    side: str,
    conf: float,
    composite: float = 45.0,
    q10=None,
    q90=None,
    coin: str = "TRUMP",
) -> GateContext:
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
        chronos_q10_path_pct=q10,
        chronos_q90_path_pct=q90,
    )


def _cfg(**over) -> dict:
    base = dict(TAIL_CFG)
    base.update(over)
    return base


# A deep-early-tail long: q10 dips to -5.7% within the first 6 steps (the
# TRUMP shape). The path MEAN of these 12 is ~-1.1% — near the mismatch
# deadband, which is exactly why the tail is the discriminating reduction.
TRUMP_Q10 = [-1.5, -3.2, -4.5, -5.4, -5.6, -5.7, -5.1, -4.4, -3.6, -2.9, -2.4, -2.1]
# A shallow-tail long: q10 only dips to -2.4% (the VVV shape) — below the 3.0
# arm threshold, so the tail trigger releases it even though its mean would
# trip the path-mean gate.
VVV_Q10 = [-0.6, -1.1, -1.7, -2.2, -2.4, -2.3, -1.9, -1.4, -0.9, -0.5, -0.2, 0.1]


# ---------------------------------------------------------------------------
# arm / veto logic
# ---------------------------------------------------------------------------


def test_long_deep_tail_low_conviction_vetoes():
    r = chronos_tail_trigger_gate(_ctx("long", 0.82, q10=TRUMP_Q10), _cfg(shadow_mode=False))
    assert r["pass"] is False
    assert "chronos_tail_trigger" in r["reason"]
    assert "long entry" in r["reason"]
    assert "0.82" in r["reason"] and "0.90" in r["reason"]


def test_long_deep_tail_released_by_confidence():
    r = chronos_tail_trigger_gate(_ctx("long", 0.90, q10=TRUMP_Q10), _cfg(shadow_mode=False))
    assert r["pass"] is True


def test_long_deep_tail_released_by_composite():
    r = chronos_tail_trigger_gate(
        _ctx("long", 0.82, composite=65.0, q10=TRUMP_Q10), _cfg(shadow_mode=False)
    )
    assert r["pass"] is True


def test_long_shallow_tail_released_even_low_conviction():
    """The whole edge: a mean-reverting winner with a shallow early tail is
    released where the path-mean gate would have vetoed it."""
    r = chronos_tail_trigger_gate(_ctx("long", 0.55, q10=VVV_Q10), _cfg())
    assert r == {"pass": True}  # no breach -> no opinion at all


def test_long_tail_beyond_window_ignored():
    """A deep dip that only appears AFTER the first 6 steps is outside the
    arm window (the entry's 30m exposure) and must not trip the gate."""
    path = [0.2, 0.1, -0.3, -0.5, -0.4, -0.6,  # first 6: shallow
            -4.0, -5.0, -5.5, -5.8, -5.2, -4.8]  # dip at steps 7-12
    r = chronos_tail_trigger_gate(_ctx("long", 0.7, q10=path), _cfg())
    assert r == {"pass": True}


def test_short_deep_tail_vetoes():
    """Mirror: a SHORT consults the ADVERSE q90 path (its upside risk). A
    q90 that climbs to +4.5% within 6 steps is the adverse tail for a short."""
    q90 = [1.0, 2.4, 3.6, 4.5, 4.2, 3.8, 3.1, 2.6, 2.2, 1.9, 1.6, 1.4]
    r = chronos_tail_trigger_gate(_ctx("short", 0.8, q90=q90), _cfg(shadow_mode=False))
    assert r["pass"] is False
    assert "short entry" in r["reason"]
    assert "max" in r["reason"]  # short's adverse tail is the MAX of the window


def test_short_ignores_q10_path():
    """A short must key off q90, not q10 — a deep q10 dip is the SHORT's
    friend (price falling, as the short wants), not its risk."""
    r = chronos_tail_trigger_gate(
        _ctx("short", 0.8, q10=TRUMP_Q10, q90=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 2),
        _cfg(),
    )
    assert r == {"pass": True}  # q90 never reaches +3.0 -> no veto


# ---------------------------------------------------------------------------
# fail-safes (no-opinion pass) — a gate failure must never block a trade
# ---------------------------------------------------------------------------


def test_disabled_passes():
    r = chronos_tail_trigger_gate(_ctx("long", 0.5, q10=TRUMP_Q10), _cfg(enabled=False))
    assert r == {"pass": True}


def test_missing_paths_pass():
    """Cold cache / pre-change signal (no per-step paths) -> no opinion."""
    r = chronos_tail_trigger_gate(_ctx("long", 0.5), _cfg())
    assert r == {"pass": True}


def test_short_path_passes():
    """A path shorter than window_steps cannot be evaluated -> no opinion."""
    r = chronos_tail_trigger_gate(_ctx("long", 0.5, q10=TRUMP_Q10[:3]), _cfg())
    assert r == {"pass": True}


def test_window_of_one_step_only_considers_first_step():
    """The arm window is `path[:window_steps]` — with window 1 only step 1's
    value (-1.5%, above the -3.0 threshold) is considered, so the deep
    dip at steps 4-6 must NOT arm the gate."""
    r = chronos_tail_trigger_gate(_ctx("long", 0.5, q10=TRUMP_Q10), _cfg(window_steps=1))
    assert r == {"pass": True}


# ---------------------------------------------------------------------------
# shadow-mode semantics (the live default)
# ---------------------------------------------------------------------------


def test_shadow_mode_marks_but_passes():
    """Shadow mode (armed 2026-08-28): MUST structurally pass and carry the
    would-block marker — no silent behavior change until the operator flips
    shadow_mode off."""
    r = chronos_tail_trigger_gate(_ctx("long", 0.82, q10=TRUMP_Q10), _cfg(shadow_mode=True))
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert "chronos_tail_trigger" in r["reason"]
    assert "via" not in r


def test_live_mode_actually_blocks():
    """With shadow_mode off the same shape must hard-block (pass False, no
    marker) — this is the promotion path once the shadow window is long enough."""
    r = chronos_tail_trigger_gate(_ctx("long", 0.82, q10=TRUMP_Q10), _cfg(shadow_mode=False))
    assert r["pass"] is False
    assert "shadow_would_block" not in r


# ---------------------------------------------------------------------------
# threshold sensitivity (X grid from the replay)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x,expected_pass",
    [
        (5.0, False),  # tail -5.7 <= -5.0 -> still vetoes at the stop-distance threshold
        (3.0, False),  # the armed threshold
        (2.5, False),  # tail -5.7 <= -2.5
        (6.0, True),   # tail -5.7 > -6.0 -> released (X beyond the realized tail)
    ],
)
def test_threshold_sensitivity(x, expected_pass):
    r = chronos_tail_trigger_gate(
        _ctx("long", 0.82, q10=TRUMP_Q10), _cfg(min_adv_path_pct=x, shadow_mode=False)
    )
    assert r["pass"] is expected_pass