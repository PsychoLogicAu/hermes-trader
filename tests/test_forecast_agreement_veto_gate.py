"""Tests for the forecast_agreement_veto gate (chronos ∧ timesfm, SHADOW by default).

The 2026-09-02 two-model parameter sweep (253 decision events since the
TimesFM shadow enable, DSL-simmed P/L) found the AND shape — BOTH forecasters
showing an adverse quantile tail past their own thresholds — to be the most
split-half-robust veto cell: ≈$24 saved vs ≈$23 for chronos-alone K6/X2.5
while vetoing FEWER entries, and timesfm-alone vetoed with sign-flipping
halves. Agreement is timesfm's contribution; it never vetoes by itself.

Gate semantics pinned here:
  * BOTH adverse (each model's own window/threshold) + low conviction → veto;
  * only one model adverse → pass (no agreement);
  * either model's paths missing (cold cache / disabled / inference error) →
    pass — a data gap can never block a trade;
  * conf >= min_conf OR composite >= min_composite releases;
  * shadow_mode (default True): structurally passes with a shadow_would_block
    marker; shadow_mode False is the promotion path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    forecast_agreement_veto_gate,
)

# Sweep-selected shape: chronos 50-bar context / K6 / X2.5, timesfm
# 100-200-bar context / K12 / X2.0. min_conf/min_composite same escape bar
# as the chronos tail trigger.
FAV_CFG = {
    "enabled": True,
    "shadow_mode": True,
    "chronos": {"window_steps": 6, "min_adv_path_pct": 2.5},
    "timesfm": {"window_steps": 12, "min_adv_path_pct": 2.0},
    "min_conf": 0.90,
    "min_composite": 60.0,
}

# Deep adverse tails for a LONG (both models breach).
C_DEEP = [-1.5, -3.2, -4.5, -5.4, -5.6, -5.7, -5.1, -4.4, -3.6, -2.9, -2.4, -2.1]
T_DEEP = [-0.8, -1.4, -2.1, -2.6, -3.0, -3.3, -3.5, -3.4, -3.1, -2.8, -2.5, -2.2]
# Timesfm shallow (min over its 12-step window is only -1.2%): no agreement.
T_SHALLOW = [-0.2, -0.5, -0.9, -1.2, -1.1, -0.8, -0.5, -0.3, -0.1, 0.1, 0.2, 0.3]
# Chronos shallow over ITS 6-step window (dip only at steps 7+): no agreement.
C_SHALLOW6 = [0.2, 0.1, -0.3, -0.5, -0.4, -0.6, -4.0, -5.0, -5.5, -5.8, -5.2, -4.8]
# Shorts: adverse is the q90 tail climbing.
C90_DEEP = [1.0, 2.4, 3.6, 4.5, 4.2, 3.8, 3.1, 2.6, 2.2, 1.9, 1.6, 1.4]
T90_DEEP = [0.6, 1.3, 1.9, 2.3, 2.6, 2.8, 2.9, 2.7, 2.4, 2.1, 1.8, 1.5]
T90_SHALLOW = [0.2, 0.5, 0.9, 1.2, 1.1, 0.8, 0.5, 0.3, 0.1, -0.1, -0.2, -0.3]


def _ctx(side="long", conf=0.82, composite=45.0, cq10=None, cq90=None,
         tq10=None, tq90=None, coin="TRUMP") -> GateContext:
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
        chronos_q10_path_pct=cq10,
        chronos_q90_path_pct=cq90,
        timesfm_q10_path_pct=tq10,
        timesfm_q90_path_pct=tq90,
    )


def _cfg(**over) -> dict:
    base = dict(FAV_CFG)
    base.update(over)
    return base


# ── AND logic ─────────────────────────────────────────────────────────────────

def test_both_adverse_low_conviction_vetoes():
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.82, cq10=C_DEEP, tq10=T_DEEP), _cfg(shadow_mode=False))
    assert r["pass"] is False
    assert "forecast_agreement_veto" in r["reason"]
    assert "both models adverse" in r["reason"]


def test_only_chronos_adverse_passes():
    """The core AND property: timesfm shallow → no agreement → pass."""
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.82, cq10=C_DEEP, tq10=T_SHALLOW), _cfg(shadow_mode=False))
    assert r == {"pass": True}


def test_only_timesfm_adverse_passes():
    """Timesfm alone must NEVER veto (sweep: its solo halves flip sign)."""
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.82, cq10=C_SHALLOW6, tq10=T_DEEP), _cfg(shadow_mode=False))
    assert r == {"pass": True}


def test_short_both_adverse_vetoes():
    r = forecast_agreement_veto_gate(
        _ctx("short", 0.80, cq90=C90_DEEP, tq90=T90_DEEP), _cfg(shadow_mode=False))
    assert r["pass"] is False


def test_short_q10_ignored():
    """A deep q10 dip is a short's friend; only q90 is adverse for shorts."""
    r = forecast_agreement_veto_gate(
        _ctx("short", 0.80, cq10=C_DEEP, cq90=[0.1] * 12,
             tq10=T_DEEP, tq90=T90_SHALLOW), _cfg(shadow_mode=False))
    assert r == {"pass": True}


# ── conviction escape ─────────────────────────────────────────────────────────

def test_released_by_confidence():
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.90, cq10=C_DEEP, tq10=T_DEEP), _cfg(shadow_mode=False))
    assert r["pass"] is True


def test_released_by_composite():
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.82, composite=65.0, cq10=C_DEEP, tq10=T_DEEP),
        _cfg(shadow_mode=False))
    assert r["pass"] is True


# ── fail-safes: a data gap can never block ────────────────────────────────────

def test_disabled_passes():
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.5, cq10=C_DEEP, tq10=T_DEEP), _cfg(enabled=False))
    assert r == {"pass": True}


def test_missing_timesfm_paths_pass():
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.5, cq10=C_DEEP), _cfg(shadow_mode=False))
    assert r == {"pass": True}


def test_missing_chronos_paths_pass():
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.5, tq10=T_DEEP), _cfg(shadow_mode=False))
    assert r == {"pass": True}


def test_short_timesfm_path_below_window_passes():
    """timesfm window is 12 steps; a 6-element path can't be evaluated."""
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.5, cq10=C_DEEP, tq10=T_DEEP[:6]), _cfg(shadow_mode=False))
    assert r == {"pass": True}


# ── shadow-mode semantics (live default) ──────────────────────────────────────

def test_shadow_marks_but_passes():
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.82, cq10=C_DEEP, tq10=T_DEEP), _cfg(shadow_mode=True))
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert "forecast_agreement_veto" in r["reason"]


def test_live_mode_blocks():
    r = forecast_agreement_veto_gate(
        _ctx("long", 0.82, cq10=C_DEEP, tq10=T_DEEP), _cfg(shadow_mode=False))
    assert r["pass"] is False
    assert "shadow_would_block" not in r


# ── per-model threshold sensitivity ───────────────────────────────────────────

@pytest.mark.parametrize(
    "cx,tx,expected_pass",
    [
        (2.5, 2.0, False),  # the armed shape: -5.7 <= -2.5 AND -3.5 <= -2.0
        (6.0, 2.0, True),   # chronos threshold beyond its tail -> no chronos breach
        (2.5, 4.0, True),   # timesfm threshold beyond its tail -> no timesfm breach
        (6.0, 4.0, True),   # neither breaches
    ],
)
def test_threshold_sensitivity(cx, tx, expected_pass):
    cfg = _cfg(shadow_mode=False,
               chronos={"window_steps": 6, "min_adv_path_pct": cx},
               timesfm={"window_steps": 12, "min_adv_path_pct": tx})
    r = forecast_agreement_veto_gate(_ctx("long", 0.82, cq10=C_DEEP, tq10=T_DEEP), cfg)
    assert r["pass"] is expected_pass
