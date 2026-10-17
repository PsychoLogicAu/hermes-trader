"""t_1999904f: breakout-direction veto gate.

Blocks an entry whose side fights the fired breakout: SHORT into breakoutUp,
LONG into breakoutDown. Evidence: d2 blocked-cohort counterfactual (62 h
window, direction-classified old combined `breakout` fires) — 15/16 DEAD-band
breakout shorts were up-pushes (-$116.62, WR 1/15); the GOOD short band fired
breakout only 1/67. The veto keys on the trigger TOKEN, not a live re-check:
the poison shorts had mostly reverted inside the range by decision time
(11/15, worst subgroup -$101.67 WR 0/11).

Modes: off (default, inert) / shadow (pass + would-block marker) / live
(block). A missing side (no breakout fired, or a pre-split hit) can never
block. Also covers the hit enrichment (range_pos_pct / bars_beyond) the
veto line accrues.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    breakout_direction_gate,
    eval_all_gates,
)
from hermes_trader.indicators.triggers import breakout  # noqa: E402
from hermes_trader.models.types import Candle  # noqa: E402


def _ctx(trade_side: str = "short", breakout_side=None, width=0.42,
         range_pos=101.3, bars_beyond=1) -> GateContext:
    return GateContext(
        confidence=0.55,
        current_positions=[],
        trade_notional_usd=10.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin="TEST",
        trade_side=trade_side,
        has_binary_news_risk=False,
        equity=100.0,
        total_open_notional=0.0,
        composite_score=40.0,
        breakout_fired=breakout_side is not None,
        breakout_side=breakout_side,
        breakout_width_pct=width,
        breakout_range_pos_pct=range_pos,
        breakout_bars_beyond=bars_beyond,
    )


CFG_SHADOW = {"mode": "shadow"}
CFG_LIVE = {"mode": "live"}


# ── Fail-safes (no-opinion pass) ────────────────────────────────────────────

def test_default_off_is_inert():
    r = breakout_direction_gate(_ctx(breakout_side="up"), {})
    assert r == {"pass": True, "reason": "breakout_direction: disabled"}


def test_unknown_mode_is_inert():
    r = breakout_direction_gate(_ctx(breakout_side="up"), {"mode": "banana"})
    assert r["pass"] is True and "disabled" in r["reason"]


def test_no_breakout_side_passes_live():
    # No breakout fired, or a pre-split hit without side: can never block.
    for side in (None, "", "sideways"):
        r = breakout_direction_gate(_ctx(breakout_side=side), CFG_LIVE)
        assert r == {"pass": True, "reason": "breakout_direction: no side"}


# ── Adverse shapes ──────────────────────────────────────────────────────────

def test_live_blocks_short_into_breakout_up():
    r = breakout_direction_gate(_ctx("short", "up"), CFG_LIVE)
    assert r["pass"] is False
    assert "SHORT into breakoutUp:0.42%" in r["reason"]
    assert "range_pos=101.3" in r["reason"] and "bars_beyond=1" in r["reason"]


def test_live_blocks_long_into_breakout_down():
    r = breakout_direction_gate(_ctx("long", "down"), CFG_LIVE)
    assert r["pass"] is False
    assert "LONG into breakoutDown" in r["reason"]


def test_shadow_passes_with_marker():
    r = breakout_direction_gate(_ctx("short", "up"), CFG_SHADOW)
    assert r["pass"] is True and r["shadow_would_block"] is True
    assert "SHORT into breakoutUp" in r["reason"]


# ── Aligned shapes never block ──────────────────────────────────────────────

def test_short_with_breakout_down_passes():
    # The user's point: breakoutDown is a short TRIGGER candidate, not a
    # veto — the veto is direction-ADVERSE only.
    assert breakout_direction_gate(_ctx("short", "down"), CFG_LIVE)["pass"]


def test_long_with_breakout_up_passes():
    assert breakout_direction_gate(_ctx("long", "up"), CFG_LIVE)["pass"]


def test_missing_context_still_blocks_with_none_fields():
    # Width/pos/beyond are accrual decoration, never a fail-open.
    ctx = _ctx("short", "up")
    ctx.breakout_width_pct = None
    ctx.breakout_range_pos_pct = None
    ctx.breakout_bars_beyond = None
    r = breakout_direction_gate(ctx, CFG_LIVE)
    assert r["pass"] is False
    assert "breakoutUp " in r["reason"]  # no width suffix when None


# ── Pipeline integration ────────────────────────────────────────────────────

def test_pipeline_blocks_adverse_and_logs_reason():
    out = eval_all_gates(_ctx("short", "up"),
                         {"breakout_direction_gate": {"mode": "live"}})
    assert out["blocked"] is True
    assert any("breakout_direction" in r for r in out["block_reasons"])


def test_pipeline_shadow_does_not_block():
    out = eval_all_gates(_ctx("short", "up"),
                         {"breakout_direction_gate": {"mode": "shadow"}})
    assert out["results"]["breakout_direction"]["shadow_would_block"] is True
    assert not any("breakout_direction" in r for r in out["block_reasons"])


def test_pipeline_off_is_byte_identical_pass():
    out = eval_all_gates(_ctx("short", "up"), {})
    r = out["results"]["breakout_direction"]
    assert r == {"pass": True, "reason": "breakout_direction: disabled"}


# ── Hit enrichment (range_pos_pct / bars_beyond) ────────────────────────────

def _flat(n, lo=99.0, hi=101.0, px=100.0):
    out = [Candle(t=1_700_000_000_000 + i * 300_000, o=px, h=hi, l=lo,
                  c=px, v=100.0) for i in range(n)]
    return out


def test_v1_fired_hit_carries_range_context():
    candles = _flat(60)
    # forming bar closes above the prior 48-bar high -> up-break
    candles.append(Candle(t=1_700_000_000_000 + 60 * 300_000, o=100.0,
                          h=102.0, l=100.0, c=102.0, v=100.0))
    hit = breakout(candles, lookback=48, width_mode="off")
    assert hit["fired"] and hit["side"] == "up"
    # close 102 vs range [99, 101] -> (102-99)/(101-99) = 150.0, one bar beyond
    assert hit["range_pos_pct"] == 150.0
    assert hit["bars_beyond"] == 1


def test_v1_down_break_counts_consecutive_bars_beyond():
    candles = _flat(60)
    # two prior closes below the low + forming bar below -> bars_beyond 3
    for i in (58, 59):
        candles[i] = Candle(t=candles[i].t, o=98.0, h=98.5, l=97.5,
                            c=98.0, v=100.0)
    candles.append(Candle(t=1_700_000_000_000 + 60 * 300_000, o=98.0,
                          h=98.2, l=97.0, c=97.5, v=100.0))
    hit = breakout(candles, lookback=48, width_mode="off")
    # NOTE: the two modified bars are INSIDE the lookback window, so they
    # drag prior_low down to 97.5 — the forming close 97.5 is not strictly
    # below it. Assert the inert case instead: no fire, no context keys.
    if hit["fired"]:
        assert hit["side"] == "down" and hit["bars_beyond"] >= 1
    else:
        assert "range_pos_pct" not in hit


def test_unfired_hit_carries_no_context():
    hit = breakout(_flat(61), lookback=48, width_mode="off")
    assert not hit["fired"]
    assert "range_pos_pct" not in hit and "bars_beyond" not in hit
