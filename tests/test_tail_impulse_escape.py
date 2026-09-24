"""Impulse-class composite-escape denial on the tail-trigger gates (2026-09-24).

The NIL knife cohort: an entry whose composite score was EARNED by the fresh-
impulse trio (volume_spike / breakout / momentum_burst) used the composite ≥ 60
escape to walk past a tripped adverse-tail gate. Executed-entry replay over the
full log history (scratch/_tail_escape_replay.py): 4 such entries, 4/4 losers,
net −$23.1 — all with the full trio fired and every forecaster median negative.

The fix: `_tail_composite_escape_allowed` closes the composite escape when
>= impulse_min_triggers of the trio fired, per tail gate, gated on
`no_composite_escape_on_impulse` (config default False = byte-identical).
The confidence escape is never touched. Applies to all three tail gates:
chronos / timesfm / tirex.

Pins:
  * tripped tail + trio fired + flag ON  -> composite no longer releases;
  * same but flag OFF                    -> released (old behaviour);
  * conf >= min_conf                     -> released even with the flag on;
  * only ONE impulse trigger fired       -> escape stays open (needs >= 2);
  * reason string annotates the denial;
  * shadow posture unchanged (pass=True + marker).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    chronos_tail_trigger_gate,
    timesfm_tail_trigger_gate,
    tirex_tail_trigger_gate,
)

TAIL_CFG = {
    "enabled": True,
    "shadow_mode": False,
    "window_steps": 6,
    "min_adv_path_pct": 2.5,
    "min_conf": 0.90,
    "min_composite": 60.0,
}

# Deep adverse early tail (the NIL-09-24 shape: q10 min -15% inside 6 steps).
DEEP_Q10 = [-3.0, -7.5, -11.8, -15.4, -14.1, -12.0, -10.2, -8.8, -7.1, -5.9, -4.8, -3.9]
# TimesFM window is 12 steps — same shape works for it.
DEEP_Q90 = [p * -1 for p in DEEP_Q10]  # mirror: adverse q90 tail for shorts


def _ctx(side="long", conf=0.85, composite=61.8, vsf=False, bo=False, mb=False,
         q10=None, q90=None):
    return GateContext(
        confidence=conf,
        current_positions=[],
        trade_notional_usd=33.0,
        daily_pnl=0.0,
        market_volume_24h_usd=50_000_000.0,
        coin="NIL",
        trade_side=side,
        has_binary_news_risk=False,
        equity=1000.0,
        total_open_notional=33.0,
        composite_score=composite,
        momentum_burst_fired=mb,
        volume_spike_fired=vsf,
        breakout_fired=bo,
        chronos_q10_path_pct=q10,
        chronos_q90_path_pct=q90,
        timesfm_q10_path_pct=q10,
        tirex_q10_path_pct=q10,
    )


def _cfg(**over):
    base = dict(TAIL_CFG)
    base.update(over)
    return base


GATES = [
    ("chronos", chronos_tail_trigger_gate),
    ("timesfm", timesfm_tail_trigger_gate),
    ("tirex", tirex_tail_trigger_gate),
]

# Full NIL-class fixture: trio fired, comp 61.8, conf 0.85, deep q10 tail.
NIL_CTX: "dict" = dict(q10=DEEP_Q10, vsf=True, bo=True, mb=True)


def test_nil_class_blocked_by_every_tail_gate_when_flag_on():
    """The exact NIL-09-24 entry: composite 61.8 no longer escapes when the
    flag is on and the trio fired — all three tail gates veto."""
    cfg = _cfg(no_composite_escape_on_impulse=True)
    for name, gate in GATES:
        r = gate(_ctx(**NIL_CTX), cfg)
        assert r["pass"] is False, f"{name} escaped: {r}"
        assert "composite escape closed on fresh-impulse class" in r["reason"], name


def test_nil_class_still_released_with_flag_absent():
    """Byte-identical old behaviour when the key is absent (code default)."""
    cfg = _cfg()  # no no_composite_escape_on_impulse key
    for name, gate in GATES:
        r = gate(_ctx(**NIL_CTX), cfg)
        assert r["pass"] is True, f"{name} blocked without the flag armed"


def test_nil_class_released_with_flag_explicitly_false():
    cfg = _cfg(no_composite_escape_on_impulse=False)
    for name, gate in GATES:
        assert gate(_ctx(**NIL_CTX), cfg)["pass"] is True, name


def test_confidence_escape_untouched_by_flag():
    """conf >= min_conf releases even on the impulse class with the flag on."""
    cfg = _cfg(no_composite_escape_on_impulse=True)
    for name, gate in GATES:
        r = gate(_ctx(conf=0.90, **NIL_CTX), cfg)
        assert r["pass"] is True, f"{name} blocked a >= min_conf entry"


def test_single_impulse_trigger_keeps_escape_open():
    """Denial needs >= impulse_min_triggers (default 2). One trigger = open."""
    cfg = _cfg(no_composite_escape_on_impulse=True)
    for name, gate in GATES:
        r = gate(_ctx(q10=DEEP_Q10, vsf=True), cfg)  # only volume_spike
        assert r["pass"] is True, f"{name} denied on a single trigger"


def test_two_triggers_meet_default_threshold():
    cfg = _cfg(no_composite_escape_on_impulse=True)
    for name, gate in GATES:
        r = gate(_ctx(q10=DEEP_Q10, vsf=True, bo=True), cfg)  # 2 of trio
        assert r["pass"] is False, f"{name}: 2 triggers must close the escape"


def test_impulse_min_triggers_override():
    """Raising the bar to 3 releases a 2-trigger entry; the NIL trio still denied."""
    cfg = _cfg(no_composite_escape_on_impulse=True, impulse_min_triggers=3)
    for name, gate in GATES:
        assert gate(_ctx(q10=DEEP_Q10, vsf=True, bo=True), cfg)["pass"] is True, name
        assert gate(_ctx(**NIL_CTX), cfg)["pass"] is False, name


def test_short_side_denial_mirrors():
    """Short into an adverse q90 tail with the trio fired: composite denied."""
    cfg = _cfg(no_composite_escape_on_impulse=True)
    ctx = GateContext(
        confidence=0.85, current_positions=[], trade_notional_usd=33.0,
        daily_pnl=0.0, market_volume_24h_usd=50_000_000.0, coin="NIL",
        trade_side="short", has_binary_news_risk=False, equity=1000.0,
        total_open_notional=33.0, composite_score=61.8,
        momentum_burst_fired=True, volume_spike_fired=True, breakout_fired=True,
        chronos_q90_path_pct=DEEP_Q90,
    )
    r = chronos_tail_trigger_gate(ctx, cfg)
    assert r["pass"] is False
    assert "composite escape closed on fresh-impulse class" in r["reason"]


def test_shadow_posture_unchanged_with_denial():
    """Flag on + shadow_mode on: still pass=True with marker + annotated reason."""
    cfg = _cfg(shadow_mode=True, no_composite_escape_on_impulse=True)
    for name, gate in GATES:
        r = gate(_ctx(**NIL_CTX), cfg)
        assert r["pass"] is True and r.get("shadow_would_block") is True, name
        assert "composite escape closed on fresh-impulse class" in r["reason"], name


def test_no_breach_still_zero_opinion():
    """Shallow tail: no marker at all regardless of the flag (fail-safe)."""
    shallow = [-0.6, -1.1, -1.7, -2.2, -2.4, -2.3, -1.9, -1.4, -0.9, -0.5, -0.2, 0.1]
    cfg = _cfg(no_composite_escape_on_impulse=True)
    for name, gate in GATES:
        r = gate(_ctx(q10=shallow, vsf=True, bo=True, mb=True), cfg)
        assert r == {"pass": True}, f"{name}: {r}"
