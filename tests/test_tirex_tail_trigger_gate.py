"""Tests for the tirex tail-trigger conviction gate (2026-09-17).

Accrual-first mirror of chronos_tail_trigger_gate on the TiRex 1.1 adverse
q-path (ctx.tirex_q10_path_pct / tirex_q90_path_pct). Same rule, same escape
bar, shadow_mode-capable like the chronos gate (unlike the pass-only timesfm
mirror): offline (scratch/eval/moirai_eval/) tirex wins stop-out AUC on both
sides but ties chronos on capture at the live operating point X=2.5 — so the
pair accrues would-block markers side by side until executed-cohort P/L picks.

Pins:
  (a) shape-fire in shadow mode -> pass True + shadow_would_block + join vars;
  (b) shadow_mode false -> REAL veto (pass False, reason);
  (c) no-shape / shallow tail -> pass True, no marker;
  (d) missing/None/short paths -> pass True, no marker, NO raise (fail-safe:
      a data gap can never block);
  (e) escape bar: conf >= 0.90 OR composite >= 60 clears the trip;
  (f) short side mirrors on q90 (max >= +x);
  (g) disabled -> plain pass;
  (h) eval_all_gates carries the tirex_tail_trigger key;
  (i) the executor shadow line fires via caplog.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    eval_all_gates,
    tirex_tail_trigger_gate,
)

# Mirror of the live chronos_tail_trigger_gate block (window 6 / X 2.5).
TAIL_CFG = {
    "enabled": True,
    "shadow_mode": True,
    "window_steps": 6,
    "min_adv_path_pct": 2.5,
    "min_conf": 0.90,
    "min_composite": 60.0,
}


def _ctx(side="long", conf=0.70, composite=45.0, q10=None, q90=None):
    return GateContext(
        confidence=conf,
        current_positions=[],
        trade_notional_usd=50.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin="TEST",
        trade_side=side,
        has_binary_news_risk=False,
        equity=1000.0,
        total_open_notional=0.0,
        composite_score=composite,
        tirex_q10_path_pct=q10,
        tirex_q90_path_pct=q90,
    )


DEEP_Q10 = [-0.5, -1.2, -3.1, -2.0, -1.0, -0.5]     # min of first 6 = -3.1
SHALLOW_Q10 = [-0.4, -0.9, -1.5, -1.2, -0.8, -0.3]  # never breaches -2.5
DEEP_Q90 = [0.3, 1.1, 2.9, 2.2, 1.0, 0.4]           # max of first 6 = +2.9


# ── (a) shadow-mode fire ------------------------------------------------------
def test_shadow_fire_marks_would_block_with_join_vars():
    r = tirex_tail_trigger_gate(_ctx(q10=DEEP_Q10), TAIL_CFG)
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert abs(r["tail_pct"] - (-3.1)) < 1e-9
    assert r["window_steps"] == 6
    assert "tirex_tail_trigger" in r["reason"]


# ── (b) live mode: real veto ---------------------------------------------------
def test_live_mode_blocks():
    cfg = dict(TAIL_CFG, shadow_mode=False)
    r = tirex_tail_trigger_gate(_ctx(q10=DEEP_Q10), cfg)
    assert r["pass"] is False
    assert "tirex_tail_trigger" in r["reason"]


# ── (c) no shape --------------------------------------------------------------
def test_shallow_tail_passes():
    r = tirex_tail_trigger_gate(_ctx(q10=SHALLOW_Q10), TAIL_CFG)
    assert r == {"pass": True}


def test_breach_outside_window_ignored():
    """Breach at step 7+ is outside window_steps — no trip."""
    path = SHALLOW_Q10 + [-9.0]
    r = tirex_tail_trigger_gate(_ctx(q10=path), TAIL_CFG)
    assert r == {"pass": True}


# ── (d) fail-safe: missing data never blocks ----------------------------------
def test_missing_paths_pass():
    for q10, q90 in [(None, None), (DEEP_Q10[:3], None), ([], [])]:
        r = tirex_tail_trigger_gate(_ctx(q10=q10, q90=q90), TAIL_CFG)
        assert r["pass"] is True and "shadow_would_block" not in r


def test_live_mode_still_passes_on_missing_data():
    cfg = dict(TAIL_CFG, shadow_mode=False)
    r = tirex_tail_trigger_gate(_ctx(q10=None), cfg)
    assert r == {"pass": True}


# ── (e) escape bar -------------------------------------------------------------
def test_high_conf_escape():
    r = tirex_tail_trigger_gate(_ctx(conf=0.95, q10=DEEP_Q10), TAIL_CFG)
    assert r == {"pass": True}


def test_high_composite_escape():
    r = tirex_tail_trigger_gate(_ctx(composite=75.0, q10=DEEP_Q10), TAIL_CFG)
    assert r == {"pass": True}


# ── (f) short side mirrors on q90 ----------------------------------------------
def test_short_side_uses_q90_max():
    r = tirex_tail_trigger_gate(_ctx(side="short", q10=DEEP_Q10, q90=DEEP_Q90), TAIL_CFG)
    assert r.get("shadow_would_block") is True
    assert abs(r["tail_pct"] - 2.9) < 1e-9


def test_short_side_ignores_q10():
    """A deep q10 (adverse for LONGS) must not trip a SHORT entry."""
    r = tirex_tail_trigger_gate(
        _ctx(side="short", q10=DEEP_Q10, q90=SHALLOW_Q10), TAIL_CFG)
    assert r == {"pass": True}


# ── (g) disabled ---------------------------------------------------------------
def test_disabled_plain_pass():
    r = tirex_tail_trigger_gate(_ctx(q10=DEEP_Q10), {"enabled": False})
    assert r == {"pass": True}


# ── (h) eval_all_gates registration -------------------------------------------
def _base_config(**tail_override):
    tail = {"enabled": False}
    tail.update(tail_override)
    return {
        "min_confidence": 0.0,
        "max_trade_notional_pct": 100.0,
        "max_total_notional_pct": 100.0,
        "tirex_tail_trigger_gate": tail,
    }


def _base_ctx(**kw):
    kw.setdefault("q10", None)
    return _ctx(**kw)


def test_eval_all_gates_carries_key():
    out = eval_all_gates(_base_ctx(), _base_config())
    assert "tirex_tail_trigger" in out["results"]


def test_eval_all_gates_live_block_surfaces_in_reasons():
    ctx = _base_ctx(q10=DEEP_Q10)
    cfg = _base_config(enabled=True, shadow_mode=False, window_steps=6,
                       min_adv_path_pct=2.5)
    out = eval_all_gates(ctx, cfg)
    assert out["blocked"] is True
    assert any("tirex_tail_trigger" in r for r in out["block_reasons"])


def test_eval_all_gates_shadow_fire_does_not_block():
    ctx = _base_ctx(q10=DEEP_Q10)
    cfg = _base_config(enabled=True, shadow_mode=True)
    out = eval_all_gates(ctx, cfg)
    assert "tirex_tail_trigger" in out["results"]
    assert out["results"]["tirex_tail_trigger"].get("shadow_would_block") is True
    # nothing else configured -> not blocked by this gate
    assert "tirex_tail_trigger" not in "".join(out["block_reasons"])
