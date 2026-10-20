"""Tests for the timestral tail-trigger conviction gate (2026-10-21).

Accrual-first mirror of chronos/tirex_tail_trigger_gate on the Timestral-209M
adverse q-path (ctx.timestral_q10_path_pct / timestral_q90_path_pct). Same
rule, same escape bar, shadow-capable: the newest forecaster's stop-out
skill on our tape is unpriced, so it accrues would-block markers beside the
armed chronos gate and the tirex mirror until executed-cohort P/L ranks them.

Pins:
  (a) shape-fire in shadow mode -> pass True + shadow_would_block + join vars;
  (b) shadow_mode false -> REAL veto (pass False, reason);
  (c) no-shape / shallow tail -> pass True, no marker;
  (d) missing/None/short paths -> pass True, no marker, NO raise (fail-safe:
      a data gap can never block);
  (e) escape bar: conf >= 0.90 OR composite >= 60 clears the trip;
  (f) short side mirrors on q90 (max >= +x);
  (g) disabled -> plain pass;
  (h) eval_all_gates carries the timestral_tail_trigger key + shadow fire
      does not block;
  (i) fresh-impulse class closes the composite escape when configured;
  (j) executor wiring pinned by AST (gate-side read + shadow line);
  (k) live config block ships enabled + shadow_mode true (accrual posture).
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    eval_all_gates,
    timestral_tail_trigger_gate,
)

REPO = Path(__file__).parent.parent

# Mirror of the shipped timestral_tail_trigger_gate block (window 6 / X 2.5,
# same operating point as the tirex mirror).
TAIL_CFG = {
    "enabled": True,
    "shadow_mode": True,
    "window_steps": 6,
    "min_adv_path_pct": 2.5,
    "min_conf": 0.90,
    "min_composite": 60.0,
}


def _ctx(side="long", conf=0.70, composite=45.0, q10=None, q90=None,
         volume_spike=False, breakout=False, momentum=False):
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
        momentum_burst_fired=momentum,
        volume_spike_fired=volume_spike,
        breakout_fired=breakout,
        timestral_q10_path_pct=q10,
        timestral_q90_path_pct=q90,
    )


DEEP_Q10 = [-0.5, -1.2, -3.1, -2.0, -1.0, -0.5]     # min of first 6 = -3.1
SHALLOW_Q10 = [-0.4, -0.9, -1.5, -1.2, -0.8, -0.3]  # never breaches -2.5
DEEP_Q90 = [0.3, 1.1, 2.9, 2.2, 1.0, 0.4]           # max of first 6 = +2.9


# ── (a) shadow-mode fire ------------------------------------------------------
def test_shadow_fire_marks_would_block_with_join_vars():
    r = timestral_tail_trigger_gate(_ctx(q10=DEEP_Q10), TAIL_CFG)
    assert r["pass"] is True
    assert r["shadow_would_block"] is True
    assert r["tail_pct"] == -3.1 and r["window_steps"] == 6
    assert "timestral_tail_trigger" in r["reason"]


# ── (b) live mode blocks ------------------------------------------------------
def test_live_mode_blocks():
    cfg = dict(TAIL_CFG, shadow_mode=False)
    r = timestral_tail_trigger_gate(_ctx(q10=DEEP_Q10), cfg)
    assert r["pass"] is False and "timestral_tail_trigger" in r["reason"]


# ── (c) shallow tail passes ---------------------------------------------------
def test_shallow_tail_passes():
    r = timestral_tail_trigger_gate(_ctx(q10=SHALLOW_Q10),
                                    dict(TAIL_CFG, shadow_mode=False))
    assert r == {"pass": True}


def test_breach_outside_window_ignored():
    q = [0.0] * 6 + [-3.0, -4.0]  # breach only beyond window_steps=6
    r = timestral_tail_trigger_gate(_ctx(q10=q), dict(TAIL_CFG, shadow_mode=False))
    assert r == {"pass": True}


# ── (d) fail-safe: data gap can never block -----------------------------------
def test_missing_paths_pass():
    cfg = dict(TAIL_CFG, shadow_mode=False)
    assert timestral_tail_trigger_gate(_ctx(q10=None), cfg) == {"pass": True}
    assert timestral_tail_trigger_gate(_ctx(q10=[]), cfg) == {"pass": True}
    assert timestral_tail_trigger_gate(_ctx(q10=[-3.0] * 5), cfg) == {"pass": True}


def test_live_mode_still_passes_on_missing_data():
    r = timestral_tail_trigger_gate(_ctx(q90=None),
                                    dict(TAIL_CFG, shadow_mode=False))
    assert r == {"pass": True}


# ── (e) escape bar -------------------------------------------------------------
def test_high_conf_escape():
    r = timestral_tail_trigger_gate(_ctx(q10=DEEP_Q10, conf=0.90),
                                    dict(TAIL_CFG, shadow_mode=False))
    assert r == {"pass": True}


def test_high_composite_escape():
    r = timestral_tail_trigger_gate(_ctx(q10=DEEP_Q10, composite=60.0),
                                    dict(TAIL_CFG, shadow_mode=False))
    assert r == {"pass": True}


# ── (f) short side mirrors on q90 ----------------------------------------------
def test_short_side_uses_q90_max():
    r = timestral_tail_trigger_gate(
        _ctx(side="short", q90=DEEP_Q90), dict(TAIL_CFG, shadow_mode=False))
    assert r["pass"] is False


def test_short_side_ignores_q10():
    r = timestral_tail_trigger_gate(
        _ctx(side="short", q10=DEEP_Q10, q90=SHALLOW_Q10),
        dict(TAIL_CFG, shadow_mode=False))
    assert r == {"pass": True}


# ── (g) disabled ---------------------------------------------------------------
def test_disabled_plain_pass():
    r = timestral_tail_trigger_gate(_ctx(q10=DEEP_Q10), {"enabled": False})
    assert r == {"pass": True}


# ── (i) impulse-class composite-escape denial ----------------------------------
def test_impulse_class_closes_composite_escape():
    cfg = dict(TAIL_CFG, shadow_mode=False, no_composite_escape_on_impulse=True)
    ctx = _ctx(q10=DEEP_Q10, composite=80.0, volume_spike=True, breakout=True)
    r = timestral_tail_trigger_gate(ctx, cfg)
    assert r["pass"] is False
    assert "composite escape closed" in r["reason"]


# ── (h) eval_all_gates registration --------------------------------------------
def _base_config(**tail_override):
    tail = {"enabled": False}
    tail.update(tail_override)
    return {
        "min_confidence": 0.0,
        "max_trade_notional_pct": 100.0,
        "max_total_notional_pct": 100.0,
        "timestral_tail_trigger_gate": tail,
    }


def test_eval_all_gates_carries_key():
    out = eval_all_gates(_ctx(q10=None), _base_config())
    assert "timestral_tail_trigger" in out["results"]


def test_eval_all_gates_live_block_surfaces_in_reasons():
    ctx = _ctx(q10=DEEP_Q10)
    cfg = _base_config(enabled=True, shadow_mode=False, window_steps=6,
                       min_adv_path_pct=2.5)
    out = eval_all_gates(ctx, cfg)
    assert out["blocked"] is True
    assert any("timestral_tail_trigger" in r for r in out["block_reasons"])


def test_eval_all_gates_shadow_fire_does_not_block():
    ctx = _ctx(q10=DEEP_Q10)
    cfg = _base_config(enabled=True, shadow_mode=True)
    out = eval_all_gates(ctx, cfg)
    assert out["results"]["timestral_tail_trigger"].get("shadow_would_block") is True
    assert "timestral_tail_trigger" not in "".join(out["block_reasons"])


# ── (j) executor wiring pinned by AST ------------------------------------------
def test_executor_wiring_present():
    tree = ast.parse((REPO / "hermes_trader/agents/executor.py").read_text())
    src = ast.unparse(tree)
    # gate-side read guarded by timestral_signal.enabled
    assert "get_timestral_signal_sync" in src
    assert "timestral_q10_path_pct=_tms_q10p" in src
    assert "timestral_q90_path_pct=_tms_q90p" in src
    # shadow accrual line
    assert "[gate][SHADOW] timestral_tail_trigger WOULD HAVE BLOCKED" in src


# ── (k) shipped config posture: accrual, not blocking ---------------------------
def test_live_config_ships_gate_in_shadow_mode():
    cfg = json.loads((REPO / ".agent-config.json").read_text())
    block = cfg.get("timestral_tail_trigger_gate") or {}
    assert block.get("enabled") is True
    assert block.get("shadow_mode") is True
