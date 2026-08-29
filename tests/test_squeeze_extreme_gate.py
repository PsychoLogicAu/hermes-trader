"""Tests for the squeeze extreme-without-breakout conviction gate + the
research-prompt `_squeeze_block` renderer.

The gate is the third member of the shadow-gate house pattern (after
chronos_mismatch and band_counter_breach): it encodes the
"chasing without confirmation" bucket from the 2026-08-27 15-day ledger
replay — entries on the candidate's side while price sits at the extreme of
the prior 48h 1h Donchian range with NO fresh aligned breakout confirming
the move. The flag is computed by squeeze_signal (one sync read per
candidate side) and fed via GateContext.squeeze_extreme_no_breakout; the
gate itself is a pure ctx function.

Shadow-mode semantics (the live default) are tested here too: the gate MUST
structurally pass and carry shadow_would_block until shadow_mode is flipped
off. Fail-safes: disabled / no flag data / not at the extreme / conviction
bar met all yield a plain pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    eval_all_gates,
    squeeze_extreme_gate,
)
from hermes_trader.agents.squeeze_signal import SqueezeSignal  # noqa: E402
import hermes_trader.agents.research as research  # noqa: E402
import hermes_trader.agents.squeeze_signal as squeeze_mod  # noqa: E402


def _ctx(
    confidence: float = 0.75,
    composite: float = 40.0,
    flag=None,
) -> GateContext:
    return GateContext(
        confidence=confidence,
        current_positions=[],
        trade_notional_usd=10.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin="TEST",
        trade_side="long",
        has_binary_news_risk=False,
        equity=100.0,
        total_open_notional=0.0,
        composite_score=composite,
        squeeze_extreme_no_breakout=flag,
    )


# ── Fail-safes (no-opinion pass) ────────────────────────────────────────────

def test_disabled_passes():
    r = squeeze_extreme_gate(_ctx(flag=True), {"enabled": False})
    assert r == {"pass": True}


def test_missing_config_passes():
    r = squeeze_extreme_gate(_ctx(flag=True), {})
    assert r == {"pass": True}


def test_no_data_passes():
    """Flag None (squeeze disabled / fetch failed) — data gap can never block."""
    r = squeeze_extreme_gate(_ctx(flag=None), {"enabled": True})
    assert r == {"pass": True}


def test_not_at_extreme_passes():
    r = squeeze_extreme_gate(_ctx(flag=False), {"enabled": True})
    assert r == {"pass": True}


# ── Shadow mode (the live default) ──────────────────────────────────────────

def test_shadow_mode_structurally_passes_with_marker():
    cfg = {"enabled": True, "shadow_mode": True, "min_conf": 0.90,
           "min_composite": 60}
    r = squeeze_extreme_gate(_ctx(confidence=0.75, composite=40.0, flag=True), cfg)
    assert r["pass"] is True
    assert r["shadow_would_block"] is True
    assert "squeeze_extreme" in r["reason"]


def test_shadow_mode_default_is_shadow():
    """shadow_mode absent from config means SHADOW (house pattern)."""
    r = squeeze_extreme_gate(_ctx(flag=True), {"enabled": True})
    assert r["pass"] is True
    assert r["shadow_would_block"] is True


def test_conviction_bar_met_passes_even_live():
    cfg = {"enabled": True, "shadow_mode": False, "min_conf": 0.90,
           "min_composite": 60}
    r = squeeze_extreme_gate(_ctx(confidence=0.91, composite=40.0, flag=True), cfg)
    assert r == {"pass": True}


def test_composite_bar_met_passes_even_live():
    cfg = {"enabled": True, "shadow_mode": False, "min_conf": 0.90,
           "min_composite": 60}
    r = squeeze_extreme_gate(_ctx(confidence=0.75, composite=61.0, flag=True), cfg)
    assert r == {"pass": True}


def test_bar_boundary_is_inclusive():
    cfg = {"enabled": True, "shadow_mode": False, "min_conf": 0.90,
           "min_composite": 60}
    assert squeeze_extreme_gate(_ctx(confidence=0.90, composite=40.0, flag=True), cfg) == {"pass": True}
    assert squeeze_extreme_gate(_ctx(confidence=0.75, composite=60.0, flag=True), cfg) == {"pass": True}


# ── Live mode (shadow_mode flipped off) ─────────────────────────────────────

def test_live_mode_blocks_low_conviction():
    cfg = {"enabled": True, "shadow_mode": False, "min_conf": 0.90,
           "min_composite": 60}
    r = squeeze_extreme_gate(_ctx(confidence=0.75, composite=40.0, flag=True), cfg)
    assert r["pass"] is False
    assert "no fresh aligned breakout" in r["reason"]


def test_eval_all_gates_wires_squeeze_extreme(monkeypatch):
    """The gate must be in eval_all_gates' results — a gate that isn't wired
    never fires (the opposite_direction_guard precedent)."""
    import hermes_trader.agents.risk_gates as rg
    src = Path(rg.__file__).read_text()
    assert 'results["squeeze_extreme"]' in src


# ── Research prompt block ───────────────────────────────────────────────────

def _sig(**kw) -> SqueezeSignal:
    base = dict(
        coin="TEST", verdict_side="long", active=False, side=None,
        score=None, close=None, chan_high=None, chan_low=None,
        ext_pct=None, atr1h=None, atr1h_pct=None, fresh_age_min=None,
        breakout_bar_t=None, lookback=48, chan_pos=None,
        extreme_no_breakout=False, logged=False, error=None,
    )
    base.update(kw)
    return SqueezeSignal(**base)


def test_squeeze_block_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(
        research, "read_agent_config",
        lambda: {"squeeze_signal": {"enabled": False}})
    assert research._squeeze_block("TEST") == ""


def test_squeeze_block_renders_caution(monkeypatch):
    monkeypatch.setattr(
        research, "read_agent_config",
        lambda: {"squeeze_signal": {"enabled": True}})
    monkeypatch.setattr(
        squeeze_mod, "get_squeeze_signal_sync",
        lambda coin, side: _sig(active=False, chan_pos=0.992,
                                extreme_no_breakout=True,
                                error="no breakout (inactive)"))
    block = research._squeeze_block("TEST")
    assert "Squeeze / 48h channel state" in block
    assert "CAUTION" in block
    assert "very top of the 48h range" in block


def test_squeeze_block_renders_fresh_breakout(monkeypatch):
    monkeypatch.setattr(
        research, "read_agent_config",
        lambda: {"squeeze_signal": {"enabled": True}})
    monkeypatch.setattr(
        squeeze_mod, "get_squeeze_signal_sync",
        lambda coin, side: _sig(active=True, side="long", ext_pct=1.8,
                                fresh_age_min=9.0, chan_pos=1.004))
    block = research._squeeze_block("TEST")
    assert "FRESH LONG breakout" in block
    assert "CAUTION" not in block


def test_squeeze_block_failure_returns_empty(monkeypatch):
    monkeypatch.setattr(
        research, "read_agent_config",
        lambda: {"squeeze_signal": {"enabled": True}})

    def _boom(coin, side):
        raise RuntimeError("candle fetch failed")

    monkeypatch.setattr(squeeze_mod, "get_squeeze_signal_sync", _boom)
    assert research._squeeze_block("TEST") == ""