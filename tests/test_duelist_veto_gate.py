"""Tests for the duelist-veto conviction gate (re-keyed 2026-09-03).

The gate fires when the primary issues a DIRECTIONAL call (LONG/SHORT) and
the A/B duelist EXPLICITLY rejects the setup — verdict VETO — or takes the
opposite side, unless the entry clears the elevated-conviction bar
(conf >= min_conf OR composite >= min_composite).

The re-key: the gate used to fire on the duelist's PASS (treating every
abstention as an objection). With VETO now a first-class verdict ("this setup
is a trap — avoid like the plague") and PASS redefined in the prompt as
neutral abstention, PASS no longer vetoes. Motivation: the 2026-09-03 headroom
replay found the PASS-keyed veto blocking 27/32 executed entries (the duelist
PASSes ~84% of everything) with a blocked cohort net +$0.57 — bare abstention
carries no quality signal.

Shadow-mode semantics (the live default) are tested here too: the gate
MUST structurally pass and carry `shadow_would_block` until shadow_mode
is flipped off. Fail-safes: disabled / no duelist verdict / duelist agrees
with the primary / duelist PASS all yield a plain pass — a data gap can
never block.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    duelist_veto_gate,
)


def _ctx(
    verdict: str | None = "VETO",
    confidence: float = 0.75,
    composite: float = 40.0,
    trade_side: str = "long",
) -> GateContext:
    return GateContext(
        confidence=confidence,
        current_positions=[],
        trade_notional_usd=10.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin="TEST",
        trade_side=trade_side,
        has_binary_news_risk=False,
        equity=100.0,
        total_open_notional=0.0,
        composite_score=composite,
        duelist_verdict=verdict,
    )


CFG_SHADOW = {"enabled": True, "shadow_mode": True}
CFG_LIVE = {"enabled": True, "shadow_mode": False}


# ── Fail-safes (no-opinion pass) ────────────────────────────────────────────

def test_disabled_passes():
    r = duelist_veto_gate(_ctx(verdict="VETO"), {"enabled": False})
    assert r == {"pass": True}


def test_missing_config_passes():
    r = duelist_veto_gate(_ctx(verdict="VETO"), {})
    assert r == {"pass": True}


def test_no_duelist_verdict_passes():
    # Duelist disabled or the second LLM call failed → no opinion.
    r = duelist_veto_gate(_ctx(verdict=None), CFG_SHADOW)
    assert r == {"pass": True}
    r2 = duelist_veto_gate(_ctx(verdict=""), CFG_SHADOW)
    assert r2 == {"pass": True}


def test_duelist_agrees_long_passes():
    r = duelist_veto_gate(_ctx(verdict="LONG", trade_side="long"), CFG_SHADOW)
    assert r == {"pass": True}


def test_duelist_agrees_short_passes():
    r = duelist_veto_gate(_ctx(verdict="SHORT", trade_side="short"), CFG_SHADOW)
    assert r == {"pass": True}


def test_unrecognised_verdict_passes():
    r = duelist_veto_gate(_ctx(verdict="MAYBE"), CFG_SHADOW)
    assert r == {"pass": True}


# ── THE re-key: PASS is neutral abstention, NOT a veto ──────────────────────

def test_duelist_pass_no_longer_vetoes():
    # The old rule vetoed here. Under the VETO key a bare PASS carries no
    # objection — "not excited" != "avoid like the plague".
    r = duelist_veto_gate(_ctx(verdict="PASS", trade_side="long"), CFG_SHADOW)
    assert r == {"pass": True}


def test_duelist_pass_no_longer_vetoes_lowered_bars():
    # Not even with the escape bars set unreachable — PASS simply never fires.
    cfg = {"enabled": True, "shadow_mode": True, "min_conf": 1.0, "min_composite": 999.0}
    r = duelist_veto_gate(_ctx(verdict="PASS", confidence=0.5, composite=10.0), cfg)
    assert r == {"pass": True}


# ── The veto shapes: explicit VETO / opposite side ──────────────────────────

def test_shadow_explicit_veto_would_block():
    r = duelist_veto_gate(_ctx(verdict="VETO", confidence=0.75, composite=40.0),
                          CFG_SHADOW)
    assert r["pass"] is True
    assert r["shadow_would_block"] is True
    assert "VETO" in r["reason"]


def test_shadow_lowercased_verdict():
    r = duelist_veto_gate(_ctx(verdict="veto"), CFG_SHADOW)
    assert r["pass"] is True
    assert r["shadow_would_block"] is True


def test_shadow_opposite_side_veto():
    # The duelist took the OPPOSITE side — stronger disagreement, same veto.
    r = duelist_veto_gate(_ctx(verdict="SHORT", trade_side="long"), CFG_SHADOW)
    assert r["pass"] is True
    assert r["shadow_would_block"] is True
    r2 = duelist_veto_gate(_ctx(verdict="LONG", trade_side="short"), CFG_SHADOW)
    assert r2["pass"] is True
    assert r2["shadow_would_block"] is True


def test_shadow_high_confidence_escape():
    # The elevated-conviction bar: conf >= min_conf (0.90 default) escapes.
    r = duelist_veto_gate(_ctx(verdict="VETO", confidence=0.92), CFG_SHADOW)
    assert r == {"pass": True}


def test_shadow_high_composite_escape():
    # OR composite >= min_composite (60.0 default) escapes.
    r = duelist_veto_gate(_ctx(verdict="VETO", composite=65.0), CFG_SHADOW)
    assert r == {"pass": True}


def test_shadow_bar_just_below():
    # Both just under the bars → the veto stands.
    r = duelist_veto_gate(_ctx(verdict="VETO", confidence=0.89, composite=59.0),
                          CFG_SHADOW)
    assert r["pass"] is True
    assert r["shadow_would_block"] is True


# ── Live mode (shadow_mode off): the veto actually blocks ───────────────────

def test_live_mode_blocks():
    r = duelist_veto_gate(_ctx(verdict="VETO", confidence=0.75, composite=40.0),
                          CFG_LIVE)
    assert r["pass"] is False
    assert "shadow_would_block" not in r
    assert "VETO" in r["reason"]


def test_live_mode_pass_still_passes():
    # The re-key holds in live mode too: PASS never blocks.
    r = duelist_veto_gate(_ctx(verdict="PASS", confidence=0.5, composite=10.0),
                          CFG_LIVE)
    assert r == {"pass": True}


def test_live_mode_escape_still_applies():
    r = duelist_veto_gate(_ctx(verdict="VETO", confidence=0.95), CFG_LIVE)
    assert r == {"pass": True}


# ── Custom bars ─────────────────────────────────────────────────────────────

def test_custom_bars():
    cfg = {"enabled": True, "shadow_mode": True,
           "min_conf": 0.80, "min_composite": 30.0}
    # conf 0.75 < 0.80 but composite 40 >= 30 → escape via composite.
    r = duelist_veto_gate(_ctx(verdict="VETO", confidence=0.75, composite=40.0),
                          cfg)
    assert r == {"pass": True}
    # Both under the custom bars → would-block.
    r2 = duelist_veto_gate(_ctx(verdict="VETO", confidence=0.70, composite=25.0),
                           cfg)
    assert r2["pass"] is True
    assert r2["shadow_would_block"] is True


# ── eval_all_gates integration ──────────────────────────────────────────────

def test_eval_all_gates_carries_duelist_veto():
    from hermes_trader.agents.risk_gates import eval_all_gates
    ctx = _ctx(verdict="VETO", confidence=0.75, composite=40.0,
               trade_side="long")
    config = {
        # High bars so the duelist veto can fire in shadow (not block).
        "min_ai_confidence": 0.5,
        "max_concurrent": 5,
        "max_trade_notional_usd": 300,
        "min_market_volume_usd": 1_000_000,
        "coin_allowlist": [],
        "coin_blocklist": [],
        "cooldown_min": 0,
        "max_crypto_long_correlated": 2,
        "max_total_notional_pct": 1.0,
        "counter_regime_min_conf": 0.0,
        "block_counter_trend_bypass": False,
        "duelist_veto_gate": {
            "enabled": True,
            "shadow_mode": True,
            "min_conf": 0.90,
            "min_composite": 60.0,
        },
    }
    out = eval_all_gates(ctx, config, last_trade_time=None)
    assert "duelist_veto" in out["results"]
    r = out["results"]["duelist_veto"]
    assert r["pass"] is True
    assert r["shadow_would_block"] is True
    # Shadow mode must NOT block the overall gate stack.
    assert not (out["blocked"] and "duelist_veto" in out["block_reasons"])
