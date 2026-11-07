"""Tests for abstention_veto_gate — the second voice's ABSTENTION veto
(t_848eb1c4, shipped shadow-only).

Shape (settled by measurement 2026-11-06/07, 72 executed entries × two
205-scan passes — evidence in t_4432a31a comments 9–12 and skill
`hermes-trader-llm-model-comparison` → references/second-voice-veto-shape.md):
the duelist seat's OPPOSITE-side disagreement was 0/72 in BOTH passes, so
`duelist_veto_gate` can never fire for it. The signal lives in LOW-confidence
declines to endorse the entry, keyed on a conf CEILING (a conf FLOOR inverts
it: pass1 floor >=0.55 → 12 fires, −$2.19 vs ceiling 0.55 → 42 fires,
+$21.22). Primary conviction escapes the veto; a per-day flow cap bounds it,
because the uncapped rule cuts 58–61% of entries.

Pinned here: the fail-open contract (no verdict / no confidence / junk
confidence = the voice has no objection), the INCLUSIVE `<=` ceiling
boundary, the OPPOSITE-side delegation to duelist_veto_gate, escape
precedence (the primary's conviction outranks the voice's hesitation, and
its marker wins), the shadow marker (structurally passes), the per-day flow
cap counting shadow would-blocks, and the eval_all_gates registration that
puts the marker in `gate_results`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import risk_gates  # noqa: E402
from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    abstention_veto_gate,
)


@pytest.fixture(autouse=True)
def _reset_flow_cap():
    """The flow-cap counter is module state keyed by UTC date — reset it so
    test order can never decide whether a cap has been reached."""
    risk_gates._abstention_veto_blocks.clear()
    yield
    risk_gates._abstention_veto_blocks.clear()


def _ctx(
    verdict: str | None = "PASS",
    duelist_conf: float | None = 0.45,
    confidence: float = 0.70,
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
        duelist_confidence=duelist_conf,
    )


# The shipped posture, at the config values from the brief.
CFG_SHADOW = {"enabled": True, "shadow_mode": True, "max_confidence": 0.55,
              "escape_min_conf": 0.78, "escape_min_composite": 60}
CFG_LIVE = dict(CFG_SHADOW, shadow_mode=False)


# ── Fail-safes: a data gap can never block ─────────────────────────────────

def test_disabled_passes():
    assert abstention_veto_gate(_ctx(), {"enabled": False}) == {"pass": True}


def test_missing_config_passes():
    assert abstention_veto_gate(_ctx(), {}) == {"pass": True}
    assert abstention_veto_gate(_ctx(), None) == {"pass": True}  # type: ignore[arg-type]


def test_no_duelist_verdict_passes():
    # Duelist disabled / second call failed → no opinion.
    assert abstention_veto_gate(_ctx(verdict=None), CFG_SHADOW) == {"pass": True}
    assert abstention_veto_gate(_ctx(verdict=""), CFG_SHADOW) == {"pass": True}


def test_missing_duelist_confidence_fails_open():
    # The conf is a NEW field: a duel row written before it existed, or a
    # seat that never emitted one, must not arm the veto.
    r = abstention_veto_gate(_ctx(duelist_conf=None), CFG_SHADOW)
    assert r == {"pass": True}
    assert "shadow_would_block" not in r


def test_junk_duelist_confidence_fails_open():
    assert abstention_veto_gate(_ctx(duelist_conf="n/a"), CFG_SHADOW) == {"pass": True}  # type: ignore[arg-type]


# ── The non-agreeing key ───────────────────────────────────────────────────

def test_agreeing_verdict_passes():
    r = abstention_veto_gate(_ctx(verdict="LONG", trade_side="long"), CFG_SHADOW)
    assert r["pass"] is True and r["via"] == "agree"
    r = abstention_veto_gate(_ctx(verdict="SHORT", trade_side="short"), CFG_SHADOW)
    assert r["pass"] is True and r["via"] == "agree"


def test_opposite_side_is_delegated_to_duelist_veto():
    # Low conf on an OPPOSITE call is not this gate's cohort — keeping the
    # two gates' would-block sets disjoint is what makes a live flip of
    # either one attributable (and non-double-counting).
    r = abstention_veto_gate(_ctx(verdict="SHORT", trade_side="long",
                                  duelist_conf=0.20), CFG_SHADOW)
    assert r["pass"] is True and r["via"] == "opposite-side"
    assert "shadow_would_block" not in r


def test_pass_abstention_fires_in_shadow():
    r = abstention_veto_gate(_ctx(verdict="PASS", duelist_conf=0.45), CFG_SHADOW)
    assert r["pass"] is True
    assert r["shadow_would_block"] is True
    assert "abstention_veto" in r["reason"]
    # the voice's own conf — absent from every other log line — is IN the
    # reason string, so the accrual is readable straight off the log
    assert "0.45" in r["reason"] and "0.55" in r["reason"]


@pytest.mark.parametrize("verdict", ["CLOSE", "VETO", "banana"])
def test_other_non_agreeing_tokens_count_as_abstention(verdict):
    # CLOSE on an entry, and unrecognised tokens (incl. the retired chat-era
    # VETO), are all "not backing this entry".
    r = abstention_veto_gate(_ctx(verdict=verdict, duelist_conf=0.40), CFG_SHADOW)
    assert r.get("shadow_would_block") is True


# ── Ceiling boundary (inclusive) ───────────────────────────────────────────

def test_ceiling_boundary_is_inclusive():
    # `<=` — conf exactly on the bar fires. Off-by-one here silently moves
    # the biggest cell of the rule (the measured refusals cluster at 0.45
    # and 0.55: see the histogram in references/second-voice-veto-shape.md).
    r = abstention_veto_gate(_ctx(duelist_conf=0.55), CFG_SHADOW)
    assert r.get("shadow_would_block") is True
    r2 = abstention_veto_gate(_ctx(duelist_conf=0.56), CFG_SHADOW)
    assert r2["pass"] is True and r2["via"] == "above-ceiling"
    assert "shadow_would_block" not in r2


def test_conf_above_ceiling_is_not_an_objection():
    # A confident abstention is the shape the floor-keyed rule used to fire
    # on — measured NET-NEGATIVE (pass1 >=0.55 → −$2.19). It must pass.
    r = abstention_veto_gate(_ctx(duelist_conf=0.82), CFG_SHADOW)
    assert r["via"] == "above-ceiling"


# ── Escape: the primary's conviction outranks the hesitation ───────────────

def test_escape_on_primary_confidence():
    r = abstention_veto_gate(_ctx(duelist_conf=0.35, confidence=0.78), CFG_SHADOW)
    assert r["pass"] is True and r["via"] == "conviction-escape"
    # just below the bar still fires
    assert abstention_veto_gate(
        _ctx(duelist_conf=0.35, confidence=0.77), CFG_SHADOW
    ).get("shadow_would_block") is True


def test_escape_on_composite():
    r = abstention_veto_gate(_ctx(duelist_conf=0.35, composite=60.0), CFG_SHADOW)
    assert r["pass"] is True and r["via"] == "conviction-escape"


def test_escape_precedence_wins_over_ceiling_marker():
    # Confident voice AND confident primary: the escape is checked first, so
    # the accrued corpus says WHY it passed without re-running the rule.
    r = abstention_veto_gate(_ctx(duelist_conf=0.90, confidence=0.95), CFG_SHADOW)
    assert r["via"] == "conviction-escape"


# ── Shadow vs live ─────────────────────────────────────────────────────────

def test_shadow_mode_structurally_passes():
    r = abstention_veto_gate(_ctx(duelist_conf=0.45), CFG_SHADOW)
    assert r["pass"] is True and r["shadow_would_block"] is True


def test_live_mode_blocks():
    r = abstention_veto_gate(_ctx(duelist_conf=0.45), CFG_LIVE)
    assert r["pass"] is False and "shadow_would_block" not in r
    assert "abstention_veto" in r["reason"]


# ── Flow cap ───────────────────────────────────────────────────────────────

def test_uncapped_by_default():
    for _ in range(30):
        assert abstention_veto_gate(_ctx(duelist_conf=0.45), CFG_SHADOW).get(
            "shadow_would_block") is True


def test_flow_cap_stops_after_n_blocks():
    cfg = dict(CFG_SHADOW, max_blocks_per_day=3)
    for _ in range(3):
        assert abstention_veto_gate(_ctx(duelist_conf=0.45), cfg).get(
            "shadow_would_block") is True
    # the 4th is suppressed: passes, and says so greppably
    r = abstention_veto_gate(_ctx(duelist_conf=0.45), cfg)
    assert r["pass"] is True and r["via"] == "flow-cap"
    assert "flow cap" in r["reason"]
    assert "shadow_would_block" not in r
    # the counter counts the SUPPRESSED decision once, not forever
    assert abstention_veto_gate(_ctx(duelist_conf=0.45), cfg)["via"] == "flow-cap"


def test_flow_cap_counts_shadow_would_blocks():
    # The cap models what a LIVE flip would block, so the shadow accrual
    # must consume the budget too — otherwise the accrued cohort is the
    # uncapped rule and the cap in the acceptance gate is fiction.
    cfg = dict(CFG_SHADOW, max_blocks_per_day=2)
    abstention_veto_gate(_ctx(duelist_conf=0.50), cfg)   # shadow fire
    abstention_veto_gate(_ctx(duelist_conf=0.50), cfg)   # shadow fire
    assert abstention_veto_gate(_ctx(duelist_conf=0.50), cfg)["via"] == "flow-cap"


def test_flow_cap_rolls_over_on_the_utc_day_key():
    risk_gates._abstention_veto_blocks["2000-01-01"] = 99
    r = abstention_veto_gate(_ctx(duelist_conf=0.45),
                             dict(CFG_SHADOW, max_blocks_per_day=1))
    assert r.get("shadow_would_block") is True


# ── The parse_verdict MASK must not read as the strongest abstention ───────

def test_masked_parse_fails_open_and_costs_no_cap_slot():
    """parse_verdict's fallback for an unparseable/truncated answer is
    verdict="PASS" + confidence=0.0 (research.py:1862-1863). PASS is a declined
    endorsement for this gate and 0.0 is the LOWEST possible confidence, so the
    mask would otherwise be the strongest possible veto — and it would burn a
    `max_blocks_per_day` slot, silently throttling the real cohort."""
    r = abstention_veto_gate(_ctx(verdict="PASS", duelist_conf=0.0), CFG_SHADOW)
    assert r["pass"] is True
    assert r.get("via") == "parse-mask"
    assert "shadow_would_block" not in r
    assert not risk_gates._abstention_veto_blocks  # the cap counter never moved
    # A genuinely unconvincing decline at a hair above the mask still fires.
    f = abstention_veto_gate(_ctx(verdict="PASS", duelist_conf=0.05), CFG_SHADOW)
    assert f.get("shadow_would_block") is True


# ── Pipeline registration: the marker must reach gate_results ──────────────

def test_eval_all_gates_carries_abstention_veto():
    ctx = _ctx(verdict="PASS", duelist_conf=0.45, confidence=0.70,
               composite=40.0, trade_side="long")
    config = {
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
        "abstention_veto_gate": dict(CFG_SHADOW),
    }
    out = risk_gates.eval_all_gates(ctx, config, last_trade_time=None)
    assert "abstention_veto" in out["results"]
    r = out["results"]["abstention_veto"]
    assert r["pass"] is True
    assert r["shadow_would_block"] is True
    assert "abstention_veto" not in out["block_reasons"]


def test_eval_all_gates_without_config_block_is_inert():
    # Deploy order pins this: the image may carry the gate while the config
    # block is still absent — the result must be a plain pass, so the
    # rebuild delta is behaviourally nil before the config flips.
    ctx = _ctx(verdict="PASS", duelist_conf=0.45)
    out = risk_gates.eval_all_gates(ctx, {"min_ai_confidence": 0.5},
                                    last_trade_time=None)
    assert out["results"]["abstention_veto"] == {"pass": True}
