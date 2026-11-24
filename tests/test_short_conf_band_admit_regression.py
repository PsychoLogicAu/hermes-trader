"""Regression: non-lane SHORTs in the 0.60-0.65 conf band are admitted after the
2026-11-24 config alignment (kanban t_c002f54c / t_e5b860cb).

The standing owner ruling (cfg 2c63752, reaffirmed t_ec17e503) sets the SHORT
conf floor at 0.60. Until 2026-11-24 the LIVE config contradicted it:
`min_ai_confidence` 0.65 (top-level AND the mammoth-1 seat-profile `gate_bars`
overlay, which wins via apply_seat_gate_bars) re-blocked at the CLASSIC gate
every short the seat's 0.60 `entry_min_short_confidence` was built to admit —
measured by t_552a0b2c: 1,210 deduped `confidence 0.6x < 0.65` refusals against
0 `short confidence` refusals in 9 days of the mammoth seat era. The fix is
config-only: classic floor 0.65->0.60; the general runner bar stays 0.65
because under deployed image f830c0a6 it is the LONG door (the conf-floor
precedence code cc86fe00 was not yet deployed — it now lives on this branch).

This test pins the PAIRED config shape end-to-end at the gate level (fake
entry, non-lane: no cohort_tag, no lane tag, no state_admission clause):

  paired (LIVE, config repo 63b6944):
    classic min_ai_confidence 0.60 (top + profile gate_bars),
    runner general min_confidence 0.65, runner min_short_confidence 0.60,
    seat profile: entry_min_confidence 0.65 / entry_min_short_confidence 0.60.
  legacy_mismatch (the config the ruling replaced):
    EVERY short bar back at 0.65 — classic floor, runner side bar, and the
    seat's entry_min_short_confidence. NOTE on shape choice: on the deployed
    image f830c0a6 the pre-fix live file kept the seat side bar at 0.60 while
    the general bar was 0.65 — the refusal came from the classic gate reading
    ONLY the general 0.65, and that exact general-bar mismatch is pinned by
    test_deployed_image_classic_floor_pin below. With cc86fe00's precedence
    merged, a 0.60 seat side bar alone now admits the band; the config shape
    that still refuses on this branch — the one the alignment removed — is
    all-bars-0.65, so that is what the legacy fixture encodes.

Acceptance (t_e5b860cb): a SHORT conf 0.62 is ADMITTED by both conf gates under
the paired config — no `confidence 0.6x < 0.65` refusal text anywhere, the
runner even logs the [gate][SEAT-BAR] ADMIT — and the SAME assertions FAIL on
the legacy shape (measured refusal `confidence 0.62 < 0.65`). The LONG side is
invariant: conf 0.62 is refused at the 0.65 long floor on every shape, pinned
by exact reason string (t_c002f54c handoff: "pin the long-block reason string
'confidence 0.62 < 0.65'").
"""

import json
import logging

import pytest

from hermes_trader.agents import config_store, executor, risk_gates

CONF = 0.62
SEAT = "seat-r"
_BAND_REFUSAL = "confidence 0.62 < 0.65"  # the audit's 1,210-refusal text
_RUNNER_REFUSAL = f"runner_gate_blocked ({_BAND_REFUSAL})"


# ── the two config shapes ────────────────────────────────────────────────────

def _cfg(*, classic_floor: float, runner_side_short: float):
    """Top-level agent config. Runner GENERAL bar is 0.65 on both shapes —
    under deployed code it is the long door (see module docstring)."""
    return {
        "min_ai_confidence": classic_floor,
        "runner_entry_gate": {
            "enabled": True,
            "min_confidence": 0.65,
            "min_short_confidence": runner_side_short,
            "min_composite": 30.0,
            "allow_shorts": True,
        },
    }


_PROFILES = {
    # paired: live shape after t_c002f54c (config repo 63b6944)
    "paired": (
        _cfg(classic_floor=0.60, runner_side_short=0.60),
        {
            "model": SEAT,
            "entry_min_confidence": 0.65,
            "entry_min_short_confidence": 0.60,
            "gate_bars": {
                "min_ai_confidence": 0.60,
                "runner_entry_gate": {
                    "min_confidence": 0.65,
                    "min_short_confidence": 0.60,
                },
            },
        },
    ),
    # legacy_mismatch: the pre-alignment config family — every short bar at
    # 0.65 (classic floor, runner side bar, seat side bar). This is the shape
    # that still refuses the band on this branch (cc86fe00 precedence would
    # only help if SOME side bar said 0.60 — none did).
    "legacy_mismatch": (
        _cfg(classic_floor=0.65, runner_side_short=0.65),
        {
            "model": SEAT,
            "entry_min_confidence": 0.65,
            "entry_min_short_confidence": 0.65,
            "gate_bars": {
                "min_ai_confidence": 0.65,
                "runner_entry_gate": {
                    "min_confidence": 0.65,
                    "min_short_confidence": 0.65,
                },
            },
        },
    ),
}


@pytest.fixture
def seat_dir(tmp_path, monkeypatch):
    """Isolated MODEL_CONFIG_DIR (same seam as tests/test_conf_floor_precedence.py)."""
    monkeypatch.setattr(config_store, "MODEL_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _install_shape(seat_dir, shape):
    cfg, profile = _PROFILES[shape]
    (seat_dir / f"{SEAT}.json").write_text(json.dumps(profile))
    return cfg


# ── fake entries (non-lane: no cohort_tag, no lane tag, no clause) ───────────

def _analysis(side, **over):
    a = {"coin": "PUMP", "side": side, "confidence": CONF, "composite_score": 40.0,
         "volume_spike_fired": True, "breakout_fired": True,
         "downtrend_momentum_fired": True, "primary_model": SEAT}
    a.update(over)
    return a


def _ctx(side, **over):
    kw = dict(confidence=CONF, current_positions=[], trade_notional_usd=50.0,
              daily_pnl=0.0, market_volume_24h_usd=500e6, coin="PUMP",
              trade_side=side, has_binary_news_risk=False, equity=1000.0,
              total_open_notional=50.0, primary_model=SEAT)
    kw.update(over)
    return risk_gates.GateContext(**kw)


# ── the regression matrix ────────────────────────────────────────────────────

def test_paired_config_admits_band_short_both_gates(seat_dir, caplog):
    cfg = _install_shape(seat_dir, "paired")
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")

    # 1. runner gate: no refusal at all — SEAT-BAR ADMIT below the general bar
    reason = executor._runner_entry_block_reason(_analysis("short"), cfg)
    assert reason == ""
    assert "[gate][SEAT-BAR] ADMIT PUMP SHORT" in caplog.text

    # 2. classic gate: conf result passes and no refusal text carries the old
    # 0.65 bar anywhere in the block reasons.
    out = risk_gates.eval_all_gates(_ctx("short"), cfg)
    assert out["results"]["confidence"]["pass"] is True
    assert not any(_BAND_REFUSAL in r for r in out["block_reasons"])


def test_legacy_config_refuses_band_short(seat_dir, caplog):
    """The acceptance test's negative control: identical assertions against the
    OLD mismatched config must fail — measured refusal `confidence 0.62 < 0.65`
    on BOTH gates — and the SEAT-BAR ADMIT never happens."""
    cfg = _install_shape(seat_dir, "legacy_mismatch")
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")

    out = risk_gates.eval_all_gates(_ctx("short"), cfg)
    assert out["results"]["confidence"]["pass"] is False
    assert out["results"]["confidence"]["reason"] == _BAND_REFUSAL
    assert any(_BAND_REFUSAL in r for r in out["block_reasons"])

    assert executor._runner_entry_block_reason(_analysis("short"), cfg) == _RUNNER_REFUSAL
    assert "[gate][SEAT-BAR] ADMIT" not in caplog.text


def test_long_band_entry_refused_on_every_shape(seat_dir):
    """The long side is INVARIANT across both shapes: the 0.65 long floor keeps
    blocking conf 0.62 longs — pinned by exact reason string on both the
    classic gate and the runner seat-bar branch."""
    for shape in ("paired", "legacy_mismatch"):
        cfg = _install_shape(seat_dir, shape)

        out = risk_gates.eval_all_gates(_ctx("long"), cfg)
        assert out["results"]["confidence"]["pass"] is False, shape
        assert out["results"]["confidence"]["reason"] == _BAND_REFUSAL, shape

        assert (executor._runner_entry_block_reason(
            _analysis("long"), cfg) == _RUNNER_REFUSAL), shape


def test_shapes_differ_only_on_the_short_band(seat_dir):
    """Discrimination proof: the paired-vs-legacy assertion pair is not
    vacuous — the SAME classic-gate call flips outcome between the shapes."""
    outcomes = {}
    for shape in ("paired", "legacy_mismatch"):
        cfg = _install_shape(seat_dir, shape)
        outcomes[shape] = risk_gates.eval_all_gates(
            _ctx("short"), cfg)["results"]["confidence"]["pass"]
    assert outcomes["paired"] is True
    assert outcomes["legacy_mismatch"] is False


# ── config-level pins (t_c002f54c handoff's corrected guard predicate) ──────

def test_effective_floors_match_live_ruling(seat_dir):
    """The EFFECTIVE short floor (seat side bar, per the cc86fe00 precedence)
    is 0.60 under the paired config and 0.65 under the legacy one; the long
    floor is 0.65 on both."""
    for shape, want in (("paired", 0.60), ("legacy_mismatch", 0.65)):
        cfg = _install_shape(seat_dir, shape)
        merged = config_store.apply_seat_gate_bars(cfg, SEAT)
        floor, _src = config_store.effective_conf_floor(merged, "short", SEAT)
        assert floor == want, shape
        long_floor, _ = config_store.effective_conf_floor(merged, "long", SEAT)
        assert long_floor == 0.65, shape
        # corrected t_5f7a5b5b predicate: the EFFECTIVE classic floor (general
        # bar AFTER the seat gate_bars overlay) must sit at or below the
        # seat's short bar, or the classic gate re-blocks what the seat admits.
        general_after_overlay = float(merged["min_ai_confidence"])
        seat_short_bar = float(
            config_store.read_model_profile_for_model(SEAT)["entry_min_short_confidence"])
        assert general_after_overlay <= seat_short_bar, shape


def test_deployed_image_classic_floor_pin(seat_dir):
    """The exact pre-fix live shape (seat side bar 0.60, general bar 0.65):
    under deployed image f830c0a6 the CLASSIC gate read ONLY the general bar
    (no side tiers — effective_conf_floor absent), so the band died at 0.65.
    Pinned here without the side tiers: the general bar after the profile
    overlay is the whole floor on that image. The alignment lowered THAT value
    to 0.60 in both files — the paired config keeps general-after-overlay <=
    the seat short bar on both code versions; the pre-fix one only worked
    above the band."""
    cfg, profile = _PROFILES["legacy_mismatch"]
    pre_fix_profile = dict(profile, entry_min_short_confidence=0.60,
                           gate_bars=dict(profile["gate_bars"],
                                          min_ai_confidence=0.65))
    (seat_dir / f"{SEAT}.json").write_text(json.dumps(pre_fix_profile))
    merged = config_store.apply_seat_gate_bars(cfg, SEAT)
    # what the DEPLOYED image's classic floor would have been: general only
    deployed_floor = float(merged["min_ai_confidence"])
    assert deployed_floor == 0.65  # -> 0.62 short refused `confidence 0.62 < 0.65`
    # under this branch's code the same file admits via the seat side bar —
    # which is exactly why the config alignment (not only the code fix) was
    # required for the hot bander on the deployed image.
    floor, src = config_store.effective_conf_floor(merged, "short", SEAT)
    assert (floor, src) == (0.60, "seat_profile")
