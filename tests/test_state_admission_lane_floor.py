"""Lane-scoped short-liquidity floor for the STATE-ADMISSION CLAUSE cohort
(t_fb899217, owner ruling 2026-11-25: `$3,000,000`).

The clause's cohort had no identity the floor resolver could match: the clause
tags its admit in `gate_results` (`clause_lane_tag`) but that is consumed only
AFTER `eval_all_gates` (executor.py), while `lane_short_floor` reads
`_lane_identity_keys(ctx)` DURING it. The change stamps `ctx.clause_tag` when
`_state_conf_admission` admits, and appends that identity LAST in
`_lane_identity_keys` (after cohort/entry_class, compound `<model>/<tag>` before
the bare tag).

Costed basis (shadow accrual 2026-11-07..25, 489 clause admits, episode-deduped,
BN2, priced at the live cap-8 clause book): alpha +$10.88 at the `$20M` seat floor
vs +$32.00 at `$3M`, marginal of the 4M->3M step +$14.69 with 0 stop-outs, and
going to `$1M` adds only +$1.0 more. The `$1M` general market floor
(`min_market_volume_usd`) is untouched, so sub-`$1M` junk still dies for every
path.

Pins: the scoped floor applies ONLY to a clause admit; absent key = byte-identical
global behaviour; a clause admit refused by the dedicated slot budget does NOT
acquire the identity; cohort/entry_class keys still win over the clause key; the
block reason carries the effective floor; malformed/non-finite/bool values keep
the global floor and stay loud; an inert clause leaks nothing to the seat path.
"""
import logging

import pytest

from hermes_trader.agents import hyperfeed, market_regime
from hermes_trader.agents import risk_gates
from hermes_trader.agents.risk_gates import GateContext, eval_all_gates

CLAUSE = {"short": {"shadow": False, "min_conf": 0.10}}


def _clause_shape(vol_usd, coin="MEGA", model="mammoth-1", cohort=None):
    """A SHORT the confidence gate blocks (conf 0.30 < 0.65) and the clause
    admits. No `cohort_tag` unless passed — that is the point: the clause cohort
    is defined by the gate that admitted it, not by a cohort tag."""
    return GateContext(
        confidence=0.30, current_positions=[], trade_notional_usd=72.0,
        daily_pnl=0.0, market_volume_24h_usd=vol_usd, coin=coin,
        trade_side="short", has_binary_news_risk=False, equity=900.0,
        total_open_notional=0.0, composite_score=40.0,
        cohort_tag=cohort, primary_model=model)


def _seat_shape(vol_usd, confidence=0.70):
    """Confidence passes on its own, so the clause never runs and never stamps."""
    return GateContext(
        confidence=confidence, current_positions=[], trade_notional_usd=700.0,
        daily_pnl=0.0, market_volume_24h_usd=vol_usd, coin="MEGA",
        trade_side="short", has_binary_news_risk=False, equity=900.0,
        total_open_notional=0.0, composite_score=40.0, primary_model="mammoth-1")


def _cfg(extra=None):
    cfg = {
        "min_ai_confidence": 0.65,
        "min_market_volume_usd": 1_000_000,
        "min_short_volume_usd": 20_000_000,
        "max_concurrent": 8,
        "runner_entry_gate": {"allow_shorts": True},
    }
    if extra:
        cfg.update(extra)
    return cfg


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    monkeypatch.setattr(market_regime, "detect_regime", lambda c: "down")
    monkeypatch.setattr(market_regime, "classify_asset", lambda c: "crypto")
    monkeypatch.setattr(market_regime, "btc_tape_activity",
                        lambda force=False: None)
    monkeypatch.setattr(market_regime, "btc_tape_history",
                        lambda force=False: None)
    monkeypatch.setattr(hyperfeed, "market_get_funding_regime",
                        lambda: {"regime": "NEUTRAL",
                                 "regimes_by_class": {"crypto": "NEUTRAL"}})
    monkeypatch.setattr(risk_gates, "read_model_profile_for_model",
                        lambda model: ({"state_admission": CLAUSE}
                                       if model in ("mammoth-1", "mechanical")
                                       else {}))
    # The clause verdict itself is pinned elsewhere (test_state_admission*.py);
    # here the subject is the IDENTITY the admit leaves behind for the floor.
    monkeypatch.setattr(risk_gates, "state_admission",
                        lambda a, c, s: (True, "unit stub: kept tape, conf 0.30", False))


def _sl(ctx, cfg):
    return eval_all_gates(ctx, cfg)["results"]["short_liquidity"]


def _conf(ctx, cfg):
    return eval_all_gates(ctx, cfg)["results"]["confidence"]


# ---- the knob does what the owner bought it for -----------------------------

def test_scoped_floor_lets_a_clause_admit_through(caplog):
    cfg = _cfg({"lane_min_short_volume_usd": {"state_admission": 3_000_000}})
    with caplog.at_level(logging.INFO):
        r = _sl(_clause_shape(6_000_000), cfg)
    assert r["pass"] is True
    assert any("[LANE-FLOOR]" in rec.message and "$3.00M" in rec.message
               and "lane=state_admission" in rec.message for rec in caplog.records)


def test_absent_key_is_byte_identical_to_the_global_floor():
    """No key = the clause short still eats the $20M seat floor, exactly as
    before the change: the stamp alone must not loosen anything."""
    r = _sl(_clause_shape(6_000_000), _cfg())
    assert r["pass"] is False and "$20M" in r["reason"]


def test_compound_key_applies_and_stays_model_exclusive():
    cfg = _cfg({"lane_min_short_volume_usd":
                {"mammoth-1/state_admission": 3_000_000}})
    assert _sl(_clause_shape(6_000_000), cfg)["pass"] is True
    other = _cfg({"lane_min_short_volume_usd":
                  {"other-model/state_admission": 3_000_000}})
    r = _sl(_clause_shape(6_000_000), other)
    assert r["pass"] is False and "$20M" in r["reason"]


def test_below_the_scoped_floor_still_blocked_and_names_that_floor():
    cfg = _cfg({"lane_min_short_volume_usd": {"state_admission": 3_000_000}})
    r = _sl(_clause_shape(2_500_000), cfg)
    assert r["pass"] is False and "short floor $3M" in r["reason"]


# ---- no leakage to any other cohort ----------------------------------------

def test_seat_path_never_sees_the_clause_floor():
    cfg = _cfg({"lane_min_short_volume_usd": {"state_admission": 3_000_000}})
    r = _sl(_seat_shape(6_000_000), cfg)
    assert r["pass"] is False and "$20M" in r["reason"]


def test_inert_clause_leaves_the_identity_unset(monkeypatch):
    """Clause refuses (inert / band out of range) -> nothing is keyable, the
    scoped floor must not apply to a non-clause short of the same model."""
    monkeypatch.setattr(risk_gates, "state_admission",
                        lambda a, c, s: (False, "unit stub: tape out of band", False))
    cfg = _cfg({"lane_min_short_volume_usd": {"state_admission": 3_000_000}})
    ctx = _clause_shape(6_000_000)
    assert _conf(ctx, cfg)["pass"] is False
    assert _sl(ctx, cfg)["pass"] is False and "$20M" in _sl(ctx, cfg)["reason"]


def test_shadow_clause_does_not_acquire_the_identity(monkeypatch):
    """Shadow mode logs WOULD ADMIT and returns the original block: it must not
    hand out a floor override either."""
    monkeypatch.setattr(risk_gates, "state_admission",
                        lambda a, c, s: (True, "unit stub: shadow", True))
    cfg = _cfg({"lane_min_short_volume_usd": {"state_admission": 3_000_000}})
    ctx = _clause_shape(6_000_000)
    assert _conf(ctx, cfg)["pass"] is False
    r = _sl(ctx, cfg)
    assert r["pass"] is False and "$20M" in r["reason"]


def test_slot_budget_refusal_does_not_acquire_the_identity(monkeypatch):
    """A clause admit refused by the dedicated clause slot book is not an
    admit: the global floor must still stand."""
    monkeypatch.setattr(
        risk_gates, "clause_slot_budget_check",
        lambda clause, side, cfg: {"allow": False, "used": 8, "limit": 8,
                                   "reason": "clause slot budget 8/8"})
    cfg = _cfg({"lane_min_short_volume_usd": {"state_admission": 3_000_000}})
    ctx = _clause_shape(6_000_000)
    assert "clause slot budget" in str(_conf(ctx, cfg).get("reason", ""))
    r = _sl(ctx, cfg)
    assert r["pass"] is False and "$20M" in r["reason"]


def test_cohort_identity_outranks_the_clause_identity(caplog):
    """First-match, most-specific-first is preserved: an entry that is BOTH an
    aligned-short-lane cohort member and a clause admit uses the LANE floor
    ($2M), not the clause floor ($3M). Discriminated by a volume that sits
    BETWEEN the two floors — it passes only if the lane key won — plus the
    canary's lane identity."""
    cfg = _cfg({"lane_min_short_volume_usd": {
        "mechanical/aligned_short_live": 2_000_000,
        "state_admission": 3_000_000}})
    ctx = _clause_shape(2_500_000, coin="ZRO", model="mechanical",
                        cohort="aligned_short_live")
    with caplog.at_level(logging.INFO):
        r = _sl(ctx, cfg)
    assert r["pass"] is True
    lines = [rec.message for rec in caplog.records if "[LANE-FLOOR]" in rec.message]
    assert lines and all("state_admission" not in m for m in lines)
    assert any("lane=mechanical/aligned_short_live" in m and "$2.00M" in m
               for m in lines)


# ---- value hygiene (same fail-safe surface as the lane floor) --------------

@pytest.mark.parametrize("bad", ["3M", -1, False, True, float("nan"), float("inf")])
def test_bad_values_keep_the_global_floor_and_stay_loud(caplog, bad):
    cfg = _cfg({"lane_min_short_volume_usd": {"state_admission": bad}})
    with caplog.at_level(logging.WARNING):
        r = _sl(_clause_shape(6_000_000), cfg)
    assert r["pass"] is False and "$20M" in r["reason"]
    assert any("LANE-FLOOR" in rec.message for rec in caplog.records)


def test_zero_disables_the_floor_for_the_clause_only():
    cfg = _cfg({"lane_min_short_volume_usd": {"state_admission": 0}})
    assert _sl(_clause_shape(1_500_000), cfg)["pass"] is True   # clause path
    assert _sl(_seat_shape(1_500_000), cfg)["pass"] is False    # seat path


# ---- the identity itself ---------------------------------------------------

def test_identity_appended_last_compound_before_bare():
    from hermes_trader.agents.risk_gates import _lane_identity_keys
    ctx = _clause_shape(6_000_000, coin="ZRO", model="mechanical",
                        cohort="aligned_short_live")
    eval_all_gates(ctx, _cfg())
    keys = _lane_identity_keys(ctx)
    assert keys == ["mechanical/aligned_short_live", "aligned_short_live",
                    "mechanical/state_admission", "state_admission"]


def test_no_identity_without_a_clause_admit():
    from hermes_trader.agents.risk_gates import _lane_identity_keys
    ctx = _seat_shape(6_000_000)
    eval_all_gates(ctx, _cfg())
    assert "state_admission" not in _lane_identity_keys(ctx)


# ---- audit follow-ups (independent review of 4126ec7d, probe P9/P10) ------

def test_stale_stamp_cannot_ride_into_a_later_decision():
    """P9: the identity is PER-EVAL, not per-object. Re-evaluating the same
    GateContext — now a seat-confidence short the clause never admitted — must
    fall back to the global $20M floor, not keep the scoped $3M."""
    cfg = _cfg({"lane_min_short_volume_usd": {"state_admission": 3_000_000}})
    ctx = _clause_shape(6_000_000)
    assert _sl(ctx, cfg)["pass"] is True          # clause admit -> $3M floor
    ctx.confidence = 0.90                        # same object, now a seat short
    r = _sl(ctx, cfg)
    assert r["pass"] is False and "$20M" in r["reason"]


def test_lane_gate_policy_cannot_release_a_gate_for_the_clause_identity():
    """P10: `lane_gate_policy` RELEASES whole gates for whoever matches its key.
    Keying it on "state_admission" would leave the clause cohort with NO
    short-liquidity floor — a bigger lever than the owner's $3M ruling, so the
    clause identity is floor-only (`include_clause=False`). The cohort key stays
    usable (positive control below), which proves the guard is scoped and not a
    broken policy path."""
    cfg = _cfg({"lane_gate_policy": {"state_admission": {"short_liquidity": False}}})
    r = _sl(_clause_shape(1_500_000), cfg)
    assert r["pass"] is False and "short floor $20M" in r["reason"]
    assert "lane_exempt" not in str(r)
    out = eval_all_gates(_clause_shape(1_500_000), cfg)
    assert not any(str(v.get("lane_exempt", "")) == "state_admission"
                   for v in out["results"].values() if isinstance(v, dict))


def test_lane_gate_policy_still_works_for_cohort_keys():
    """Positive control for the guard above: cohort-keyed policy is unaffected."""
    cfg = _cfg({"lane_gate_policy":
                {"aligned_short_live": {"short_liquidity": False}}})
    ctx = _clause_shape(1_500_000, coin="ZRO", model="mechanical",
                        cohort="aligned_short_live")
    out = eval_all_gates(ctx, cfg)
    r = out["results"]["short_liquidity"]
    assert r["pass"] is True and r.get("via") == "lane_gate_policy"
