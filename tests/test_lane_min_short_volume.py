"""Lane-scoped short-liquidity floor override — config
`lane_min_short_volume_usd` (t_42e148eb, owner decision 2026-11-25).

The $20M floor was calibrated for seat-sized shorts, but for the aligned-short
lane's mechanical cohort it voided 91% of a measured 32-day book while its
protective value ABOVE $1M did not survive that sample (knives distribute
across liquidity: scratch/_lane_floor_grid.py). Owner set a $2M lane floor;
the global floor and the $1M market floor stay untouched for every path.

Pins: absent key = byte-identical global behavior; override applies ONLY to
the lane identity (model/cohort or cohort); seat shapes never see it; 0 =
disabled for that lane; malformed block/value = loud warning + global stands;
block reasons carry the effective floor (greppable evidence).
"""
import logging

import pytest

from hermes_trader.agents import hyperfeed, market_regime
from hermes_trader.agents.risk_gates import GateContext, eval_all_gates


def _lane_shape(vol_usd, coin="ZRO"):
    """Aligned-short lane live-admit shape (conf 0.0 by construction, cohort
    carries it into maybe_execute)."""
    return GateContext(
        confidence=0.0, current_positions=[], trade_notional_usd=100.0,
        daily_pnl=0.0, market_volume_24h_usd=vol_usd, coin=coin,
        trade_side="short", has_binary_news_risk=False, equity=900.0,
        total_open_notional=0.0, composite_score=40.0,
        cohort_tag="aligned_short_live", primary_model="mechanical")


def _seat_shape(vol_usd):
    """Seat-path short: no cohort identity."""
    return GateContext(
        confidence=0.70, current_positions=[], trade_notional_usd=700.0,
        daily_pnl=0.0, market_volume_24h_usd=vol_usd, coin="ZRO",
        trade_side="short", has_binary_news_risk=False, equity=900.0,
        total_open_notional=0.0, composite_score=40.0)


def _cfg(extra=None):
    cfg = {
        "min_ai_confidence": 0.65,
        "min_market_volume_usd": 1_000_000,
        "min_short_volume_usd": 20_000_000,
        "max_concurrent": 8,
        "runner_entry_gate": {"cohort_admissions":
                              {"mechanical/aligned_short_live": 0.0}},
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


def _sl(ctx, cfg):
    return eval_all_gates(ctx, cfg)["results"]["short_liquidity"]


def test_absent_key_lane_shape_blocked_by_global_floor():
    """No key = byte-identical: the $6M lane short still eats the $20M floor."""
    r = _sl(_lane_shape(6_000_000), _cfg())
    assert r["pass"] is False and "$20M" in r["reason"]


def test_override_lets_lane_admit_through(caplog):
    cfg = _cfg({"lane_min_short_volume_usd":
                {"mechanical/aligned_short_live": 2_000_000}})
    with caplog.at_level(logging.INFO):
        r = _sl(_lane_shape(6_000_000), cfg)
    assert r["pass"] is True
    assert any("[LANE-FLOOR]" in rec.message and "$2.00M" in rec.message
               for rec in caplog.records)


def test_cohort_only_identity_also_matches():
    cfg = _cfg({"lane_min_short_volume_usd": {"aligned_short_live": 2_000_000}})
    assert _sl(_lane_shape(6_000_000), cfg)["pass"] is True


def test_override_never_touches_seat_path():
    cfg = _cfg({"lane_min_short_volume_usd":
                {"mechanical/aligned_short_live": 2_000_000}})
    r = _sl(_seat_shape(6_000_000), cfg)
    assert r["pass"] is False and "$20M" in r["reason"]


def test_zero_disables_floor_for_the_lane_only():
    cfg = _cfg({"lane_min_short_volume_usd":
                {"mechanical/aligned_short_live": 0}})
    assert _sl(_lane_shape(1_500_000), cfg)["pass"] is True
    assert _sl(_seat_shape(1_500_000), cfg)["pass"] is False


def test_block_reason_carries_override_floor():
    """Greppable evidence: a lane reject reads 'short floor $2M', so live
    logs distinguish lane rejects from seat rejects by the floor value."""
    cfg = _cfg({"lane_min_short_volume_usd":
                {"mechanical/aligned_short_live": 2_000_000}})
    r = _sl(_lane_shape(1_500_000), cfg)
    assert r["pass"] is False and "short floor $2M" in r["reason"]


def test_malformed_value_keeps_global_floor_and_is_loud(caplog):
    cfg = _cfg({"lane_min_short_volume_usd":
                {"mechanical/aligned_short_live": "2M"}})
    with caplog.at_level(logging.WARNING):
        r = _sl(_lane_shape(6_000_000), cfg)
    assert r["pass"] is False and "$20M" in r["reason"]
    assert any("LANE-FLOOR" in rec.message and "global floor retained" in rec.message
               for rec in caplog.records)


def test_negative_value_keeps_global_floor():
    cfg = _cfg({"lane_min_short_volume_usd":
                {"mechanical/aligned_short_live": -1}})
    assert _sl(_lane_shape(6_000_000), cfg)["pass"] is False


def test_non_object_block_keeps_global_floor(caplog):
    cfg = _cfg({"lane_min_short_volume_usd": 2_000_000})
    with caplog.at_level(logging.WARNING):
        r = _sl(_lane_shape(6_000_000), cfg)
    assert r["pass"] is False
    assert any("LANE-FLOOR" in rec.message for rec in caplog.records)


def test_boolean_value_never_disables_the_floor(caplog):
    """JSON `false` floats to 0.0 and would silently kill the gate (bool is a
    subclass of int) — the exact fail-open the guard exists for
    (independent review 2026-11-25, S1)."""
    for bad in (False, True):
        cfg = _cfg({"lane_min_short_volume_usd":
                    {"mechanical/aligned_short_live": bad}})
        with caplog.at_level(logging.WARNING):
            r = _sl(_lane_shape(6_000_000), cfg)
        assert r["pass"] is False and "$20M" in r["reason"]
    assert any("LANE-FLOOR" in rec.message and "finite number >= 0"
               in rec.message for rec in caplog.records)


def test_nan_and_inf_values_keep_global_floor():
    for bad in (float("nan"), float("inf")):
        cfg = _cfg({"lane_min_short_volume_usd":
                    {"mechanical/aligned_short_live": bad}})
        r = _sl(_lane_shape(6_000_000), cfg)
        assert r["pass"] is False and "$20M" in r["reason"]


def test_config_key_case_is_normalized_like_policy():
    """'Mechanical/Aligned_Short_Live' must match, same normalization
    lane_gate_policy applies (review S2)."""
    cfg = _cfg({"lane_min_short_volume_usd":
                {"  Mechanical/Aligned_Short_Live ": 2_000_000}})
    assert _sl(_lane_shape(6_000_000), cfg)["pass"] is True
