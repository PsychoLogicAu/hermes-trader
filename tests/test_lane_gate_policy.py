"""Lane-scoped gate allow/disallow — config `lane_gate_policy` (t_42e148eb).

Owner decision 2026-11-24: the aligned-short lane's mechanical cohort carries
conf 0.00 by construction (cohort_admissions floor) and C.6 closed the
composite escape, so the counter-regime gate is a HARD WALL for its admits
whenever the BTC 1h-proxy reads up — 0 conversions since lane re-arm, and the
only wall-blocked admits that clear the short-liquidity floor (ZEC/LIT 11-23,
green) were blocked by this gate ALONE. `lane_gate_policy` lets a lane's admit
identity (cohort_tag / "primary_model/cohort_tag" / entry_class) disallow a
specific execute-stage gate. Fail-safe: unknown lane key, unknown gate name,
non-False value, malformed block = gate stays ON.

The short-liquidity floor stays ON as a GATE for every lane (owner: "the
liquidity floor is fine") — no policy below releases it. (Update 2026-11-25:
the floor VALUE is separately lane-overridable via
`lane_min_short_volume_usd`, see test_lane_min_short_volume.py.)
"""
import logging

import pytest

from hermes_trader.agents import hyperfeed, market_regime
from hermes_trader.agents.risk_gates import GateContext, eval_all_gates


def _zec_shape():
    """ZEC 2026-11-23 21:53Z live-admit shape: conf 0.00, comp 35.5,
    $339.6M vol (clears the $20M short floor), regime up, funding NEUTRAL."""
    return GateContext(
        confidence=0.0, current_positions=[], trade_notional_usd=72.0,
        daily_pnl=0.0, market_volume_24h_usd=339.6e6, coin="ZEC",
        trade_side="short", has_binary_news_risk=False, equity=900.0,
        total_open_notional=0.0, composite_score=35.5,
        cohort_tag="aligned_short_live", primary_model="mechanical")


def _cfg(extra=None):
    cfg = {
        "min_ai_confidence": 0.65,
        "counter_regime_min_conf": 0.85,
        "counter_regime_no_composite_escape": True,
        "min_market_volume_usd": 5_000_000,
        "min_short_volume_usd": 20_000_000,
        "runner_entry_gate": {"cohort_admissions":
                              {"mechanical/aligned_short_live": 0.0}},
    }
    if extra:
        cfg.update(extra)
    return cfg


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    monkeypatch.setattr(market_regime, "detect_regime", lambda c: "up")
    monkeypatch.setattr(market_regime, "classify_asset", lambda c: "crypto")
    monkeypatch.setattr(market_regime, "btc_tape_activity",
                        lambda force=False: None)  # quiet_tape: gap -> pass
    monkeypatch.setattr(market_regime, "btc_tape_history",
                        lambda force=False: None)
    monkeypatch.setattr(hyperfeed, "market_get_funding_regime",
                        lambda: {"regime": "NEUTRAL",
                                 "regimes_by_class": {"crypto": "NEUTRAL"}})


def test_no_policy_wall_still_blocks_the_lane_shape():
    """Byte-identical without the key: COUNTER-TREND-BLOCK stands."""
    out = eval_all_gates(_zec_shape(), _cfg())
    assert out["results"]["market_regime"]["pass"] is False
    assert out["blocked"] is True
    assert any("counter-regime" in r for r in out["block_reasons"])


def test_lane_policy_false_releases_market_regime_only():
    pol = {"mechanical/aligned_short_live": {"market_regime": False}}
    out = eval_all_gates(_zec_shape(), _cfg({"lane_gate_policy": pol}))
    mr = out["results"]["market_regime"]
    assert mr["pass"] is True and mr["via"] == "lane_gate_policy"
    assert mr["lane_exempt"] == "mechanical/aligned_short_live"
    assert mr["prior"]["pass"] is False  # prior verdict preserved for accrual
    assert out["blocked"] is False


def test_lane_exempt_log_is_loud_and_greppable(caplog):
    pol = {"mechanical/aligned_short_live": {"market_regime": False}}
    with caplog.at_level(logging.WARNING):
        eval_all_gates(_zec_shape(), _cfg({"lane_gate_policy": pol}))
    lines = [r.getMessage() for r in caplog.records if "LANE-EXEMPT" in r.getMessage()]
    assert len(lines) == 1
    assert "gate=market_regime" in lines[0]
    assert "DISALLOWED for lane=mechanical/aligned_short_live" in lines[0]


def test_bare_cohort_tag_key_matches_too():
    pol = {"aligned_short_live": {"market_regime": False}}
    out = eval_all_gates(_zec_shape(), _cfg({"lane_gate_policy": pol}))
    assert out["results"]["market_regime"]["pass"] is True


def test_other_lane_not_released():
    pol = {"vwap-fade-short": {"market_regime": False}}
    out = eval_all_gates(_zec_shape(), _cfg({"lane_gate_policy": pol}))
    assert out["results"]["market_regime"]["pass"] is False
    assert out["blocked"] is True


def test_allow_true_is_explicit_noop():
    pol = {"mechanical/aligned_short_live": {"market_regime": True}}
    out = eval_all_gates(_zec_shape(), _cfg({"lane_gate_policy": pol}))
    assert out["results"]["market_regime"]["pass"] is False


def test_malformed_values_fail_safe():
    """String "false", unknown gate, junk lane block: gate stays ON."""
    for pol in (
        {"mechanical/aligned_short_live": {"market_regime": "false"}},
        {"mechanical/aligned_short_live": {"nonexistent_gate": False}},
        {"mechanical/aligned_short_live": "nope"},
        "not a dict",
    ):
        out = eval_all_gates(_zec_shape(), _cfg({"lane_gate_policy": pol}))
        assert out["results"]["market_regime"]["pass"] is False, pol


def test_released_result_carries_gate_annotations_top_level():
    """Independent-review finding: divergent_sizing_shadow (LIVE) and the C.16
    Trade-result annotations read results["market_regime"]["regime"/
    "divergence"/...] at TOP LEVEL — the released dict must keep them."""
    pol = {"mechanical/aligned_short_live": {"market_regime": False}}
    out = eval_all_gates(_zec_shape(), _cfg({"lane_gate_policy": pol}))
    mr = out["results"]["market_regime"]
    assert mr["pass"] is True and mr["via"] == "lane_gate_policy"
    assert mr["regime"] == "up"          # annotation survives
    assert "funding" in mr and "counter_trend" in mr and "divergence" in mr
    assert mr["prior"]["pass"] is False  # original verdict preserved


def test_unmatched_gate_name_warns_loudly(caplog):
    """A gate-name typo in the policy must NOT fail silently: loud
    [gate][LANE-POLICY] line, gate untouched."""
    pol = {"mechanical/aligned_short_live": {"market-regime": False}}
    with caplog.at_level(logging.WARNING):
        out = eval_all_gates(_zec_shape(), _cfg({"lane_gate_policy": pol}))
    assert out["results"]["market_regime"]["pass"] is False  # still blocked
    msgs = [r.getMessage() for r in caplog.records]
    assert any("LANE-POLICY" in m and "NO effect" in m
               and "gate=market-regime" in m for m in msgs)


def test_liquidity_floor_never_exempted_by_this_policy_shape():
    """The thin-market short floor is NOT in the policy — a THIN lane admit
    is still blocked by short_liquidity even with market_regime disallowed."""
    ctx = _zec_shape()
    ctx.market_volume_24h_usd = 7.8e6  # ZRO 11-23 shape
    pol = {"mechanical/aligned_short_live": {"market_regime": False}}
    out = eval_all_gates(ctx, _cfg({"lane_gate_policy": pol}))
    assert out["results"]["market_regime"]["pass"] is True  # released
    assert out["results"]["short_liquidity"]["pass"] is False  # still blocked
    assert out["blocked"] is True
