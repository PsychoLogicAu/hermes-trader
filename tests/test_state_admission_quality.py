"""Short-side state-admission QUALITY gate (t_860f95f8, 2026-10-29).

Cohort evidence: on crash-episode band days the short population
oversupplies the 10-slot book ~7:1; expert consensus spread + chronos
upside tail + tirex median separate the tastiest trades (43-trade book,
42/43 winners, 0 max_loss). The gate lives ONLY on the short side of
``state_admission``; absent ``quality`` sub-block = pre-filter behavior.
"""
import pytest

from hermes_trader.agents.risk_gates import (
    state_admission,
    inject_state_quality,
)

CLAUSE = {
    "shadow": True,
    "short": {
        "btc24_min_pct": -2.0,
        "btc24_max_pct": 0.0,
        "min_conf": 0.0,
        "quality": {
            "max_expert_spread_pct": 1.15,
            "max_chronos_tail_pct": 0.45,
            "max_tirex_median_pct": 0.0,
        },
    },
}


def _short(qual_vals=None, **over):
    a = {"side": "short", "confidence": 0.20, "btc_drift_24h": -1.0}
    a.update(over or {})
    for k, v in (qual_vals or {}).items():
        a[k] = v
    return a


GOOD = {"expert_spread_pct": 0.90, "chronos_tail_pct": 0.3,
        "tirex_median_pct": -0.1}


def test_quality_all_pass_admits():
    admit, why, shadow = state_admission(_short(GOOD), CLAUSE, True)
    assert admit and shadow
    assert "quality [" in why


def test_absent_quality_block_is_byte_identical_old_path():
    clause = {"shadow": True,
              "short": {"btc24_min_pct": -2.0, "btc24_max_pct": 0.0,
                        "min_conf": 0.0}}
    admit, why, shadow = state_admission(_short(), clause, True)
    assert admit and "quality" not in why


def test_expert_spread_over_max_skips():
    vals = dict(GOOD, expert_spread_pct=1.20)
    admit, why, _ = state_admission(_short(vals), CLAUSE, True)
    assert not admit and "QUALITY-SKIP: expert consensus spread +1.20 > max +1.15" in why


def test_chronos_tail_over_max_skips():
    vals = dict(GOOD, chronos_tail_pct=0.5)
    admit, why, _ = state_admission(_short(vals), CLAUSE, True)
    assert not admit and "QUALITY-SKIP: chronos upside tail +0.50 > max +0.45" in why


def test_tirex_median_over_max_skips():
    vals = dict(GOOD, tirex_median_pct=0.1)
    admit, why, _ = state_admission(_short(vals), CLAUSE, True)
    assert not admit and "QUALITY-SKIP: tirex median forecast +0.10 > max +0.00" in why


def test_leg_key_set_but_read_missing_skips():
    vals = {k: v for k, v in GOOD.items() if k != "tirex_median_pct"}
    admit, why, _ = state_admission(_short(vals), CLAUSE, True)
    assert not admit and "QUALITY-SKIP: no tirex median forecast read" in why


def test_partial_quality_block_needs_only_present_legs():
    clause = {"shadow": True, "short": {
        "btc24_min_pct": -2.0, "btc24_max_pct": 0.0, "min_conf": 0.0,
        "quality": {"max_expert_spread_pct": 1.15}}}
    admit, why, _ = state_admission(_short({"expert_spread_pct": 0.5}), clause, True)
    assert admit


def test_tape_band_checked_before_quality():
    # drift outside band -> TAPE-SKIP (no quality reads needed at all)
    admit, why, _ = state_admission(_short(btc_drift_24h=+1.0), CLAUSE, True)
    assert not admit and "TAPE-SKIP" in why


def test_long_side_ignores_short_quality():
    clause = {"shadow": True,
              "long": {"btc24_max_pct": 0.0},
              "short": CLAUSE["short"]}
    a = {"side": "long", "confidence": 0.20, "btc_drift_24h": -1.0}
    admit, why, _ = state_admission(a, clause, True)
    assert admit and "rescue tape" in why


def test_malformed_quality_limit_inert():
    clause = {"shadow": True, "short": {
        "btc24_min_pct": -2.0, "btc24_max_pct": 0.0, "min_conf": 0.0,
        "quality": {"max_tirex_median_pct": "banana"}}}
    admit, why, _ = state_admission(_short(GOOD), clause, True)
    assert not admit and "malformed clause values" in why


def test_boundary_values_pass():
    vals = {"expert_spread_pct": 1.15, "chronos_tail_pct": 0.45,
            "tirex_median_pct": 0.0}
    admit, _, _ = state_admission(_short(vals), CLAUSE, True)
    assert admit  # limits are maxima (inclusive): value <= max passes


class _Sig:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_inject_maps_and_rounds(monkeypatch):
    import hermes_trader.agents.expert_select as es_mod
    import hermes_trader.agents.chronos_signal as cs_mod
    import hermes_trader.agents.tirex_signal as ts_mod
    monkeypatch.setattr(es_mod, "peek_expert_select",
                        lambda coin: _Sig(spread_pct=0.83))
    monkeypatch.setattr(cs_mod, "peek_chronos",
                        lambda coin: _Sig(q90_path_pct=[0.1, 0.2, 0.44, 9.9, 9.9, 9.9]))
    monkeypatch.setattr(ts_mod, "peek_tirex",
                        lambda coin: _Sig(median_pct=-0.04))
    out = inject_state_quality({}, "ARB", "short",
                               cfg={"chronos_tail_trigger_gate": {"window_steps": 6}})
    assert out["expert_spread_pct"] == 0.83
    # window cut: 9.9s are outside the 6-step window... they ARE inside (len 6)
    assert out["chronos_tail_pct"] == 9.9  # max of first 6
    assert out["tirex_median_pct"] == 0.0  # 0.1-rounded like the trade-result field


def test_inject_window_and_missing_reads(monkeypatch):
    import hermes_trader.agents.expert_select as es_mod
    import hermes_trader.agents.chronos_signal as cs_mod
    import hermes_trader.agents.tirex_signal as ts_mod
    monkeypatch.setattr(es_mod, "peek_expert_select", lambda coin: None)
    monkeypatch.setattr(cs_mod, "peek_chronos",
                        lambda coin: _Sig(q90_path_pct=[0.1, 0.2]))  # shorter than window
    monkeypatch.setattr(ts_mod, "peek_tirex", lambda coin: _Sig(median_pct=None))
    out = inject_state_quality({}, "ARB", "short",
                               cfg={"chronos_tail_trigger_gate": {"window_steps": 6}})
    assert out == {}  # all peeks miss -> fields absent -> clause SKIPs set legs


def test_inject_peek_crash_is_swallowed(monkeypatch):
    import hermes_trader.agents.expert_select as es_mod
    def boom(coin):
        raise RuntimeError("cache exploded")
    monkeypatch.setattr(es_mod, "peek_expert_select", boom)
    out = inject_state_quality({"a": 1}, "ARB", "short", cfg={})
    assert out["a"] == 1  # never raises on the trade path
