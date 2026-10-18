"""Tests for the runner gate's state-admission clause (kanban t_1999904f).

Sub-conf-floor rescue read from the ANSWERING model's profile
(config-models/<primary_model>.json -> "state_admission"): LONG admits iff
NOT dailyMover-fired (the crowded-extension class is the poison), SHORT
admits iff the 24h decline is in the half-open band
[min_decline_pct, max_decline_pct). shadow:true is log-only AND the
kill-switch; only an explicit false goes live. Covers the pure helper
(inert cases, band edges, mover rejection, sanity floor, shadow
resolution) plus the gate call site (full-pass early return, shadow
logs-but-still-blocks, absent profile block byte-identical, at/above the
general floor the clause is never consulted, allow_shorts honored).
"""

import json
import logging

from hermes_trader.agents import config_store, executor, risk_gates


def _gate(**over):
    g = {
        "enabled": True,
        "min_confidence": 0.65,
        "min_composite": 30.0,
        "allow_shorts": True,
    }
    g.update(over)
    return {"runner_entry_gate": g}


_CLAUSE = {
    "shadow": False,
    "long": {"min_conf": 0.40},
    "short": {"min_conf": 0.40, "min_decline_pct": 5.0, "max_decline_pct": 8.0},
}


def _analysis(**over):
    a = {
        "coin": "TEST",
        "side": "long",
        "confidence": 0.50,
        "composite_score": 0.0,
        "primary_model": "test-model",
        "daily_move_pct": 3.0,
        "daily_mover_fired": False,
    }
    a.update(over)
    return a


def _short(dmove, conf=0.47):
    return _analysis(side="short", confidence=conf, daily_move_pct=dmove)


def _stub_profile(monkeypatch, clause):
    monkeypatch.setattr(
        executor, "read_model_profile_for_model",
        lambda name: {"state_admission": clause} if clause is not None else {})


# ── helper: inert cases ──────────────────────────────────────────────────

def test_helper_absent_clause_inert():
    assert risk_gates.state_admission(_analysis(), None, True) == (False, "", True)


def test_helper_malformed_clause_inert():
    admit, why, shadow = risk_gates.state_admission(_analysis(), "yes", True)
    assert (admit, why, shadow) == (False, "", True)


def test_helper_shadow_defaults_true_and_only_explicit_false_goes_live():
    for val, expect in ((None, True), (True, True), ("true", True),
                        ("banana", True), (False, False), ("false", False),
                        ("False", False)):
        clause = dict(_CLAUSE)
        if val is None:
            clause.pop("shadow")
        else:
            clause["shadow"] = val
        _, _, shadow = risk_gates.state_admission(_analysis(), clause, True)
        assert shadow is expect, repr(val)


def test_helper_nondirectional_side_inert():
    assert risk_gates.state_admission(_analysis(side="none"),
                                     dict(_CLAUSE, shadow=True),
                                     True) == (False, "", True)


# ── helper: long clause ──────────────────────────────────────────────────

def test_helper_long_admits_non_mover_at_any_dmove():
    # No dmove condition on the long clause: >10% THIN movers (mover flag
    # off) are the best cell in the window and must stay admissible.
    for dmove in (-9.0, 0.0, 7.0, 15.0, 40.0):
        admit, why, _ = risk_gates.state_admission(
            _analysis(daily_move_pct=dmove, daily_mover_fired=False), _CLAUSE, True)
        assert admit, dmove


def test_helper_long_rejects_mover_class():
    admit, why, _ = risk_gates.state_admission(
        _analysis(daily_mover_fired=True), _CLAUSE, True)
    assert not admit and "dailyMover" in why


def test_helper_long_sanity_floor():
    admit, why, _ = risk_gates.state_admission(
        _analysis(confidence=0.39), _CLAUSE, True)
    assert not admit and "sanity floor" in why


def test_helper_long_unset_subblock_inert():
    assert risk_gates.state_admission(_analysis(), {"short": {}}, True)[:2] == (False, "")


def test_helper_long_ceiling_is_half_open_and_absent_means_none():
    cl = dict(_CLAUSE, long={"min_conf": 0.40, "max_conf": 0.50})
    assert risk_gates.state_admission(_analysis(confidence=0.499), cl, True)[0]
    admit, why, _ = risk_gates.state_admission(_analysis(confidence=0.50), cl, True)
    assert not admit and "ceiling" in why          # upper edge OUT: [0.40, 0.50)
    assert not risk_gates.state_admission(_analysis(confidence=0.64), cl, True)[0]
    # absent ceiling = no ceiling (back-compat with the deployed profile)
    assert risk_gates.state_admission(_analysis(confidence=0.64), _CLAUSE, True)[0]


# ── helper: short clause ─────────────────────────────────────────────────

def test_helper_short_band_is_half_open():
    assert risk_gates.state_admission(_short(-5.0), _CLAUSE, True)[0]        # lower edge IN
    assert risk_gates.state_admission(_short(-7.9), _CLAUSE, True)[0]
    assert not risk_gates.state_admission(_short(-4.99), _CLAUSE, True)[0]   # flat tape OUT
    assert not risk_gates.state_admission(_short(-8.0), _CLAUSE, True)[0]    # upper edge OUT
    assert not risk_gates.state_admission(_short(3.0), _CLAUSE, True)[0]     # shorting a rally OUT


def test_helper_short_respects_allow_shorts():
    admit, why, _ = risk_gates.state_admission(_short(-6.0), _CLAUSE, False)
    assert not admit and "allow_shorts" in why


def test_helper_short_missing_dmove_inert():
    admit, why, _ = risk_gates.state_admission(
        _analysis(side="short", daily_move_pct=None), _CLAUSE, True)
    assert (admit, why) == (False, "")


def test_helper_short_unset_band_inert():
    assert not risk_gates.state_admission(_short(-6.0), {"long": {}}, True)[0]


def test_helper_short_malformed_band_inert():
    clause = {"short": {"min_decline_pct": "five", "max_decline_pct": 8.0}}
    assert not risk_gates.state_admission(_short(-6.0), clause, True)[0]


def test_helper_short_sanity_floor():
    admit, why, _ = risk_gates.state_admission(_short(-6.0, conf=0.39), _CLAUSE, True)
    assert not admit and "sanity floor" in why


# ── gate call site ───────────────────────────────────────────────────────

def test_gate_live_admit_is_full_runner_gate_pass(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE))
    # trend-only long (no fresh impulse): the general path would re-block
    # at structure — the clause must early-return "" instead.
    assert executor._runner_entry_block_reason(_analysis(), _gate()) == ""
    assert "[gate][STATE-ADMIT] ADMIT TEST LONG" in caplog.text


def test_gate_shadow_logs_would_admit_and_still_blocks(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE, shadow=True))
    reason = executor._runner_entry_block_reason(_analysis(), _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.65)"
    assert "[gate][STATE-ADMIT][SHADOW] WOULD ADMIT" in caplog.text


def test_gate_absent_profile_block_is_byte_identical(monkeypatch):
    _stub_profile(monkeypatch, None)
    assert executor._runner_entry_block_reason(
        _analysis(), _gate()) == "runner_gate_blocked (confidence 0.50 < 0.65)"


def test_gate_at_or_above_general_floor_never_consults_clause(monkeypatch):
    def _boom(name):
        raise AssertionError("clause must not be read at/above the general floor")
    monkeypatch.setattr(executor, "read_model_profile_for_model", _boom)
    reason = executor._runner_entry_block_reason(_analysis(confidence=0.70), _gate())
    assert reason != ""  # normal structure path still governs above-floor entries


def test_gate_without_primary_model_never_reads_profile(monkeypatch):
    def _boom(name):
        raise AssertionError("must not read a profile without primary_model")
    monkeypatch.setattr(executor, "read_model_profile_for_model", _boom)
    reason = executor._runner_entry_block_reason(_analysis(primary_model=""), _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.65)"


def test_gate_clause_found_when_profile_filename_differs_from_model_key(
        tmp_path, monkeypatch):
    # Regression (pplx cutover, 2026-10-18): the profile FILE name may differ
    # from the answering model KEY (pplx-decider-27b.json holds
    # "model": "pplx-decider") — the clause lookup must resolve by key.
    monkeypatch.setattr(config_store, "MODEL_CONFIG_DIR", str(tmp_path))
    (tmp_path / "seat-x-file.json").write_text(json.dumps(
        {"model": "seat-X", "state_admission": _CLAUSE}))
    reason = executor._runner_entry_block_reason(
        _analysis(primary_model="seat-X"), _gate())
    assert reason == ""  # clause rescued conf 0.50 < 0.65


def test_gate_short_admit_requires_allow_shorts(monkeypatch):
    _stub_profile(monkeypatch, dict(_CLAUSE))
    reason = executor._runner_entry_block_reason(_short(-6.0), _gate(allow_shorts=False))
    assert reason == "runner_gate_blocked (confidence 0.47 < 0.65)"


def test_gate_short_band_admit(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE))
    assert executor._runner_entry_block_reason(_short(-6.0), _gate()) == ""
    assert "[gate][STATE-ADMIT] ADMIT TEST SHORT" in caplog.text


# ── gate call site: SKIP accrual for everything the clause declines ─────

def test_gate_logs_skip_line_for_ceiling_excluded_long(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE, long={"min_conf": 0.40, "max_conf": 0.50}))
    reason = executor._runner_entry_block_reason(_analysis(confidence=0.55), _gate())
    assert reason == "runner_gate_blocked (confidence 0.55 < 0.65)"
    assert "[gate][STATE-ADMIT][SKIP] TEST LONG" in caplog.text
    assert "ceiling" in caplog.text


def test_gate_logs_skip_line_for_mover_excluded_long(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE))
    executor._runner_entry_block_reason(_analysis(daily_mover_fired=True), _gate())
    assert "[gate][STATE-ADMIT][SKIP] TEST LONG" in caplog.text
    assert "dailyMover" in caplog.text


def test_gate_no_skip_line_when_clause_absent(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, None)
    executor._runner_entry_block_reason(_analysis(), _gate())
    assert "STATE-ADMIT" not in caplog.text


# ── classic conf gate hook (the live no-op fix, 2026-10-17) ─────────────

def _ctx(**over):
    from typing import Any, Dict
    kw: Dict[str, Any] = dict(
        confidence=0.50, current_positions=[], trade_notional_usd=100.0,
        daily_pnl=0.0, market_volume_24h_usd=50_000_000.0, coin="TEST",
        trade_side="long", has_binary_news_risk=False, equity=1000.0,
        total_open_notional=0.0, primary_model="test-model",
        daily_move_pct=3.0, daily_mover_fired=False,
    )
    kw.update(over)
    return risk_gates.GateContext(**kw)


_CFG = {"runner_entry_gate": {"allow_shorts": True}, "min_ai_confidence": 0.65}


def _stub_rg_profile(monkeypatch, clause):
    monkeypatch.setattr(
        risk_gates, "read_model_profile_for_model",
        lambda name: {"state_admission": clause} if clause is not None else {})


def _classic(ctx, clause, monkeypatch, caplog=None):
    if caplog is not None:
        caplog.set_level(logging.INFO, logger="hermes_trader.agents.risk_gates")
    _stub_rg_profile(monkeypatch, clause)
    return risk_gates._state_conf_admission(ctx, _CFG,
                                            risk_gates.confidence_gate(ctx, 0.65), 0.65)


def test_classic_gate_admits_clause_long(monkeypatch, caplog):
    res = _classic(_ctx(), dict(_CLAUSE), monkeypatch, caplog)
    assert res.get("pass") and res.get("via") == "state_admission"
    assert "[gate][STATE-ADMIT] ADMIT (classic conf gate) TEST LONG" in caplog.text


def test_classic_gate_shadow_still_blocks(monkeypatch, caplog):
    res = _classic(_ctx(), dict(_CLAUSE, shadow=True), monkeypatch, caplog)
    assert not res.get("pass")
    assert "[gate][STATE-ADMIT][SHADOW] WOULD ADMIT (classic conf gate)" in caplog.text


def test_classic_gate_ceiling_excluded_long_skips(monkeypatch, caplog):
    cl = dict(_CLAUSE, long={"min_conf": 0.40, "max_conf": 0.50})
    res = _classic(_ctx(confidence=0.55), cl, monkeypatch, caplog)
    assert not res.get("pass") and res.get("reason") == "confidence 0.55 < 0.65"
    assert "[gate][STATE-ADMIT][SKIP] (classic conf gate)" in caplog.text
    assert "ceiling" in caplog.text


def test_classic_gate_mover_long_still_blocked(monkeypatch):
    res = _classic(_ctx(daily_mover_fired=True), dict(_CLAUSE), monkeypatch)
    assert not res.get("pass")


def test_classic_gate_short_band_admits(monkeypatch):
    res = _classic(_ctx(trade_side="short", confidence=0.47, daily_move_pct=-6.0),
                   dict(_CLAUSE), monkeypatch)
    assert res.get("pass") and res.get("via") == "state_admission"


def test_helper_short_ceiling_is_half_open(monkeypatch):
    cl = dict(_CLAUSE, short={"min_conf": 0.40, "max_conf": 0.50,
                              "min_decline_pct": 5.0, "max_decline_pct": 8.0})
    a = risk_gates.state_admission(
        {"side": "short", "confidence": 0.499, "daily_move_pct": -6.0}, cl, True)
    assert a[0]
    b = risk_gates.state_admission(
        {"side": "short", "confidence": 0.50, "daily_move_pct": -6.0}, cl, True)
    assert not b[0] and "ceiling" in b[1]
    c = risk_gates.state_admission(
        {"side": "short", "confidence": 0.62, "daily_move_pct": -6.0}, cl, True)
    assert not c[0] and "ceiling" in c[1]


def test_classic_gate_short_ceiling_skips(monkeypatch, caplog):
    cl = dict(_CLAUSE, short={"min_conf": 0.40, "max_conf": 0.50,
                              "min_decline_pct": 5.0, "max_decline_pct": 8.0})
    res = _classic(_ctx(trade_side="short", confidence=0.55, daily_move_pct=-6.0),
                   cl, monkeypatch, caplog)
    assert not res.get("pass")
    assert "[gate][STATE-ADMIT][SKIP] (classic conf gate)" in caplog.text
    assert "ceiling" in caplog.text


def test_classic_gate_absent_clause_byte_identical(monkeypatch):
    res = _classic(_ctx(), None, monkeypatch)
    assert res == {"pass": False, "reason": "confidence 0.50 < 0.65"}


def test_classic_gate_no_model_byte_identical(monkeypatch):
    res = _classic(_ctx(primary_model=None), dict(_CLAUSE), monkeypatch)
    assert res == {"pass": False, "reason": "confidence 0.50 < 0.65"}
