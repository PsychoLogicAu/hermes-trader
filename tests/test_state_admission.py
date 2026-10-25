"""Tests for the runner gate's state-admission clause (kanban t_1999904f;
tape re-key 2026-10-25, kanban t_05afd186).

Sub-conf-floor rescue read from the ANSWERING model's profile
(config-models/<primary_model>.json -> "state_admission"), gated on TAPE
condition: LONG admits iff BTC trail-24h drift <= btc24_max_pct AND NOT
dailyMover-fired; SHORT admits iff BTC drift is in the half-open window
[btc24_min_pct, btc24_max_pct). The old conf bands (min/max_conf) and the
24h decline band were removed after failing cross-corpus re-derivation;
min_conf survives only as an optional re-narrowing knob (default 0.0).
shadow:true is log-only AND the kill-switch; only an explicit false goes
live. Covers the pure helper (inert cases, tape edges, mover rejection,
knob, shadow resolution, caller-injected btc_drift_24h) plus both gate
call sites (full-pass early return, shadow logs-but-still-blocks, absent
profile block byte-identical, at/above the general floor the clause is
never consulted, allow_shorts honored, SKIP accrual labelled).
"""

import json
import logging

from hermes_trader.agents import config_store, executor, market_regime, risk_gates


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
    "long": {"btc24_max_pct": -3.0},
    "short": {"btc24_min_pct": -3.0, "btc24_max_pct": 0.0},
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
        "btc_drift_24h": -4.0,  # rescue tape by default
    }
    a.update(over)
    return a


def _short(btc=-2.0, conf=0.47, **over):
    return _analysis(side="short", confidence=conf, btc_drift_24h=btc, **over)


def _stub_profile(monkeypatch, clause):
    monkeypatch.setattr(
        executor, "read_model_profile_for_model",
        lambda name: {"state_admission": clause} if clause is not None else {})


def _stub_tape(monkeypatch, drift=-4.0):
    monkeypatch.setattr(market_regime, "btc_tape_activity",
                        lambda *a, **k: {"vol": 1.2, "drift": drift})


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
    admit, why, shadow = risk_gates.state_admission(_analysis(side="none"),
                                                    dict(_CLAUSE, shadow=True),
                                                    True)
    assert (admit, shadow) == (False, True) and "non-directional" in why


# ── helper: long clause ──────────────────────────────────────────────────

def test_helper_long_admits_on_rescue_tape_at_any_dmove_and_conf():
    # No conf band and no dmove condition anymore: the tape key + mover
    # flag are the whole predicate. Thin movers (flag off) stay admissible.
    for dmove in (-9.0, 0.0, 7.0, 15.0, 40.0):
        assert risk_gates.state_admission(
            _analysis(daily_move_pct=dmove, daily_mover_fired=False),
            _CLAUSE, True)[0], dmove
    for conf in (0.05, 0.39, 0.64):  # open envelope: no floor, no ceiling
        assert risk_gates.state_admission(
            _analysis(confidence=conf), _CLAUSE, True)[0], conf


def test_helper_long_admits_at_tape_edge_inclusive():
    # drift <= btc24_max_pct (edge is IN)
    assert risk_gates.state_admission(_analysis(btc_drift_24h=-3.0), _CLAUSE, True)[0]


def test_helper_long_rejects_outside_rescue_tape():
    for drift in (-2.99, 0.0, 3.0, 12.0):
        admit, why, _ = risk_gates.state_admission(
            _analysis(btc_drift_24h=drift), _CLAUSE, True)
        assert not admit and "TAPE-SKIP" in why, drift


def test_helper_long_rejects_mover_class():
    admit, why, _ = risk_gates.state_admission(
        _analysis(daily_mover_fired=True), _CLAUSE, True)
    assert not admit and "dailyMover" in why


def test_helper_long_min_conf_knob_default_open():
    # absent min_conf = 0.0 (was 0.40 in the band shape)
    assert risk_gates.state_admission(
        _analysis(confidence=0.10), {"long": {"btc24_max_pct": -3.0}}, True)[0]
    # knob present -> enforced (re-narrowing point)
    admit, why, _ = risk_gates.state_admission(
        _analysis(confidence=0.39), {"long": {"btc24_max_pct": -3.0,
                                              "min_conf": 0.40}}, True)
    assert not admit and "sanity floor" in why


def test_helper_long_unset_subblock_inert():
    # Inert STILL carries a labelled why: the call sites log a SKIP line for
    # any clause dict, so a blank why breaks the greppable skip accrual.
    admit, why = risk_gates.state_admission(_analysis(), {"short": {}}, True)[:2]
    assert not admit and "no 'long' block" in why


def test_helper_long_tape_key_unset_inert():
    admit, why = risk_gates.state_admission(_analysis(), {"long": {}}, True)[:2]
    assert not admit and "long tape key unset" in why


def test_helper_long_no_tape_read_inert():
    # a data gap can never trigger a rescue (fail-open to the normal block)
    a = _analysis()
    a.pop("btc_drift_24h")
    admit, why, _ = risk_gates.state_admission(a, _CLAUSE, True)
    assert not admit and "no BTC trail-24h drift read" in why


# ── helper: short clause ─────────────────────────────────────────────────

def test_helper_short_tape_window_is_half_open():
    assert risk_gates.state_admission(_short(btc=-3.0), _CLAUSE, True)[0]   # lower edge IN
    assert risk_gates.state_admission(_short(btc=-0.01), _CLAUSE, True)[0]
    assert not risk_gates.state_admission(_short(btc=0.0), _CLAUSE, True)[0]    # upper edge OUT
    assert not risk_gates.state_admission(_short(btc=-3.01), _CLAUSE, True)[0]  # past rescue
    assert not risk_gates.state_admission(_short(btc=2.0), _CLAUSE, True)[0]    # up tape OUT


def test_helper_short_respects_allow_shorts():
    admit, why, _ = risk_gates.state_admission(_short(), _CLAUSE, False)
    assert not admit and "allow_shorts" in why


def test_helper_short_unset_tape_keys_inert():
    admit, why = risk_gates.state_admission(_short(), {"long": {}}, True)[:2]
    assert not admit and "no 'short' block" in why
    admit, why = risk_gates.state_admission(_short(), {"short": {"min_conf": 0.40}}, True)[:2]
    assert not admit and "short tape keys unset" in why
    # half the keys is still inert — never an unbounded window
    admit, why = risk_gates.state_admission(
        _short(), {"short": {"btc24_min_pct": -3.0}}, True)[:2]
    assert not admit and "short tape keys unset" in why


def test_helper_short_no_tape_read_inert():
    a = _short()
    a.pop("btc_drift_24h")
    admit, why, _ = risk_gates.state_admission(a, _CLAUSE, True)
    assert not admit and "no BTC trail-24h drift read" in why


def test_helper_short_malformed_keys_inert():
    clause = {"short": {"btc24_min_pct": "three", "btc24_max_pct": 0.0}}
    admit, why = risk_gates.state_admission(_short(), clause, True)[:2]
    assert not admit and "malformed clause values" in why


def test_helper_short_min_conf_knob_default_open():
    assert risk_gates.state_admission(_short(conf=0.05), _CLAUSE, True)[0]
    admit, why, _ = risk_gates.state_admission(
        _short(conf=0.39), {"short": {"btc24_min_pct": -3.0,
                                      "btc24_max_pct": 0.0, "min_conf": 0.40}}, True)
    assert not admit and "sanity floor" in why


# ── gate call site (runner) ─────────────────────────────────────────────

def test_gate_live_admit_is_full_runner_gate_pass(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE))
    # trend-only long on rescue tape (no fresh impulse): the general path
    # would re-block at structure — the clause must early-return "" instead.
    assert executor._runner_entry_block_reason(_analysis(), _gate()) == ""
    assert "[gate][STATE-ADMIT] ADMIT TEST LONG" in caplog.text


def test_gate_injects_btc_tape_when_analysis_lacks_it(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE))
    monkeypatch.setattr(executor, "_late_chase_tape_read",
                        lambda: {"vol": 1.1, "drift": -5.0})
    a = _analysis()
    a.pop("btc_drift_24h")
    assert executor._runner_entry_block_reason(a, _gate()) == ""
    assert "BTC 24h -5.00% <= -3.0%" in caplog.text


def test_gate_tape_read_gap_leaves_clause_inert(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE))
    monkeypatch.setattr(executor, "_late_chase_tape_read", lambda: None)
    a = _analysis()
    a.pop("btc_drift_24h")
    reason = executor._runner_entry_block_reason(a, _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.65)"
    assert "no BTC trail-24h drift read" in caplog.text


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
    reason = executor._runner_entry_block_reason(_short(), _gate(allow_shorts=False))
    assert reason == "runner_gate_blocked (confidence 0.47 < 0.65)"


def test_gate_short_tape_admit(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE))
    assert executor._runner_entry_block_reason(_short(), _gate()) == ""
    assert "[gate][STATE-ADMIT] ADMIT TEST SHORT" in caplog.text


# ── gate call site: SKIP accrual for everything the clause declines ─────

def test_gate_logs_skip_line_for_tape_excluded_long(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE))
    reason = executor._runner_entry_block_reason(_analysis(btc_drift_24h=-1.0), _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.65)"
    assert "[gate][STATE-ADMIT][SKIP] TEST LONG" in caplog.text
    assert "TAPE-SKIP" in caplog.text


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


def test_gate_inert_short_clause_skip_line_is_labelled(monkeypatch, caplog):
    # Live regression 2026-10-23: a long-only state_admission profile used to
    # log "[gate][STATE-ADMIT][SKIP] <coin> SHORT:" with NOTHING after the
    # colon. Inert must carry a labelled reason.
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, {"long": {"btc24_max_pct": -3.0}})
    reason = executor._runner_entry_block_reason(_short(), _gate())
    assert reason == "runner_gate_blocked (confidence 0.47 < 0.65)"
    assert "[gate][STATE-ADMIT][SKIP] TEST SHORT: clause has no 'short' block" in caplog.text


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


def _classic(ctx, clause, monkeypatch, caplog=None, tape_drift=-4.0):
    if caplog is not None:
        caplog.set_level(logging.INFO, logger="hermes_trader.agents.risk_gates")
    _stub_rg_profile(monkeypatch, clause)
    if tape_drift is not None:
        _stub_tape(monkeypatch, tape_drift)
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


def test_classic_gate_tape_excluded_long_skips(monkeypatch, caplog):
    res = _classic(_ctx(), dict(_CLAUSE), monkeypatch, caplog, tape_drift=-1.0)
    assert not res.get("pass") and res.get("reason") == "confidence 0.50 < 0.65"
    assert "[gate][STATE-ADMIT][SKIP] (classic conf gate)" in caplog.text
    assert "TAPE-SKIP" in caplog.text


def test_classic_gate_mover_long_still_blocked(monkeypatch):
    res = _classic(_ctx(daily_mover_fired=True), dict(_CLAUSE), monkeypatch)
    assert not res.get("pass")


def test_classic_gate_short_tape_admits(monkeypatch):
    res = _classic(_ctx(trade_side="short", confidence=0.47),
                   dict(_CLAUSE), monkeypatch, tape_drift=-2.0)
    assert res.get("pass") and res.get("via") == "state_admission"


def test_classic_gate_tape_gap_inert(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.risk_gates")
    _stub_rg_profile(monkeypatch, dict(_CLAUSE))
    monkeypatch.setattr(market_regime, "btc_tape_activity",
                        lambda *a, **k: None)
    res = risk_gates._state_conf_admission(_ctx(), _CFG,
                                           risk_gates.confidence_gate(_ctx(), 0.65), 0.65)
    assert not res.get("pass")
    assert "no BTC trail-24h drift read" in caplog.text


def test_classic_gate_absent_clause_byte_identical(monkeypatch):
    res = _classic(_ctx(), None, monkeypatch)
    assert res == {"pass": False, "reason": "confidence 0.50 < 0.65"}


def test_classic_gate_no_model_byte_identical(monkeypatch):
    res = _classic(_ctx(primary_model=None), dict(_CLAUSE), monkeypatch)
    assert res == {"pass": False, "reason": "confidence 0.50 < 0.65"}
