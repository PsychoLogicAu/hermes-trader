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

import logging

from hermes_trader.agents import executor


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
        executor, "read_model_profile",
        lambda name: {"state_admission": clause} if clause is not None else {})


# ── helper: inert cases ──────────────────────────────────────────────────

def test_helper_absent_clause_inert():
    assert executor._state_admission(_analysis(), None, True) == (False, "", True)


def test_helper_malformed_clause_inert():
    admit, why, shadow = executor._state_admission(_analysis(), "yes", True)
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
        _, _, shadow = executor._state_admission(_analysis(), clause, True)
        assert shadow is expect, repr(val)


def test_helper_nondirectional_side_inert():
    assert executor._state_admission(_analysis(side="none"),
                                     dict(_CLAUSE, shadow=True),
                                     True) == (False, "", True)


# ── helper: long clause ──────────────────────────────────────────────────

def test_helper_long_admits_non_mover_at_any_dmove():
    # No dmove condition on the long clause: >10% THIN movers (mover flag
    # off) are the best cell in the window and must stay admissible.
    for dmove in (-9.0, 0.0, 7.0, 15.0, 40.0):
        admit, why, _ = executor._state_admission(
            _analysis(daily_move_pct=dmove, daily_mover_fired=False), _CLAUSE, True)
        assert admit, dmove


def test_helper_long_rejects_mover_class():
    admit, why, _ = executor._state_admission(
        _analysis(daily_mover_fired=True), _CLAUSE, True)
    assert not admit and "dailyMover" in why


def test_helper_long_sanity_floor():
    admit, why, _ = executor._state_admission(
        _analysis(confidence=0.39), _CLAUSE, True)
    assert not admit and "sanity floor" in why


def test_helper_long_unset_subblock_inert():
    assert executor._state_admission(_analysis(), {"short": {}}, True)[:2] == (False, "")


# ── helper: short clause ─────────────────────────────────────────────────

def test_helper_short_band_is_half_open():
    assert executor._state_admission(_short(-5.0), _CLAUSE, True)[0]        # lower edge IN
    assert executor._state_admission(_short(-7.9), _CLAUSE, True)[0]
    assert not executor._state_admission(_short(-4.99), _CLAUSE, True)[0]   # flat tape OUT
    assert not executor._state_admission(_short(-8.0), _CLAUSE, True)[0]    # upper edge OUT
    assert not executor._state_admission(_short(3.0), _CLAUSE, True)[0]     # shorting a rally OUT


def test_helper_short_respects_allow_shorts():
    admit, why, _ = executor._state_admission(_short(-6.0), _CLAUSE, False)
    assert not admit and "allow_shorts" in why


def test_helper_short_missing_dmove_inert():
    admit, why, _ = executor._state_admission(
        _analysis(side="short", daily_move_pct=None), _CLAUSE, True)
    assert (admit, why) == (False, "")


def test_helper_short_unset_band_inert():
    assert not executor._state_admission(_short(-6.0), {"long": {}}, True)[0]


def test_helper_short_malformed_band_inert():
    clause = {"short": {"min_decline_pct": "five", "max_decline_pct": 8.0}}
    assert not executor._state_admission(_short(-6.0), clause, True)[0]


def test_helper_short_sanity_floor():
    admit, why, _ = executor._state_admission(_short(-6.0, conf=0.39), _CLAUSE, True)
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
    monkeypatch.setattr(executor, "read_model_profile", _boom)
    reason = executor._runner_entry_block_reason(_analysis(confidence=0.70), _gate())
    assert reason != ""  # normal structure path still governs above-floor entries


def test_gate_without_primary_model_never_reads_profile(monkeypatch):
    def _boom(name):
        raise AssertionError("must not read a profile without primary_model")
    monkeypatch.setattr(executor, "read_model_profile", _boom)
    reason = executor._runner_entry_block_reason(_analysis(primary_model=""), _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.65)"


def test_gate_short_admit_requires_allow_shorts(monkeypatch):
    _stub_profile(monkeypatch, dict(_CLAUSE))
    reason = executor._runner_entry_block_reason(_short(-6.0), _gate(allow_shorts=False))
    assert reason == "runner_gate_blocked (confidence 0.47 < 0.65)"


def test_gate_short_band_admit(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _stub_profile(monkeypatch, dict(_CLAUSE))
    assert executor._runner_entry_block_reason(_short(-6.0), _gate()) == ""
    assert "[gate][STATE-ADMIT] ADMIT TEST SHORT" in caplog.text
