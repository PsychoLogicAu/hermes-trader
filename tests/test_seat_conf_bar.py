"""Tests for the per-seat conf bar (model-profile entry_min_confidence).

The runner gate's conf floor is a property of the ANSWERING seat's conf
scale, not of the market: 0.65 tuned to one seat mis-scales the next (the
nano fix 2026-10-13, then the same failure on d2 2026-10-16 — d2's blocked
cohort is positive in EVERY band below 0.65). The seat's profile file
(config-models/<model>.json) may now set entry_min_confidence /
entry_min_short_confidence; absent key or no profile = byte-identical
general path (fail-open). Covers the gate function plus the
model-name -> profile-file resolution (names may differ from file names).
"""

import json

import pytest

from hermes_trader.agents import config_store, executor


@pytest.fixture
def seat_dir(tmp_path, monkeypatch):
    """Isolated MODEL_CONFIG_DIR with one seat profile."""
    monkeypatch.setattr(config_store, "MODEL_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _write_profile(tmp_path, fname, **keys):
    (tmp_path / fname).write_text(json.dumps(keys))


def _gate(**over):
    g = {
        "enabled": True,
        "min_confidence": 0.65,
        "min_short_confidence": 0.65,
        "min_composite": 30.0,
        "allow_shorts": True,
    }
    g.update(over)
    return {"runner_entry_gate": g}


def _long_analysis(conf=0.50, **over):
    # structured_runner shape: fresh impulse + score, so ONLY the conf
    # floor decides.
    a = {
        "coin": "PUMP",
        "side": "long",
        "confidence": conf,
        "composite_score": 40.0,
        "volume_spike_fired": True,
        "breakout_fired": True,
        "primary_model": "seat-x",
    }
    a.update(over)
    return a


def _short_analysis(conf=0.50, **over):
    a = _long_analysis(conf=conf, side="short", downtrend_momentum_fired=True)
    return a


def test_absent_key_is_byte_identical_general_path(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x")
    reason = executor._runner_entry_block_reason(_long_analysis(), _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.65)"


def test_no_profile_at_all_is_noop(seat_dir):
    reason = executor._runner_entry_block_reason(_long_analysis(), _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.65)"


def test_seat_bar_admits_below_general_bar(seat_dir, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_confidence=0.45)
    assert executor._runner_entry_block_reason(_long_analysis(conf=0.50), _gate()) == ""
    assert "[gate][SEAT-BAR] ADMIT PUMP LONG" in caplog.text


def test_seat_bar_still_filters_below_itself(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_confidence=0.45)
    reason = executor._runner_entry_block_reason(_long_analysis(conf=0.40), _gate())
    assert reason == "runner_gate_blocked (confidence 0.40 < 0.45)"


def test_malformed_seat_bar_fails_open_to_general(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_confidence="banana")
    reason = executor._runner_entry_block_reason(_long_analysis(), _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.65)"


def test_seat_bar_only_applies_to_its_own_seat(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_confidence=0.45)
    reason = executor._runner_entry_block_reason(
        _long_analysis(primary_model="other-seat"), _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.65)"


def test_model_name_resolves_to_differently_named_profile(seat_dir):
    # qwen-style: file name != model key
    _write_profile(seat_dir, "qwen3.8-27b-iq2xs.json",
                   model="Qwen3.8-27B-GSQ-RCO-GGUF-IQ2_XS-mtp",
                   entry_min_confidence=0.45)
    a = _long_analysis(primary_model="Qwen3.8-27B-GSQ-RCO-GGUF-IQ2_XS-mtp")
    assert executor._runner_entry_block_reason(a, _gate()) == ""


def test_short_bar_independent_of_long_bar(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_confidence=0.60,
                   entry_min_short_confidence=0.45)
    # long 0.50: below its seat bar 0.60 -> blocked (effective bar reported)
    assert (executor._runner_entry_block_reason(_long_analysis(conf=0.50), _gate())
            == "runner_gate_blocked (confidence 0.50 < 0.60)")
    # short 0.50: above its seat bar 0.45 -> admitted (downtrend structure ok)
    assert executor._runner_entry_block_reason(_short_analysis(conf=0.50), _gate()) == ""
    # short 0.40: below its seat bar -> blocked, reporting the EFFECTIVE
    # (seat) bar
    reason = executor._runner_entry_block_reason(_short_analysis(conf=0.40), _gate())
    assert reason == "runner_gate_blocked (confidence 0.40 < 0.45)"


def test_short_bar_absent_falls_back_to_gate_value(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_confidence=0.45)  # long only
    reason = executor._runner_entry_block_reason(_short_analysis(conf=0.50), _gate())
    # long seat bar admits past the general bar, then the gate's
    # min_short_confidence 0.65 still applies to the short
    assert reason == "runner_gate_blocked (short confidence 0.50 < 0.65)"


def test_seat_bar_does_not_exempt_structure_checks(seat_dir):
    # conf passes the seat bar but the long has no fresh impulse -> still
    # blocked by the structure requirement (bar is conf-only)
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_confidence=0.45)
    a = _long_analysis(conf=0.50, volume_spike_fired=False, breakout_fired=False,
                       composite_score=0.0)
    reason = executor._runner_entry_block_reason(a, _gate())
    assert reason.startswith("runner_gate_blocked (needs volume+breakout/burst")
