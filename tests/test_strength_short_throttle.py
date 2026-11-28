"""t_c3e7ab77 (2026-11-28): strength-entry SHORT throttle — positive-drift twin
of the tape-recovery short veto.

Evidence (board t_c3e7ab77 #14): on the s8 full set the deployed 0.46 book
prices BTC24 >= +1% entries NEGATIVE (-$21.8/13d BN2, alpha -3.13, h2 -44.2)
while dip/band entries carry the whole increment; strength entries are ~10%
of admits. Owner directive: SHADOW LOG FIRST — accrue the counterfactual,
cut nothing live on one era of data.

Rule: SHORT + BTC trail-24h drift >= btc24_min_pct -> throttle. The knob is a
FLOOR, not a band, and carries no arm condition (positive drift IS the
trigger). btc24_min_pct must be > 0: a 0/negative value would throttle the
entire short book, so it is refused loudly and the gate stays INERT.

Modes: off (default, ZERO fetches, byte-identical) / shadow (admit +
WOULD-SKIP marker) / live (block). Same placement law and fail-safe asymmetry
as the tape-recovery twin: drift gap / malformed keys = INERT.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from hermes_trader.agents import executor  # noqa: E402
from hermes_trader.agents.executor import _runner_entry_block_reason  # noqa: E402

BLOCK = "runner_gate_blocked (strength short throttle)"


def _block(mode, mn=1.0):
    b = {"mode": mode}
    if mn is not None:
        b["btc24_min_pct"] = mn
    return b


def _config(mode, sth_min=1.0, **over):
    gate = {"enabled": True, "min_confidence": 0.65, "min_composite": 30.0,
            "allow_shorts": True}
    gate.update(over)
    cfg = {"runner_entry_gate": gate}
    if mode is not None:
        cfg["strength_short_throttle_gate"] = _block(mode, sth_min)
    return cfg


def _analysis(**over) -> dict:
    a = {
        "coin": "TEST", "side": "short", "confidence": 0.85,
        "composite_score": 50.0, "daily_mover_fired": False,
        "volume_spike_fired": False, "breakout_fired": False,
        "momentum_burst_fired": False, "uptrend_momentum_fired": False,
        "downtrend_momentum_fired": True, "slow_burn_count": 0,
        "whale_signal": False, "reasoning": "",
    }
    a.update(over)
    return a


def _stub_tape(monkeypatch, drift):
    calls = []

    def _read():
        calls.append(1)
        return None if drift is None else {"vol": 1.0, "drift": drift}

    monkeypatch.setattr(executor, "_late_chase_tape_read", _read)
    return calls


# ── inert paths: zero fetches, byte-identical ───────────────────────────────

def test_absent_block_zero_fetches(monkeypatch):
    tc = _stub_tape(monkeypatch, 2.5)  # drift WOULD trigger if consulted
    assert _runner_entry_block_reason(_analysis(), _config(None)) == ""
    assert tc == []


def test_mode_off_zero_fetches(monkeypatch):
    tc = _stub_tape(monkeypatch, 2.5)
    assert _runner_entry_block_reason(_analysis(), _config("off")) == ""
    assert tc == []


def test_long_side_untouched(monkeypatch):
    tc = _stub_tape(monkeypatch, 2.5)
    cfg = _config("live")
    cfg["runner_entry_gate"]["min_confidence"] = 0.40
    r = _runner_entry_block_reason(_analysis(side="long", confidence=0.85), cfg)
    # Whatever the LONG gates decide, the throttle must never be involved
    # and must never consult the tape on the long path.
    assert r != BLOCK
    assert tc == []


# ── shadow: admit + marker, never block ─────────────────────────────────────

def test_shadow_admits_and_logs(monkeypatch, caplog):
    _stub_tape(monkeypatch, 1.4)
    with caplog.at_level("WARNING"):
        r = _runner_entry_block_reason(_analysis(), _config("shadow"))
    assert r == ""  # admits — shadow never cuts
    assert "[STRENGTH-THROTTLE][SHADOW] WOULD SKIP" in caplog.text
    assert "NOT skipping" in caplog.text


def test_shadow_negative_drift_silent(monkeypatch, caplog):
    _stub_tape(monkeypatch, -0.7)
    with caplog.at_level("WARNING"):
        assert _runner_entry_block_reason(_analysis(), _config("shadow")) == ""
    assert "STRENGTH-THROTTLE" not in caplog.text


# ── live: blocks, mechanical across conf bands ──────────────────────────────

def test_live_blocks_positive_drift(monkeypatch):
    _stub_tape(monkeypatch, 1.0)  # exactly on the floor: >= fires
    assert _runner_entry_block_reason(_analysis(), _config("live")) == BLOCK


def test_live_blocks_high_conf_too(monkeypatch):
    # Mechanical like its tape-recovery twin: the >=0.65 core population is
    # NOT exempt. Pinning this design choice.
    _stub_tape(monkeypatch, 3.2)
    a = _analysis(confidence=0.71)
    assert _runner_entry_block_reason(a, _config("live")) == BLOCK


# ── fail-safe asymmetry: gaps and misconfigs are INERT ──────────────────────

def test_drift_gap_inert_even_live(monkeypatch):
    _stub_tape(monkeypatch, None)
    assert _runner_entry_block_reason(_analysis(), _config("live")) == ""


def test_malformed_min_inert(monkeypatch):
    cfg = _config(None)
    cfg["strength_short_throttle_gate"] = {"mode": "live", "btc24_min_pct": "banana"}
    _stub_tape(monkeypatch, 5.0)
    assert _runner_entry_block_reason(_analysis(), cfg) == ""


def test_nonpositive_min_refused_not_throttle_all(monkeypatch, caplog):
    # btc24_min_pct 0 would throttle the ENTIRE short book — must refuse the
    # misconfig and stay inert, never throttle everything.
    _stub_tape(monkeypatch, 0.0)
    with caplog.at_level("WARNING"):
        r = _runner_entry_block_reason(_analysis(), _config("live", sth_min=0.0))
    assert r == ""
    assert "must be > 0" in caplog.text
