"""t_860f95f8 (2026-11-14): tape-recovery SHORT veto — mirror-family refusal
of the mild-tape LONG veto for the conf>=0.65 core population.

Evidence: the live facepalms XMR/SOL 11-13 (via=aligned, conf 0.65/0.67
exactly on the bar, btc_regime=down stamped while the tape was recovering)
are the crash-aftermath residue the state_admission arm-gate refuses in
shadow; the core lane had no tape-drift check at all.

Rule: SHORT + drift in band [btc24_min_pct, btc24_max_pct) + tape_state_arms
shows NO active arm (fresh venture through the band floor within its clock)
-> the band reading is recovery/LHS-slide residue -> veto. The crossing
clock is the SAME machinery as the state_admission arm-gate (deep -3/48h,
shallow -2/12h defaults, disarm_confirm_bars persistence, disarm_hard_above_pct
insurance) with its own config block `tape_recovery_short_veto_gate`.

Modes: off (default, ZERO fetches, byte-identical) / shadow (admit +
WOULD-VETO marker) / live (block). Placement law: above EVERY early-return
admission in the runner gate (cohort, seat-bar, state-clause, mover bypass).
Fail-safe asymmetry (pinned): drift gap / history gap / malformed keys /
malformed history = INERT — a data gap may fail an entry gate closed but
may never fire a protective veto.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from hermes_trader.agents import executor  # noqa: E402
from hermes_trader.agents.executor import _runner_entry_block_reason  # noqa: E402

BLOCK = "runner_gate_blocked (tape-recovery short veto)"
TAPE_STATE = {"deep_arm_pct": -3.0, "deep_arm_clock_h": 48,
              "shallow_arm_pct": -2.0, "shallow_arm_clock_h": 12,
              "disarm_confirm_bars": 2, "disarm_hard_above_pct": 0.5}


def _block(mode):
    return {"mode": mode, "btc24_min_pct": -2.0, "btc24_max_pct": 0.0,
            "tape_state": dict(TAPE_STATE)}


def _config(mode, **gate_over):
    gate = {"enabled": True, "min_confidence": 0.65, "min_composite": 30.0,
            "allow_shorts": True}
    gate.update(gate_over)
    cfg = {"runner_entry_gate": gate}
    if mode is not None:
        cfg["tape_recovery_short_veto_gate"] = _block(mode)
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


def _hist(drift_series):
    """(t_ms, close) 1h bars realizing drift(t) == drift_series(t) exactly:
    c(t) = c(t-24h) * (1 + d(t)/100). Len > 30 required by tape_state_arms."""
    t0 = 1_790_000_000_000
    closes = []
    by_t = {}
    for i, d in enumerate(drift_series):
        t = t0 + i * 3_600_000
        if i < 24:
            c = 100.0
        else:
            c = by_t[t - 86_400_000] * (1.0 + d / 100.0)
        by_t[t] = c
        closes.append((t, c))
    return closes


def _series(n, segments):
    """segments = [(start_idx, drift), ...]; flat between."""
    out, cur = [], 0.0
    marks = dict(segments)
    for i in range(n):
        if i in marks:
            cur = marks[i]
        out.append(cur)
    return out


def _stubs(monkeypatch, drift, hist):
    tape_calls, hist_calls = [], []

    def _read():
        tape_calls.append(1)
        return None if drift is None else {"vol": 1.0, "drift": drift}

    def _hist_read():
        hist_calls.append(1)
        return hist
    monkeypatch.setattr(executor, "_late_chase_tape_read", _read)
    monkeypatch.setattr(executor, "_tape_history_read", _hist_read)
    return tape_calls, hist_calls


N = 120
NOW = N - 1

# fresh crash: drift crossed -2 SIX bars ago and sits in-band -> ARMED short
FRESH = _series(N, [(N - 8, -2.4), (N - 3, -1.8)])
# stale crash: crossed -2 ~68h ago (ages beyond both clocks), now -1.3 residue
STALE = _series(N, [(N - 70, -2.5), (N - 40, -1.3)])
# disarmed LHS slide: crash -3.5, sustained green escape (+0.6 x3 >= conf 2),
# slide back to -1.6 WITHOUT crossing the floor again
SLIDE = _series(N, [(N - 80, -3.5), (N - 50, 0.6), (N - 20, -1.6)])


# ── inert paths: zero fetches, byte-identical ───────────────────────────────

def test_absent_block_zero_fetches(monkeypatch):
    tc, hc = _stubs(monkeypatch, -1.5, _hist(STALE))
    assert _runner_entry_block_reason(_analysis(), _config(None)) == ""
    assert tc == [] and hc == []


def test_mode_off_zero_fetches(monkeypatch):
    tc, hc = _stubs(monkeypatch, -1.5, _hist(STALE))
    assert _runner_entry_block_reason(_analysis(), _config("off")) == ""
    assert tc == [] and hc == []


def test_unknown_mode_inert(monkeypatch):
    tc, hc = _stubs(monkeypatch, -1.5, _hist(STALE))
    cfg = _config(None)
    cfg["tape_recovery_short_veto_gate"] = {"mode": "banana",
                                            "btc24_min_pct": -2.0,
                                            "btc24_max_pct": 0.0}
    assert _runner_entry_block_reason(_analysis(), cfg) == ""
    assert tc == []


def test_long_never_reaches_short_veto(monkeypatch):
    tc, hc = _stubs(monkeypatch, -1.5, _hist(STALE))
    a = _analysis(side="long", downtrend_momentum_fired=False,
                  volume_spike_fired=True, breakout_fired=True,
                  slow_burn_count=1)
    assert _runner_entry_block_reason(a, _config("live")) == ""
    assert tc == [] and hc == []


def test_malformed_band_keys_inert(monkeypatch):
    tc, hc = _stubs(monkeypatch, -1.5, _hist(STALE))
    cfg = _config("live")
    cfg["tape_recovery_short_veto_gate"]["btc24_min_pct"] = "minus two"
    assert _runner_entry_block_reason(_analysis(), cfg) == ""


def test_nonnumeric_drift_value_inert(monkeypatch):
    # reviewer-pass hardening: the sensor contract is numeric, but a string
    # in the dict must degrade to inert, never raise out of the runner gate.
    def _read():
        return {"vol": 1.0, "drift": "down"}
    monkeypatch.setattr(executor, "_late_chase_tape_read", _read)
    hc = []
    monkeypatch.setattr(executor, "_tape_history_read",
                        lambda: hc.append(1) or _hist(STALE))
    assert _runner_entry_block_reason(_analysis(), _config("live")) == ""
    assert hc == []


# ── band + arm semantics ─────────────────────────────────────────────────────

def test_drift_out_of_band_no_history_fetch(monkeypatch):
    tc, hc = _stubs(monkeypatch, -2.5, _hist(STALE))  # below floor: crash tape
    assert _runner_entry_block_reason(_analysis(), _config("live")) == ""
    assert tc == [1] and hc == []          # history only read in-band
    _stubs(monkeypatch, 0.5, _hist(STALE))  # above ceiling
    assert _runner_entry_block_reason(_analysis(), _config("live")) == ""


def test_fresh_crash_armed_not_vetoed(monkeypatch):
    _stubs(monkeypatch, -1.8, _hist(FRESH))
    assert _runner_entry_block_reason(_analysis(), _config("live")) == ""


def test_stale_residue_vetoed_live(monkeypatch):
    _stubs(monkeypatch, -1.3, _hist(STALE))
    assert _runner_entry_block_reason(_analysis(), _config("live")) == BLOCK


def test_disarmed_lhs_slide_vetoed_live(monkeypatch):
    # green escape >= 2 confirmed closes then a slide back in WITHOUT a fresh
    # floor venture — the facepalm tape shape.
    _stubs(monkeypatch, -1.6, _hist(SLIDE))
    assert _runner_entry_block_reason(_analysis(), _config("live")) == BLOCK


def test_history_gap_fail_open_inert(monkeypatch):
    # short history -> arms None: unlike the ENTRY arm-gate (fails closed),
    # a protective VETO must never fire on a data gap.
    _stubs(monkeypatch, -1.3, [("t", 100.0)] * 10)
    assert _runner_entry_block_reason(_analysis(), _config("live")) == ""
    _stubs(monkeypatch, -1.3, None)
    assert _runner_entry_block_reason(_analysis(), _config("live")) == ""


def test_malformed_history_fail_open(monkeypatch):
    _stubs(monkeypatch, -1.3, [("t", "bad")] * 60)
    assert _runner_entry_block_reason(_analysis(), _config("live")) == ""


# ── shadow mode ──────────────────────────────────────────────────────────────

def test_shadow_logs_and_admits(monkeypatch, caplog):
    import logging as _lg
    _stubs(monkeypatch, -1.3, _hist(STALE))
    with caplog.at_level(_lg.WARNING):
        assert _runner_entry_block_reason(_analysis(), _config("shadow")) == ""
    assert "[gate][TAPE-RECOVERY-SHORT][SHADOW] WOULD VETO" in caplog.text
    assert "tape [deep=" in caplog.text  # stamp rides the line for pricing


# ── placement law: above every early-return admission ────────────────────────

def test_veto_precedes_cohort_admission(monkeypatch):
    _stubs(monkeypatch, -1.3, _hist(STALE))
    cfg = _config("live")
    cfg["runner_entry_gate"]["cohort_admissions"] = {"seat-x/fade": 0.40}
    a = _analysis(confidence=0.40, cohort_tag="fade", primary_model="seat-x")
    assert _runner_entry_block_reason(a, cfg) == BLOCK


def test_veto_precedes_state_admission_clause(monkeypatch):
    _stubs(monkeypatch, -1.3, _hist(STALE))

    def _prof(model):
        return {"state_admission": {
            "shadow": False,
            "short": {"btc24_min_pct": -2.0, "btc24_max_pct": 0.0,
                      "min_conf": 0.10}}}
    monkeypatch.setattr(executor, "read_model_profile_for_model", _prof)
    a = _analysis(confidence=0.30, btc_drift_24h=-1.3, primary_model="seat-x")
    assert _runner_entry_block_reason(a, _config("live")) == BLOCK


def test_mover_bypass_population_covered(monkeypatch):
    _stubs(monkeypatch, -1.3, _hist(STALE))
    a = _analysis(daily_mover_fired=True, volume_spike_fired=True)
    assert _runner_entry_block_reason(a, _config("live")) == BLOCK
