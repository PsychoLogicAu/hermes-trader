"""t_104c00f2 (2026-11-14): TAPE-ARMED short admit for the structure clause.

Evidence: mammoth-era funnel 11-14 — conf>=0.60 SHORTs on a low-vol grind
decline were killed 50x in 4.4h by `short needs downtrend momentum or fresh
impulse+structure` and priced +$44.25 (WR 44/50, zero -5% stops). The
structure disjuncts require vol-spike evidence; a grind decline has none,
so the gate starves the follow-through class it was never built to cull.

Rule: short reaches conf floor, fails structured_short, lane
`short_tape_structure` (tri-state via lane_mode) is live, BTC trail-24h
drift sits in the SHARED band [btc24_min_pct, btc24_max_pct) of
tape_recovery_short_veto_gate AND tape_state_arms shows an ACTIVE arm
(fresh venture through the band floor inside its clock) -> the decline is
mechanically structured; admit with an ADMIT marker. shadow logs
WOULD-ADMIT but keeps blocking. off (default) = zero tape fetches,
byte-identical.

Fail-safe mirror (pinned, symmetric with the veto): a data gap may never
fire an ADMIT either — drift gap / history gap / malformed keys = inert.
Mutual exclusion with the veto is structural: the veto fires exactly when
NO arm is active; this admits exactly when an arm IS active — the facepalm
class stays culled by the structure bar even while the veto is shadowed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from hermes_trader.agents import executor  # noqa: E402
from hermes_trader.agents.executor import _runner_entry_block_reason  # noqa: E402

STRUCT_BLOCK = "runner_gate_blocked (short needs downtrend momentum or fresh impulse+structure; score=10, slow=0)"
TAPE_STATE = {"deep_arm_pct": -3.0, "deep_arm_clock_h": 48,
              "shallow_arm_pct": -2.0, "shallow_arm_clock_h": 12,
              "disarm_confirm_bars": 2, "disarm_hard_above_pct": 0.5}


def _gate(mode=None, **over):
    gate = {"enabled": True, "min_confidence": 0.65, "min_composite": 30.0,
            "allow_shorts": True, "min_short_confidence": 0.60,
            "min_short_composite": 25.0}
    if mode is not None:
        gate["short_tape_structure"] = mode
    gate.update(over)
    return gate


def _config(mode=None, veto_block=True, **gate_over):
    cfg = {"runner_entry_gate": _gate(mode, **gate_over)}
    if veto_block:
        # veto mode "off": the admit reads the block's band/clock keys
        # independently of the veto's own mode (asserted by every live test
        # still fetching tape below with the veto silent)
        cfg["tape_recovery_short_veto_gate"] = {
            "mode": "off", "btc24_min_pct": -2.0, "btc24_max_pct": 0.0,
            "tape_state": dict(TAPE_STATE)}
    return cfg


def _analysis(**over) -> dict:
    # unstructured fade candidate: no triggers, low score, conf clears floor
    a = {
        "coin": "TEST", "side": "short", "confidence": 0.65,
        "composite_score": 10.0, "daily_mover_fired": False,
        "volume_spike_fired": False, "breakout_fired": False,
        "momentum_burst_fired": False, "uptrend_momentum_fired": False,
        "downtrend_momentum_fired": False, "slow_burn_count": 0,
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


N = 120
NOW = N - 1

# fresh crash: drift crossed -2 SIX bars ago and sits in-band -> ARMED
FRESH = _series(N, [(N - 8, -2.4), (N - 3, -1.8)])
# stale crash: crossed -2 ~68h ago (aged beyond both clocks), -1.3 residue -> NO arm
STALE = _series(N, [(N - 70, -2.5), (N - 40, -1.3)])
# disarmed LHS slide: crash -3.5, sustained green escape, slide back to -1.6
# without re-crossing the floor -> NO arm
SLIDE = _series(N, [(N - 80, -3.5), (N - 50, 0.6), (N - 20, -1.6)])


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


# ── inert paths: zero fetches, byte-identical ───────────────────────────────

def test_key_absent_zero_fetches(monkeypatch):
    tc, hc = _stubs(monkeypatch, -1.8, _hist(FRESH))
    assert _runner_entry_block_reason(_analysis(), _config(None)) == STRUCT_BLOCK
    assert tc == [] and hc == []


def test_mode_off_zero_fetches(monkeypatch):
    tc, hc = _stubs(monkeypatch, -1.8, _hist(FRESH))
    assert _runner_entry_block_reason(_analysis(), _config("off")) == STRUCT_BLOCK
    assert tc == [] and hc == []


def test_invalid_mode_inert(monkeypatch):
    tc, hc = _stubs(monkeypatch, -1.8, _hist(FRESH))
    assert _runner_entry_block_reason(_analysis(), _config("banana")) == STRUCT_BLOCK
    assert tc == []


def test_no_veto_block_inert(monkeypatch):
    # band/clock lives in tape_recovery_short_veto_gate; absent block = inert
    tc, hc = _stubs(monkeypatch, -1.8, _hist(FRESH))
    cfg = _config("live", veto_block=False)
    assert _runner_entry_block_reason(_analysis(), cfg) == STRUCT_BLOCK
    assert tc == []


# ── live admit ──────────────────────────────────────────────────────────────

def test_live_armed_admits(monkeypatch, caplog):
    _stubs(monkeypatch, -1.8, _hist(FRESH))
    with caplog.at_level("INFO"):
        assert _runner_entry_block_reason(_analysis(), _config("live")) == ""
    assert "SHORT-TAPE-STRUCTURE" in caplog.text and "ADMIT" in caplog.text


def test_live_stale_still_blocked(monkeypatch):
    # band in-range but NO active arm -> recovery/LHS residue stays culled
    _stubs(monkeypatch, -1.3, _hist(STALE))
    assert _runner_entry_block_reason(_analysis(), _config("live")) == STRUCT_BLOCK


def test_live_slide_still_blocked(monkeypatch):
    _stubs(monkeypatch, -1.6, _hist(SLIDE))
    assert _runner_entry_block_reason(_analysis(), _config("live")) == STRUCT_BLOCK


def test_live_out_of_band_still_blocked(monkeypatch):
    # green tape: drift >= 0 -> band guard, never admits
    _stubs(monkeypatch, +0.4, _hist(FRESH))
    assert _runner_entry_block_reason(_analysis(), _config("live")) == STRUCT_BLOCK


def test_live_drift_gap_never_admits(monkeypatch):
    _stubs(monkeypatch, None, _hist(FRESH))
    assert _runner_entry_block_reason(_analysis(), _config("live")) == STRUCT_BLOCK


def test_live_history_gap_never_admits(monkeypatch):
    _stubs(monkeypatch, -1.8, None)
    assert _runner_entry_block_reason(_analysis(), _config("live")) == STRUCT_BLOCK


def test_live_malformed_drift_never_admits(monkeypatch):
    _stubs(monkeypatch, "banana", _hist(FRESH))
    assert _runner_entry_block_reason(_analysis(), _config("live")) == STRUCT_BLOCK


# ── shadow: WOULD-ADMIT accrual, still blocked ──────────────────────────────

def test_shadow_armed_logs_would_admit_but_blocks(monkeypatch, caplog):
    _stubs(monkeypatch, -1.8, _hist(FRESH))
    with caplog.at_level("INFO"):
        assert _runner_entry_block_reason(_analysis(), _config("shadow")) == STRUCT_BLOCK
    assert "SHORT-TAPE-STRUCTURE][SHADOW]" in caplog.text
    assert "WOULD ADMIT" in caplog.text


def test_shadow_stale_logs_nothing_armed(monkeypatch, caplog):
    _stubs(monkeypatch, -1.3, _hist(STALE))
    with caplog.at_level("INFO"):
        assert _runner_entry_block_reason(_analysis(), _config("shadow")) == STRUCT_BLOCK
    assert "WOULD ADMIT" not in caplog.text


# ── scope guards ────────────────────────────────────────────────────────────

def test_conf_floor_still_applies(monkeypatch):
    # the admit exempts ONLY the structure clause — conf floor precedes it
    _stubs(monkeypatch, -1.8, _hist(FRESH))
    a = _analysis(confidence=0.55)
    out = _runner_entry_block_reason(a, _config("live"))
    assert "confidence" in out and "0.55" in out


def test_structured_shorts_never_touch_tape_reads(monkeypatch):
    # downtrend fired -> structured via first disjunct -> zero tape fetches
    tc, hc = _stubs(monkeypatch, -1.8, _hist(FRESH))
    a = _analysis(downtrend_momentum_fired=True)
    assert _runner_entry_block_reason(a, _config("live")) == ""
    assert tc == [] and hc == []
