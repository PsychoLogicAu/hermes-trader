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


def _block(mode, mn: object = 1.0) -> dict:
    b = {"mode": mode}
    if mn is not None:
        b["btc24_min_pct"] = mn
    return b


def _config(mode, sth_min: object = 1.0, **over) -> dict:
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


# ── independent-audit pins (deleg_f9116435, 2026-11-28) ─────────────────────
# The audit proved two raise surfaces escaped the runner gate (no per-gate
# try upstream -> trading_loop watchdog aborts the whole scan cycle): a
# truthy NON-DICT top-level value (.get AttributeError, both sides, before
# the side guard) and huge-int values (float(10**400) OverflowError past the
# bare (TypeError, ValueError)). Plus polarity: bool floor silent-arm,
# +inf drift firing, armed-mode-without-floor silent no-op. All pinned here.

def test_top_level_bool_inert(monkeypatch):
    tc = _stub_tape(monkeypatch, 5.0)
    cfg = _config(None)
    cfg["strength_short_throttle_gate"] = True  # truthy non-dict typo
    assert _runner_entry_block_reason(_analysis(), cfg) == ""
    assert tc == []


def test_top_level_list_inert(monkeypatch):
    tc = _stub_tape(monkeypatch, 5.0)
    cfg = _config(None)
    cfg["strength_short_throttle_gate"] = [{"mode": "live"}]
    assert _runner_entry_block_reason(_analysis(side="short"), cfg) == ""
    assert tc == []


def test_top_level_string_inert_long_side_too(monkeypatch):
    # pre-side-guard raise: the LONG path must be equally protected
    tc = _stub_tape(monkeypatch, 5.0)
    cfg = _config(None)
    cfg["strength_short_throttle_gate"] = "shadow"
    assert _runner_entry_block_reason(_analysis(side="long"), cfg) != BLOCK
    assert tc == []


def test_top_level_false_silent_off(monkeypatch, caplog):
    tc = _stub_tape(monkeypatch, 5.0)
    cfg = _config(None)
    cfg["strength_short_throttle_gate"] = False
    with caplog.at_level("WARNING"):
        assert _runner_entry_block_reason(_analysis(), cfg) == ""
    assert "STRENGTH-THROTTLE" not in caplog.text
    assert tc == []


def test_huge_int_floor_inert_no_raise(monkeypatch, caplog):
    # float(10**400) raises OverflowError — the _sd_finite class of defect
    tc = _stub_tape(monkeypatch, 5.0)
    cfg = _config(None)
    cfg["strength_short_throttle_gate"] = {"mode": "live",
                                          "btc24_min_pct": 10 ** 400}
    with caplog.at_level("WARNING"):
        assert _runner_entry_block_reason(_analysis(), cfg) == ""
    assert "not a finite number" in caplog.text
    assert tc == []


def test_huge_int_drift_inert_no_raise(monkeypatch):
    tc = _stub_tape(monkeypatch, 10 ** 400)
    assert _runner_entry_block_reason(_analysis(), _config("live")) == ""
    assert len(tc) == 1  # consulted, parsed to None, inert


def test_inf_drift_does_not_fire(monkeypatch):
    # polarity vs sibling: garbage +inf sensor value must NOT arm a live
    # block (pre-fix float('inf') >= 1.0 fired)
    _stub_tape(monkeypatch, float("inf"))
    assert _runner_entry_block_reason(_analysis(), _config("live")) == ""


def test_bool_floor_refused(monkeypatch, caplog):
    # S1 precedent (t_fb899217): bool shorthand refused explicitly, never
    # silently arms at float(True)==1.0
    tc = _stub_tape(monkeypatch, 5.0)
    with caplog.at_level("WARNING"):
        r = _runner_entry_block_reason(_analysis(), _config("live", sth_min=True))
    assert r == ""
    assert "boolean" in caplog.text
    assert tc == []


def test_nan_floor_inert(monkeypatch, caplog):
    tc = _stub_tape(monkeypatch, 5.0)
    cfg = _config(None)
    cfg["strength_short_throttle_gate"] = {"mode": "shadow",
                                          "btc24_min_pct": float("nan")}
    with caplog.at_level("WARNING"):
        assert _runner_entry_block_reason(_analysis(), cfg) == ""
    assert "not a finite number" in caplog.text
    assert tc == []


def test_armed_without_floor_warns(monkeypatch, caplog):
    # mode live + no floor must NOT be a silent dead gate
    tc = _stub_tape(monkeypatch, 5.0)
    with caplog.at_level("WARNING"):
        r = _runner_entry_block_reason(_analysis(), _config("live", sth_min=None))
    assert r == ""
    assert "absent" in caplog.text
    assert tc == []
