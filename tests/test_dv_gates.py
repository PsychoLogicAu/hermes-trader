"""Tests for the clef decision-voice GATES (clef-gate scope Changes 1-3).

Two enforcement points, both shadow-first per the house pattern
(.hermes/plans/2026-10-07-clef-gate-close-voice-scope.md):

  1. entry trap-veto (executor._dv_trap_veto_reason) — shadow ON logs
     WOULD-BE-BLOCKED and allows; shadow OFF hard-blocks with the exact
     reason string through the runner_gate forensics path; NO dv row is
     fail-open (a failed observer is 'no observation', never a block).
  2. close_veto accrual (research._dv_close_exit_accrual) — held +
     close_now >= threshold + primary != CLOSE -> WOULD-HAVE-CLOSED line;
     grace_minutes suppresses young positions.

Plus the whitelist regression (the 2026-08 pitfall): trap/close_now survive
analysis -> entry-context -> close-row join.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import executor, research  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

def _cfg(threshold=0.45, shadow=True, enabled=True, close_thr=0.50,
         close_shadow=True, grace=90):
    return {
        "decision_voice_gate": {
            "enabled": enabled,
            "trap_veto": {"shadow_mode": shadow, "threshold": threshold},
            "close_veto": {"shadow_mode": close_shadow, "threshold": close_thr,
                           "grace_minutes": grace},
        }
    }


def _analysis(trap=0.52, close_now=None, **over):
    a = {
        "id": "a1", "coin": "TEST", "verdict": "LONG", "side": "long",
        "confidence": 0.80, "composite_score": 55, "entry_px": 100.0,
        "perception_id": "pid-dv-1",
        "decision_voice_at_entry": {
            "model": "clef-flash", "verdict": "LONG", "confidence": 0.7,
            "side": "long", "trap": trap, "close_now": close_now,
        },
    }
    a.update(over)
    return a


# ── 1. trap-veto: hard block, exact reason string ─────────────────────────

def test_trap_veto_hard_block_reason_exact():
    r = executor._dv_trap_veto_reason(_analysis(trap=0.52), _cfg(shadow=False))
    assert r == "dv_trap_veto (trap 0.52 >= 0.45)"


def test_trap_veto_below_threshold_passes():
    assert executor._dv_trap_veto_reason(_analysis(trap=0.40), _cfg()) == ""


# ── 2. trap-veto: shadow ON -> allowed + accrual line ─────────────────────

def test_trap_veto_shadow_logs_and_allows(caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.executor"):
        r = executor._dv_trap_veto_reason(_analysis(trap=0.52), _cfg(shadow=True))
    assert r == ""  # trade allowed
    lines = [rec.getMessage() for rec in caplog.records
             if "dv_trap_veto WOULD BE BLOCKED" in rec.getMessage()]
    assert len(lines) == 1
    msg = lines[0]
    assert "TEST LONG" in msg
    assert "trap 0.52 >= 0.45" in msg
    assert "conf 0.80" in msg
    assert "composite 55" in msg
    assert "pid-dv-1" in msg


# ── 3. fail-open: no dv row / missing scalar / gate off ───────────────────

def test_trap_veto_fail_open_no_dv_row():
    a = _analysis()
    a["decision_voice_at_entry"] = None
    assert executor._dv_trap_veto_reason(a, _cfg(shadow=False)) == ""


def test_trap_veto_fail_open_missing_trap_scalar():
    a = _analysis()
    a["decision_voice_at_entry"] = {"model": "clef-flash", "verdict": "LONG"}
    assert executor._dv_trap_veto_reason(a, _cfg(shadow=False)) == ""


def test_trap_veto_gate_disabled_no_opinion():
    assert executor._dv_trap_veto_reason(_analysis(trap=0.99),
                                         _cfg(enabled=False, shadow=False)) == ""


# ── 4. close_veto accrual: held + >= thr + primary != CLOSE ───────────────

class _FakeTracker:
    def __init__(self, entry_px, entry_time, peak_px, mark_px):
        self.entry_px = entry_px
        self.entry_time = entry_time
        self.peak_px = peak_px
        self.last_mark_px = mark_px


def _dv_row(close_now=0.72, pid="pid-dv-9"):
    return {"coin": "TEST", "perception_id": pid, "dv_close_now": close_now}


def _held(coin="TEST", side="long"):
    return [{"coin": coin, "side": side, "size_usd": 33.0}]


@pytest.fixture
def _gate_cfg(monkeypatch):
    """Write the decision_voice_gate block into the agent config (hot read)."""
    import hermes_trader.agents.config_store as cs
    from pathlib import Path

    def _write(**kw):
        cfg_path = Path(cs.CONFIG_PATH)
        cfg_path.unlink(missing_ok=True)
        with open(cfg_path, "w") as f:
            json.dump({"mode": "LIVE", **_cfg(**kw)}, f)
    yield _write
    Path(cs.CONFIG_PATH).unlink(missing_ok=True)


def test_close_veto_fires_held_split(caplog, _gate_cfg):
    _gate_cfg()
    trk = _FakeTracker(100.0, time.time() - 120 * 60, 101.0, 97.0)
    from hermes_trader.agents import dsl_exit as _dslx
    _dslx._active_positions["TEST_long"] = trk
    try:
        with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
            research._dv_close_exit_accrual("TEST", _dv_row(0.72), _held())
        lines = [r.getMessage() for r in caplog.records
                 if "dv_close_veto WOULD HAVE CLOSED" in r.getMessage()]
        assert len(lines) == 1
        msg = lines[0]
        assert "TEST long" in msg
        assert "close_now 0.72 >= 0.50" in msg
        assert "120min" in msg
        assert "-3.00%" in msg      # spot vs entry (mark 97 vs entry 100)
        assert "pid-dv-9" in msg
    finally:
        _dslx._active_positions.pop("TEST_long", None)


def test_close_veto_silent_below_threshold(caplog, _gate_cfg):
    _gate_cfg()
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
        research._dv_close_exit_accrual("TEST", _dv_row(0.40), _held())
    assert not [r for r in caplog.records
                if "dv_close_veto" in r.getMessage()]


def test_close_veto_silent_when_unheld(caplog, _gate_cfg):
    _gate_cfg()
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
        research._dv_close_exit_accrual("TEST", _dv_row(0.90), [])
    assert not [r for r in caplog.records
                if "dv_close_veto" in r.getMessage()]


def test_close_veto_silent_when_gate_disabled(caplog, _gate_cfg):
    _gate_cfg(enabled=False)
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
        research._dv_close_exit_accrual("TEST", _dv_row(0.90), _held())
    assert not [r for r in caplog.records
                if "dv_close_veto" in r.getMessage()]


# ── 5. grace_minutes suppresses young positions ───────────────────────────

def test_close_veto_grace_suppresses_young_position(caplog, _gate_cfg):
    _gate_cfg(grace=90)
    trk = _FakeTracker(100.0, time.time() - 30 * 60, 101.0, 97.0)  # 30min old
    from hermes_trader.agents import dsl_exit as _dslx
    _dslx._active_positions["TEST_long"] = trk
    try:
        with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
            research._dv_close_exit_accrual("TEST", _dv_row(0.90), _held())
        assert not [r for r in caplog.records
                    if "dv_close_veto" in r.getMessage()]
    finally:
        _dslx._active_positions.pop("TEST_long", None)


def test_close_veto_shadow_off_logs_but_does_not_act(caplog, _gate_cfg):
    """Promotion NOT wired yet: shadow OFF must log LOUDLY and still not act
    (the synthetic-CLOSE injection is a separate reviewed change)."""
    _gate_cfg(close_shadow=False)
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.research"):
        research._dv_close_exit_accrual("TEST", _dv_row(0.72), _held())
    lines = [r.getMessage() for r in caplog.records
             if "dv_close_veto FIRED (shadow OFF, action NOT wired)" in r.getMessage()]
    assert len(lines) == 1
    assert not [r for r in caplog.records
                if "WOULD HAVE CLOSED" in r.getMessage()]


# ── 6. whitelist regression: analysis -> entry-context -> close row ───────

def test_trap_close_scalars_survive_entry_context_join(monkeypatch):
    """The 2026-08 pitfall test: a field not in the analysis whitelist
    silently never reaches the executor. Pin the full ride:
    analysis.decision_voice_at_entry -> memory entry-context 'decision_voice'
    -> close-row 'decision_voice_at_entry'."""
    from hermes_trader.agents.memory import AgentMemory

    a = _analysis(trap=0.61, close_now=0.55)

    # executor snapshots analysis -> entry context (the dict literal at the
    # record_entry_context call site): pin the mapping directly.
    m = AgentMemory()
    m._initialized = True
    ctx = {
        "decision_voice": a.get("decision_voice_at_entry"),
        "perception_id": a.get("perception_id"),
    }
    m.record_entry_context("TEST", "long", ctx)
    ec = m.pop_entry_context("TEST", "long")
    assert ec["decision_voice"]["trap"] == 0.61
    assert ec["decision_voice"]["close_now"] == 0.55

    # And the research() whitelist itself carries the scalars (source-level
    # pin — the dict literal must name both keys).
    src = Path(research.__file__).read_text()
    wl = src.split('"decision_voice_at_entry"', 1)[1].split("if dv_row", 1)[0]
    assert '"trap": dv_row["dv_trap"]' in wl
    assert '"close_now": dv_row["dv_close_now"]' in wl
