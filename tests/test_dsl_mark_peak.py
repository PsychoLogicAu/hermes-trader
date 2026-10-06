"""mark_peak_px — the SAMPLED-mark peak, separate from the wick-ratcheted peak_px.

CHIP 2026-10-06 01:20 UTC: an intrabar wick ratcheted peak_px to exactly the
scalp protect_pct (1.0%) while the best sampled mark was +0.77% — the phase-2
trailing floor had NEVER armed (it arms in check() on the mark), yet the
ai_close slot-defer guard's peak_px-based "armed" test read the wick as armed
and let an LLM stale-close through at +0.12% spot. Fix: trackers keep
mark_peak_px (updated only by check()/refresh_entry_basis, never by
ratchet_peak), and the guard tests that.
"""
from __future__ import annotations

import time

import pytest

from hermes_trader.agents import dsl_exit
from hermes_trader.agents.dsl_exit import DSLTracker, ExitPolicy


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(dsl_exit, "DSL_STATE_FILE", str(tmp_path / "dsl_state.json"))
    with dsl_exit._registry_lock:
        dsl_exit._active_positions.clear()
    dsl_exit._loaded_from_disk = False
    yield
    with dsl_exit._registry_lock:
        dsl_exit._active_positions.clear()
    dsl_exit._loaded_from_disk = False


def _tracker(coin="FOO", side="long", entry=100.0, protect=1.0):
    return DSLTracker(coin, side, entry, time.time(),
                      ExitPolicy(protect_pct=protect))


def test_check_updates_mark_peak_and_ratchet_does_not():
    t = _tracker()
    t.check(100.77)                 # sampled mark -> both peaks move
    assert t.mark_peak_px == 100.77 and t.peak_px == 100.77
    t.ratchet_peak(hi=101.0, lo=100.1)  # intrabar wick -> only peak_px moves
    assert t.peak_px == 101.0
    assert t.mark_peak_px == 100.77     # the seam: wick must NOT count here


def test_mark_peak_never_regresses():
    t = _tracker()
    t.check(100.77)
    t.check(100.10)
    assert t.mark_peak_px == 100.77


def test_short_mark_peak_tracks_low():
    t = _tracker(side="short")
    t.check(99.25)
    t.ratchet_peak(hi=99.5, lo=99.0)   # wick down to -1.0%
    assert t.peak_px == 99.0
    assert t.mark_peak_px == 99.25


def test_state_roundtrip_preserves_mark_peak():
    t = _tracker()
    t.check(100.77)
    t.ratchet_peak(hi=101.0, lo=100.1)
    with dsl_exit._registry_lock:
        dsl_exit._active_positions["FOO_long"] = t
    dsl_exit._save_state()
    d = dsl_exit._tracker_from_dict(dsl_exit._tracker_to_dict(t))
    assert d.peak_px == 101.0 and d.mark_peak_px == 100.77


def test_legacy_state_without_mark_peak_falls_back_conservatively():
    # Pre-fix state files: mark_peak_px absent -> seed at entry (reads as
    # never-armed to the guard until the next check() tick re-seeds it).
    d = {"coin": "FOO", "side": "long", "entry_px": 100.0,
         "entry_time": time.time(), "peak_px": 101.0}
    t = dsl_exit._tracker_from_dict(d)
    assert t.peak_px == 101.0
    assert t.mark_peak_px == 100.0


def test_add_refresh_clamps_mark_peak_to_new_basis():
    t = _tracker()
    t.check(100.5)
    t.refresh_entry_basis(101.0, 200.0)   # add above the old mark peak
    assert t.mark_peak_px == 101.0        # clamped, never negative range
