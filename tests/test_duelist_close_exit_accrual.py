"""Tests for the B.10 duelist exit-voice shadow accrual (research.py).

The accrual is LOG-ONLY: when the A/B duelist calls CLOSE on a HELD coin and
the primary did NOT close it, a `[gate][SHADOW] duelist_close_exit WOULD HAVE
CLOSED ...` line is logged with the fields the offline join needs (dl_conf,
age, spot, peak, entry/mark, pid). It never acts on the book, never raises,
and fires on EVERY held-CLOSE split (no conf cutoff in code — the trigger
variants are priced offline from the logged fields).

Design + in-sample replay: .hermes/WATCHLIST.md §B.10.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import research  # noqa: E402


def _row(verdict="CLOSE", conf=0.85, pid="pid-1"):
    return {
        "coin": "TEST",
        "perception_id": pid,
        "duelist_verdict": verdict,
        "duelist_confidence": conf,
    }


def _held(coin="TEST", side="long"):
    return [{"coin": coin, "side": side, "size_usd": 33.0}]


class _FakeTracker:
    def __init__(self, entry_px, entry_time, peak_px, mark_px):
        self.entry_px = entry_px
        self.entry_time = entry_time
        self.peak_px = peak_px
        self.mark_peak_px = peak_px
        self.last_mark_px = mark_px


def _capture(caplog):
    return caplog


def test_held_close_split_logs(caplog):
    import time as _t
    from hermes_trader.agents import dsl_exit as _dslx
    trk = _FakeTracker(100.0, _t.time() - 45 * 60, 101.2, 99.0)
    _dslx._active_positions["TEST_long"] = trk
    try:
        with caplog.at_level("INFO", logger="hermes_trader.agents.research"):
            research._duelist_close_exit_accrual("TEST", _row(conf=0.90), _held())
        lines = [r.message for r in caplog.records
                 if "duelist_close_exit WOULD HAVE CLOSED" in r.message]
        assert len(lines) == 1
        msg = lines[0]
        assert "TEST long" in msg
        assert "dl_conf 0.90" in msg
        assert "45min" in msg          # age from tracker entry_time
        assert "-1.00%" in msg         # spot vs entry (long, mark 99 vs 100)
        assert "+1.20%" in msg         # one-way peak
        assert "pid-1" in msg
    finally:
        _dslx._active_positions.pop("TEST_long", None)


def test_short_side_signs(caplog):
    import time as _t
    from hermes_trader.agents import dsl_exit as _dslx
    # short: entry 100, peak (favourable) 98, mark 101 -> spot -1.00%, peak +2.00%
    trk = _FakeTracker(100.0, _t.time() - 10 * 60, 98.0, 101.0)
    _dslx._active_positions["TEST_short"] = trk
    try:
        with caplog.at_level("INFO", logger="hermes_trader.agents.research"):
            research._duelist_close_exit_accrual(
                "TEST", _row(conf=0.70), _held(side="short"))
        lines = [r.message for r in caplog.records
                 if "duelist_close_exit WOULD HAVE CLOSED" in r.message]
        assert len(lines) == 1
        assert "TEST short" in lines[0]
        assert "-1.00%" in lines[0]
        assert "+2.00%" in lines[0]
    finally:
        _dslx._active_positions.pop("TEST_short", None)


def test_non_close_verdict_silent(caplog):
    with caplog.at_level("INFO", logger="hermes_trader.agents.research"):
        for v in ("PASS", "VETO", "LONG", "SHORT", ""):
            research._duelist_close_exit_accrual("TEST", _row(verdict=v), _held())
    assert not [r for r in caplog.records
                if "duelist_close_exit" in r.message]


def test_unheld_coin_silent(caplog):
    # Defensive: parse_verdict's close-guard already downgrades unheld CLOSEs,
    # but the accrual must not log for a coin that isn't in open_positions.
    with caplog.at_level("INFO", logger="hermes_trader.agents.research"):
        research._duelist_close_exit_accrual(
            "TEST", _row(), [{"coin": "OTHER", "side": "long", "size_usd": 10.0}])
    assert not [r for r in caplog.records
                if "duelist_close_exit" in r.message]


def test_missing_tracker_still_logs(caplog):
    # No DSL tracker (e.g. rehydrated position not yet tracked): the line
    # still fires with n/a fields — the join needs the split, not the detail.
    from hermes_trader.agents import dsl_exit as _dslx
    _dslx._active_positions.pop("TEST_long", None)
    with caplog.at_level("INFO", logger="hermes_trader.agents.research"):
        research._duelist_close_exit_accrual("TEST", _row(conf=0.55), _held())
    lines = [r.message for r in caplog.records
             if "duelist_close_exit WOULD HAVE CLOSED" in r.message]
    assert len(lines) == 1
    assert "n/a" in lines[0]


def test_never_raises_on_bad_tracker(caplog):
    class _Broken:
        entry_px = 100.0
        @property
        def entry_time(self):
            raise RuntimeError("tracker exploded")
    from hermes_trader.agents import dsl_exit as _dslx
    _dslx._active_positions["TEST_long"] = _Broken()
    try:
        with caplog.at_level("DEBUG", logger="hermes_trader.agents.research"):
            research._duelist_close_exit_accrual("TEST", _row(), _held())
        # inner tracker read swallowed -> line still logged with n/a age
        lines = [r.message for r in caplog.records
                 if "duelist_close_exit WOULD HAVE CLOSED" in r.message]
        assert len(lines) == 1
    finally:
        _dslx._active_positions.pop("TEST_long", None)
