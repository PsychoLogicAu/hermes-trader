"""Tape-state stamp tests (t_860f95f8, 2026-10-30).

The stamp is LOG-ONLY: it labels which tape-phase (deep-arm / shallow-arm
crossing clock) a short-clause decision sits in; it must never change a
verdict, must be byte-identical when the config block is absent, and must
fail open (no-history) on candle gaps.
"""
import datetime as dt

import pytest

from hermes_trader.agents.risk_gates import state_admission, tape_state_stamp

H = 3_600_000
T0 = int(dt.datetime(2026, 10, 20, tzinfo=dt.UTC).timestamp() * 1000)


def hist(closes):
    return [(T0 + i * H, c) for i, c in enumerate(closes)]


def ramp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def crash_then_recover():
    # flat 50h, 100 -> 94 over 24h (drift bottoms near -6), recover to 98.5
    return [100.0] * 50 + ramp(100.0, 94.0, 24) + ramp(94.0, 98.5, 24)


CFG = {"deep_arm_pct": -3.0, "deep_arm_clock_h": 48,
       "shallow_arm_pct": -2.0, "shallow_arm_clock_h": 12}


def test_stamp_renders_both_legs_after_crash_recovery():
    s = tape_state_stamp(hist(crash_then_recover()), CFG)
    assert s != "no-history"
    deep, shallow = s.split()
    assert deep.startswith("deep=") and shallow.startswith("shallow=")
    # recovery is recent: deep leg must be armed (+); ages monotonic: the
    # shallower depth is crossed later, so its age is smaller.
    assert "+" in deep
    d_age = float(deep.split("=")[1].split("h")[0])
    s_age = float(shallow.split("=")[1].split("h")[0])
    assert s_age < d_age


def test_stamp_marks_stale_crossings_past_clock():
    closes = ([100.0] * 50 + ramp(100.0, 96.0, 10) + ramp(96.0, 100.0, 10)
              + [100.0] * 60)
    s = tape_state_stamp(hist(closes), CFG)
    assert "x<=48" in s and "x<=12" in s  # both crossings older than clocks


def test_stamp_flat_tape_no_crossings():
    s = tape_state_stamp(hist([100.0] * 100), CFG)
    assert s == "deep=n shallow=n"


def test_stamp_fail_open_on_gaps():
    assert tape_state_stamp(None, CFG) == "no-history"
    assert tape_state_stamp([], CFG) == "no-history"
    assert tape_state_stamp(hist([100.0] * 10), CFG) == "no-history"  # too short
    assert tape_state_stamp([("x", "y")] * 40, CFG) == "malformed-history"


def _short_clause(with_tape):
    sub = {"btc24_min_pct": -2.0, "btc24_max_pct": 0.0, "min_conf": 0.0,
           "quality": {"max_chronos_tail_pct": 0.45, "max_tirex_median_pct": 0.0}}
    if with_tape:
        sub["tape_state"] = dict(CFG)
    return {"shadow": True, "short": sub}


def _analysis(with_history):
    a = {"side": "short", "confidence": 0.30, "btc_drift_24h": -1.2,
         "chronos_tail_pct": 0.3, "tirex_median_pct": -0.1}
    if with_history:
        a["btc_history_1h"] = hist(crash_then_recover())
    return a


def test_admit_carries_stamp_only_when_block_present():
    admit, why, shadow = state_admission(_analysis(True), _short_clause(True), True)
    assert admit and shadow and " tape [deep=" in why
    admit2, why2, _ = state_admission(_analysis(True), _short_clause(False), True)
    assert admit2 and "tape [" not in why2


def test_quality_skip_lines_also_stamped():
    a = _analysis(True)
    a["chronos_tail_pct"] = 0.6
    admit, why, _ = state_admission(a, _short_clause(True), True)
    assert not admit and why.startswith("QUALITY-SKIP: chronos upside tail") and " tape [" in why


def test_stamp_fail_open_admit_still_works():
    # block present but caller injected no history (candle gap): stamp
    # degrades, verdict unchanged.
    admit, why, _ = state_admission(_analysis(False), _short_clause(True), True)
    assert admit and " tape [no-history]" in why


def test_long_side_untouched():
    a = {"side": "long", "confidence": 0.30, "btc_drift_24h": -1.2}
    clause = {"shadow": True, "long": {"btc24_max_pct": 0.0},
              "short": {"btc24_min_pct": -2.0, "btc24_max_pct": 0.0,
                        "tape_state": dict(CFG)}}
    admit, why, _ = state_admission(a, clause, True)
    assert admit and "tape [" not in why
