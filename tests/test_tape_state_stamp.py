"""Tape-state stamp tests (t_860f95f8, 2026-10-30).

The stamp is LOG-ONLY: it labels which tape-phase (deep-arm / shallow-arm
crossing clock) a short-clause decision sits in; it must never change a
verdict, must be byte-identical when the config block is absent, and must
fail open (no-history) on candle gaps.
"""
import datetime as dt

import pytest

from hermes_trader.agents.risk_gates import (
    state_admission,
    tape_state_arms,
    tape_state_stamp,
)

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


# ---- arm GATE (require_arm) — owner ruling 2026-10-30: no entry until the
# drift has gone through the band floor and recovered back. Gate FAILS CLOSED.

def drift_hist(dvals):
    """Close series whose hourly 24h-lag drift EQUALS dvals (bar 24 onward):
    closes[t] = closes[t-24] * (1 + d_t/100). Drift paths explicit, no
    ramp-overshoot artifacts."""
    c = [100.0] * 24
    for i, dv in enumerate(dvals):
        c.append(c[i] * (1.0 + dv / 100.0))
    return hist(c)


_FRESH = [0.0] * 40 + ramp(0.0, -8.0, 24) + ramp(-8.0, -1.0, 8)
# deep+shallow up-crossings land near the series end; drift never reaches 0.


def _clause(require_arm, tape=True):
    sub = {"btc24_min_pct": -2.0, "btc24_max_pct": 0.0, "min_conf": 0.0}
    if tape:
        sub["tape_state"] = {"deep_arm_pct": -3.0, "deep_arm_clock_h": 48.0,
                             "shallow_arm_pct": -2.0, "shallow_arm_clock_h": 12.0,
                             "require_arm": require_arm}
    return {"shadow": True, "short": sub}


def _a(hist=None):
    a = {"side": "short", "confidence": 0.30, "btc_drift_24h": -1.0}
    if hist is not None:
        a["btc_history_1h"] = hist
    return a


_FRESH_CRASH = [100.0] * 60 + ramp(100.0, 92.0, 24) + ramp(92.0, 99.0, 20)
# drift goes ~-8 (through the floor), then recovers back across -3/-2 near
# the series end -> deep and shallow arms both active.


_FRESH_CRASH = None  # superseded by drift-path _FRESH below


def test_require_arm_admits_after_crash_recovery():
    adm, why, _ = state_admission(_a(drift_hist(_FRESH)), _clause(True), True)
    assert adm and "kept tape" in why and "deep=" in why


def test_require_arm_refuses_slide_from_above_never_below_floor():
    # tape slid into the band from flat: drift never breached the floor
    slide = [100.0] * 60 + ramp(100.0, 98.6, 60)  # drift bottoms ~-1.4
    adm, why, _ = state_admission(_a(hist(slide)), _clause(True), True)
    assert not adm and "ARM-SKIP" in why
    # ...and the very same analysis is ADMITTED with the gate off (stamp only)
    adm2, _, _ = state_admission(_a(hist(slide)), _clause(False), True)
    assert adm2


def test_require_arm_refuses_stale_and_no_crossings():
    stale = crash_then_recover() + [98.5] * 120  # recovery crossing ages out
    adm, why, _ = state_admission(_a(hist(stale)), _clause(True), True)
    assert not adm and "ARM-SKIP" in why and "hd<=" in why  # stale -> past-clock or disarmed markers
    adm, why, _ = state_admission(_a(hist([100.0] * 150)), _clause(True), True)
    assert not adm and "ARM-SKIP" in why


def test_require_arm_fail_closed_on_history_gaps():
    for bad in (None, [], [("x", "y")] * 40):
        a = _a(None)
        if bad is not None:
            a["btc_history_1h"] = bad
        adm, why, _ = state_admission(a, _clause(True), True)
        assert not adm and "ARM-SKIP" in why
    # stamp-only mode keeps the fail-OPEN no-history admit (unchanged)
    adm, why, _ = state_admission(_a(None), _clause(False), True)
    assert adm and "no-history" in why


def test_gate_off_is_byte_identical_stamp_only():
    slide = [100.0] * 60 + ramp(100.0, 98.6, 60)
    a_no_block = _a(hist(slide))
    adm1, why1, _ = state_admission(a_no_block, {"shadow": True, "short": {
        "btc24_min_pct": -2.0, "btc24_max_pct": 0.0, "min_conf": 0.0}}, True)
    adm2, why2, _ = state_admission(a_no_block, _clause(False), True)
    assert adm1 and adm2 and why2.startswith(why1)


def test_tape_state_arms_data_matches_stamp():
    H = drift_hist(_FRESH)
    cfg = {"deep_arm_clock_h": 48.0, "shallow_arm_clock_h": 48.0}
    arms = tape_state_arms(H, cfg, band_max=0.0)
    assert arms["deep"]["active"] and arms["shallow"]["active"]
    assert arms["deep"]["age"] is not None and arms["deep"]["age"] <= 48
    stamp = tape_state_stamp(H, cfg, band_max=0.0)
    assert "deep=" in stamp and stamp.count("+") == 2


# ---- DISARM rule (owner ruling 2026-10-30): drift venturing out of the band
# above the top (>0) disarms prior crossings; re-arm needs a fresh venture
# below the floor. Stamp marker 'd' = crossed-but-disarmed.

_TOP_EXIT = (_FRESH + ramp(-1.0, 3.0, 10) + [3.0] * 20
             + ramp(3.0, -1.5, 15))
# crash -> recover (crossing) -> drift exits above band top 0 (disarm)
# -> slides back INTO band from above, never dipping below the floor.


def test_band_top_venture_disarms_prior_crossing():
    H = drift_hist(_TOP_EXIT)
    arms = tape_state_arms(H, {}, band_max=0.0)
    assert not arms["deep"]["active"] and not arms["shallow"]["active"]
    assert arms["shallow"]["disarmed"] and arms["deep"]["disarmed"]
    adm, why, _ = state_admission(_a(H), _clause(True), True)
    assert not adm and "ARM-SKIP" in why
    stamp = tape_state_stamp(H, {}, band_max=0.0)
    assert "d<=" in stamp  # crossed but disarmed, not armed '+' nor stale 'x'


def test_fresh_dip_after_disarm_rearms():
    rearm = _TOP_EXIT + ramp(-1.5, -4.0, 10) + ramp(-4.0, -1.2, 10)
    H = drift_hist(rearm)
    arms = tape_state_arms(H, {}, band_max=0.0)
    assert arms["shallow"]["active"] and not arms["shallow"]["disarmed"]
    adm, why, _ = state_admission(_a(H), _clause(True), True)
    assert adm and "kept tape" in why


def test_disarm_above_pct_knob_overrides_band_top():
    gentle = _FRESH + ramp(-1.0, -0.6, 10)  # recovers to -0.6, never >= 0
    H = drift_hist(gentle)
    a_ok = tape_state_arms(H, {}, band_max=0.0)
    assert not a_ok["shallow"]["disarmed"] and a_ok["shallow"]["active"]
    a_k = tape_state_arms(H, {"disarm_above_pct": -1.0}, band_max=0.0)
    assert a_k["shallow"]["disarmed"] and not a_k["shallow"]["active"]


def test_no_band_max_back_compat_no_disarm():
    arms = tape_state_arms(drift_hist(_TOP_EXIT), {})  # no band context
    assert not arms["shallow"]["disarmed"]
    assert arms["shallow"]["age"] is not None
