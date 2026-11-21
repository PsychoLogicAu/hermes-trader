"""Stale-depth veto on the tape-armed short admit (2026-11-21 flood).

The `short_tape_structure` arm admits a SHORT on tape alone (ACTIVE arm =
fresh venture through the band floor). That says the decline is
mechanically structured; it does not say the decline still has room. This
veto rides that ADMIT only — never a seat-driven entry, and the general
structure clause above it is untouched (owner ruling 2026-11-21: 'wire
that up as live do skip, and let's leave the general clause open').

Cell (three legs, all required, from ONE btc_tape_activity read):
  drift <= btc24_deep_pct (deep) AND net6h <= net6h_max_pct (the 6h leg is
  already realized) AND low4h_age_min >= low_age_min_min (no new 4h local
  low for a while — the decline's front passed).

Evidence: 133 closed shorts Sep 1 - Nov 20 (+$7.64 book) contain ZERO
instances in the production cell (-8/-4/30min); the nine 05:13-06:14 knife
entries of 2026-11-21 all sit in it (-$34.40 incl. two still open at
-$27.47 marked) while the seven profitable pre-dawn entries sit at
local-low age 0-10 min. n=4 out of sample, so the lever is tri-state and
logs every refusal.

The integration tests below run the harness's synthetic 120-bar tape, which
arms at the -2/-3 crossing levels, so they use test-local depth values; the
production cell (-8/-4/30) is pinned against the helper directly, and the
scalar itself is pinned against the candle series.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from hermes_trader.agents import executor  # noqa: E402
from hermes_trader.agents.executor import (  # noqa: E402
    _runner_entry_block_reason, _stale_depth_squeeze)
from hermes_trader.models.types import Candle  # noqa: E402

STRUCT_BLOCK = ("runner_gate_blocked (short needs downtrend momentum or "
                "fresh impulse+structure; score=10, slow=0)")
TAPE_STATE = {"deep_arm_pct": -3.0, "deep_arm_clock_h": 48,
              "shallow_arm_pct": -2.0, "shallow_arm_clock_h": 12,
              "disarm_confirm_bars": 2, "disarm_hard_above_pct": 0.5}
CELL = {"btc24_deep_pct": -1.5, "net6h_max_pct": -4.0, "low_age_min_min": 30}
PROD_CELL = {"btc24_deep_pct": -8.0, "net6h_max_pct": -4.0, "low_age_min_min": 30}


def _series(n, segments):
    out, cur = [], 0.0
    marks = dict(segments)
    for i in range(n):
        if i in marks:
            cur = marks[i]
        out.append(cur)
    return out


def _hist(drift_series):
    """(t_ms, close) 1h bars realizing drift(t) == drift_series(t):
    c(t) = c(t-24h) * (1 + d(t)/100)."""
    t0 = 1_790_000_000_000
    closes, by_t = [], {}
    for i, d in enumerate(drift_series):
        t = t0 + i * 3_600_000
        c = 100.0 if i < 24 else by_t[t - 86_400_000] * (1.0 + d / 100.0)
        by_t[t] = c
        closes.append((t, c))
    return closes


N = 120
NOW = N - 1
# fresh crash: crossed -2 six bars ago, sits in band -> shallow ARM ACTIVE
# (the same series the short_tape_structure harness proves arms the arm)
FRESH = _series(N, [(N - 8, -2.4), (N - 3, -1.8)])
HIST = _hist(FRESH)


def _config(stale_mode=None, cell=None, arm="live"):
    gate = {"enabled": True, "min_confidence": 0.65, "min_composite": 30.0,
            "allow_shorts": True, "min_short_confidence": 0.60,
            "min_short_composite": 25.0, "short_tape_structure": arm}
    if stale_mode is not None:
        gate["short_tape_stale_depth_veto"] = stale_mode
    cfg = {"runner_entry_gate": gate,
           "tape_recovery_short_veto_gate": {
               "mode": "off", "btc24_min_pct": -2.0, "btc24_max_pct": 0.0,
               "tape_state": dict(TAPE_STATE)}}
    cfg["short_tape_stale_depth_gate"] = dict(CELL if cell is None else cell)
    return cfg


def _analysis(**over) -> dict:
    a = {"coin": "TEST", "side": "short", "confidence": 0.65,
         "composite_score": 10.0, "daily_mover_fired": False,
         "volume_spike_fired": False, "breakout_fired": False,
         "momentum_burst_fired": False, "uptrend_momentum_fired": False,
         "downtrend_momentum_fired": False, "slow_burn_count": 0,
         "whale_signal": False, "reasoning": ""}
    a.update(over)
    return a


def _stubs(monkeypatch, drift=-1.8, net6h=-5.1, low_age=95):
    """The tape read as production returns it. net6h/low4h_age_min may be
    absent or None (older read / short history) — both must keep the veto
    inert."""
    tape = {"vol": 1.0, "drift": drift}
    if net6h is not None:
        tape["net6h"] = net6h
    if low_age is not None:
        tape["low4h_age_min"] = low_age
    monkeypatch.setattr(executor, "_late_chase_tape_read", lambda: tape)
    monkeypatch.setattr(executor, "_tape_history_read", lambda: HIST)
    return tape


def _reason(**kw):
    return _runner_entry_block_reason(_analysis(), _config(**kw))


# ── the veto is a rider on the arm's admit: off/absent = byte-identical ────

def test_veto_key_absent_admits(monkeypatch, caplog):
    _stubs(monkeypatch)
    with caplog.at_level("INFO"):
        assert _reason() == ""
    assert "ADMIT" in caplog.text and "STALE-DEPTH" not in caplog.text


def test_veto_off_admits(monkeypatch, caplog):
    _stubs(monkeypatch)
    with caplog.at_level("INFO"):
        assert _reason(stale_mode="off") == ""
    assert "ADMIT" in caplog.text and "STALE-DEPTH" not in caplog.text


# ── live: the cell skips the admit ─────────────────────────────────────────

def test_live_cell_hit_skips(monkeypatch, caplog):
    _stubs(monkeypatch)
    with caplog.at_level("INFO"):
        got = _reason(stale_mode="live")
    assert got == STRUCT_BLOCK
    assert "[STALE-DEPTH]" in caplog.text and "SKIP" in caplog.text
    assert "ADMIT" not in caplog.text


def test_shadow_logs_would_skip_and_still_admits(monkeypatch, caplog):
    _stubs(monkeypatch)
    with caplog.at_level("INFO"):
        assert _reason(stale_mode="shadow") == ""
    assert "WOULD SKIP" in caplog.text and "ADMIT" in caplog.text


# ── each leg is required; the front clock is what separates 11-21's cohorts ─

def test_fresh_local_low_admits(monkeypatch):
    # 11-14's founder class: deep tape at a NEW local low -> follow-through
    # is still live and the admit stands
    _stubs(monkeypatch, low_age=0)
    assert _reason(stale_mode="live") == ""


def test_flat_6h_leg_admits(monkeypatch):
    # 11-20's grind class: old low, but the 6h leg is not realized yet ->
    # still harvesting, not the squeeze
    _stubs(monkeypatch, net6h=-1.2)
    assert _reason(stale_mode="live") == ""


def test_shallow_tape_admits(monkeypatch):
    _stubs(monkeypatch, drift=-1.2)
    assert _reason(stale_mode="live") == ""


def test_leg_boundaries_inclusive(monkeypatch):
    # drift/net6h/age at the exact configured level all veto (same
    # inclusive vocabulary as the lane's net6h_max_pct)
    _stubs(monkeypatch, drift=-1.5, net6h=-4.0, low_age=30)
    assert _reason(stale_mode="live") == STRUCT_BLOCK


def test_age_just_under_the_floor_admits(monkeypatch):
    _stubs(monkeypatch, low_age=25)
    assert _reason(stale_mode="live") == ""


# ── fail-safe: a gap never vetoes; it admits exactly as shipped ────────────

def test_missing_front_clock_admits(monkeypatch):
    _stubs(monkeypatch, net6h=-5.1, low_age=None)  # no low4h_age_min in read
    assert _reason(stale_mode="live") == ""


def test_missing_net6h_admits(monkeypatch):
    _stubs(monkeypatch, net6h=None, low_age=95)
    assert _reason(stale_mode="live") == ""


def test_tape_gap_still_blocks_structure_only(monkeypatch):
    # a data gap may never ADMIT either (the arm's own fail-safe pin)
    monkeypatch.setattr(executor, "_late_chase_tape_read", lambda: None)
    monkeypatch.setattr(executor, "_tape_history_read", lambda: HIST)
    assert _reason(stale_mode="live") == STRUCT_BLOCK


def test_malformed_config_falls_back_to_documented_cell(monkeypatch, caplog):
    # A typo in a LIVE restrictive knob must not silently disarm the veto:
    # the documented cell (-8/-4/30) is used and the WARNING names the key.
    # Here the harness tape (drift -1.8) is shallower than the documented
    # depth leg, so the admit stands — but loudly, and the veto is proven
    # still armed at the documented cell below.
    _stubs(monkeypatch)
    cfg = _config(stale_mode="live")
    cfg["short_tape_stale_depth_gate"] = {"btc24_deep_pct": "deep",
                                          "net6h_max_pct": -4.0,
                                          "low_age_min_min": 30}
    with caplog.at_level("WARNING"):
        assert _runner_entry_block_reason(_analysis(), cfg) == ""
    assert "short_tape_stale_depth_gate" in caplog.text
    assert "absent or malformed" in caplog.text
    # same malformed config, a tape inside the documented cell -> still vetoed
    assert _stale_depth_squeeze({"drift": -9.4, "net6h": -5.1,
                                 "low4h_age_min": 95},
                                cfg["short_tape_stale_depth_gate"]) == (
        -9.4, -5.1, 95.0)


def test_block_absent_uses_documented_cell_with_warning(monkeypatch, caplog):
    # veto explicitly live but the params block is missing: the shipped cell
    # applies (never inert), and the gap is logged, not swallowed
    _stubs(monkeypatch)
    cfg = _config(stale_mode="live")
    del cfg["short_tape_stale_depth_gate"]
    with caplog.at_level("WARNING"):
        assert _runner_entry_block_reason(_analysis(), cfg) == ""
    assert "absent or malformed" in caplog.text
    assert _stale_depth_squeeze({"drift": -9.4, "net6h": -5.1,
                                 "low4h_age_min": 95}, None) == (-9.4, -5.1, 95.0)


# ── blast radius: the general clause stays open (owner ruling) ─────────────

def test_seat_structure_entry_untouched(monkeypatch, caplog):
    # A short the SEAT structured on its own downtrend momentum never
    # reaches the arm branch, so the cell cannot skip it however deep it
    # sits — the general clause stays open by design.
    _stubs(monkeypatch)
    with caplog.at_level("INFO"):
        got = _runner_entry_block_reason(
            _analysis(downtrend_momentum_fired=True), _config(stale_mode="live"))
    assert got == "" and "STALE-DEPTH" not in caplog.text


def test_arm_off_leaves_no_veto_surface(monkeypatch, caplog):
    # arm off -> there is no admit to veto: the trade stays blocked by the
    # structure clause and the veto code never runs
    _stubs(monkeypatch)
    with caplog.at_level("INFO"):
        got = _runner_entry_block_reason(
            _analysis(), _config(arm="off", stale_mode="live"))
    assert got == STRUCT_BLOCK and "STALE-DEPTH" not in caplog.text


# ── the cell as a pure function, at production values ──────────────────────

def test_helper_prod_cell_needs_all_three_legs():
    assert _stale_depth_squeeze(
        {"drift": -9.4, "net6h": -5.1, "low4h_age_min": 95}, PROD_CELL) == (
        -9.4, -5.1, 95.0)          # 2026-11-21 05:45 LIT profile
    assert _stale_depth_squeeze(
        {"drift": -9.4, "net6h": -5.1, "low4h_age_min": 10}, PROD_CELL) is None
    assert _stale_depth_squeeze(
        {"drift": -7.9, "net6h": -5.1, "low4h_age_min": 95}, PROD_CELL) is None
    assert _stale_depth_squeeze(
        {"drift": -9.4, "net6h": -3.2, "low4h_age_min": 95}, PROD_CELL) is None
    assert _stale_depth_squeeze(
        {"drift": -9.4, "net6h": -5.1, "low4h_age_min": 25}, PROD_CELL) is None


def test_helper_inert_on_gaps_and_junk():
    assert _stale_depth_squeeze({"drift": -9.4, "net6h": None,
                                 "low4h_age_min": 95}, PROD_CELL) is None
    assert _stale_depth_squeeze({"drift": -9.4, "net6h": -5.1}, PROD_CELL) is None
    assert _stale_depth_squeeze({"drift": None, "net6h": -5.1,
                                 "low4h_age_min": 95}, PROD_CELL) is None
    assert _stale_depth_squeeze(None, PROD_CELL) is None
    assert _stale_depth_squeeze("tape gap", PROD_CELL) is None


def test_helper_junk_tape_scalars_never_veto_never_raise():
    # the tape read is a sensor: a non-numeric, huge or non-finite scalar is a
    # gap (no opinion). It may not arm a veto and may not raise out of the
    # runner gate (no per-gate try/except upstream).
    junk = ["abc", "None", None, 10 ** 400, float("nan"), float("inf"),
            float("-inf"), "Infinity", ""]
    for val in junk:
        for key in ("drift", "net6h", "low4h_age_min"):
            tape = {"drift": -9.4, "net6h": -5.1, "low4h_age_min": 95}
            tape[key] = val
            assert _stale_depth_squeeze(tape, PROD_CELL) is None, (key, val)


def test_helper_infinity_config_never_misarms_the_cell():
    # `float("Infinity")` parses, and `drift <= inf` is true for ANY tape: an
    # unguarded cast would veto shallow entries the owner never meant to
    # touch. Non-finite config falls back to the documented leg instead.
    for bad in ("Infinity", float("inf"), "-Infinity", float("nan"), "1e999",
                10 ** 400):
        block = {"btc24_deep_pct": bad, "net6h_max_pct": -4.0,
                 "low_age_min_min": 30}
        assert _stale_depth_squeeze({"drift": -1.8, "net6h": -5.1,
                                     "low4h_age_min": 95}, block) is None, bad
        assert _stale_depth_squeeze({"drift": -9.4, "net6h": -5.1,
                                     "low4h_age_min": 95}, block) == (
            -9.4, -5.1, 95.0), bad
    # and the same for the other two legs: an -inf leg ceiling would never
    # veto anything, an inf age floor would veto everything — both fall back
    # to the documented leg, so the knife is still refused
    assert _stale_depth_squeeze({"drift": -9.4, "net6h": -5.1,
                                 "low4h_age_min": 95},
                                {"btc24_deep_pct": -8.0, "net6h_max_pct": -4.0,
                                 "low_age_min_min": float("inf")}) == (
        -9.4, -5.1, 95.0)
    assert _stale_depth_squeeze({"drift": -9.4, "net6h": -5.1,
                                 "low4h_age_min": 95},
                                {"btc24_deep_pct": -8.0,
                                 "net6h_max_pct": float("-inf"),
                                 "low_age_min_min": 30}) == (-9.4, -5.1, 95.0)


# ── the new scalar: one fetch, the LOW series, the 4h window ───────────────

def _tape(monkeypatch, candles):
    from hermes_trader.agents import market_regime as mr
    monkeypatch.setattr(mr, "fetch_hl_candles",
                        lambda coin, interval="5m", count=290: candles)
    monkeypatch.setattr(mr, "_tape_cache", (None, 0.0))
    res = mr.btc_tape_activity(force=True)
    assert res is not None
    return res


def _bars(low_age_bars, n=434):
    """Flat 100.0 closes; the lowest LOW of the trailing 4h sits
    `low_age_bars` bars back from the last bar."""
    return [Candle(t=i * 300_000, o=100.0, h=100.0,
                   l=90.0 if i == n - 1 - low_age_bars else 100.0,
                   c=100.0, v=1.0) for i in range(n)]


def test_low_age_zero_when_printing_a_new_local_low(monkeypatch):
    assert _tape(monkeypatch, _bars(0))["low4h_age_min"] == 0.0


def test_low_age_counts_bars_since_the_local_low(monkeypatch):
    assert _tape(monkeypatch, _bars(12))["low4h_age_min"] == 60.0
    assert _tape(monkeypatch, _bars(19))["low4h_age_min"] == 95.0


def test_low_age_window_is_four_hours_not_the_whole_fetch(monkeypatch):
    # the only sub-100 low sits 60 bars back: outside the 4h window, so the
    # reading is bounded by the window (max 235 min), never 300. Pinned
    # EXACTLY at both ends of the window: a quietly-shortened window (say 40
    # bars, max 195 min) must fail here, not pass a one-sided bound.
    assert _tape(monkeypatch, _bars(47))["low4h_age_min"] == 235.0
    assert _tape(monkeypatch, _bars(60))["low4h_age_min"] == 235.0
    assert _tape(monkeypatch, _bars(300))["low4h_age_min"] == 235.0
    assert _tape(monkeypatch, _bars(48))["low4h_age_min"] == 235.0


def test_low_age_survives_a_zero_close_hole(monkeypatch):
    # bars with c == 0 are dropped by the tape read; if that breaks the
    # 48-bar window the scalar degrades to None (no opinion), not a signal
    bars = _bars(3)
    bars[-5].c = 0.0
    assert _tape(monkeypatch, bars)["low4h_age_min"] is None


def test_low_age_present_when_the_12h_trail_is_too_short(monkeypatch):
    # 290 bars feed the 4h low window even though drift_min12 degrades to
    # None — the veto loses the leg it needs, never the whole read
    tape = _tape(monkeypatch, [Candle(t=i * 300_000, o=100.0, h=100.0, l=100.0,
                                      c=100.0, v=1.0) for i in range(290)])
    assert tape["drift_min12"] is None
    assert tape["low4h_age_min"] is not None


def test_unreadable_low_column_costs_only_the_new_key(monkeypatch):
    # a candle source without a LOW column must not take the tape read down
    # with it: vol/drift/net6h stay exactly as shipped
    class _NoLow:
        def __init__(self, c):
            self.c = c

    tape = _tape(monkeypatch, [_NoLow(100.0 + (i % 3)) for i in range(290)])
    assert tape["low4h_age_min"] is None
    assert tape["drift"] == pytest.approx(1.0)
    assert tape["vol"] > 0 and tape["net6h"] == pytest.approx(0.0)
