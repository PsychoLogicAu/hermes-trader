"""Sustained-fuel requirement on the late-chase bypass
(`bypass_late_chase_require_fuel`, 2026-10-29).

B.23 rerun owner ruling (t_54d26fee, "this class of trade should be shut
off" after SPX 10-29): the bypass cohort (n=94, 09-18→10-29, net −$12.27)
splits on ENTRY FUEL — daily_mover-backed admits net +$5.51 (n=30, 10%
knife) vs −$17.78 (n=64, 22% knife, the entire October bleed); the split
holds inside the mom24h 10-20% band (+3.05 vs −13.89). The require-trigger
floor (B.49) does NOT catch the class: GRASS 10-05 / INJ 10-21 / SPX 10-29
all passed it on a fired breakout (2 knives / 3). The knob denies the bypass
unless a trigger in `bypass_late_chase_fuel_triggers` (default ["mover"])
fires; deny suppresses even a LIVE bypass (standard late-chase block reason
stands, byte-identical for parsers) with a loud [gate][FUEL] DENIED line;
shadow accruals carry FUEL=allow|deny. Knob off/absent ⇒ byte-identical.
"""
import logging

from hermes_trader.agents import executor

import types


def _gate(**over):
    g = {
        "enabled": True,
        "min_confidence": 0.70,
        "min_composite": 30.0,
        "min_hip3_composite": 50.0,
        "bypass_late_trend_chase": True,
        "bypass_late_trend_chase_min_conf": 0.90,
        "late_chase_dynamic_per_signal_drop": 0.10,
        "late_chase_timesfm_vote": False,
        "late_chase_timesfm_drop": 0.0,
    }
    g.update(over)
    return {"runner_entry_gate": g}


def _analysis(conf=0.82, **over):
    a = {
        "coin": "SKR",
        "side": "long",
        "confidence": conf,
        "composite_score": 0.0,
        "uptrend_momentum_fired": True,
    }
    a.update(over)
    return a


def _chronos_aligned(monkeypatch):
    sig = types.SimpleNamespace(median_pct=0.2, spread_pct=0.5, error=None)
    monkeypatch.setattr(executor, "get_chronos_signal_sync", lambda c, s: sig)


# ── knob off: byte-identical — breakout-only bypass still releases ────────────

def test_off_by_default_breakout_only_bypass_still_releases(monkeypatch):
    _chronos_aligned(monkeypatch)
    assert executor._runner_entry_block_reason(
        _analysis(breakout_fired=True), _gate()) == ""


def test_off_explicit_false_breakout_only_bypass_still_releases(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=False)
    assert executor._runner_entry_block_reason(
        _analysis(breakout_fired=True), g) == ""


# ── knob on: mover fuel allows ────────────────────────────────────────────────

def test_mover_fuel_allows_bypass(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True)
    assert executor._runner_entry_block_reason(
        _analysis(daily_mover_fired=True), g) == ""


def test_mover_plus_breakout_allows(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True)
    assert executor._runner_entry_block_reason(
        _analysis(daily_mover_fired=True, breakout_fired=True), g) == ""


# ── knob on: the shut-off class — blip-only admits denied ─────────────────────

def test_breakout_only_denied(monkeypatch, caplog):
    # SPX 10-29 shape: conf over bar, fresh breakout, no sustained fuel.
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True)
    reason = executor._runner_entry_block_reason(_analysis(breakout_fired=True), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    denied = [r for r in caplog.records if "[gate][FUEL]" in r.getMessage()]
    assert len(denied) == 1
    msg = denied[0].getMessage()
    assert "late_chase_bypass DENIED for SKR LONG" in msg
    assert "conf 0.82 >= bar 0.80" in msg
    assert "no FUEL trigger fired" in msg
    assert "fired=breakout" in msg


def test_volspk_only_denied_on_default_fuel(monkeypatch):
    # Default fuel set is ["mover"]; the cohort's vol-only cell (n=7,
    # −$1.45) is on the deny side until widened by config.
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True)
    assert executor._runner_entry_block_reason(
        _analysis(volume_spike_fired=True), g).startswith("runner_gate_blocked")


def test_zero_triggers_denied(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True)
    assert executor._runner_entry_block_reason(_analysis(), g).startswith(
        "runner_gate_blocked")


def test_missing_flags_fail_closed_to_deny(monkeypatch):
    # Deliberate fail-safe: absent flags read as not fired = deny side.
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True)
    assert executor._runner_entry_block_reason(
        _analysis(volume_spike_fired=None), g).startswith("runner_gate_blocked")


# ── fuel set is configurable ──────────────────────────────────────────────────

def test_widened_fuel_set_allows_volspk(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True,
              bypass_late_chase_fuel_triggers=["mover", "volspk"])
    assert executor._runner_entry_block_reason(
        _analysis(volume_spike_fired=True), g) == ""


def test_burst_fuel_only_when_configured(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True)
    assert executor._runner_entry_block_reason(
        _analysis(momentum_burst_fired=True), g).startswith("runner_gate_blocked")
    g2 = _gate(bypass_late_chase_require_fuel=True,
               bypass_late_chase_fuel_triggers=["mover", "burst"])
    assert executor._runner_entry_block_reason(
        _analysis(momentum_burst_fired=True), g2) == ""


# ── coexists with the stricter-than-trigger doctrine (B.49 gate stays on) ─────

def test_both_gates_on_breakout_only_fuel_denies(monkeypatch, caplog):
    # require_trigger ALLOWS the breakout blip; require_fuel then denies it.
    # Exactly one DENIED line, and it's the FUEL one.
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=True,
              bypass_late_chase_require_fuel=True)
    reason = executor._runner_entry_block_reason(_analysis(breakout_fired=True), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert not [r for r in caplog.records if "[gate][TRIGGER]" in r.getMessage()]
    assert len([r for r in caplog.records if "[gate][FUEL]" in r.getMessage()]) == 1


# ── only consulted at bypass candidates ───────────────────────────────────────

def test_not_consulted_below_bar(monkeypatch, caplog):
    # conf 0.78 < dynamic bar 0.80: never a bypass candidate → the knob must
    # not even log; the block reason is the pre-feature text.
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True)
    reason = executor._runner_entry_block_reason(_analysis(conf=0.78), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert not [r for r in caplog.records if "[gate][FUEL]" in r.getMessage()]


# ── shadow mode: outcome never changes, accrual carries the verdict ───────────

def test_shadow_no_fuel_annotated_deny(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_trend_chase=False,
              bypass_late_trend_chase_shadow_mode=True,
              bypass_late_chase_require_fuel=True)
    reason = executor._runner_entry_block_reason(_analysis(breakout_fired=True), g)
    assert reason.startswith("runner_gate_blocked")  # shadow: still blocked
    shadow = [r for r in caplog.records if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert len(shadow) == 1
    assert "FUEL=deny(fired=breakout, fuel-required=mover)" in shadow[0].getMessage()
    # no live DENIED line — nothing was being bypassed to deny
    assert not [r for r in caplog.records if "[gate][FUEL]" in r.getMessage()]


def test_shadow_with_mover_fuel_annotated_allow(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_trend_chase=False,
              bypass_late_trend_chase_shadow_mode=True,
              bypass_late_chase_require_fuel=True)
    reason = executor._runner_entry_block_reason(
        _analysis(daily_mover_fired=True), g)
    assert reason.startswith("runner_gate_blocked")  # shadow: still blocked
    shadow = [r for r in caplog.records if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert len(shadow) == 1
    assert "FUEL=allow(mover)" in shadow[0].getMessage()


def test_bypass_log_line_carries_fuel_allow_note(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True)
    assert executor._runner_entry_block_reason(
        _analysis(daily_mover_fired=True), g) == ""
    bypassed = [r for r in caplog.records
                if "late-trend chase bypassed on SKR" in r.getMessage()]
    assert len(bypassed) == 1
    assert "FUEL=allow(mover)" in bypassed[0].getMessage()
