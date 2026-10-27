"""Fired-trigger evidence floor on the late-chase bypass
(`bypass_late_chase_require_trigger`, 2026-10-27).

B.49 owner ruling ("both live"): the bypass-cohort survey (t_54d26fee,
2026-09-18→10-26, n=85 net −$13.49) splits cleanly on trigger evidence —
admits with ≥1 fired entry trigger net −$0.23 (n=31); the zero-trigger
admits carry the entire loss (−$13.26, n=54). The knob denies the bypass to
candidates with NO fired trigger (volume_spike / breakout / momentum_burst /
daily_mover / whale); slow-burn does NOT count (structure, not a trigger).
Deny suppresses even a LIVE bypass (standard late-chase block reason then
stands, byte-identical for parsers) and logs a loud [gate][TRIGGER] DENIED
line; shadow accruals carry TRIGGER=allow|deny. Knob off/absent ⇒
byte-identical to pre-feature.
"""
import types

from hermes_trader.agents import executor


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
    # spread_pct clears the vote deadband (0.2/0.5 = 0.4 >= 0.25).
    sig = types.SimpleNamespace(median_pct=0.2, spread_pct=0.5, error=None)
    monkeypatch.setattr(executor, "get_chronos_signal_sync", lambda c, s: sig)


# ── knob off: byte-identical — zero-trigger bypass still releases ─────────────

def test_off_by_default_zero_trigger_bypass_still_releases(monkeypatch):
    _chronos_aligned(monkeypatch)
    assert executor._runner_entry_block_reason(_analysis(), _gate()) == ""


def test_off_explicit_false_zero_trigger_bypass_still_releases(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=False)
    assert executor._runner_entry_block_reason(_analysis(), g) == ""


# ── knob on: trigger evidence allows ──────────────────────────────────────────

def test_volume_spike_counts_as_trigger(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=True)
    assert executor._runner_entry_block_reason(
        _analysis(volume_spike_fired=True), g) == ""


def test_whale_counts_as_trigger(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=True)
    assert executor._runner_entry_block_reason(
        _analysis(whale_signal=True), g) == ""


def test_mover_counts_as_trigger(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=True)
    assert executor._runner_entry_block_reason(
        _analysis(daily_mover_fired=True), g) == ""


def test_any_single_trigger_suffices(monkeypatch):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=True)
    assert executor._runner_entry_block_reason(
        _analysis(breakout_fired=True), g) == ""


# ── knob on: zero triggers denied ─────────────────────────────────────────────

def test_zero_triggers_denied(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=True)
    reason = executor._runner_entry_block_reason(_analysis(), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    denied = [r for r in caplog.records if "[gate][TRIGGER]" in r.getMessage()]
    assert len(denied) == 1
    msg = denied[0].getMessage()
    assert "late_chase_bypass DENIED for SKR LONG" in msg
    assert "conf 0.82 >= bar 0.80" in msg
    assert "NO trigger fired" in msg


def test_slow_burn_alone_is_not_a_trigger(monkeypatch):
    # slow_burn is structure, not a fired trigger — the survey's
    # triggered/untriggered split keys on the five fire flags.
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=True)
    assert executor._runner_entry_block_reason(
        _analysis(slow_burn_count=2), g).startswith("runner_gate_blocked")


def test_missing_flags_fail_closed_to_deny(monkeypatch):
    # Deliberate fail-safe: absent flags read as not fired = deny side.
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=True)
    assert executor._runner_entry_block_reason(
        _analysis(), g).startswith("runner_gate_blocked")


# ── only consulted at bypass candidates ───────────────────────────────────────

def test_not_consulted_below_bar(monkeypatch, caplog):
    # conf 0.78 < dynamic bar 0.80: never a bypass candidate → the knob must
    # not even log; the block reason is the pre-feature text.
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=True)
    reason = executor._runner_entry_block_reason(_analysis(conf=0.78), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert not [r for r in caplog.records if "[gate][TRIGGER]" in r.getMessage()]


# ── shadow mode: outcome never changes, accrual carries the verdict ───────────

def test_shadow_zero_triggers_annotated_deny(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_trend_chase=False,
              bypass_late_trend_chase_shadow_mode=True,
              bypass_late_chase_require_trigger=True)
    reason = executor._runner_entry_block_reason(_analysis(), g)
    assert reason.startswith("runner_gate_blocked")  # shadow: still blocked
    shadow = [r for r in caplog.records if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert len(shadow) == 1
    assert "TRIGGER=deny(none fired)" in shadow[0].getMessage()
    # no live DENIED line — nothing was being bypassed to deny
    assert not [r for r in caplog.records if "[gate][TRIGGER]" in r.getMessage()]


def test_shadow_with_trigger_annotated_allow(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_trend_chase=False,
              bypass_late_trend_chase_shadow_mode=True,
              bypass_late_chase_require_trigger=True)
    reason = executor._runner_entry_block_reason(
        _analysis(volume_spike_fired=True), g)
    assert reason.startswith("runner_gate_blocked")  # shadow: still blocked
    shadow = [r for r in caplog.records if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert len(shadow) == 1
    assert "TRIGGER=allow(volspk)" in shadow[0].getMessage()


def test_bypass_log_line_carries_trigger_allow_note(monkeypatch, caplog):
    import logging
    # The bypass line is INFO — caplog defaults to WARNING, so raise the
    # executor logger's capture level explicitly.
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_trigger=True)
    # NOTE: volume+breakout together make fresh_impulse=True — the candidate
    # then admits via the structured-runner path and never reaches the
    # bypass branch. Use a trigger that doesn't feed fresh_impulse (mover).
    assert executor._runner_entry_block_reason(
        _analysis(daily_mover_fired=True), g) == ""
    bypassed = [r for r in caplog.records if "late-trend chase bypassed on SKR" in r.getMessage()]
    assert len(bypassed) == 1
    assert "TRIGGER=allow(mover)" in bypassed[0].getMessage()
