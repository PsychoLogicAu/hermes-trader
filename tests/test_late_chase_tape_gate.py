"""BTC-loudness meta-gate on the late-chase bypass (`late_chase_tape_gate`).

Spec: .hermes/specs/late-chase-btc-tape-gate.md (2026-09-13). Evidence: the
shadow bypass cohort's P/L tracks BTC's tape — BTC trail24h vol < 1.5% at
entry netted +$109 (n=81) vs −$80 for vol ≥ 1.5 (n=70); signed drift < −1%
was −$25 vs +$30 above +1%. The gate therefore allows the bypass only when

    vol < late_chase_tape_vol_max_pct AND drift > late_chase_tape_drift_min_pct

consumed from the SAME cached btc_tape_activity() read quiet_tape uses (raw
numbers — opposite polarity to that gate's verdict). Deny suppresses even a
LIVE bypass; data gap fails safe to DENY; shadow mode never changes the
outcome, only annotates the accrual line with TAPE=<verdict>. Master flag
off ⇒ no tape fetch at all, behavior byte-identical to pre-feature.
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
    sig = types.SimpleNamespace(median_pct=0.2, error=None)
    monkeypatch.setattr(executor, "get_chronos_signal_sync", lambda c, s: sig)


def _tape(monkeypatch, vol=None, drift=None, raise_exc=False):
    calls = []

    def fake():
        calls.append(1)
        if raise_exc:
            raise RuntimeError("boom")
        if vol is None:
            return None
        return {"vol": vol, "drift": drift}
    monkeypatch.setattr(executor, "_late_chase_tape_read", fake)
    return calls


# ── master flag off: byte-identical, zero fetches ─────────────────────────────

def test_off_by_default_no_fetch_and_bypass_still_releases(monkeypatch):
    _chronos_aligned(monkeypatch)
    calls = _tape(monkeypatch, vol=9.9, drift=-9.9)  # would deny if consulted
    assert executor._runner_entry_block_reason(_analysis(conf=0.82), _gate()) == ""
    assert calls == []


def test_off_explicit_false_no_fetch(monkeypatch):
    _chronos_aligned(monkeypatch)
    calls = _tape(monkeypatch, vol=9.9, drift=-9.9)
    g = _gate(late_chase_tape_gate=False)
    assert executor._runner_entry_block_reason(_analysis(conf=0.82), g) == ""
    assert calls == []


# ── on: allow / deny band ─────────────────────────────────────────────────────

def test_calm_tape_allows_bypass(monkeypatch):
    _chronos_aligned(monkeypatch)
    _tape(monkeypatch, vol=1.2, drift=-0.4)
    g = _gate(late_chase_tape_gate=True)
    assert executor._runner_entry_block_reason(_analysis(conf=0.82), g) == ""


def test_loud_tape_denies_bypass(monkeypatch):
    _chronos_aligned(monkeypatch)
    _tape(monkeypatch, vol=2.6, drift=0.1)
    g = _gate(late_chase_tape_gate=True)
    reason = executor._runner_entry_block_reason(_analysis(conf=0.82), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")


def test_falling_btc_denies_even_at_low_vol(monkeypatch):
    _chronos_aligned(monkeypatch)
    _tape(monkeypatch, vol=1.2, drift=-1.5)  # calm but falling hard
    g = _gate(late_chase_tape_gate=True)
    assert executor._runner_entry_block_reason(
        _analysis(conf=0.82), g).startswith("runner_gate_blocked")


def test_rising_btc_at_boundary_drift_denies_strict(monkeypatch):
    # drift exactly at the boundary is NOT > min → deny (strict inequality).
    _chronos_aligned(monkeypatch)
    _tape(monkeypatch, vol=1.2, drift=-1.0)
    g = _gate(late_chase_tape_gate=True)
    assert executor._runner_entry_block_reason(
        _analysis(conf=0.82), g).startswith("runner_gate_blocked")


def test_custom_thresholds_respected(monkeypatch):
    _chronos_aligned(monkeypatch)
    _tape(monkeypatch, vol=2.6, drift=0.1)
    g = _gate(late_chase_tape_gate=True, late_chase_tape_vol_max_pct=3.0)
    assert executor._runner_entry_block_reason(_analysis(conf=0.82), g) == ""


def test_data_gap_fails_safe_to_deny(monkeypatch):
    _chronos_aligned(monkeypatch)
    _tape(monkeypatch, vol=None)  # btc_tape_activity() → None
    g = _gate(late_chase_tape_gate=True)
    assert executor._runner_entry_block_reason(
        _analysis(conf=0.82), g).startswith("runner_gate_blocked")


def test_tape_read_exception_fails_safe_to_deny(monkeypatch):
    # Exception INSIDE btc_tape_activity must be swallowed by the indirection
    # wrapper (→ None → gap-deny), never propagate out of the gate.
    _chronos_aligned(monkeypatch)
    import hermes_trader.agents.market_regime as mr
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(mr, "btc_tape_activity", boom)
    g = _gate(late_chase_tape_gate=True)
    assert executor._runner_entry_block_reason(
        _analysis(conf=0.82), g).startswith("runner_gate_blocked")


# ── only consulted at bypass candidates ───────────────────────────────────────

def test_not_consulted_below_bar(monkeypatch):
    # conf 0.78 < dynamic bar 0.80: never a bypass candidate → no tape fetch,
    # block reason identical to the pre-feature text (parsers unaffected).
    _chronos_aligned(monkeypatch)
    calls = _tape(monkeypatch, vol=1.2, drift=0.0)
    g = _gate(late_chase_tape_gate=True)
    reason = executor._runner_entry_block_reason(_analysis(conf=0.78), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert calls == []


# ── shadow mode: outcome never changes, accrual carries the verdict ───────────

def test_shadow_loud_tape_still_blocks_with_tape_note(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    _tape(monkeypatch, vol=2.6, drift=0.1)
    g = _gate(bypass_late_trend_chase=False,
              bypass_late_trend_chase_shadow_mode=True,
              late_chase_tape_gate=True)
    reason = executor._runner_entry_block_reason(_analysis(conf=0.82), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    shadow = [r for r in caplog.records if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert len(shadow) == 1
    assert "TAPE=deny(vol 2.60%, drift +0.10%," in shadow[0].getMessage()
    # no live DENIED line — nothing was being bypassed to deny
    assert not [r for r in caplog.records if "[gate][TAPE]" in r.getMessage()]


def test_shadow_calm_tape_note_allows_line(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    _tape(monkeypatch, vol=1.2, drift=-0.4)
    g = _gate(bypass_late_trend_chase=False,
              bypass_late_trend_chase_shadow_mode=True,
              late_chase_tape_gate=True)
    reason = executor._runner_entry_block_reason(_analysis(conf=0.82), g)
    assert reason.startswith("runner_gate_blocked")  # shadow: still blocked
    shadow = [r for r in caplog.records if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert len(shadow) == 1
    assert "TAPE=allow(vol 1.20%, drift -0.40%)" in shadow[0].getMessage()


def test_live_deny_emits_gate_tape_log_line(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    _tape(monkeypatch, vol=2.6, drift=-1.5)
    g = _gate(late_chase_tape_gate=True)
    assert executor._runner_entry_block_reason(
        _analysis(conf=0.82), g).startswith("runner_gate_blocked")
    denied = [r for r in caplog.records if "[gate][TAPE]" in r.getMessage()]
    assert len(denied) == 1
    msg = denied[0].getMessage()
    assert "late_chase_bypass DENIED for SKR LONG" in msg
    assert "vol 2.60%, drift -1.50%" in msg
