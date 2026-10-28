"""B.53 (t_5384f2ce): mild-down-tape LONG veto.

Blocks a LONG while BTC trail-24h drift sits in the mild-down band
[btc24_min_pct, btc24_max_pct) — the mirror of the state-admission SHORT
tape-keys band. Evidence (SUBFLOOR-short cohort dissection, 2026-10-20..25
shadow window): on that tape the mirror long classes were flat-negative —
ceiling-skip -37.92 (n=42, 14 max_loss) and sub-floor rescue -26.10
(n=43, 16 max_loss) — while clause shorts banked (occupancy-real +29.48
band (-2,0)). Longs were negative in EVERY non-crash band of that 5-day
window, so the band edges are provisional: promotion bar on the board is
n>=50 in-band would-blocks + blocked set negative BN1+BN2 + >=3 distinct
regime episodes. Accrual-first by design.

Modes: off (default, inert) / shadow (pass + would-block marker) / live
(block). LONG only; categorical on side+tape (no conf exemption — conf
rode the extension that lost money). Covers BOTH call sites: the classic
gate (risk_gates.mild_tape_long_veto_gate via eval_all_gates) and the
runner-gate twin (executor._runner_entry_block_reason), which MUST fire
before the structured_daily_mover bypass and the sub-floor state-clause
early-returns. Fail-safe everywhere: mode off = zero tape fetches; tape
gap / unset / malformed keys = inert (a gap never fires a protective veto).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import executor, market_regime  # noqa: E402
from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    eval_all_gates,
    mild_tape_long_veto_gate,
)

CFG_SHADOW = {"mode": "shadow", "btc24_min_pct": -2.0, "btc24_max_pct": 0.0}
CFG_LIVE = {"mode": "live", "btc24_min_pct": -2.0, "btc24_max_pct": 0.0}


def _ctx(trade_side: str = "long", confidence: float = 0.70) -> GateContext:
    return GateContext(
        confidence=confidence,
        current_positions=[],
        trade_notional_usd=10.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin="TEST",
        trade_side=trade_side,
        has_binary_news_risk=False,
        equity=100.0,
        total_open_notional=0.0,
        composite_score=40.0,
    )


def _stub_classic_tape(monkeypatch, drift):
    monkeypatch.setattr(market_regime, "btc_tape_activity",
                        lambda *a, **k: {"vol": 1.2, "drift": drift})


# ── classic gate: fail-safes ────────────────────────────────────────────────

def test_default_off_is_inert():
    r = mild_tape_long_veto_gate(_ctx(), {})
    assert r["pass"] and "disabled" in r["reason"]
    assert "shadow_would_block" not in r


def test_unknown_mode_is_inert():
    r = mild_tape_long_veto_gate(_ctx(), {"mode": "banana",
                                          "btc24_min_pct": -2.0,
                                          "btc24_max_pct": 0.0})
    assert r["pass"] and "disabled" in r["reason"]


def test_short_never_vetoed():
    r = mild_tape_long_veto_gate(_ctx(trade_side="short"), CFG_LIVE)
    assert r["pass"] and "not long" in r["reason"]


def test_unset_tape_keys_inert(monkeypatch):
    _stub_classic_tape(monkeypatch, -1.0)  # in-band drift; keys must still gate
    r = mild_tape_long_veto_gate(_ctx(), {"mode": "live"})
    assert r["pass"] and "tape keys unset" in r["reason"]


def test_malformed_tape_keys_inert(monkeypatch):
    _stub_classic_tape(monkeypatch, -1.0)
    r = mild_tape_long_veto_gate(
        _ctx(), {"mode": "live", "btc24_min_pct": "minus two",
                 "btc24_max_pct": 0.0})
    assert r["pass"] and "malformed" in r["reason"]


def test_tape_gap_inert(monkeypatch):
    _stub_classic_tape(monkeypatch, None)
    r = mild_tape_long_veto_gate(_ctx(), CFG_LIVE)
    assert r["pass"] and "no BTC drift read" in r["reason"]


def test_drift_below_band_passes_live(monkeypatch):
    # deep-down rescue tape (below the floor): the veto must not touch it —
    # that is the state clause's long-rescue territory (disjoint band).
    _stub_classic_tape(monkeypatch, -2.5)
    r = mild_tape_long_veto_gate(_ctx(), CFG_LIVE)
    assert r["pass"] and "outside" in r["reason"]


def test_drift_at_ceiling_passes_live(monkeypatch):
    # half-open [min, max): drift == ceiling is OUT (up-tape not vetoed).
    _stub_classic_tape(monkeypatch, 0.0)
    r = mild_tape_long_veto_gate(_ctx(), CFG_LIVE)
    assert r["pass"] and "outside" in r["reason"]


def test_drift_at_floor_blocks_live(monkeypatch):
    # drift == floor is IN the band.
    _stub_classic_tape(monkeypatch, -2.0)
    r = mild_tape_long_veto_gate(_ctx(), CFG_LIVE)
    assert r["pass"] is False and "mild_tape_long" in r["reason"]


# ── classic gate: shadow / live ─────────────────────────────────────────────

def test_shadow_passes_with_marker(monkeypatch):
    _stub_classic_tape(monkeypatch, -1.0)
    r = mild_tape_long_veto_gate(_ctx(), CFG_SHADOW)
    assert r["pass"] and r.get("shadow_would_block") is True
    assert "mild_tape_long" in r["reason"]


def test_live_blocks_long_on_mild_tape(monkeypatch):
    _stub_classic_tape(monkeypatch, -1.0)
    r = mild_tape_long_veto_gate(_ctx(), CFG_LIVE)
    assert r["pass"] is False and "mild_tape_long" in r["reason"]


def test_registered_in_eval_all_gates_and_blocks(monkeypatch):
    _stub_classic_tape(monkeypatch, -1.0)
    out = eval_all_gates(
        _ctx(), {"mild_tape_long_veto_gate": dict(CFG_LIVE)})
    assert out["results"]["mild_tape_long"]["pass"] is False
    assert out["blocked"] is True
    assert any("mild_tape_long" in r for r in out["block_reasons"])


def test_eval_all_gates_default_off_inert():
    out = eval_all_gates(_ctx(), {})
    r = out["results"]["mild_tape_long"]
    assert r == {"pass": True, "reason": "mild_tape_long: disabled"}
    assert not any("mild_tape_long" in r2 for r2 in out["block_reasons"])


# ── runner-gate twin (executor) ─────────────────────────────────────────────
from hermes_trader.agents.executor import _runner_entry_block_reason  # noqa: E402

def _analysis(**over) -> dict:
    a = {
        "coin": "TEST", "side": "long", "confidence": 0.85,
        "composite_score": 50.0, "daily_mover_fired": False,
        "volume_spike_fired": False, "breakout_fired": False,
        "momentum_burst_fired": False, "uptrend_momentum_fired": False,
        "downtrend_momentum_fired": False, "slow_burn_count": 0,
        "whale_signal": False, "reasoning": "",
    }
    a.update(over)
    return a


def _config(mode):
    cfg = {"runner_entry_gate": {"enabled": True, "min_confidence": 0.65,
                                 "min_composite": 30.0}}
    if mode is not None:
        cfg["mild_tape_long_veto_gate"] = {"mode": mode,
                                           "btc24_min_pct": -2.0,
                                           "btc24_max_pct": 0.0}
    return cfg


def _stub_runner_tape(monkeypatch, drift):
    def _read():
        calls.append(1)
        return None if drift is None else {"vol": 1.2, "drift": drift}
    calls = []
    monkeypatch.setattr(executor, "_late_chase_tape_read", _read)
    return calls


def test_runner_live_vetoes_before_bypass(monkeypatch):
    # mover bypass population (conf 0.85 + score 50 + mover): the mild-tape
    # veto sits above it too — the tape, not the mover flag, decides here.
    _stub_runner_tape(monkeypatch, -1.0)
    a = _analysis(daily_mover_fired=True, volume_spike_fired=True)
    assert _runner_entry_block_reason(
        a, _config("live")) == "runner_gate_blocked (mild-down tape long veto)"


def test_runner_shadow_logs_and_admits(monkeypatch, caplog):
    _stub_runner_tape(monkeypatch, -1.5)
    a = _analysis(volume_spike_fired=True, breakout_fired=True,
                  slow_burn_count=1)
    import logging as _lg
    with caplog.at_level(_lg.WARNING):
        assert _runner_entry_block_reason(a, _config("shadow")) == ""
    assert "[gate][MILD-TAPE-LONG][SHADOW] WOULD VETO" in caplog.text


def test_runner_off_zero_fetch_byte_identical(monkeypatch):
    calls = _stub_runner_tape(monkeypatch, -1.0)
    a = _analysis(volume_spike_fired=True, breakout_fired=True,
                  slow_burn_count=1)  # fresh-impulse admit path
    assert _runner_entry_block_reason(a, _config("off")) == ""
    assert calls == []          # disabled path makes ZERO tape fetches
    assert _runner_entry_block_reason(a, _config(None)) == ""
    assert calls == []          # absent block = byte-identical


def test_runner_tape_gap_inert(monkeypatch):
    _stub_runner_tape(monkeypatch, None)
    a = _analysis(volume_spike_fired=True, breakout_fired=True,
                  slow_burn_count=1)
    assert _runner_entry_block_reason(a, _config("live")) == ""


def test_runner_out_of_band_admits(monkeypatch):
    _stub_runner_tape(monkeypatch, -2.5)  # below floor = rescue tape
    a = _analysis(volume_spike_fired=True, breakout_fired=True,
                  slow_burn_count=1)
    assert _runner_entry_block_reason(a, _config("live")) == ""
    _stub_runner_tape(monkeypatch, +0.5)  # above ceiling = up-tape
    assert _runner_entry_block_reason(a, _config("live")) == ""


def test_runner_malformed_keys_inert(monkeypatch):
    _stub_runner_tape(monkeypatch, -1.0)
    cfg = _config("live")
    cfg["mild_tape_long_veto_gate"]["btc24_min_pct"] = "three"
    a = _analysis(volume_spike_fired=True, breakout_fired=True,
                  slow_burn_count=1)
    assert _runner_entry_block_reason(a, cfg) == ""


def test_runner_short_path_never_reaches_veto(monkeypatch):
    calls = _stub_runner_tape(monkeypatch, -1.0)
    a = _analysis(side="short", confidence=0.85,
                  downtrend_momentum_fired=True)
    cfg = _config("live")
    cfg["runner_entry_gate"]["allow_shorts"] = True
    assert _runner_entry_block_reason(a, cfg) == ""
    assert calls == []  # short path returns before the long section
