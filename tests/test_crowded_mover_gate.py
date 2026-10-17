"""t_1999904f: crowded-mover long veto.

Blocks a LONG entered on a fired CROWDED dailyMover flag (24h move past the
configured bar AND volume >= $3M). Evidence: d2 seat window 2026-10-14..17 —
blocked mover longs n=158, -$2.08/trade, 28 max_loss; executed mover longs
(the structured_daily_mover bypass path) n=29, -$2.64, 4 max_loss rescued
only by the 10-16 up-tape. Discriminator is CROWDEDNESS (the flag), never a
raw dmove threshold: >10% THIN movers are the best cell (n=32, 91% WR).

Modes: off (default, inert) / shadow (pass + would-block marker) / live
(block). LONG only; categorical (no conf exemption — conf is the poison's
carrier). Covers BOTH call sites: the classic gate (risk_gates.
crowded_mover_gate via eval_all_gates) and the runner-gate twin
(executor._runner_entry_block_reason), which MUST fire before the
structured_daily_mover bypass it overrides.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    crowded_mover_gate,
    eval_all_gates,
)

CFG_SHADOW = {"mode": "shadow"}
CFG_LIVE = {"mode": "live"}


def _ctx(trade_side: str = "long", mover: bool = True,
         confidence: float = 0.85) -> GateContext:
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
        composite_score=50.0,
        daily_mover_fired=mover,
    )


# ── classic gate: fail-safes ────────────────────────────────────────────────

def test_default_off_is_inert():
    r = crowded_mover_gate(_ctx(), {})
    assert r["pass"] and "disabled" in r["reason"]
    assert "shadow_would_block" not in r


def test_unknown_mode_is_inert():
    r = crowded_mover_gate(_ctx(), {"mode": "banana"})
    assert r["pass"] and "disabled" in r["reason"]


def test_short_with_mover_passes_live():
    r = crowded_mover_gate(_ctx(trade_side="short"), CFG_LIVE)
    assert r["pass"] and "not long" in r["reason"]


def test_long_without_mover_passes_live():
    r = crowded_mover_gate(_ctx(mover=False), CFG_LIVE)
    assert r["pass"] and "mover not fired" in r["reason"]


# ── classic gate: shadow / live ─────────────────────────────────────────────

def test_shadow_passes_with_marker():
    r = crowded_mover_gate(_ctx(), CFG_SHADOW)
    assert r["pass"] and r.get("shadow_would_block") is True
    assert "crowded_mover" in r["reason"]


def test_live_blocks_long_on_mover():
    r = crowded_mover_gate(_ctx(), CFG_LIVE)
    assert r["pass"] is False
    assert "crowded_mover" in r["reason"]


def test_registered_in_eval_all_gates_and_blocks():
    out = eval_all_gates(
        _ctx(), {"crowded_mover_gate": {"mode": "live"}},
    )
    assert out["results"]["crowded_mover"]["pass"] is False
    assert out["blocked"] is True
    assert any("crowded_mover" in r for r in out["block_reasons"])


def test_eval_all_gates_default_off_inert():
    out = eval_all_gates(_ctx(), {})
    r = out["results"]["crowded_mover"]
    assert r == {"pass": True, "reason": "crowded_mover: disabled"}
    assert not any("crowded_mover" in r2 for r2 in out["block_reasons"])


# ── runner-gate twin (executor) ─────────────────────────────────────────────

from hermes_trader.agents.executor import _runner_entry_block_reason  # noqa: E402


def _analysis(**over) -> dict:
    a = {
        "coin": "TEST", "side": "long", "confidence": 0.85,
        "composite_score": 50.0, "daily_mover_fired": True,
        "volume_spike_fired": False, "breakout_fired": False,
        "momentum_burst_fired": False, "uptrend_momentum_fired": False,
        "downtrend_momentum_fired": False, "slow_burn_count": 0,
        "whale_signal": False, "reasoning": "",
    }
    a.update(over)
    return a


def _config(mode: str) -> dict:
    return {
        "runner_entry_gate": {"enabled": True, "min_confidence": 0.65,
                              "min_composite": 30.0},
        "crowded_mover_gate": {"mode": mode},
    }


def test_runner_gate_live_vetoes_mover_long():
    # conf 0.85 + score 50 = exactly the structured_daily_mover bypass
    # population; the veto must fire BEFORE the bypass admits it.
    assert _runner_entry_block_reason(
        _analysis(), _config("live")
    ) == "runner_gate_blocked (crowded daily mover veto)"


def test_runner_gate_shadow_does_not_block():
    # fresh impulse so the fall-through path passes: shadow must only log.
    a = _analysis(volume_spike_fired=True, breakout_fired=True,
                  slow_burn_count=1)
    assert _runner_entry_block_reason(a, _config("shadow")) == ""


def test_runner_gate_off_admits_via_bypass():
    # byte-identical pre-change behavior: mover bypass admits this entry.
    a = _analysis(volume_spike_fired=True)  # satisfies bypass structure term
    assert _runner_entry_block_reason(a, _config("off")) == ""


def test_runner_gate_live_passes_non_mover_long():
    a = _analysis(daily_mover_fired=False, volume_spike_fired=True,
                  breakout_fired=True, slow_burn_count=1)
    assert _runner_entry_block_reason(a, _config("live")) == ""


def test_runner_gate_live_passes_short_on_mover():
    # short path returns before the long-side veto (breakout gate owns it)
    a = _analysis(side="short", confidence=0.85,
                  downtrend_momentum_fired=True)
    cfg = _config("live")
    cfg["runner_entry_gate"]["allow_shorts"] = True
    assert _runner_entry_block_reason(a, cfg) == ""
