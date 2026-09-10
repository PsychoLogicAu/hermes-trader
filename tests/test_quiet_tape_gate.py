"""Tests for the quiet broad-tape entry gate (2026-09-10, WATCHLIST §B.17).

The gate blocks a NEW entry when the BROAD tape (BTC) is quiet — trailing-24h
realized vol < `vol_pct` AND |trailing-24h drift| < `drift_pct`. It is SHADOW
by default: structurally pass + `shadow_would_block` marker until
`shadow_mode` is flipped in .agent-config.json. Fail-safes: disabled /
data gap (btc_tape_activity() → None) always pass — a data gap can never
block a trade.

`btc_tape_activity()` is monkeypatched so the tests never touch the network;
the vol/drift math itself is the sweep's math (scratch/_quiet_tape_sweep.py)
and is exercised here only via the gate's threshold comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    quiet_tape_gate,
)


def _ctx(
    confidence: float = 0.80,
    composite: float = 40.0,
    trade_side: str = "long",
) -> GateContext:
    return GateContext(
        confidence=confidence,
        current_positions=[],
        trade_notional_usd=100.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin="TEST",
        trade_side=trade_side,
        has_binary_news_risk=False,
        equity=1000.0,
        total_open_notional=0.0,
        composite_score=composite,
    )


def _tape(vol: float, drift: float) -> dict:
    return {"vol": vol, "drift": drift}


def _run(gate_cfg, tape, ctx=None):
    """Run the gate with btc_tape_activity monkeypatched to return `tape`."""
    # the gate imports from hermes_trader.agents.market_regime — patch there
    from hermes_trader.agents import market_regime
    orig = market_regime.btc_tape_activity
    market_regime.btc_tape_activity = (lambda force=False: tape)
    try:
        return quiet_tape_gate(ctx or _ctx(), gate_cfg)
    finally:
        market_regime.btc_tape_activity = orig


CFG_SHADOW = {"enabled": True, "shadow_mode": True, "vol_pct": 2.5, "drift_pct": 2.0}
CFG_LIVE = {"enabled": True, "shadow_mode": False, "vol_pct": 2.5, "drift_pct": 2.0}


def test_disabled_passes():
    r = _run({"enabled": False}, _tape(0.5, 0.1))
    assert r == {"pass": True}


def test_data_gap_never_blocks_even_when_live():
    """A data gap can never block a trade — the fail-safe contract."""
    r = _run(CFG_LIVE, None)
    assert r == {"pass": True}


def test_quiet_tape_shadow_marker():
    """Quiet tape + shadow mode: structurally passes, carries the marker."""
    r = _run(CFG_SHADOW, _tape(0.8, 0.3))
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert "quiet_tape" in r["reason"]
    assert "0.80%" in r["reason"]  # vol in the reason (join/audit key)


def test_quiet_tape_live_blocks():
    r = _run(CFG_LIVE, _tape(0.8, 0.3))
    assert r["pass"] is False
    assert "quiet_tape" in r["reason"]


def test_active_tape_passes_vol_above():
    """Vol above the threshold → not quiet → pass (drift irrelevant)."""
    r = _run(CFG_LIVE, _tape(3.0, 0.0))
    assert r == {"pass": True}


def test_active_tape_passes_drift_above():
    """Drift above the threshold → not quiet → pass (vol irrelevant)."""
    r = _run(CFG_LIVE, _tape(0.5, 5.0))
    assert r == {"pass": True}


def test_drift_sign_agnostic():
    """|drift| — a down-tape with strong drift is ACTIVE, not quiet."""
    r = _run(CFG_LIVE, _tape(0.5, -5.0))
    assert r == {"pass": True}


def test_boundary_is_strict():
    """vol == threshold or |drift| == threshold → NOT quiet (strict <)."""
    r_at_vol = _run(CFG_LIVE, _tape(2.5, 0.0))
    assert r_at_vol == {"pass": True}
    r_at_drift = _run(CFG_LIVE, _tape(0.0, 2.0))
    assert r_at_drift == {"pass": True}
    r_just_below = _run(CFG_LIVE, _tape(2.49, 1.99))
    assert r_just_below["pass"] is False


def test_negative_thresholds_disable():
    """vol_pct/drift_pct <= 0 → gate off (no opinion), never blocks."""
    r = _run({"enabled": True, "shadow_mode": False, "vol_pct": 0, "drift_pct": 2.0},
             _tape(0.1, 0.1))
    assert r == {"pass": True}


def test_default_thresholds_are_25_20():
    """Omitted vol_pct/drift_pct fall back to the sweep's 2.5/2.0."""
    r = _run({"enabled": True, "shadow_mode": False}, _tape(2.0, 1.5))
    assert r["pass"] is False  # below both defaults
    r2 = _run({"enabled": True, "shadow_mode": False}, _tape(3.0, 1.5))
    assert r2 == {"pass": True}  # vol above default vol_pct


def test_side_irrelevant():
    """The gate is side-independent — shorts get the same quiet-tape block."""
    r_long = _run(CFG_LIVE, _tape(0.5, 0.2), _ctx(trade_side="long"))
    r_short = _run(CFG_LIVE, _tape(0.5, 0.2), _ctx(trade_side="short"))
    assert r_long["pass"] is False
    assert r_short["pass"] is False


def test_tape_activity_math_on_synthetic_candles():
    """btc_tape_activity vol/drift math on a series with known per-bar returns.

    251 closes built from exactly alternating ±0.5% log-returns (125 of each):
      vol   = pstdev(returns)×sqrt(288)×100 = 0.005×16.9706×100 ≈ 8.485%
      drift = last/first − 1 = exp(0) − 1 = 0 (pairs cancel)
    """
    import math
    from hermes_trader.agents import market_regime

    class FakeCandle:
        def __init__(self, c):
            self.c = c

    s = 0.0
    closes = [100.0]
    for i in range(1, 251):
        s += 0.005 if i % 2 == 0 else -0.005
        closes.append(100.0 * math.exp(s))
    candles = [FakeCandle(c) for c in closes]
    orig = market_regime.fetch_hl_candles
    market_regime.fetch_hl_candles = (lambda coin, interval="5m", count=100, fresh=False: candles)
    market_regime._tape_cache = (None, 0.0)  # reset module cache
    try:
        tape = market_regime.btc_tape_activity(force=True)
        assert tape is not None
        assert abs(tape["vol"] - 0.005 * math.sqrt(288) * 100) < 0.05, tape
        assert abs(tape["drift"]) < 0.01, tape
    finally:
        market_regime.fetch_hl_candles = orig
        market_regime._tape_cache = (None, 0.0)


def test_tape_activity_steady_tape_has_near_zero_vol():
    """A steady-drift tape has near-zero realized vol (bar-to-bar variance is
    ~0) and the expected drift — the two axes are independent."""
    import math
    from hermes_trader.agents import market_regime

    class FakeCandle:
        def __init__(self, c):
            self.c = c

    closes = [100.0 + i * 0.01 for i in range(250)]
    candles = [FakeCandle(c) for c in closes]
    orig = market_regime.fetch_hl_candles
    market_regime.fetch_hl_candles = (lambda coin, interval="5m", count=100, fresh=False: candles)
    market_regime._tape_cache = (None, 0.0)
    try:
        tape = market_regime.btc_tape_activity(force=True)
        assert tape is not None
        assert tape["vol"] < 0.01, tape  # steady tape ≈ 0 realized vol
        assert abs(tape["drift"] - (closes[-1] / closes[0] - 1) * 100) < 0.01, tape
    finally:
        market_regime.fetch_hl_candles = orig
        market_regime._tape_cache = (None, 0.0)


def test_tape_activity_insufficient_history_returns_none():
    from hermes_trader.agents import market_regime

    class FakeCandle:
        def __init__(self, c):
            self.c = c

    candles = [FakeCandle(100.0)] * 50  # < _TAPE_MIN_BARS
    orig = market_regime.fetch_hl_candles
    market_regime.fetch_hl_candles = (lambda coin, interval="5m", count=100, fresh=False: candles)
    market_regime._tape_cache = (None, 0.0)
    try:
        assert market_regime.btc_tape_activity(force=True) is None
    finally:
        market_regime.fetch_hl_candles = orig
        market_regime._tape_cache = (None, 0.0)


def test_tape_activity_fetch_failure_returns_none():
    from hermes_trader.agents import market_regime

    def boom(coin, interval="5m", count=100, fresh=False):
        raise RuntimeError("429")

    orig = market_regime.fetch_hl_candles
    market_regime.fetch_hl_candles = boom
    market_regime._tape_cache = (None, 0.0)
    try:
        assert market_regime.btc_tape_activity(force=True) is None
    finally:
        market_regime.fetch_hl_candles = orig
        market_regime._tape_cache = (None, 0.0)
