"""Tests for the quiet broad-tape entry gate (2026-09-10, WATCHLIST §B.17).

Multi-variant since 2026-09-11: the gate evaluates a `btc` variant (BTC
proxy, the original) and an `alt` variant (equal-weight alt-basket index —
the tape the bot actually trades). Each variant blocks a NEW entry when its
trailing-24h realized vol < `vol_pct` AND |trailing-24h drift| < `drift_pct`.

Merge semantics: entry blocked iff any LIVE (shadow_mode false) variant
reads quiet; a shadow variant never blocks — it carries `shadow_would_block`
+ `shadow_reasons` (one `quiet_tape[<name>]` reason per firing shadow
variant), which the executor logs loudly. Fail-safes: disabled variant,
data gap (tape fetch → None), or non-positive thresholds ALWAYS pass for
that variant — a data gap can never block a trade.

Both tape-activity functions are monkeypatched so the tests never touch the
network; the vol/drift math is the sweep's math (scratch/_quiet_tape_sweep.py
/ _quiet_tape_alt_sweep.py) and is exercised here only via synthetic candles.
"""
from __future__ import annotations

import math
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


# Side channel for fetch-observation tests (avoids mutating the result).
_LAST_FETCHED = {"btc": False, "alt": False}


def _run(gate_cfg: dict, btc_tape, alt_tape: object = "absent",
         ctx: "GateContext | None" = None):
    """Run the gate with both tape functions monkeypatched.

    `btc_tape` / `alt_tape`: tape dict, None (data gap), or "absent"
    (default — that variant must not be fetched). Fetch observations land
    in `_LAST_FETCHED` (test-only).
    """
    from hermes_trader.agents import market_regime

    orig_btc = market_regime.btc_tape_activity
    orig_alt = market_regime.alt_basket_tape_activity
    _LAST_FETCHED.update({"btc": False, "alt": False})
    if btc_tape != "absent":
        market_regime.btc_tape_activity = (
            lambda force=False: (_LAST_FETCHED.__setitem__("btc", True), btc_tape)[1])
    if alt_tape != "absent":
        market_regime.alt_basket_tape_activity = (
            lambda force=False, **kw: (_LAST_FETCHED.__setitem__("alt", True), alt_tape)[1])
    try:
        return quiet_tape_gate(ctx or _ctx(), gate_cfg)
    finally:
        market_regime.btc_tape_activity = orig_btc
        market_regime.alt_basket_tape_activity = orig_alt


class _FakeCandle:
    def __init__(self, c):
        self.c = c


def _alt_tape_on(coins_closes, blocklist=None):
    """Run alt_basket_tape_activity with universe/candles monkeypatched.

    `coins_closes`: {coin: [closes]} — universe vol assigned so every coin is
    above the floor (rank by dict order). Returns the tape dict or None.
    """
    from hermes_trader.agents import market_regime
    from hermes_trader.client import universe as uni_mod

    vol = 1e9
    fake_universe = [
        {"coin": c, "type": "perp", "dex": None, "dayNtlVlm": vol - i}
        for i, c in enumerate(coins_closes)
    ]
    orig_universe = uni_mod.get_universe
    orig_fetch = market_regime.fetch_hl_candles
    uni_mod.get_universe = (lambda **kw: fake_universe)
    market_regime.fetch_hl_candles = lambda coin, interval="5m", count=100, fresh=False: (
        [_FakeCandle(x) for x in coins_closes[coin]] if coin in coins_closes else [])
    market_regime._alt_basket_cache = (None, 0.0)
    try:
        return market_regime.alt_basket_tape_activity(
            force=True, blocklist=tuple(blocklist or ()))
    finally:
        uni_mod.get_universe = orig_universe
        market_regime.fetch_hl_candles = orig_fetch
        market_regime._alt_basket_cache = (None, 0.0)


# ── Legacy flat config shape (backward compat — BTC variant only) ──────

CFG_SHADOW = {"enabled": True, "shadow_mode": True, "vol_pct": 2.5, "drift_pct": 2.0}
CFG_LIVE = {"enabled": True, "shadow_mode": False, "vol_pct": 2.5, "drift_pct": 2.0}


def test_flat_disabled_passes():
    r = _run({"enabled": False}, _tape(0.5, 0.1))
    assert r == {"pass": True}


def test_flat_data_gap_never_blocks_even_when_live():
    r = _run(CFG_LIVE, None)
    assert r == {"pass": True}


def test_flat_quiet_shadow_marker():
    r = _run(CFG_SHADOW, _tape(0.8, 0.3))
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert "quiet_tape[btc]" in r["reason"]
    assert "0.80%" in r["reason"]  # vol in the reason (join/audit key)


def test_flat_quiet_live_blocks():
    r = _run(CFG_LIVE, _tape(0.8, 0.3))
    assert r["pass"] is False
    assert "quiet_tape[btc]" in r["reason"]
    assert not r.get("shadow_would_block")


def test_flat_active_tape_passes_vol_above():
    r = _run(CFG_LIVE, _tape(3.0, 0.0))
    assert r == {"pass": True}


def test_flat_active_tape_passes_drift_above():
    r = _run(CFG_LIVE, _tape(0.5, 5.0))
    assert r == {"pass": True}


def test_flat_drift_sign_agnostic():
    r = _run(CFG_LIVE, _tape(0.5, -5.0))
    assert r == {"pass": True}


def test_flat_boundary_is_strict():
    assert _run(CFG_LIVE, _tape(2.5, 0.0)) == {"pass": True}
    assert _run(CFG_LIVE, _tape(0.0, 2.0)) == {"pass": True}
    assert _run(CFG_LIVE, _tape(2.49, 1.99))["pass"] is False


def test_flat_negative_thresholds_disable():
    r = _run({"enabled": True, "shadow_mode": False, "vol_pct": 0, "drift_pct": 2.0},
             _tape(0.1, 0.1))
    assert r == {"pass": True}


def test_flat_default_thresholds_are_25_20():
    assert _run({"enabled": True, "shadow_mode": False}, _tape(2.0, 1.5))["pass"] is False
    assert _run({"enabled": True, "shadow_mode": False}, _tape(3.0, 1.5)) == {"pass": True}


def test_flat_side_irrelevant():
    r_long = _run(CFG_LIVE, _tape(0.5, 0.2), ctx=_ctx(trade_side="long"))
    r_short = _run(CFG_LIVE, _tape(0.5, 0.2), ctx=_ctx(trade_side="short"))
    assert r_long["pass"] is False
    assert r_short["pass"] is False


def test_flat_shape_never_fetches_alt():
    """Legacy flat config must not trigger the alt-basket fetch."""
    _run(CFG_LIVE, _tape(0.8, 0.3), alt_tape=_tape(0.8, 0.3))
    assert _LAST_FETCHED["alt"] is False
    assert _LAST_FETCHED["btc"] is True


# ── Multi-variant config shape ──────────────────────────────────────────

BQT = {"enabled": True, "shadow_mode": False, "vol_pct": 2.5, "drift_pct": 2.0}
ALTQ = {"enabled": True, "shadow_mode": True, "vol_pct": 4.5, "drift_pct": 2.0}


def test_multi_btc_live_alt_shadow_btc_quiet_blocks():
    """BTC live+quiet blocks; alt also reads quiet but only accrues."""
    r = _run({"btc": BQT, "alt": ALTQ}, _tape(1.0, 0.5), _tape(3.0, 1.0))
    assert r["pass"] is False
    assert r.get("shadow_would_block") is True
    assert "quiet_tape[btc]" in r["reason"]
    assert "quiet_tape[alt]" in r["reason"]
    assert len(r["shadow_reasons"]) == 1
    assert "quiet_tape[alt]" in r["shadow_reasons"][0]


def test_multi_alt_live_quiet_blocks_even_when_btc_active():
    """The alt variant has real live capability (parity with BTC)."""
    r = _run({"btc": {**BQT}, "alt": {**ALTQ, "shadow_mode": False}},
             _tape(3.0, 5.0), _tape(3.0, 1.0))
    assert r["pass"] is False
    assert "quiet_tape[alt]" in r["reason"]
    assert "quiet_tape[btc]" not in r["reason"]
    assert not r.get("shadow_would_block")


def test_multi_both_shadow_never_blocks():
    r = _run({"btc": {**BQT, "shadow_mode": True}, "alt": ALTQ},
             _tape(1.0, 0.5), _tape(3.0, 1.0))
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert len(r["shadow_reasons"]) == 2
    assert "quiet_tape[btc]" in r["shadow_reasons"][0]
    assert "quiet_tape[alt]" in r["shadow_reasons"][1]


def test_multi_alt_disabled_btc_alone():
    r = _run({"btc": BQT, "alt": {"enabled": False, "vol_pct": 4.5, "drift_pct": 2.0}},
             _tape(1.0, 0.5), _tape(3.0, 1.0))
    assert r["pass"] is False  # BTC live+quiet still blocks
    assert "quiet_tape[alt]" not in r["reason"]
    assert _LAST_FETCHED["alt"] is False  # disabled variant never fetched


def test_multi_alt_data_gap_never_blocks():
    """Alt tape data gap → alt variant has no opinion; BTC alone decides."""
    r = _run({"btc": {**BQT, "shadow_mode": True}, "alt": {**ALTQ, "shadow_mode": False}},
             _tape(3.0, 5.0), None)
    assert r == {"pass": True}  # BTC active, alt no opinion
    r2 = _run({"btc": BQT, "alt": {**ALTQ, "shadow_mode": False}},
              _tape(1.0, 0.5), None)
    assert r2["pass"] is False  # BTC LIVE+quiet still blocks
    assert "quiet_tape[alt]" not in r2["reason"]
    assert not r2.get("shadow_would_block")  # alt fired nothing to accrue


def test_multi_neither_quiet_passes():
    r = _run({"btc": BQT, "alt": {**ALTQ, "shadow_mode": False}},
             _tape(3.0, 5.0), _tape(5.0, -7.0))
    assert r == {"pass": True}


def test_multi_all_variants_disabled_passes():
    r = _run({"btc": {"enabled": False}, "alt": {"enabled": False}},
             _tape(0.5, 0.1), _tape(0.5, 0.1))
    assert r == {"pass": True}


# ── alt_basket_tape_activity: math + fail-safes (synthetic candles) ─────

STEADY = [100.0 + i * 0.05 for i in range(250)]  # steady +1.24% tape


def test_alt_index_equal_weight_math():
    """Opposite 10% moves across two coins → equal-weight index is flat:
    vol ≈ 0, drift ≈ 0. Both axes computed on the INDEX, not per-coin."""
    up = [100.0] * 248 + [110.0] * 2   # +10% over the window
    down = [100.0] * 248 + [90.0] * 2
    tape = _alt_tape_on({"A": up, "B": down})
    assert tape is not None
    assert tape["vol"] < 0.5, tape
    assert abs(tape["drift"]) < 0.5, tape


def test_alt_index_steady_tape_near_zero_vol_expected_drift():
    """A steady-drift tape has near-zero realized vol and the expected
    drift — same independence contract as the BTC tape."""
    tape = _alt_tape_on({"A": STEADY, "B": STEADY})
    assert tape is not None
    assert tape["vol"] < 0.05, tape
    expected = (STEADY[-1] / STEADY[0] - 1) * 100
    assert abs(tape["drift"] - expected) < 0.01, tape


def test_alt_btc_excluded_from_basket():
    """BTC is the measure, not the basket — it must be dropped even when
    it is the highest-volume name."""
    tape = _alt_tape_on({"BTC": STEADY, "A": STEADY, "B": STEADY})
    assert tape is not None  # still has 2 alts
    # a BTC-only + 1-alt universe has only one basket coin → no opinion
    assert _alt_tape_on({"BTC": STEADY, "A": STEADY}) is None


def test_alt_blocklist_respected():
    """Blocklisted coins leave the basket; a blocked universe with <2 alts
    → no opinion (None)."""
    assert _alt_tape_on({"A": STEADY, "B": STEADY, "TON": STEADY},
                        blocklist=("TON",)) is not None
    # TON is the only non-blocklisted coin left → <2 basket coins → None
    assert _alt_tape_on({"A": STEADY, "TON": STEADY},
                        blocklist=("A",)) is None


def test_alt_insufficient_history_returns_none():
    short = [100.0] * 50  # < 100 bars in the window
    assert _alt_tape_on({"A": short, "B": short}) is None


def test_alt_universe_fetch_failure_returns_none():
    from hermes_trader.agents import market_regime
    from hermes_trader.client import universe as uni_mod

    orig_universe = uni_mod.get_universe
    orig_fetch = market_regime.fetch_hl_candles
    uni_mod.get_universe = (lambda **kw: (_ for _ in ()).throw(RuntimeError("429")))
    market_regime._alt_basket_cache = (None, 0.0)
    try:
        assert market_regime.alt_basket_tape_activity(force=True) is None
    finally:
        uni_mod.get_universe = orig_universe
        market_regime.fetch_hl_candles = orig_fetch
        market_regime._alt_basket_cache = (None, 0.0)


# ── btc_tape_activity math (unchanged contract) ─────────────────────────

def test_btc_tape_activity_math_on_synthetic_candles():
    """251 closes from alternating ±0.5% log-returns:
      vol = pstdev(returns)×sqrt(288)×100 ≈ 8.485%; drift = 0 (pairs cancel)."""
    from hermes_trader.agents import market_regime

    s = 0.0
    closes = [100.0]
    for i in range(1, 251):
        s += 0.005 if i % 2 == 0 else -0.005
        closes.append(100.0 * math.exp(s))
    candles = [_FakeCandle(c) for c in closes]
    orig = market_regime.fetch_hl_candles
    market_regime.fetch_hl_candles = (lambda coin, interval="5m", count=100, fresh=False: candles)
    market_regime._tape_cache = (None, 0.0)
    try:
        tape = market_regime.btc_tape_activity(force=True)
        assert tape is not None
        assert abs(tape["vol"] - 0.005 * math.sqrt(288) * 100) < 0.05, tape
        assert abs(tape["drift"]) < 0.01, tape
    finally:
        market_regime.fetch_hl_candles = orig
        market_regime._tape_cache = (None, 0.0)


def test_btc_tape_activity_insufficient_history_returns_none():
    from hermes_trader.agents import market_regime

    candles = [_FakeCandle(100.0)] * 50  # < _TAPE_MIN_BARS
    orig = market_regime.fetch_hl_candles
    market_regime.fetch_hl_candles = (lambda coin, interval="5m", count=100, fresh=False: candles)
    market_regime._tape_cache = (None, 0.0)
    try:
        assert market_regime.btc_tape_activity(force=True) is None
    finally:
        market_regime.fetch_hl_candles = orig
        market_regime._tape_cache = (None, 0.0)
