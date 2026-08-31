"""Tests for the band counter-trend breach conviction gate.

Deterministic encoding of the shape the band-snapback prompt note targets:
price bouncing OFF a DRIFTING band's upper edge while the band drifts DOWN
(a 1h "relief bounce against the trend") — and the mirror (dip below the
lower edge of an UP-drifting band). When that shape is present, a NEW
entry on the bounce/dip side needs >= min_conf (0.90) conviction to pass.

Shadow-mode semantics (the live default) are tested here too: the gate
MUST log the would-block and MUST structurally pass until shadow_mode
is flipped off.

Replay shape (from the 2026-08-26 GRASS incident): 4h relief rally inside
a downswing — 1h band EMA/16 drifting down, price bouncing back above the
upper edge. The LLM read the counter-trend band line as bullish and
entered at conf 0.82, which the trade then lost (-9.6% ROE).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    band_counter_breach_gate,
)
from hermes_trader.indicators.triggers import band_state  # noqa: E402
from hermes_trader.models.types import Candle  # noqa: E402


def _candles(prices: list) -> list[Candle]:
    """Synthetic 1h candles: flat body at each price, tiny wicks."""
    out = []
    for i, p in enumerate(prices):
        out.append(
            Candle(
                t=1_700_000_000_000 + i * 3_600_000,
                o=p,
                h=p * 1.001,
                l=p * 0.999,
                c=p,
                v=100.0,
            )
        )
    return out


def _grass_shape(prices: list | None = None) -> list[Candle]:
    """Down-trending 1h band over its OWN 16-bar window + a short relief
    bounce back above the upper edge — the GRASS 2026-08-26 setup.

    The direction verdict is measured over the band's own window only (the
    band's 16-bar EMA span), so the rally must be short enough that the
    16-bar edge still reads DOWN (a 4-bar bounce is well inside one 16-bar
    window; a 12-bar rally would dominate the window and read as UP — the
    old 48-bar drift reference is gone)."""
    if prices is not None:
        return _candles(prices)
    prices = [1.0 - 0.002 * i for i in range(96)]  # ~-19% downswing
    prices += [prices[-1] + 0.008 * (k + 1) for k in range(4)]  # 4-bar relief rally
    return _candles(prices)


def _chop_prices() -> list:
    """Flat band (drift ~0) with price oscillating around the edge — the
    non-trending case that must never arm the gate."""
    import math
    prices = [1.0 + 0.01 * math.sin(i / 3.0) for i in range(100)]
    return prices


# ---------------------------------------------------------------------------
# band_state — the drift-gate math (shared with band_snapback)
# ---------------------------------------------------------------------------


def test_band_state_grass_shape_detects_down_drift_and_breach():
    bs = band_state(_grass_shape(), band_span=16, max_drift_pct=1.5, ma_type="ema")
    assert bs["trending"] is True
    assert bs["direction"] == "DOWN"
    assert bs["drift_pct"] > 1.5
    assert bs["px_upper_pct"] > 0.0  # price past the UPPER edge
    assert bs["breach_opposite_pct"] == pytest.approx(
        bs["px_upper_pct"], rel=1e-9
    )  # DOWN drift: opposite edge = upper
    assert bs["breach_opposite_pct"] > 1.0  # comfortably past the 1% arm threshold


def test_band_state_mirror_up_drift_dip_below_lower():
    prices = [1.0 + 0.0025 * i for i in range(94)]  # uptrend over the window
    prices += [prices[-1] - 0.008 * (k + 1) for k in range(6)]  # 6-bar dip below the lower edge
    bs = band_state(_candles(prices), band_span=16, max_drift_pct=1.5, ma_type="ema")
    assert bs["trending"] is True
    assert bs["direction"] == "UP"
    assert bs["breach_opposite_pct"] == pytest.approx(-bs["px_lower_pct"], rel=1e-9)


def test_band_state_chop_never_trends():
    bs = band_state(_candles(_chop_prices()), band_span=16, max_drift_pct=1.5)
    assert bs["trending"] is False
    # breach only matters when trending; on a flat band the gate must have
    # no opinion regardless of the px position


def test_band_state_insufficient_history():
    bs = band_state(_grass_shape()[:30], band_span=16)
    assert bs is None  # < 2*band_span bars -> no opinion


def test_band_state_partial_bar_included():
    """include_partial=True must use the forming bar's close (the live
    perception semantics) — the verdict reference for a mid-entry call."""
    full = _grass_shape()
    with_partial = band_state(full, band_span=16, max_drift_pct=1.5, include_partial=True)
    without = band_state(full, band_span=16, max_drift_pct=1.5, include_partial=False)
    # both should agree on direction here; the partial call must not error
    # and must reference the last (partial) bar's close
    assert with_partial["direction"] == without["direction"]


# ---------------------------------------------------------------------------
# drift_ref — the gate-only longer drift-reference lag (2026-08-31)
# ---------------------------------------------------------------------------

def _gentle_downswing_shape() -> list[Candle]:
    """A GRASS-shaped downswing GENTLE enough that the band's own 16-bar
    window reads drift 0.74% (chop -> the reworked default sleeps) while a
    32-bar drift reference reads 1.72% (trending -> the gate arms). This is
    the exact asymmetry drift_ref exists to recover: late-chase bounces off
    a slow, long trend. Bounce carries px ~1.45% past the upper edge."""
    prices = [1.0 - 0.0006 * i for i in range(90)]
    prices += [prices[-1] + 0.005 * (k + 1) for k in range(4)]
    return _candles(prices)


def test_band_state_drift_ref_default_equals_span():
    """drift_ref=None must be byte-identical to drift_ref=band_span — the
    trigger's single-window semantics are the untouched default."""
    cs = _grass_shape()
    a = band_state(cs, band_span=16, max_drift_pct=1.5)
    b = band_state(cs, band_span=16, max_drift_pct=1.5, drift_ref=16)
    assert a == b


def test_band_state_drift_ref_arms_gentle_trend_the_span_alone_misses():
    cs = _gentle_downswing_shape()
    own = band_state(cs, band_span=16, max_drift_pct=1.5)          # ref=16
    long_ref = band_state(cs, band_span=16, max_drift_pct=1.5, drift_ref=32)
    assert own["trending"] is False          # own-window chop: gate sleeps
    assert long_ref["trending"] is True      # 32-bar ref: the trend is seen
    assert long_ref["direction"] == "DOWN"
    # edges are the SAME MA — px-vs-edge and breach barely move with ref
    assert abs(long_ref["px_upper_pct"] - own["px_upper_pct"]) < 0.15
    assert long_ref["breach_opposite_pct"] > 1.0


def test_band_state_drift_ref_chop_stays_chop():
    """A longer reference lag must not manufacture trend out of flat chop."""
    import math
    prices = [1.0 + 0.01 * math.sin(i / 3.0) for i in range(120)]
    bs = band_state(_candles(prices), band_span=16, max_drift_pct=1.5, drift_ref=32)
    assert bs["trending"] is False


def test_band_state_drift_ref_history_boundary():
    """ref=32 needs span+ref+2 = 50 bars (include_partial); one short -> None."""
    cs = _grass_shape()
    assert band_state(cs[:49], band_span=16, drift_ref=32) is None
    assert band_state(cs[:50], band_span=16, drift_ref=32) is not None


# ---------------------------------------------------------------------------
# the gate itself
# ---------------------------------------------------------------------------

# The live .agent-config.json band_snapback block (GRASS: no per-coin
# override -> base settings: EMA/16 on 1h, 16-bar window, 1.5% drift gate).
BAND_CFG = {
    "enabled": True,
    "ma_type": "ema",
    "band_span": 16,
    "max_drift_pct": 1.5,
    "min_poke_atr": 0.75,
    "max_project_atr": 0.25,
    "interval": "1h",
}


def _ctx(side: str, conf: float) -> GateContext:
    """Real GateContext; the gate's two I/O boundaries are mocked per-test
    (read_agent_config -> band_snapback config, fetch_hl_candles ->
    synthetic candles)."""
    return GateContext(
        confidence=conf,
        current_positions=[],
        trade_notional_usd=50.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin="GRASS",
        trade_side=side,
        has_binary_news_risk=False,
        equity=1000.0,
        total_open_notional=50.0,
    )


def _wire(monkeypatch, candles: list[Candle], band_cfg: dict | None = None):
    """Point the gate's I/O at the synthetic candles and a live-config-shaped
    band_snapback block (ema/16, 1h, span 16 — GRASS's live settings)."""
    agent_cfg = {"band_snapback": band_cfg if band_cfg is not None else BAND_CFG}
    monkeypatch.setattr(
        "hermes_trader.agents.config_store.read_agent_config",
        lambda: agent_cfg,
    )
    monkeypatch.setattr(
        "hermes_trader.client.hl_client.fetch_hl_candles",
        lambda coin, interval="1h", count=200, **kw: candles,
    )


def _gate_cfg(**over) -> dict:
    base = {"enabled": True, "shadow_mode": False, "min_conf": 0.9, "min_breach_pct": 1.0}
    base.update(over)
    return base


def test_gate_grass_replay_blocked_at_082(monkeypatch):
    """The exact GRASS trade: conf 0.82 long, band DOWN, px past the upper
    edge -> would-have-been blocked (0.82 < 0.90)."""
    _wire(monkeypatch, _grass_shape())
    r = band_counter_breach_gate(_ctx("long", 0.82), _gate_cfg())
    assert r["pass"] is False
    assert "GRASS long" in r["reason"]
    assert r["reason"].startswith("[gate:band_counter_breach]")
    assert "0.82" in r["reason"] and "0.90" in r["reason"]


def test_gate_same_shape_passes_at_090(monkeypatch):
    _wire(monkeypatch, _grass_shape())
    r = band_counter_breach_gate(_ctx("long", 0.90), _gate_cfg())
    assert r["pass"] is True, r
    assert r.get("via") == "confidence"


def test_gate_short_entry_not_counter_trend(monkeypatch):
    """A SHORT when the band is DOWN-drifting with px above the upper edge
    is WITH the drift — the bounce is the short's friend, not its risk."""
    _wire(monkeypatch, _grass_shape())
    r = band_counter_breach_gate(_ctx("short", 0.6), _gate_cfg())
    assert r["pass"] is True
    assert r == {"pass": True}  # no opinion at all — not the shape


def test_gate_mirror_up_drift_dip_blocks_short_below_lower(monkeypatch):
    prices = [1.0 + 0.0025 * i for i in range(94)]
    prices += [prices[-1] - 0.008 * (k + 1) for k in range(6)]
    _wire(monkeypatch, _candles(prices))
    r = band_counter_breach_gate(_ctx("short", 0.8), _gate_cfg())
    # dip below the lower edge of an UP-drifting band + SHORT = counter-trend
    assert r["pass"] is False
    assert "lower" in r["reason"]
    assert "UP-drifting" in r["reason"]


def test_gate_chop_passes(monkeypatch):
    _wire(monkeypatch, _candles(_chop_prices()))
    r = band_counter_breach_gate(_ctx("long", 0.6), _gate_cfg())
    assert r == {"pass": True}  # band not trending -> no opinion


def test_gate_disabled_passes(monkeypatch):
    _wire(monkeypatch, _grass_shape())
    r = band_counter_breach_gate(_ctx("long", 0.5), _gate_cfg(enabled=False))
    assert r == {"pass": True}


def test_gate_band_trigger_disabled_in_agent_config_passes(monkeypatch):
    """The gate defers to the band_snapback trigger config: if the trigger
    itself is off, the gate has no opinion."""
    _wire(monkeypatch, _grass_shape(), band_cfg={**BAND_CFG, "enabled": False})
    r = band_counter_breach_gate(_ctx("long", 0.5), _gate_cfg())
    assert r == {"pass": True}


def test_gate_sub_threshold_breach_passes(monkeypatch):
    """Band is trending (down) but price is INSIDE the band (pullback within
    the bounce) — breach clamps to 0 < min_breach_pct -> shape not armed,
    trade proceeds at any confidence."""
    prices = [1.0 - 0.002 * i for i in range(88)]
    prices += [0.800, 0.806, 0.812, 0.818, 0.815, 0.812,
               0.810, 0.808, 0.806, 0.804, 0.802, 0.800]
    _wire(monkeypatch, _candles(prices))
    r = band_counter_breach_gate(_ctx("long", 0.5), _gate_cfg(min_breach_pct=1.0))
    assert r["pass"] is True


def test_gate_shadow_mode_logs_but_passes(monkeypatch, caplog):
    """Shadow mode (the live default): MUST log a would-block and MUST
    structurally pass — no silent behavior change."""
    _wire(monkeypatch, _grass_shape())
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        r = band_counter_breach_gate(_ctx("long", 0.82), _gate_cfg(shadow_mode=True))
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert "via" not in r
    assert "GRASS long" in r["reason"]
    logged = [rec for rec in caplog.records if "band_counter_breach would-block" in rec.getMessage()]
    assert logged, "shadow mode must log the would-block"
    assert "0.82" in logged[0].getMessage()
    assert "GRASS" in logged[0].getMessage()


def test_gate_candle_fetch_failure_passes(monkeypatch):
    """Fail-safe: a candle fetch error must never block a trade."""
    agent_cfg = {"band_snapback": BAND_CFG}
    monkeypatch.setattr(
        "hermes_trader.agents.config_store.read_agent_config",
        lambda: agent_cfg,
    )
    def _boom(*a, **k):
        raise RuntimeError("api down")
    monkeypatch.setattr("hermes_trader.client.hl_client.fetch_hl_candles", _boom)
    r = band_counter_breach_gate(_ctx("long", 0.5), _gate_cfg())
    assert r == {"pass": True}


def test_gate_per_coin_override_drives_fetch(monkeypatch):
    """A per-coin band_snapback override (interval/band_span) must drive the
    gate's candle fetch (count = 2*band_span + 4), like the live trigger's
    resolution — recorded via the mocked fetch."""
    ov_cfg = {**BAND_CFG, "overrides": {"GRASS": {"interval": "15m", "band_span": 24}}}
    _wire(monkeypatch, _grass_shape(), band_cfg=ov_cfg)
    import hermes_trader.client.hl_client as hlc
    calls = []
    monkeypatch.setattr(
        hlc, "fetch_hl_candles",
        lambda coin, interval="1h", count=200, **kw: calls.append((coin, interval, count)) or _grass_shape(),
    )
    band_counter_breach_gate(_ctx("long", 0.5), _gate_cfg())
    assert calls, "gate must fetch candles on the band interval"
    coin, interval, count = calls[0]
    assert coin == "GRASS"
    assert interval == "15m"      # override applied
    assert count == 2 * 24 + 4    # override band_span applied


# ---------------------------------------------------------------------------
# drift_ref_span — the gate-only longer drift reference (2026-08-31)
# ---------------------------------------------------------------------------

def test_gate_drift_ref_span_arms_the_slow_late_chase(monkeypatch):
    """The whole point of the key: a GRASS-shaped bounce off a GENTLE, long
    downswing. With the key absent the 16-bar own-window drift (0.74%) reads
    chop and the 0.82-long passes; with drift_ref_span=32 the same candles
    read trending DOWN 1.72%, the breach arms, and 0.82 < 0.90 blocks."""
    cs = _gentle_downswing_shape()
    _wire(monkeypatch, cs)
    r_absent = band_counter_breach_gate(_ctx("long", 0.82), _gate_cfg())
    assert r_absent == {"pass": True}          # own-window chop: no opinion
    r_ref = band_counter_breach_gate(
        _ctx("long", 0.82), _gate_cfg(drift_ref_span=32))
    assert r_ref["pass"] is False              # the late-chase shape is armed
    assert "band_counter_breach" in r_ref["reason"]


def test_gate_drift_ref_span_scales_fetch_and_keeps_escape(monkeypatch):
    """The fetch grows to span + drift_ref_span + 4, and the conviction
    escape still applies at the longer ref (0.90 passes the armed shape)."""
    _wire(monkeypatch, _gentle_downswing_shape())
    import hermes_trader.client.hl_client as hlc
    calls = []
    monkeypatch.setattr(
        hlc, "fetch_hl_candles",
        lambda coin, interval="1h", count=200, **kw: calls.append((coin, interval, count)) or _gentle_downswing_shape(),
    )
    cfg = _gate_cfg(drift_ref_span=32)
    r = band_counter_breach_gate(_ctx("long", 0.90), cfg)
    assert r["pass"] is True and r.get("via") == "confidence"
    assert calls and calls[0][2] == 16 + 32 + 4