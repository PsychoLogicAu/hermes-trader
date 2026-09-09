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


# ---------------------------------------------------------------------------
# drift_confirmed_release (2026-09-06) — the trend-leg escape
# ---------------------------------------------------------------------------
# The 28-block cohort counterfactual: the conf>=0.90 escape never fired
# (max observed conf 0.85), so the gate re-timed every confirmed-trend
# entry deeper into the rip. The release quadrant: band drift >= 2.5% AND
# NO fresh 5m extreme (the price is extending a 4h range, not popping).
# The fresh-5m-extreme rows were the pops the gate exists to kill (LIT
# max-loss, DOGE stale-flat) — they must keep blocking.

RELEASE = {"enabled": True, "min_drift_pct": 2.5, "pop_lookback_5m": 48}


def _m5m(px: float, pop: bool, side: str = "long", n_closed: int = 49) -> list:
    """Synthetic 5m candles: `n_closed` closed bars + 1 still-forming bar
    whose OPEN is the live price `px`. `pop=True` -> every prior closed
    bar's extreme sits BEYOND px (a fresh extreme is in flight);
    `pop=False` -> at least one prior extreme is beyond px (the price is
    grinding under an established 4h range)."""
    out = []
    for i in range(n_closed):
        if side == "long":
            h, l = (px * 0.99, px * 0.985) if pop else (px * 1.01, px * 0.99)
        else:
            # short pop: prior lows ALL above px (price fresh below them);
            # short grind: a prior LOW sits below px (established range)
            h, l = (px * 1.01, px * 1.015) if pop else (px * 1.01, px * 0.99)
        out.append(Candle(t=1_700_000_000_000 + i * 300_000,
                          o=px, h=h, l=l, c=px, v=100.0))
    # forming bar: open == the live mid at the scan tick
    out.append(Candle(t=1_700_000_000_000 + n_closed * 300_000,
                      o=px, h=px, l=px, c=px, v=100.0))
    return out


def _wire5m(monkeypatch, band_candles: list[Candle], candles5m,
            band_cfg: dict | None = None):
    """_wire plus a routed 5m fetch (the release guard's second I/O)."""
    agent_cfg = {"band_snapback": band_cfg if band_cfg is not None else BAND_CFG}
    monkeypatch.setattr(
        "hermes_trader.agents.config_store.read_agent_config",
        lambda: agent_cfg,
    )

    def _fetch(coin, interval="1h", count=200, **kw):
        if interval == "5m":
            return candles5m
        return band_candles

    monkeypatch.setattr(
        "hermes_trader.client.hl_client.fetch_hl_candles", _fetch
    )


def _gate_cfg_release(**over) -> dict:
    cfg = _gate_cfg(drift_ref_span=32)  # live value — the drift the release checks
    cfg["drift_confirmed_release"] = dict(RELEASE)
    cfg.update(over)
    return cfg


def test_release_trend_leg_no_pop_passes_at_low_conf(monkeypatch):
    """The ARB shape: band drift-confirmed (ref 32 reads 6.94% >= 2.5),
    price grinding under the 4h high (no fresh 5m extreme) -> the 0.82
    long passes WITHOUT the unreachable conf>=0.90."""
    _wire5m(monkeypatch, _grass_shape(), _m5m(px=0.85, pop=False))
    r = band_counter_breach_gate(_ctx("long", 0.82), _gate_cfg_release())
    assert r["pass"] is True, r
    assert r.get("via") == "drift_confirmed"
    assert "RELEASING (drift-confirmed)" in r["reason"]
    assert "6.9" in r["reason"] or "drift" in r["reason"]


def test_release_fresh_pop_still_blocks(monkeypatch):
    """The GRASS/LIT/DOGE shape: same band, but the price IS at a fresh 5m
    extreme (the pop the gate exists to kill) -> the release must NOT
    fire; the original conf block stands."""
    _wire5m(monkeypatch, _grass_shape(), _m5m(px=0.85, pop=True))
    r = band_counter_breach_gate(_ctx("long", 0.82), _gate_cfg_release())
    assert r["pass"] is False
    assert "RELEASING" not in r["reason"]
    assert "conf 0.82 < 0.90" in r["reason"]


def test_release_drift_below_threshold_still_blocks(monkeypatch):
    """Band not drift-confirmed (gentle swing reads 1.72% < 2.5 at ref 32)
    even with no pop -> unconfirmed bounce = the gate's original job."""
    _wire5m(monkeypatch, _gentle_downswing_shape(), _m5m(px=0.85, pop=False))
    r = band_counter_breach_gate(_ctx("long", 0.82), _gate_cfg_release())
    assert r["pass"] is False
    assert "RELEASING" not in r["reason"]


def test_release_fail_closed_on_5m_fetch_error(monkeypatch):
    """Guard fetch failure = NO OPINION, never a release — the pop side is
    the gate's whole reason to exist. Must block exactly as before."""
    _wire(monkeypatch, _grass_shape())
    import hermes_trader.client.hl_client as hlc

    def _fetch(coin, interval="1h", count=200, **kw):
        if interval == "5m":
            raise RuntimeError("api down")
        return _grass_shape()

    monkeypatch.setattr(hlc, "fetch_hl_candles", _fetch)
    r = band_counter_breach_gate(_ctx("long", 0.82), _gate_cfg_release())
    assert r["pass"] is False
    assert "RELEASING" not in r["reason"]


def test_release_fail_closed_on_insufficient_5m_history(monkeypatch):
    """< lookback+1 5m bars (fresh listing) -> None -> block, not release."""
    _wire5m(monkeypatch, _grass_shape(), _m5m(px=0.85, pop=False, n_closed=10))
    r = band_counter_breach_gate(_ctx("long", 0.82), _gate_cfg_release())
    assert r["pass"] is False
    assert "RELEASING" not in r["reason"]


def test_release_mirror_short_side(monkeypatch):
    """Dip below the lower edge of an UP-drifting band (mirror shape):
    drift 5.59% >= 2.5 + no fresh 5m LOW (a prior low sits below the live
    price = grinding under the range, not popping down) -> short passes."""
    prices = [1.0 + 0.0025 * i for i in range(94)]
    prices += [prices[-1] - 0.008 * (k + 1) for k in range(6)]
    _wire5m(monkeypatch, _candles(prices), _m5m(px=0.95, pop=False, side="short"))
    r = band_counter_breach_gate(_ctx("short", 0.8), _gate_cfg_release())
    assert r["pass"] is True, r
    assert r.get("via") == "drift_confirmed"


def test_release_shadow_mode_never_releases(monkeypatch, caplog):
    """Shadow-mode contract: with shadow_mode ON the release must not fire
    AND the would-block must still log (the release is a live-execution
    escape — it cannot suppress the shadow accrual)."""
    _wire5m(monkeypatch, _grass_shape(), _m5m(px=0.85, pop=False))
    cfg = _gate_cfg_release(shadow_mode=True)
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        r = band_counter_breach_gate(_ctx("long", 0.82), cfg)
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert "via" not in r
    logged = [rec for rec in caplog.records if "would-block" in rec.getMessage()]
    assert logged, "shadow mode must log the would-block even when a live release would apply"


def test_release_disabled_key_keeps_legacy_block(monkeypatch):
    """drift_confirmed_release.enabled=False == the pre-change behaviour,
    even with a drift-confirmed no-pop shape."""
    cfg = _gate_cfg(drift_ref_span=32)
    cfg["drift_confirmed_release"] = dict(RELEASE, enabled=False)
    _wire5m(monkeypatch, _grass_shape(), _m5m(px=0.85, pop=False))
    r = band_counter_breach_gate(_ctx("long", 0.82), cfg)
    assert r["pass"] is False


def test_fresh_5m_extreme_long_pop_and_grind(monkeypatch):
    """Unit test of the guard: forming open beyond every prior 5m high =
    pop (True); a prior high above the live price = grind (False)."""
    from hermes_trader.agents.risk_gates import _fresh_5m_extreme
    px = 1.03
    pop = _m5m(px=px, pop=True)
    grind = _m5m(px=px, pop=False)
    monkeypatch.setattr(
        "hermes_trader.client.hl_client.fetch_hl_candles",
        lambda coin, interval="5m", count=50, **kw: pop,
    )
    assert _fresh_5m_extreme("X", "long", 48, px) is True
    monkeypatch.setattr(
        "hermes_trader.client.hl_client.fetch_hl_candles",
        lambda coin, interval="5m", count=50, **kw: grind,
    )
    assert _fresh_5m_extreme("X", "long", 48, px) is False


def test_fresh_5m_extreme_short_side(monkeypatch):
    from hermes_trader.agents.risk_gates import _fresh_5m_extreme
    px = 0.97
    pop = _m5m(px=px, pop=True, side="short")
    grind = _m5m(px=px, pop=False, side="short")
    monkeypatch.setattr(
        "hermes_trader.client.hl_client.fetch_hl_candles",
        lambda coin, interval="5m", count=50, **kw: pop,
    )
    assert _fresh_5m_extreme("X", "short", 48, px) is True
    monkeypatch.setattr(
        "hermes_trader.client.hl_client.fetch_hl_candles",
        lambda coin, interval="5m", count=50, **kw: grind,
    )
    assert _fresh_5m_extreme("X", "short", 48, px) is False


def test_fresh_5m_extreme_fail_closed(monkeypatch):
    """Fetch error / empty / short history -> None (no opinion), never
    a True/False the gate could misread as 'no pop'."""
    from hermes_trader.agents.risk_gates import _fresh_5m_extreme

    def _boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setattr("hermes_trader.client.hl_client.fetch_hl_candles", _boom)
    assert _fresh_5m_extreme("X", "long", 48, 1.0) is None
    monkeypatch.setattr(
        "hermes_trader.client.hl_client.fetch_hl_candles",
        lambda coin, interval="5m", count=50, **kw: [],
    )
    assert _fresh_5m_extreme("X", "long", 48, 1.0) is None
    monkeypatch.setattr(
        "hermes_trader.client.hl_client.fetch_hl_candles",
        lambda coin, interval="5m", count=50, **kw: _m5m(px=1.0, pop=False, n_closed=10),
    )
    assert _fresh_5m_extreme("X", "long", 48, 1.0) is None


# ---------------------------------------------------------------------------
# P3 — band_snapback.shadow_mode (the single umbrella flag)
#
# The gate's own shadow decision now ORs in `band_snapback.shadow_mode`, so
# the single flag the operator flips to shadow the whole band-snapback feature
# drives this gate to would-block-only even with the gate's own
# band_counter_breach_gate.shadow_mode left at its default. The gate's own key
# stays a secondary (belt-and-braces): shadow = own OR umbrella.
# ---------------------------------------------------------------------------


def _gate_cfg_no_shadow_key(**over) -> dict:
    """Gate cfg with the gate's OWN shadow_mode ABSENT -> code default True.
    (Distinct from _gate_cfg, which sets shadow_mode explicitly.)"""
    base = {"enabled": True, "min_conf": 0.9, "min_breach_pct": 1.0}
    base.update(over)
    return base


def test_gate_umbrella_shadow_drives_would_block(monkeypatch):
    """THE decisive umbrella test: band_snapback.shadow_mode=True drives the
    gate to would-block-only EVEN WHEN the gate's own shadow_mode is armed
    (False). Without the umbrella this exact shape BLOCKS (pass:False); with
    it, the single flag rescues to pass:True + shadow_would_block:True."""
    _wire(monkeypatch, _grass_shape(),
          band_cfg={**BAND_CFG, "shadow_mode": True})
    cfg = _gate_cfg(shadow_mode=False)  # gate's own armed
    r = band_counter_breach_gate(_ctx("long", 0.82), cfg)
    assert r["pass"] is True, r
    assert r.get("shadow_would_block") is True
    assert "via" not in r
    assert "GRASS long" in r["reason"]


def test_gate_umbrella_with_gate_own_at_default(monkeypatch):
    """The literal requirement: band_snapback.shadow_mode=True with the gate's
    own shadow_mode at its code default (absent -> True) -> still pass:True."""
    _wire(monkeypatch, _grass_shape(),
          band_cfg={**BAND_CFG, "shadow_mode": True})
    cfg = _gate_cfg_no_shadow_key()  # own shadow_mode absent -> default True
    r = band_counter_breach_gate(_ctx("long", 0.82), cfg)
    assert r["pass"] is True, r
    assert r.get("shadow_would_block") is True


def test_gate_umbrella_absent_keeps_armed_block(monkeypatch):
    """No-op guarantee: with band_snapback.shadow_mode ABSENT (code default
    False) and the gate's own shadow_mode armed (False), the gate BLOCKS
    exactly as before — the umbrella must never force shadow when it is off,
    and never leak a shadow_would_block marker into a hard block."""
    _wire(monkeypatch, _grass_shape())  # BAND_CFG, no shadow_mode key
    cfg = _gate_cfg(shadow_mode=False)
    r = band_counter_breach_gate(_ctx("long", 0.82), cfg)
    assert r["pass"] is False, r
    assert "shadow_would_block" not in r
    assert r["reason"].startswith("[gate:band_counter_breach]")


def test_gate_umbrella_shadow_never_releases(monkeypatch, caplog):
    """Umbrella shadow must ALSO suppress the drift-confirmed release escape
    (a live-execution release cannot fire while the umbrella shadows) and the
    would-block still accrues — mirroring the gate's own shadow contract.
    Gate's own shadow_mode armed (False) + release enabled + umbrella on:
    without the umbrella the release would return via=drift_confirmed with no
    shadow marker; with it the gate stays in the shadow would-block path."""
    _wire5m(monkeypatch, _grass_shape(), _m5m(px=0.85, pop=False),
            band_cfg={**BAND_CFG, "shadow_mode": True})
    cfg = _gate_cfg_release(shadow_mode=False)  # own armed; release enabled
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        r = band_counter_breach_gate(_ctx("long", 0.82), cfg)
    assert r["pass"] is True, r
    assert r.get("shadow_would_block") is True
    assert "via" not in r, "the drift-confirmed release must not fire under the umbrella"
    logged = [rec for rec in caplog.records if "would-block" in rec.getMessage()]
    assert logged, "shadow mode must log the would-block even when a live release would apply"