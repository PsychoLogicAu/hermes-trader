"""Tests for the band_snapback trigger (MA-band poke-out + snapback).

Pure synthetic candles, no network. Covers:
- lower wick poke + snapback in chop -> LONG fires (EMA band)
- upper wick poke + snapback in chop -> SHORT fires
- SMA band variant (ma_type="sma") fires the same way
- trending band -> stays silent (poke is continuation, not fade)
- poke with close still OUTSIDE the band (breakout) -> silent
- shallow poke below min_poke_atr -> silent
- insufficient history (fewer than 2*window bars) -> silent
- live (include_partial) vs backtest (include_partial=False) framing
- current_px override (price drifted back outside) -> silent
- weight-0: a fired bandSnapback contributes nothing to composite_score
- deeper poke scores higher
- linear 1-bar forward projection de-lags the poke judge (stale-edge
  crossing rejected, deep poke still fires, flat band unaffected)
- projection cap (max_project_atr): clamps the de-lagged delta in the
  accelerating-band tail — a wick between the capped and uncapped judge
  lines fires with the cap, stays silent with it disabled
"""

import math

from hermes_trader.indicators.triggers import band_snapback, composite_score
from hermes_trader.models.types import Candle

T0 = 1_700_000_000_000
STEP = 900_000  # 15m


def _candle(i, o, h, l, c):
    return Candle(t=T0 + i * STEP, o=o, h=h, l=l, c=c, v=1000.0)


# The trigger needs 2*window bars before the poke (window for the band edges
# + window for the drift reference). Default window=24 -> need 48 bars.
WIN = 24
NEED = 2 * WIN  # 48


def chop(n=None, mid=100.0, half=0.3):
    """Flat oscillating chop: closes pinned near the mid, wicks to +/-`half`.

    For flat chop the MA-of-highs sits at ~mid+half, MA-of-lows at ~mid-half.
    The band is essentially flat (tiny oscillation), so the drift gate passes.
    """
    n = n or NEED
    out = []
    for i in range(n):
        c = mid + (0.05 if i % 2 == 0 else -0.05)
        o = mid + (-0.05 if i % 2 == 0 else 0.05)
        out.append(_candle(i, o, c + half, c - half, c))
    return out


def test_lower_poke_snaps_back_fires_long():
    """EMA band: wick pierces the lower MA edge, close snaps back inside."""
    cs = chop()
    poke = _candle(NEED, 99.9, 100.0, 99.2, 99.85)  # wick ~0.5 below ~99.7
    hit = band_snapback(cs + [poke], window=WIN, include_partial=False)
    assert hit["fired"], hit
    assert "long" in hit["reason"], hit["reason"]
    assert "EMA" in hit["reason"], hit["reason"]
    assert hit["score"] > 0


def test_lower_poke_sma_band_fires_long():
    """SMA band variant: same geometry, ma_type='sma'."""
    cs = chop()
    poke = _candle(NEED, 99.9, 100.0, 99.2, 99.85)
    hit = band_snapback(cs + [poke], window=WIN, ma_type="sma",
                        include_partial=False)
    assert hit["fired"], hit
    assert "long" in hit["reason"]
    assert "SMA" in hit["reason"], hit["reason"]


def test_upper_poke_snaps_back_fires_short():
    """EMA band: wick pierces the upper MA edge, close snaps back inside."""
    cs = chop()
    poke = _candle(NEED, 100.1, 100.8, 99.9, 100.15)  # wick ~0.5 above ~100.3
    hit = band_snapback(cs + [poke], window=WIN, include_partial=False)
    assert hit["fired"], hit
    assert "short" in hit["reason"], hit["reason"]
    assert hit["score"] > 0


def test_trending_band_stays_silent():
    """Strong uptrend: the MA band itself drifts > max_drift_pct. A 'poke'
    below the lower band is continuation, not a fade — the drift gate vetoes."""
    cs = []
    for i in range(NEED):
        base = 100.0 + i * 0.5  # strong uptrend: 0.5 per bar
        cs.append(_candle(i, base - 0.1, base + 0.3, base - 0.5, base))
    poke = _candle(NEED, 124.5, 124.9, 123.9, 124.6)
    hit = band_snapback(cs + [poke], window=WIN, include_partial=False)
    assert not hit["fired"], hit
    assert "trending" in hit["reason"], hit["reason"]


def test_band_span_tighter_still_fires_in_chop():
    """band_span (8) << window (24): the band hugs price tightly but the poke
    still snaps back -> fires, and the reason surfaces the span override."""
    cs = chop()
    poke = _candle(NEED, 99.9, 100.0, 99.2, 99.85)
    hit = band_snapback(cs + [poke], window=WIN, band_span=8,
                        include_partial=False)
    assert hit["fired"], hit
    assert "long" in hit["reason"]
    assert "/span8" in hit["reason"], hit["reason"]


def test_band_span_tight_drift_gate_still_vetoes_trend():
    """The KEY property: a tight band_span (8) hugs the trend closely, but the
    drift gate still measures the band edge over the FULL window (24) — so a
    real trend is STILL vetoed, not faded. Shrinking band_span to cut lag does
    not weaken trend detection."""
    cs = []
    for i in range(NEED):
        base = 100.0 + i * 0.5  # strong uptrend: 0.5 per bar
        cs.append(_candle(i, base - 0.1, base + 0.3, base - 0.5, base))
    poke = _candle(NEED, 124.5, 124.9, 123.9, 124.6)
    hit = band_snapback(cs + [poke], window=WIN, band_span=8,
                        include_partial=False)
    assert not hit["fired"], hit
    assert "trending" in hit["reason"], hit["reason"]


def test_poke_without_snapback_stays_silent():
    """Wick pokes below but the CLOSE also stays below the band: breakout, not fade."""
    cs = chop()
    poke = _candle(NEED, 99.8, 99.9, 99.2, 99.55)  # close below ~99.7 band
    hit = band_snapback(cs + [poke], window=WIN, include_partial=False)
    assert not hit["fired"], hit


def test_shallow_poke_below_atr_threshold_stays_silent():
    """Wick barely kisses the band: no meaningful poke depth."""
    cs = chop()
    poke = _candle(NEED, 99.9, 100.0, 99.6, 99.9)
    hit = band_snapback(cs + [poke], window=WIN, include_partial=False)
    assert not hit["fired"], hit


def test_insufficient_history():
    """Fewer than 2*window bars before the poke -> insufficient_history."""
    hit = band_snapback(chop(10), window=WIN, include_partial=False)
    assert not hit["fired"]
    assert hit["reason"] == "insufficient_history"


def test_live_include_partial_framing():
    """Live scan: the poke is the last CLOSED bar (candles[-2]); the in-progress
    bar (candles[-1]) carries the current price."""
    cs = chop()
    poke = _candle(NEED, 99.9, 100.0, 99.2, 99.85)
    partial = _candle(NEED + 1, 99.85, 99.9, 99.8, 99.88)  # current px ~99.88
    hit_live = band_snapback(cs + [poke, partial], window=WIN, include_partial=True)
    assert hit_live["fired"], hit_live
    assert "long" in hit_live["reason"]


def test_current_px_overrides_poke_close():
    """Wick poked below and the poke close was inside, but the current price
    (current_px) has since dropped back OUTSIDE the band — snapback no longer holds."""
    cs = chop()
    poke = _candle(NEED, 99.9, 100.0, 99.2, 99.85)
    hit = band_snapback(cs + [poke], window=WIN, include_partial=False,
                        current_px=99.55)  # dropped below the ~99.7 lower edge
    assert not hit["fired"], hit


def test_deeper_poke_scores_higher():
    cs = chop()
    shallow = band_snapback(cs + [_candle(NEED, 99.9, 100.0, 99.35, 99.85)],
                            window=WIN, include_partial=False)
    deep = band_snapback(cs + [_candle(NEED, 99.9, 100.0, 99.05, 99.85)],
                         window=WIN, include_partial=False)
    assert shallow["fired"] and deep["fired"], (shallow, deep)
    assert deep["score"] > shallow["score"]
    assert shallow["score"] <= 10.0 and deep["score"] <= 10.0


def test_weight_zero_does_not_affect_composite():
    cs = chop()
    poke = _candle(NEED, 99.9, 100.0, 99.2, 99.85)
    hit = band_snapback(cs + [poke], window=WIN, include_partial=False)
    assert hit["fired"]
    weights = {"bandSnapback": 0.0, "trendStrength": 0.55}
    assert composite_score([hit], weights) == 0


def test_no_poke_at_all_stays_silent():
    """Flat chop with no wick poke: no signal."""
    cs = chop(NEED + 1)  # one extra flat bar as the 'poke'
    hit = band_snapback(cs, window=WIN, include_partial=False)
    assert not hit["fired"], hit


# --- Linear forward projection of the band edge (de-lagged poke) ----------

def test_project_band_edge_linear():
    """The pure helper: projects the last true reading one bar at the
    edge's own last gradient (edge + (edge[-1] - edge[-2]))."""
    from hermes_trader.indicators.triggers import _project_band_edge
    assert abs(_project_band_edge([10.0, 10.2, 10.4]) - 10.6) < 1e-9
    assert _project_band_edge([10.0, 10.0, 10.0]) == 10.0  # flat -> itself
    assert _project_band_edge([9.8]) == 9.8                # <2 pts -> last
    assert abs(_project_band_edge([10.4, 10.2]) - 10.0) < 1e-9  # falling edge


def _falling_ramp(n=NEED, start=103.0, step=0.05):
    """Gentle downtrend: lows fall `step`/bar (band edge has a negative
    gradient), closes sit just above the low. Drift over the window is
    ~1.1% < the 1.5% gate, so the chop gate passes — the band drifts."""
    out = []
    for i in range(n):
        lo = start - step * i
        cl = lo + 0.5
        out.append(_candle(i, cl - 0.1, cl + 0.3, lo, cl))
    return out


def test_stale_edge_crossing_is_now_rejected():
    """THE DE-LAG: on a falling band, a wick that only pokes the STALE
    (bar-before) edge was judged as a poke before; the projected edge is
    lower, so the poke no longer crosses it -> silent."""
    from hermes_trader.indicators.math import atr as _atr
    from hermes_trader.indicators.triggers import _band_ma
    cs = _falling_ramp()
    fit = cs  # the 48 bars before the poke
    lo_ma = _band_ma(fit, WIN, "ema")[1]
    stale = lo_ma[-1]
    proj = lo_ma[-1] + (lo_ma[-1] - lo_ma[-2])
    a = _atr(fit[-WIN:] + [cs[-1]], 14)[-1]
    min_depth = 0.5 * a
    assert proj < stale  # falling band: projection extends the decline
    pl = (stale - min_depth + proj - min_depth) / 2  # crosses stale zone only
    px = stale + 0.1
    poke = _candle(NEED, px, px + 0.1, pl, px)
    hit = band_snapback(cs + [poke], window=WIN, include_partial=False)
    assert not hit["fired"], hit
    assert "no snapback" in hit["reason"], hit["reason"]


def test_deeper_poke_still_fires_against_projected_edge():
    """A wick that genuinely crosses the PROJECTED edge still fires — the
    de-lag tightened the judge line, it didn't kill the signal."""
    from hermes_trader.indicators.math import atr as _atr
    from hermes_trader.indicators.triggers import _band_ma
    cs = _falling_ramp()
    lo_ma = _band_ma(cs, WIN, "ema")[1]
    proj = lo_ma[-1] + (lo_ma[-1] - lo_ma[-2])
    a = _atr(cs[-WIN:] + [cs[-1]], 14)[-1]
    min_depth = 0.5 * a
    pl = proj - min_depth - 0.1   # clearly past the projected edge
    px = proj + 0.1               # snapped back inside the projected edge
    poke = _candle(NEED, px, px + 0.1, pl, px)
    hit = band_snapback(cs + [poke], window=WIN, include_partial=False)
    assert hit["fired"], hit
    assert "long" in hit["reason"] and "projected" in hit["reason"], hit["reason"]


def test_flat_band_projection_is_a_noop():
    """Flat chop: the last two true readings are (near-)equal, so the
    projection moves the edge ~0 and existing flat-band behavior holds."""
    from hermes_trader.indicators.triggers import _project_band_edge, _band_ma
    cs = chop()
    fit = cs
    lo_ma = _band_ma(fit, WIN, "ema")[1]
    delta = _project_band_edge(lo_ma) - lo_ma[-1]
    # chop() oscillates closes +/-0.05, so the EMA-of-lows carries a tiny
    # residual wiggle (~0.004/bar). The point: the projection moves the edge
    # negligibly vs the poke depth (~0.4 ATR), so flat-band behavior holds.
    assert abs(delta) < 0.01, delta
    poke = _candle(NEED, 99.9, 100.0, 99.2, 99.85)
    hit = band_snapback(cs + [poke], window=WIN, include_partial=False)
    assert hit["fired"], hit


# --- Projection cap (max_project_atr) -------------------------------------

def _accelerating_tail(n=NEED, crash=2.5, chop=0.6):
    """Flat chop (range `chop` around 100) with the LAST fit bar crashing
    `crash` below the chop low. The crash makes the EMA-of-lows 1-bar
    projection overshoot the decline, while the drift over the window stays
    tiny (the crash is one bar) so the chop gate still passes — the exact
    'accelerating band edge' tail the cap guards against."""
    out = []
    for i in range(n - 1):
        mid = 100.0 + (0.05 if i % 2 == 0 else -0.05)
        lo = mid - chop / 2
        out.append(_candle(i, mid - 0.02, mid + 0.02, lo, mid))
    last_lo = (100.0 - chop / 2) - crash
    last_cl = last_lo + 0.3
    out.append(_candle(n - 1, last_cl + 0.1, last_cl + 0.1, last_lo, last_cl))
    return out


def test_projection_cap_trims_over_projected_signal():
    """THE CAP: after a crash bar, the RAW 1-bar projection of the lower
    edge overshoots the decline (judge line dragged to ~99.30). A poke whose
    close sits just above that over-projected line (99.3076) would falsely
    'snap back inside' and FIRE with the cap disabled. The default cap
    (0.25*ATR) pulls the judge line back toward the true reading (~99.32),
    so the same poke correctly stays SILENT. Proves the cap is conservative:
    it trims the aggressive accelerating-band tail and never ADDS signals.
    (Geometry verified end-to-end with the trigger as oracle; the cap is
    provably binding here: |raw delta| 0.20 > cap 0.18.)"""
    from hermes_trader.indicators.math import atr as _atr
    from hermes_trader.indicators.triggers import _band_ma, _project_band_edge
    fit = _accelerating_tail()
    lo_ma = _band_ma(fit, WIN, "ema")[1]
    stale, raw = lo_ma[-1], _project_band_edge(lo_ma)
    # the test premise itself: the raw projection overshoots the decline by
    # MORE than the cap, so the clamp actually binds
    poke = _candle(NEED, 99.3076, 99.3076, fit[-1].l - 0.9, 99.3076)
    a = _atr(fit[-WIN:] + [poke], 14)[-1]
    assert raw < stale - 0.25 * a, (raw, stale, a)  # cap binds
    # and the drift gate passes (one-bar crash, tiny window drift)
    assert abs(stale - lo_ma[-1 - WIN]) / poke.c * 100 < 1.5

    # default (cap ON): the over-projected false 'snapback' is trimmed
    hit_capped = band_snapback(fit + [poke], window=WIN, include_partial=False)
    assert not hit_capped["fired"], hit_capped
    # cap OFF: the raw projection fires the false positive
    hit_raw = band_snapback(fit + [poke], window=WIN, include_partial=False,
                            max_project_atr=None)
    assert hit_raw["fired"] and "long" in hit_raw["reason"], hit_raw