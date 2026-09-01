"""Deep-analysis pipeline: perception -> multi-timeframe indicators -> AI verdict -> persist."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import httpx

from hermes_trader.agents.config_store import read_agent_config
from hermes_trader.agents.duel_store import (
    call_duelist,
    duelist_config,
    duelist_enabled,
    record_duel,
    resolve_max_tokens,
)
from hermes_trader.agents.memory import memory
from hermes_trader.agents.system_prompt import build_system_prompt
from hermes_trader.client.hl_client import (
    fetch_account_state,
    fetch_funding_history,
    fetch_hl_candles,
    resolve_user_address,
)
from hermes_trader.indicators.math import adx, atr, candle_val, ema, rsi
from hermes_trader.models.types import Candle

logger = logging.getLogger(__name__)


def _compute_indicators(candles: List[Candle]) -> Dict[str, Any]:
    """Compute EMA8/21, RSI, ATR, ADX for a set of candles."""
    if not candles:
        return {
            "ema8": None, "ema21": None, "slope_up": None,
            "rsi14": None, "atr14": None, "adx14": None,
            "last_close": 0, "last_time": 0,
        }

    closes = [candle_val(c, "c") for c in candles]

    if len(closes) < 21:
        return {
            "ema8": None, "ema21": None, "slope_up": None,
            "rsi14": None, "atr14": None, "adx14": None,
            "last_close": closes[-1],
            "last_time": candles[-1].t,
        }

    ema8_arr = ema(closes, 8)
    ema21_arr = ema(closes, 21)

    last_ema8 = ema8_arr[-1] if ema8_arr else None
    last_ema21 = ema21_arr[-1] if ema21_arr else None

    slope_up = None
    if last_ema8 is not None and len(ema8_arr) >= 3:
        slope_up = last_ema8 > ema8_arr[-3]

    rsi_arr = rsi(candles, 14)
    atr_arr = atr(candles, 14)
    adx_arr = adx(candles, 14)

    return {
        "ema8": last_ema8 if last_ema8 is not None and math.isfinite(last_ema8) else None,
        "ema21": last_ema21 if last_ema21 is not None and math.isfinite(last_ema21) else None,
        "slope_up": slope_up,
        "rsi14": rsi_arr[-1] if rsi_arr and math.isfinite(rsi_arr[-1]) else None,
        "atr14": atr_arr[-1] if atr_arr and math.isfinite(atr_arr[-1]) else None,
        "adx14": adx_arr[-1] if adx_arr and math.isfinite(adx_arr[-1]) else None,
        "last_close": closes[-1],
        "last_time": candles[-1].t,
    }


def _fetch_funding_rate(coin: str) -> str:
    """Latest hourly funding rate for a coin, or 'N/A' if unavailable."""
    start_time = int(time.time() * 1000) - 86_400_000
    history = fetch_funding_history(coin, start_time)
    if history:
        rate = float(history[-1].get("fundingRate", "0"))
        if math.isfinite(rate):
            return f"{rate * 100:.4f}%/hr"
    return "N/A"


# Only surface news from the last N days. Without this, Brave returned
# year-old articles (e.g. AIXBT's 2025 hack) that then tripped the binary-news
# gate on a fresh 2026 trade. The gate reasons about *imminent* event risk, so
# stale headlines are noise — both to the gate and to the LLM prompt.
NEWS_FRESHNESS_DAYS = 2


def _fetch_news(coin: str) -> str:
    """Recent (last NEWS_FRESHNESS_DAYS) news headlines for a coin via the
    Brave Search API.

    Returns a compact ' | '-joined headline string, or 'no news' when no
    BRAVE_API_KEY is set or the request fails — news is a supplementary
    signal, so a fetch failure degrades gracefully and never blocks research.
    """
    key = os.environ.get("BRAVE_API_KEY", "")
    if not key:
        return "no news"
    # Brave `freshness` takes a YYYY-MM-DDtoYYYY-MM-DD range; a 2-day window
    # approximates "within 48h" (the closest the API offers to an hour-precise
    # bound without per-result age filtering).
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=NEWS_FRESHNESS_DAYS)
    freshness = f"{start.isoformat()}to{today.isoformat()}"
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/news/search",
            params={"q": f"{coin} crypto", "count": 5, "freshness": freshness},
            headers={"X-Subscription-Token": key, "Accept": "application/json"},
            timeout=10.0,
        )
        if not resp.is_success:
            return "no news"
        results = resp.json().get("results", []) or []
        headlines = [str(r.get("title", "")).strip() for r in results if r.get("title")]
        return " | ".join(headlines[:5]) if headlines else "no news"
    except Exception:
        return "no news"


def _signals_block(coin: str) -> str:
    """Free positioning signals (GEX / aggTrades whale / FINRA short-vol / news
    catalyst) formatted for the AI prompt. The executor already uses these, but
    the AI was BLIND to them — so it kept PASSing rippers it couldn't see the
    bullish context for. Surfacing them here lets the AI's own verdict reflect
    them. Wrapped so any signal outage never breaks research."""
    is_hip3 = ":" in (coin or "")
    lines: List[str] = []
    try:
        if is_hip3:
            from hermes_trader.agents.options_gex import gex_signal_cached
            g = gex_signal_cached(coin)
            if g:
                lines.append(
                    f"  - Dealer gamma (GEX): {g.regime}; call wall {g.call_wall} (overhead "
                    f"resistance / ride target), put wall {g.put_wall} (support), spot {g.spot:g}. "
                    + ("Negative gamma = squeeze-prone, lets moves RUN."
                       if g.regime == "trend_short_gamma"
                       else "Long gamma = pins/mean-reverts near the walls.")
                )
            from hermes_trader.agents.short_volume import short_volume_signal
            sv = short_volume_signal(coin)
            if sv:
                lines.append(
                    f"  - Short volume (FINRA): {sv.ratio * 100:.0f}% ({sv.regime}, {sv.trend})."
                    + (" Crowded short = SQUEEZE FUEL for a long."
                       if sv.regime == "crowded_short_squeeze_fuel" else "")
                )
        else:
            from hermes_trader.agents.crypto_whale import crypto_whale_signal
            w = crypto_whale_signal(coin, window_minutes=15)
            if w and w.whale_n:
                lines.append(
                    f"  - Whale order-flow (Binance aggTrades, 15m): {w.bias}, net "
                    f"${w.net_usd:+,.0f} across {w.whale_n} large prints."
                    + (" Large buyers stepping in (bullish)." if w.bias == "whale_buying"
                       else " Large sellers hitting bids (bearish)." if w.bias == "whale_selling"
                       else "")
                )
        from hermes_trader.agents.news_catalyst import catalyst_scan
        base = coin.split(":", 1)[1] if ":" in coin else coin
        n = catalyst_scan(base, timespan="1h")
        if n and (n.breaking or n.surge_x >= 1.5):
            top = n.headlines[0].title[:90] if n.headlines else ""
            lines.append(
                f"  - News catalyst: {'BREAKING' if n.breaking else f'elevated ({n.surge_x}x coverage)'}"
                f" — {top!r}"
            )
    except Exception as e:
        logger.debug(f"[research] signals block failed for {coin}: {e}")
    if not lines:
        return "Positioning signals (GEX/whale/short-vol/news): none flagged"
    return ("Positioning signals (free data — weigh these in your verdict):\n"
            + "\n".join(lines))


def _chronos_block(coin: str) -> str:
    """Chronos-2 forward forecast for the AI prompt, when
    `chronos_signal.in_prompt` is true. Calls the sync wrapper so the line is
    DETERMINISTIC per coin: it renders whenever the signal is enabled and the
    model is loaded (the model is preloaded at app init; steady-state cost is
    ~200ms/coin, dominated by the HL candle fetch). A disabled flag or a
    failed/absent forecast returns '' so the prompt shape is unchanged.

    The `median` figure is the PATH AVERAGE over the horizon (the mean of the
    12 step-by-step forecasts), NOT the point price at the horizon — the
    2026-08-28 60-flag replay showed the median path mean-reverts (TRUMP's
    q50 dipped to -1.4% and recovered to -0.6% by step 12), so the endpoint
    is nearly useless as a read. The replay ranked the reductions by loss
    avoidance: EARLY ADVERSE TAIL > skew > path-mean. This block therefore
    also renders the early tail — the extreme the adverse quantile reaches
    within the first 6 steps (~30 min), the same shape the
    `chronos_tail_trigger_gate` consumes — so the LLM can see the stop-risk
    the median understates (the replay's q10 tail overstated realized
    magnitude ~2x, so treat it as a fat-tail RISK flag, not a magnitude).

    Labeled as a DECAY/continuation warning on purpose: the bot's exit policy is
    scalp, so the horizon is LONGER than the hold. A negative call means the
    move is likely to fade over the window — drop conviction for a scalp, don't
    auto-PASS a confirmed fresh breakout. A positive call means the model sees
    continuation over the window — supports holding through noise.
    """
    try:
        cfg = read_agent_config().get("chronos_signal", {})
        if not cfg.get("in_prompt", False) or not cfg.get("enabled", False):
            return ""
        # Lazy import keeps the torch/chronos dep off the module-load path;
        # module-attr call (not `from ... import`) so tests can patch the sync
        # wrapper. The model is preloaded at init, so this rarely pays more
        # than the ~200ms candle fetch.
        from hermes_trader.agents import chronos_signal as _cs
        sig = _cs.get_chronos_signal_sync(coin, "long")
        if sig is None or sig.error or sig.median_pct is None:
            return ""
        med = sig.median_pct
        low = sig.q_low
        high = sig.q_high
        # Quantiles are absolute prices; convert to % vs context_last for the
        # prompt (the log line uses context_last too, so both are comparable).
        if low is not None and high is not None and sig.context_last:
            lo_pct = (low - sig.context_last) / sig.context_last * 100
            hi_pct = (high - sig.context_last) / sig.context_last * 100
            span = f", p10 avg {lo_pct:+.1f}% / p90 avg {hi_pct:+.1f}%"
        else:
            span = ""
        # Early adverse tail: the extreme the adverse quantile reaches within
        # the first 6 steps (~30 min) — the replay's most loss-avoiding read,
        # and the same value the tail-trigger gate consumes. The forecast is
        # side-independent (the cache is coin-keyed), so render both
        # directions: p10 path min (long-side stop risk) and p90 path max
        # (short-side stop risk).
        tail_bits = []
        _tail_steps = 6
        p10p = sig.q10_path_pct or []
        p90p = sig.q90_path_pct or []
        if p10p[:_tail_steps]:
            tail_bits.append(f"p10 min {min(p10p[:_tail_steps]):+.1f}%")
        if p90p[:_tail_steps]:
            tail_bits.append(f"p90 max {max(p90p[:_tail_steps]):+.1f}%")
        tail_s = (
            f"; early tail, first {_tail_steps * 5}m (stop-risk read — the "
            f"median understates it): {' / '.join(tail_bits)}"
            if tail_bits else ""
        )
        # Chronos runs on 5m candles; horizon bars * 5m = the forward window.
        hours = sig.horizon * 5 / 60
        hours_s = f"{hours:.0f}" if hours == int(hours) else f"{hours:g}"
        # Confidence floor (2026-08-30 HEMI replay): the FADE/continuation
        # notes are interpretive CLAIMS, so they only render when the median
        # clears a quarter of the model's own p10-p90 band (min_conf_ratio).
        # All 8 hourly HEMI forecasts at the live config were under the floor
        # — three of them rendered the full FADE warning at |median| 0.08–1.70%
        # inside 5–8% bands, and the LLM leaned on those for six hours of a
        # +38% rip. Below the floor the note says no-confident-direction and
        # the data lines above still carry the numbers for the LLM's own
        # judgement. Fail-safe: no spread -> ratio 0.0 -> neutral note
        # (never claim a fade we have no basis for).
        from hermes_trader.agents import chronos_signal as _cs2
        min_conf_ratio = _cs2.resolve_min_conf_ratio(cfg)
        ratio = _cs2.confidence_ratio(sig)
        if ratio < min_conf_ratio:
            spread_s = (f"{sig.spread_pct:.1f}%"
                        if sig.spread_pct is not None else "no band")
            note = (f"the model has no confident direction over the next ~{hours_s}h — the "
                    f"median ({med:+.2f}%) sits inside its own p10-p90 band "
                    f"({spread_s}, ratio {ratio:.2f} < {min_conf_ratio:.2f}); "
                    "treat it as noise, not a fade or continuation.")
        elif med > 0:
            note = (f"the model sees continuation for the next ~{hours_s}h — supports "
                    "holding through pullbacks, favours swing over scalp conviction.")
        elif med < 0:
            note = (f"the model expects the move to FADE within ~{hours_s}h — this is a "
                    "decay/continuation warning: lower conviction on late entries, "
                    f"prefer a tight scalp TP over holding for the {hours_s}h horizon; it is "
                    "NOT an instruction to auto-PASS a confirmed fresh breakout.")
        else:
            note = f"the model is directionally neutral over the next ~{hours_s}h."
        return (
            "Chronos forecast (shadow signal — weigh, don't obey):\n"
            f"  - Path-average price over the next ~{hours_s}h: {med:+.2f}%{span}{tail_s}. "
            + note
        )
    except Exception as e:
        logger.debug(f"[research] chronos block failed for {coin}: {e}")
        return ""


def _squeeze_block(coin: str) -> str:
    """Squeeze-breakout (1h Donchian) state for the AI prompt, when
    `squeeze_signal` is enabled. Same determinism contract as _chronos_block:
    sync, cache-first read (the executor's attach + gate-side read share the
    300s per-coin cache, so this is normally a pure dict read); '' on disabled
    / failed / absent data so the prompt shape is unchanged.

    Rendered as CONTEXT to weigh, not a trigger: the breakout rule's OOS edge
    came from FRESH, aligned breakouts, and the research's worst-loss bucket
    was same-side re-entries AT THE EXTREME with no fresh breakout
    ("chasing without confirmation"). The `extreme_no_breakout` flag is the
    deterministic encoding of that bucket and is also the input the
    squeeze_extreme shadow gate consumes — the prompt surfaces it so the LLM
    sees the same shape the gate does.
    """
    try:
        cfg = read_agent_config().get("squeeze_signal", {})
        if not cfg.get("enabled", False):
            return ""
        from hermes_trader.agents.squeeze_signal import get_squeeze_signal_sync
        sig = get_squeeze_signal_sync(coin, "long")
        if sig is None:
            return ""
        lines = ["Squeeze / 48h channel state (1h Donchian — weigh, don't obey):"]
        if sig.active:
            side_word = "long" if sig.side == "long" else "short"
            lines.append(
                f"  - FRESH {side_word.upper()} breakout: the last confirmed 1h close "
                f"broke the prior 48h {'high' if sig.side == 'long' else 'low'} by "
                f"{sig.ext_pct:+.2f}% ({sig.fresh_age_min:.0f}m ago, decisive body). "
                "This is the shape the breakout rule rewards — a fresh aligned "
                "breakout CONFIRMS a same-side entry at the extreme."
            )
        else:
            lines.append(f"  - No fresh 1h breakout (last check: {sig.error}).")
        if sig.chan_pos is not None:
            pos = sig.chan_pos
            if pos > 1.0:
                zone = "ABOVE the 48h channel high"
            elif pos < 0.0:
                zone = "BELOW the 48h channel low"
            elif pos > 0.95:
                zone = f"at the very top of the 48h range ({pos:.0%})"
            elif pos < 0.05:
                zone = f"at the very bottom of the 48h range ({pos:.0%})"
            else:
                zone = f"mid-range (48h position {pos:.0%})"
            lines.append(f"  - Price now {zone}.")
        if sig.extreme_no_breakout:
            lines.append(
                "  - CAUTION: the candidate side is at the channel extreme with NO "
                "fresh breakout confirming it — the 'chasing without confirmation' "
                "shape the 15-day ledger's worst losses came from. Down-weight "
                "conviction on a same-side late entry; a fresh aligned breakout "
                "(above) is what distinguishes continuation from the chase."
            )
        return "\n".join(lines)
    except Exception as e:
        logger.debug(f"[research] squeeze block failed for {coin}: {e}")
        return ""


def _build_user_message(
    coin: str,
    perception: Dict[str, Any],
    tf1h: Dict[str, Any],
    tf4h: Dict[str, Any],
    tf1d: Dict[str, Any],
    funding_rate: str,
    news: str,
    equity: float,
    open_positions: List[Dict[str, Any]],
    mode: str,
    dex_equity: Dict[str, float] | None = None,
    recent_candles: List[Candle] | None = None,
) -> str:
    """Build the user message passed to the LLM."""
    trigger_summary = (
        ", ".join(
            f"{t['name']}: {t['reason']}"
            for t in perception.get("triggers", [])
            if t.get("fired")
        )
        or "no triggers fired"
    )

    # 1h-structure block — accumulation/exhaustion patterns the multi-tf
    # indicator blocks miss. Surfaced as an ENTRY-TIMING signal to be combined
    # WITH the 4h/1d trend, not as a reason to trade against it: in an uptrend a
    # 1h accumulation times a long pullback-entry; in a downtrend a 1h bounce is
    # a short entry (sell the rip), NOT a counter-trend dip-buy.
    _slow_burn_names = {"volumeBuildup1h", "trendFlip1h", "higherLows1h"}
    slow_burn_hits = [
        t for t in perception.get("triggers", [])
        if t.get("name") in _slow_burn_names and t.get("fired")
    ]
    if slow_burn_hits:
        structure_lines = ["1h structure signals (entry-timing — apply IN the 4h/1d trend direction):"]
        for t in slow_burn_hits:
            structure_lines.append(f"  - {t['name']}: {t['reason']}")
        structure_lines.append(
            "Use these to time the entry, not to pick the side. If 4h/1d are bullish, this is a "
            "long pullback-entry; if 4h/1d are bearish, a 1h pop is a SHORT entry (sell the rip) — "
            "do not buy the dip into a downtrend."
        )
        structure_block = "\n".join(structure_lines)
    else:
        structure_block = "1h structure signals: none fired (no accumulation / breakout setup detected)"

    # Whale-accumulation block: oi_funding_anomaly flag (deep-negative funding +
    # flat price + high OI = whales loading while retail shorts). When present
    # this is a strong LONG-bias signal — don't fight it.
    whale = perception.get("whale_signal")
    if whale:
        whale_block = (
            "Whale accumulation flag (oi_funding_anomaly):\n"
            f"  - funding rate: {whale.get('funding_rate', 0):.6f} (deeply negative = retail shorting)\n"
            f"  - 24h price change: {whale.get('price_24h_change_pct', 0):+.2f}% (relatively flat)\n"
            f"  - open interest: ${whale.get('oi', 0):,.0f}\n"
            f"  - confidence: {whale.get('confidence', 0):.2f}\n"
            "Interpretation: smart money is building long positions while retail pays them "
            "to short. When the shorts cover, price tends to squeeze UP. Bias LONG unless "
            "structure is overwhelmingly bearish."
        )
    else:
        whale_block = "Whale accumulation flag: not flagged for this coin"

    # Band-snapback COUNTER-signal block: a wick that poked OUT of the
    # moving-average band and snapped back INSIDE is a short-term mean-reversion
    # tell in the fade direction (lower poke + snap-back → LONG; upper poke +
    # snap-back → SHORT). Fed to the LLM as a VETO/weighting signal, NOT a
    # standalone entry. The OOS edge is better than a coin flip but fee-thin, so
    # its value is telling the LLM "don't fade this" — the observed dumb-trade
    # mode was the LLM taking the OPPOSITE side of an imminent snapback (e.g.
    # shorting into a lower-poke snap-back that wanted to revert long).
    _snap = next((t for t in perception.get("triggers", [])
                  if t.get("name") == "bandSnapback"), None)
    if _snap and _snap.get("fired"):
        _snap_reason = _snap.get("reason", "")
        _snap_side = "SHORT" if _snap_reason.startswith("short") else "LONG"
        _opp = "LONG" if _snap_side == "SHORT" else "SHORT"
        snapback_block = (
            "Band snapback counter-signal (a wick poked OUT of the moving-average band and "
            f"snapped back INSIDE → a short-term mean reversion is set up to the {_snap_side}):"
            f"\n  - signal detail: {_snap_reason}"
            "\n  - How to weight it: this is a COUNTER-signal, not an entry. Its out-of-sample "
            "backtest edge is better than a coin flip but thin (fees eat most of it as a "
            f"standalone trade). If your own analysis is leaning the OPPOSITE side (you're "
            f"about to go {_opp} while the snapback points {_snap_side}), treat that as a red "
            "flag: you'd be fading an imminent mean-reversion. Require strong independent "
            "confirmation (higher-TF trend, funding, structure) before taking that opposite "
            f"side. Use this signal to raise confidence in the {_snap_side} direction and to "
            "lower confidence in the opposite one — not to open a trade by itself."
        )
    elif _snap and _snap.get("reason", "").startswith("band trending"):
        # Silent but INFORMATIVE: the drift gate vetoes the fade because the
        # band itself is trending. That direction is exactly the context the
        # LLM was missing on counter-trend scalps (e.g. shorting while the 1h
        # band is drifting up with price at its lower edge = fading the local
        # structure). Render it as band context, with the counter-trend
        # caveat, rather than hiding it behind "not present".
        _bt_reason = _snap.get("reason", "")
        _bt_dir = "DOWN" if "trending DOWN" in _bt_reason else "UP"
        # Extension check: when price sits BEYOND the drift-side edge (drift
        # UP -> above the upper edge; drift DOWN -> below the lower edge), a
        # new entry ON the drift side is chasing the extension — the nearest
        # mean-reversion move is back toward the band, i.e. AGAINST the entry.
        # This is the mirror image of the counter-trend case above (LIT short
        # into an up-drift) and the exact shape of the 00:46 TRUMP short,
        # which entered with the 1h band 5.1% below its lower edge.
        _ext_note = ""
        _m = re.search(
            r"px\s+([+-]?\d+(?:\.\d+)?)%\s+vs\s+upper\s+edge,\s+"
            r"([+-]?\d+(?:\.\d+)?)%\s+vs\s+lower\s+edge", _bt_reason)
        if _m:
            _px_up, _px_lo = float(_m.group(1)), float(_m.group(2))
            _ext = _px_up if _bt_dir == "UP" else -_px_lo
            if _ext >= 1.0:
                _ext_note = (
                    f"\n  - price is {_ext:.1f}% BEYOND the drift-side band edge "
                    f"(extended past the band in the {_bt_dir} direction): a NEW entry on the "
                    f"{_bt_dir} side here is chasing the extension — the nearest "
                    "short-term reversion is back toward the band, so your stop sits on the "
                    "far side of the move. Prefer waiting for a retest of the band edge, or "
                    "demand a tight invalidation and favorable risk/reward (stop distance < "
                    "target distance).")
            else:
                # Counter-trend edge breach — the MIRROR the drift-side note above misses:
                # the band drifts ONE way but price is extended PAST THE OPPOSITE edge.
                # 1h band trending DOWN while price bounces ABOVE the upper edge = top of a
                # relief rally (the start-of-downswing shape); a NEW LONG there buys the
                # bounce against the drift and the nearest reversion is back toward the band.
                # This is the exact shape the LLM misread as "drift already priced in →
                # continuation" (GRASS long 2026-08-26 19:07, 1h band DOWN 6.4% drift,
                # price +6.6% above the upper edge).
                if _bt_dir == "DOWN":
                    _ct_ext, _ct_edge = _px_up, "upper"
                    _ct_shape = ("a bounce ABOVE the upper edge of a DOWN-drifting band — the "
                                 "TOP of a relief rally, i.e. the START of the next down-swing, "
                                 "not a continuation")
                else:  # UP
                    _ct_ext, _ct_edge = -_px_lo, "lower"
                    _ct_shape = ("a dip BELOW the lower edge of an UP-drifting band — the BOTTOM "
                                 "of a pullback, i.e. the START of the next up-swing, not a "
                                 "continuation")
                if _ct_ext >= 1.0:
                    _ext_note = (
                        f"\n  - price is {_ct_ext:.1f}% beyond the {_ct_edge} band edge on the "
                        f"OPPOSITE side of the band's drift: this is {_ct_shape}. The band's "
                        f"short-term mean-reversion pressure points WITH the drift ({_bt_dir}), "
                        f"so price extended past the {_ct_edge} edge is OVER-extended against the "
                        "drift — a NEW entry on the bounce/dip side here is a counter-trend chase, "
                        "and the nearest short-term reversion is back toward the band, AGAINST the "
                        "entry (stop on the far side of the move). Do NOT read 'price is beyond "
                        f"the band edge' as 'the {_bt_dir} drift is already priced in → "
                        "continuation': it is the OPPOSITE — the extension is what reverts. Prefer "
                        "waiting for the band edge to be reclaimed or a confirmed retest, or "
                        "explicit reversal structure, before entering the opposite-of-drift side.")
        snapback_block = (
            "Band context (no snapback signal — the MA band is TRENDING, which vetoes the "
            f"short-term fade in this band's timeframe):\n  - band state: {_bt_reason}"
            f"\n  - How to read it: the band's short-term mean-reversion pressure points WITH "
            f"the drift ({_bt_dir}). A position AGAINST that direction is a counter-trend "
            "scalp — it only pays when the local wick wins against an established drift. If "
            "your thesis is on the opposite side of the band drift, require explicit "
            "counter-trend evidence (reversal structure, divergence, funding flip) and a tight "
            "stop; do not size it as a trend position."
            + _ext_note
        )
    else:
        _other_reason = (_snap or {}).get("reason", "")
        snapback_block = (
            "Band snapback counter-signal: not present (no wick poke-and-snapback in the band "
            f"timeframe this scan"
            + (f"; band state: {_other_reason}" if _other_reason and _other_reason != "flat" else "")
            + ")"
        )

    def _fmt_px(p: float) -> str:
        """Adaptive precision so sub-cent coins (HMSTR at $0.000173 etc.) don't
        all read as '0.0002' to the LLM. Without this the AI returned identical
        entry/sl/tp on cheap coins because the prompt rounded them to the same
        4-decimal value."""
        if p == 0:
            return "0"
        ap = abs(p)
        if ap >= 1:
            return f"{p:.4f}"
        if ap >= 0.01:
            return f"{p:.5f}"
        if ap >= 0.0001:
            return f"{p:.6f}"
        return f"{p:.8f}"

    def _indicator_block(label: str, snap: Dict[str, Any]) -> str:
        parts = []
        if snap.get("ema8") is not None and snap.get("ema21") is not None:
            direction = "bullish" if snap["ema8"] > snap["ema21"] else "bearish"
            parts.append(
                f"EMA8={_fmt_px(snap['ema8'])}, EMA21={_fmt_px(snap['ema21'])}, {direction}"
            )
        if snap.get("slope_up") is not None:
            parts.append(f"EMA8 slope: {'rising' if snap['slope_up'] else 'falling'}")
        if snap.get("rsi14") is not None:
            parts.append(f"RSI(14)={snap['rsi14']:.1f}")
        if snap.get("atr14") is not None:
            parts.append(f"ATR(14)={_fmt_px(snap['atr14'])}")
        if snap.get("adx14") is not None:
            parts.append(f"ADX(14)={snap['adx14']:.1f}")
        parts.append(f"last close={_fmt_px(snap.get('last_close', 0))}")
        return f"{label}: {' | '.join(parts)}"

    # Only the coins/sides we already hold — purely so the LLM doesn't
    # double-trade a coin or can CLOSE one. Deliberately NO dollar sizes:
    # account notional/leverage must not influence the verdict (sizing and
    # every risk cap live in the execution gates, not here).
    position_block = (
        "Open positions (do not re-enter these; CLOSE only if structure flipped): "
        + ", ".join(f"{p['coin']} {p['side']}" for p in open_positions)
        if open_positions
        else "Open positions: none"
    )

    # Raw recent price action so the LLM can read candlestick/chart patterns
    # directly (shooting star, hammer, engulfing, flags) — the indicator blocks
    # above summarize away the candle bodies/wicks that patterns live in.
    def _ohlc_block(candles: List[Candle] | None, n: int = 12) -> str:
        if not candles:
            return ""
        rows = []
        for i, c in enumerate(candles[-n:]):
            idx = -(len(candles[-n:]) - i)  # ... -2, -1 (newest = last closed)
            o, h, l, cl = (candle_val(c, k) for k in ("o", "h", "l", "c"))
            rows.append(f"  [{idx:>3}] O={_fmt_px(o)} H={_fmt_px(h)} L={_fmt_px(l)} C={_fmt_px(cl)}")
        return ("Recent 1h candles (oldest→newest, last row = most recent closed bar):\n"
                + "\n".join(rows))

    ohlc_block = _ohlc_block(recent_candles)

    return "\n".join([
        f"Candidate: {coin} (HL {perception.get('type', 'perp')}-PERP)",
        f"Current mid: ${_fmt_px(perception.get('mid', 0))}",
        f"Perception score: {perception.get('composite_score', 0)}/100",
        f"Fired triggers: {trigger_summary}",
        "",
        "Market context (multi-timeframe):",
        _indicator_block("1h", tf1h),
        _indicator_block("4h", tf4h),
        _indicator_block("1d", tf1d),
        "",
        ohlc_block,
        "" if ohlc_block else "",
        structure_block,
        "",
        whale_block,
        "",
        snapback_block,
        "",
        _signals_block(coin),
        _chronos_block(coin),
        _squeeze_block(coin),
        "",
        f"Funding rate (latest): {funding_rate}",
        f"Recent news: {news}",
        position_block,
        "",
        f"Mode: {mode} — {'your verdict will execute against real funds' if mode == 'LIVE' else 'analysis only, no execution'}",
        "",
        'Respond with 3-5 bullet points of reasoning, then output your decision as VALID JSON on the very last line:',
        '{"verdict":"PASS"|"LONG"|"SHORT"|"CLOSE","confidence":0.0-1.0,"side":"long"|"short"|"null","entryPx":number,"stopPx":number,"tpPx":number,"reasoning":"brief"}',
        "Nothing after the JSON.",
    ])


def _call_ai(
    system_prompt: str,
    user_message: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """Call the LLM API (runs the async client in a fresh event loop).

    Endpoint resolution: explicit args win, otherwise the primary LLM_* env
    vars. The duelist (duel_store.call_duelist) uses its own LLM_DUEL_* values
    through this same code path — the prompt is byte-identical, which is what
    makes the A/B verdicts comparable.
    """
    if api_key is None:
        api_key = os.environ.get("LLM_API_KEY", os.environ.get("OPENROUTER_API_KEY", ""))
    if base_url is None:
        base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    if model is None:
        model = os.environ.get("LLM_MODEL", os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4.3"))

    if not api_key:
        logger.warning("[research] LLM_API_KEY not set — returning empty response")
        return ""

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_async_do_call(api_key, base_url, model, system_prompt, user_message))
    except httpx.TimeoutException:
        # Single retry on timeout (2026-08-27, user-requested), matching the
        # duelist behavior. A ReadTimeout here means the server was still
        # generating at 120s — either a genuinely slow healthy call or a
        # runaway; one identical retry is the whole policy (same budget,
        # same params), bounded at ~2 x 120s. The second timeout falls
        # through to the generic handler below. This preserves the
        # historical contract — _call_ai never raises, a failed call yields
        # "" and parse_verdict's ai_down PASS — while converting a
        # transiently-busy server from a lost coin to a recovered one.
        logger.warning(
            f"[research] LLM call timed out after 120s — retrying once "
            f"(server may be mid-generation)"
        )
        try:
            return loop.run_until_complete(_async_do_call(api_key, base_url, model, system_prompt, user_message))
        except httpx.TimeoutException:
            logger.warning(
                f"[research] LLM call TIMED OUT on both attempts (~240s total) — "
                f"coin defaults to PASS ai_down this cycle"
            )
            return ""
    except Exception as e:  # noqa: BLE001 — research worker must survive any LLM fault
        # Non-timeout fault (ConnectionError to a dead endpoint, bad JSON,
        # …). LOUD: previously these returned "" via the same silent path
        # as a 402 — a dead LLM endpoint looked like "no setups".
        logger.warning(f"[research] LLM call failed (non-fatal): {type(e).__name__}: {e}")
        return ""
    finally:
        loop.close()


def _duelist_verdict(
    system_prompt: str,
    user_message: str,
    coin: str,
    perception: Dict[str, Any],
    primary_verdict: str,
    primary_confidence: float,
    primary_ms: int = 0,
) -> Dict[str, Any] | None:
    """Run the A/B duelist: the SAME prompt to the second model, recorded but
    never used. Returns the parsed verdict dict (for the session-log event +
    the entry-context snapshot), or None when the duelist is disabled/failed.

    Best-effort end to end: a duelist outage logs a warning and returns None —
    the primary path above has already produced the verdict that executes, so
    nothing downstream depends on this.
    """
    try:
        if not duelist_enabled():
            return None
        cfg = duelist_config()
        _dl_t0 = time.monotonic()
        dl_text = call_duelist(cfg["api_key"], cfg["base_url"], cfg["model"],
                               system_prompt, user_message,
                               max_tokens=cfg["max_tokens"])
        duelist_ms = int((time.monotonic() - _dl_t0) * 1000)
        # Empty text = the duelist call failed (402/429/timeout — call_duelist
        # swallows errors and returns ""). parse_verdict would tag it
        # verdict=PASS/ai_down, which would log a bogus "AGREE" row against a
        # primary that said PASS. A failed duelist is "no observation", not an
        # opinion — record nothing.
        if not (dl_text or "").strip():
            logger.warning(
                f"[duel] {coin}: duelist {cfg['model']} returned no text — not recorded"
            )
            return None
        dl_parsed = parse_verdict(dl_text, coin, perception)
        row = {
            "coin": coin,
            "perception_id": perception.get("id", "unknown"),
            "mode": str(read_agent_config().get("mode", "OFF")),
            "primary_model": os.environ.get("LLM_MODEL", os.environ.get("OPENROUTER_MODEL", "")),
            "duelist_model": cfg["model"],
            "primary_verdict": primary_verdict,
            "primary_confidence": primary_confidence,
            "duelist_verdict": dl_parsed["verdict"],
            "duelist_confidence": dl_parsed["confidence"],
            "duelist_side": dl_parsed["side"],
            "duelist_reasoning": (dl_parsed["reasoning"] or "")[:300],
            # Wall time of each LLM call (ms) — the same prompt to both
            # models, so latency is a directly comparable A/B dimension
            # (a faster model with equal accuracy can change cycle budget).
            "primary_ms": int(primary_ms or 0),
            "duelist_ms": duelist_ms,
        }
        record_duel(row)
        logger.info(
            f"[duel] {coin}: primary {row['primary_model']} {primary_verdict} "
            f"(conf {primary_confidence:.2f}, {row['primary_ms']}ms) "
            f"vs duelist {row['duelist_model']} {row['duelist_verdict']} "
            f"(conf {row['duelist_confidence']:.2f}, {row['duelist_ms']}ms) — "
            f"{'AGREE' if row['duelist_verdict'] == primary_verdict else 'SPLIT'}"
        )
        return row
    except Exception as e:  # noqa: BLE001 — the primary verdict is already in hand
        logger.warning(f"[duel] duelist hook failed for {coin} (non-fatal): {e!r}")
        return None


async def _async_do_call(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_message: str,
) -> str:
    """Async POST to the chat-completions endpoint.

    On a 402 that includes an affordability hint ("can only afford N tokens"),
    retries ONCE with max_tokens shrunk to the affordable budget. During the
    2026-06-11 credit drought the bot sat fully blind for ~12h while OpenRouter
    was offering 842 affordable tokens per call — enough for a non-truncated
    verdict on most prompts. Degraded thinking beats no thinking; if the
    shrunken reply still truncates, parse_verdict falls back to PASS exactly
    as before (no new failure mode).
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:

        # Completion budget: operator-tunable via LLM_MAX_TOKENS (default
        # 32768, read at call time). It caps the RESPONSE length only — the
        # prompt size is governed by the model server's context window.
        default_max_toks = resolve_max_tokens("LLM_MAX_TOKENS")

        async def _post(max_toks: int):
            url = base_url.rstrip("/") + "/chat/completions"
            return await client.post(
                url,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "max_tokens": max_toks,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )

        resp = await _post(default_max_toks)
        if resp.status_code == 402:
            # "...You requested up to 2048 tokens, but can only afford 842..."
            m = re.search(r"can only afford (\d+)", resp.text or "")
            if m and int(m.group(1)) >= 500:
                budget = int(m.group(1)) - 50  # headroom for billing jitter
                logger.warning(
                    f"[research] 402 with affordability hint — retrying DEGRADED "
                    f"at max_tokens={budget} (add credits to restore full reasoning)"
                )
                resp = await _post(budget)

        if resp.is_success:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                text = msg.get("content") or msg.get("reasoning") or ""
                return text
            logger.error("[research] LLM returned 200 but no choices — empty response")
            return ""
        # LOUD failure. A non-200 (esp. 402 Payment Required = out of OpenRouter
        # credits, or 401/429) previously returned "" silently → parse_verdict
        # defaulted every coin to PASS conf 0.0, so a billing/API outage looked
        # identical to "no setups" and the bot sat blind for hours. Make it scream.
        body = resp.text[:200] if resp.text else ""
        logger.error(
            f"[research] LLM call FAILED: HTTP {resp.status_code} — AI research is "
            f"DOWN, all verdicts will default to PASS until fixed. {body}"
        )
    return ""


def parse_verdict(
    ai_text: str,
    coin: str,
    perception: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse the AI response: JSON on the last line, with a regex fallback."""
    if not ai_text:
        ai_text = ""

    logger = logging.getLogger("hermes_trader.agents.research")
    logger.info(f"[parse_verdict] {coin} raw AI text (last 800 chars):\n{ai_text[-800:]}" if len(ai_text) > 800 else f"[parse_verdict] {coin} raw AI text:\n{ai_text}")

    verdict = "PASS"
    confidence = 0.0
    side = None
    entry_px = perception.get("mid", 0)
    stop_px = 0.0
    tp_px = 0.0
    news_risk = "none"
    reasoning = ai_text.strip()

    lines = ai_text.strip().split("\n")

    # Find JSON on the last line
    json_str = ""
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{") and "verdict" in line and line.endswith("}"):
            json_str = line
            break

    # Fallback: regex match
    if not json_str:
        match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', ai_text)
        if match:
            json_str = match.group(0)

    if json_str:
        try:
            cleaned = re.sub(r'```json\s*|```', '', json_str).strip()
            parsed = json.loads(cleaned)

            raw = str(parsed.get("verdict", "")).upper()
            if raw == "LONG":
                verdict = "LONG"
            elif raw == "SHORT":
                verdict = "SHORT"
            elif raw == "CLOSE":
                verdict = "CLOSE"

            confidence = parsed.get("confidence", 0)
            side = parsed.get("side") if parsed.get("side") in ("long", "short") else None
            entry_px = parsed.get("entry_px") or parsed.get("entryPx", perception.get("mid", 0))
            stop_px = parsed.get("stop_px") or parsed.get("stopPx", 0)
            tp_px = parsed.get("tp_px") or parsed.get("tpPx", 0)
            nr = str(parsed.get("news_risk") or parsed.get("newsRisk") or "none").lower()
            news_risk = nr if nr in ("none", "positive", "negative") else "none"
            reasoning = parsed.get("reasoning", ai_text[:500])
        except json.JSONDecodeError:
            first_line = lines[0] if lines else ""
            if re.search(r"LONG", first_line, re.IGNORECASE):
                verdict = "LONG"
            elif re.search(r"SHORT", first_line, re.IGNORECASE):
                verdict = "SHORT"
            elif re.search(r"CLOSE", first_line, re.IGNORECASE):
                verdict = "CLOSE"

    # Coerce confidence to a clamped float — the LLM occasionally returns it
    # as a string ("0.8") or out of range; a string would TypeError at the
    # gate comparison (`ctx.confidence >= 0.85`) on a live trade.
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    # Derive side from verdict when the LLM omitted/nulled the side field.
    # Without this a SHORT verdict with side=None falls through to the
    # executor's `or "long"` default and executes in the WRONG direction.
    if verdict == "LONG":
        side = "long"
    elif verdict == "SHORT":
        side = "short"
    # CLOSE/PASS keep whatever side was parsed (unused downstream).

    return {
        "verdict": verdict,
        "confidence": confidence,
        "side": side,
        "entry_px": entry_px,
        "stop_px": stop_px,
        "tp_px": tp_px,
        "news_risk": news_risk,
        "reasoning": reasoning,
        # Empty ai_text = the LLM call failed (402/429/timeout) and this PASS is
        # an ERROR CODE, not an opinion. Tagged so the executor's structural/whale
        # override won't upgrade a failure-PASS into a blind LONG — on 2026-06-11
        # a 402 window let the override shotgun 8 PASS→LONG upgrades in one
        # minute, filling the book with unvetted longs that then blocked real
        # AI SHORT signals on the movers.
        "ai_down": not ai_text.strip(),
    }


def research(coin: str, perception: Dict[str, Any]) -> Dict[str, Any]:
    """Full AI research pipeline for a perception — returns an analysis dict."""
    c1h = fetch_hl_candles(coin, "1h", 100)
    c4h = fetch_hl_candles(coin, "4h", 100)
    c1d = fetch_hl_candles(coin, "1d", 60)

    funding_raw = _fetch_funding_rate(coin)
    news = _fetch_news(coin)

    # Thin-history guard: multi-timeframe TA is meaningless without enough 4h
    # bars (EMA21/ADX need history). A near-empty series produced confident-
    # looking but baseless entries (e.g. WLD entered at 0.68 conf on "0 candles"
    # then ran straight to the stop). Decline outright — PASS, no LLM call, no entry.
    if len(c4h) < 30:
        logger.warning(f"[research] thin 4h history for {coin}: only {len(c4h)} candles — PASS (skip)")
        analysis = {
            "id": str(uuid.uuid4()), "perception_id": perception.get("id", "unknown"),
            "coin": coin, "verdict": "PASS", "confidence": 0.0, "side": None,
            "entry_px": perception.get("mid", 0), "stop_px": 0.0, "tp_px": 0.0,
            "reasoning": f"insufficient 4h history ({len(c4h)} candles) for reliable multi-TF TA",
            "news_context": news, "news_risk": "none",
            "created_at": int(time.time() * 1000),
            "composite_score": float(perception.get("composite_score", 0) or 0),
            "momentum_burst_fired": False, "slow_burn_fired": False,
            "slow_burn_count": 0,
            "daily_mover_fired": any(
                t.get("name") == "dailyMover" and t.get("fired")
                for t in (perception.get("triggers") or [])
            ),
            "whale_signal": perception.get("whale_signal"),
        }
        memory.record_analysis(analysis)
        return analysis

    tf1h = _compute_indicators(c1h)
    tf4h = _compute_indicators(c4h)
    tf1d = _compute_indicators(c1d)

    config = read_agent_config()
    mode = str(config.get("mode", "OFF"))

    equity = 0.0
    dex_equity: Dict[str, float] = {}
    open_positions: List[Dict[str, Any]] = []
    user = resolve_user_address()

    if user:
        # Aggregated equity so the LLM sees true capital across main + HIP-3
        # dexes when reasoning about risk caps for HIP-3 candidates.
        state = fetch_account_state(user, include_hip3=True)
        equity = float(state.get("equity", "0"))
        dex_equity = state.get("dex_equity") or {}
        memory.update_equity(equity)

        open_positions = [
            {
                "coin": p.get("position", {}).get("coin", ""),
                "side": "long" if float(p.get("position", {}).get("szi", "0")) > 0 else "short",
                "size_usd": float(p.get("position", {}).get("positionValue", "0")) or (
                    abs(float(p.get("position", {}).get("szi", "0"))) *
                    float(p.get("position", {}).get("entryPx", "0"))
                ),
            }
            for p in (state.get("asset_positions") or [])
            if float(p.get("position", {}).get("szi", "0")) != 0
        ]

    wr = memory.get_win_rate()
    system_prompt = build_system_prompt(mode, wr.get("rate", 0), int(wr.get("total", 0)))
    user_message = _build_user_message(
        coin, perception, tf1h, tf4h, tf1d,
        funding_raw, news, equity, open_positions, mode,
        dex_equity=dex_equity, recent_candles=c1h,
    )

    ai_t0 = time.monotonic()
    ai_text = _call_ai(system_prompt, user_message)
    primary_ms = int((time.monotonic() - ai_t0) * 1000)
    parsed = parse_verdict(ai_text, coin, perception)

    # A/B duelist: the SAME prompt to the second model, recorded but never
    # used. Runs AFTER the primary verdict so a duelist outage (slow 9B server
    # included) can never delay or break execution. `duelist_at_entry` is
    # carried into the analysis dict → executor's entry-context snapshot →
    # the outcome store's close row, which is the join the A/B report
    # (duel_store.aggregate) keys on.
    duelist_row = _duelist_verdict(
        system_prompt, user_message, coin, perception,
        parsed["verdict"], parsed["confidence"],
        primary_ms=primary_ms,
    )
    if duelist_row is not None:
        try:
            from hermes_trader.session_log import append as _log_event
            _log_event({
                "event": "duel", "coin": coin,
                "primary_verdict": duelist_row["primary_verdict"],
                "duelist_verdict": duelist_row["duelist_verdict"],
                "primary_model": duelist_row["primary_model"],
                "duelist_model": duelist_row["duelist_model"],
                "agree": duelist_row["duelist_verdict"] == duelist_row["primary_verdict"],
                "primary_ms": duelist_row["primary_ms"],
                "duelist_ms": duelist_row["duelist_ms"],
            })
        except Exception:
            pass

    analysis = {
        "id": str(uuid.uuid4()),
        "perception_id": perception.get("id", "unknown"),
        "coin": coin,
        "verdict": parsed["verdict"],
        "confidence": parsed["confidence"],
        "side": parsed["side"],
        "entry_px": parsed["entry_px"],
        "stop_px": parsed["stop_px"],
        "tp_px": parsed["tp_px"],
        "reasoning": parsed["reasoning"],
        "news_context": news,
        # AI's good/bad judgment of the recent news — drives the news gate
        # (only "negative" stands the trade down; an earnings beat is fine).
        "news_risk": parsed["news_risk"],
        # Failure-PASS marker — must survive this whitelist or the executor's
        # override guard never sees it (it didn't, on first deploy).
        "ai_down": bool(parsed.get("ai_down")),
        # A/B duelist verdict (None when the duelist is disabled/failed). Must
        # survive this whitelist: the executor snapshots it into the entry
        # context so the close row can attribute the same trade to the second
        # model. Carries only verdict-level data — the full reasoning lives in
        # the duel JSONL to keep the analysis (and the memory file) lean.
        "duelist_at_entry": (
            {
                "model": duelist_row["duelist_model"],
                "verdict": duelist_row["duelist_verdict"],
                "confidence": duelist_row["duelist_confidence"],
                "side": duelist_row["duelist_side"],
            }
            if duelist_row is not None else None
        ),
        "created_at": int(time.time() * 1000),
        # Carry forward so risk gates can read own-coin signal strength.
        "composite_score": float(perception.get("composite_score", 0) or 0),
        "momentum_burst_fired": any(
            t.get("name") == "momentumBurst" and t.get("fired")
            for t in (perception.get("triggers") or [])
        ),
        "slow_burn_fired": any(
            t.get("name") in ("volumeBuildup1h", "trendFlip1h", "higherLows1h")
            and t.get("fired")
            for t in (perception.get("triggers") or [])
        ),
        "slow_burn_count": sum(
            1 for t in (perception.get("triggers") or [])
            if t.get("name") in ("volumeBuildup1h", "trendFlip1h", "higherLows1h")
            and t.get("fired")
        ),
        # O'Neil breakout pair — feeds the breakout force-execute (a hedged AI
        # PASS on a 20-period-high break WITH a volume surge gets upgraded;
        # XPL +32% 2026-06-12 was researched 38x, PASSed 21x, never traded
        # while both of these were fired hours before the move).
        "breakout_fired": any(
            t.get("name") == "breakout" and t.get("fired")
            for t in (perception.get("triggers") or [])
        ),
        "volume_spike_fired": any(
            t.get("name") == "volumeSpike" and t.get("fired")
            for t in (perception.get("triggers") or [])
        ),
        "uptrend_momentum_fired": any(
            t.get("name") == "uptrendMomentum" and t.get("fired")
            for t in (perception.get("triggers") or [])
        ),
        "downtrend_momentum_fired": any(
            t.get("name") == "downtrendMomentum" and t.get("fired")
            for t in (perception.get("triggers") or [])
        ),
        "daily_mover_fired": any(
            t.get("name") == "dailyMover" and t.get("fired")
            for t in (perception.get("triggers") or [])
        ),
        # OI+funding accumulation signal (oi_funding_anomaly). When present,
        # the coin shows whale-loading patterns (high OI, negative funding,
        # flat price). Used as a counter-regime bypass for LONGs.
        "whale_signal": perception.get("whale_signal"),
    }

    # Chronos shadow logging moved to executor.py (trade result path).
    # Keeping this would duplicate logs; executor owns the signal now.
    if parsed["verdict"] in ("LONG", "SHORT"):
        pass  # Chronos handled in executor.py

    memory.record_analysis(analysis)
    return analysis


def _fire_chronos_shadow(coin: str, side: str) -> None:
    """Fire a Chronos-2 forecast in the background for shadow-mode logging.

    Runs async on a daemon thread; NEVER blocks research or the execute hot path.
    The signal is LOGGED only — NOT fed into the LLM prompt, NOT used for gating.
    This is pure forward-validation so we can measure forecast quality before
    any decision integration.
    """
    try:
        from hermes_trader.agents.chronos_signal import get_chronos_signal_async
        side_str = side or "long" if side else "long"
        get_chronos_signal_async(coin, side_str)
    except Exception as e:
        logger.debug(f"[research] chronos shadow failed for {coin}: {e}")
