"""Risk gates — every gate is a pure function returning {pass, reason?}.

All gates are evaluated; results are collected for telemetry (no short-circuit).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

GateResult = Dict[str, Any]  # {pass: bool, reason?: str}

logger = logging.getLogger(__name__)


class GateContext:
    """Context passed to all risk gates."""
    def __init__(
        self,
        confidence: float,
        current_positions: List[Dict[str, Any]],
        trade_notional_usd: float,
        daily_pnl: float,
        market_volume_24h_usd: float,
        coin: str,
        trade_side: str,  # 'long' or 'short'
        has_binary_news_risk: bool,
        equity: float,
        total_open_notional: float,
        composite_score: float = 0.0,
        momentum_burst_fired: bool = False,
        slow_burn_fired: bool = False,
        whale_signal_fired: bool = False,
        binary_news_match: str = "",
        peak_daily_pnl: float = 0.0,
        chronos_median_pct: Optional[float] = None,
        chronos_spread_pct: Optional[float] = None,
        chronos_q10_path_pct: Optional[List[float]] = None,
        chronos_q90_path_pct: Optional[List[float]] = None,
        squeeze_extreme_no_breakout: Optional[bool] = None,
        duelist_verdict: Optional[str] = None,
    ):
        self.confidence = confidence
        self.current_positions = current_positions
        self.trade_notional_usd = trade_notional_usd
        self.daily_pnl = daily_pnl
        self.peak_daily_pnl = peak_daily_pnl
        self.market_volume_24h_usd = market_volume_24h_usd
        self.coin = coin
        self.trade_side = trade_side
        self.has_binary_news_risk = has_binary_news_risk
        self.equity = equity
        self.total_open_notional = total_open_notional
        self.composite_score = composite_score
        self.momentum_burst_fired = momentum_burst_fired
        # True iff any 1h slow-burn trigger fired (volumeBuildup1h /
        # trendFlip1h / higherLows1h). Used as a counter-regime bypass: a
        # clean 1h accumulation pattern overrides the slow BTC proxy.
        self.slow_burn_fired = slow_burn_fired
        # True iff whale_index oi_funding_anomaly flagged this coin
        # (negative funding + flat price + high OI = whale accumulation).
        # Same gate-bypass role as slow_burn_fired; orthogonal signal.
        self.whale_signal_fired = whale_signal_fired
        # The headline + matched term that tripped the binary-news gate, for
        # log visibility ("which article blocked this?").
        self.binary_news_match = binary_news_match
        # Median move of this coin's Chronos-2 forecast, % vs last close.
        # Side-independent; fed by maybe_execute via get_chronos_signal_sync
        # (warm cache hit, or one bounded compute on a cold/expired cache —
        # never a cache-only peek, which raced the attach compute).
        # None = no usable forecast → chronos_mismatch_gate
        # has no opinion and passes.
        self.chronos_median_pct = chronos_median_pct
        # Chronos-2 p10-p90 spread, % vs last close (same warm sync read as
        # the median above). Fed ONLY to the ratio-aware deadband
        # COUNTERFACTUAL in chronos_mismatch_gate — it never gates live
        # execution, it just records that a ratio-aware deadband would have
        # rescued a fixed-deadband block (HEMI replay, 2026-08-30).
        # None = no spread (error signal) → counterfactual is inert.
        self.chronos_spread_pct = chronos_spread_pct
        # Per-step Chronos-2 quantile paths, % vs last close (same sync read;
        # None on error signals / old shapes). Fed to
        # chronos_tail_trigger_gate — the shape-based counter-forecast veto
        # validated by the 2026-08-28 60-flag replay (adverse-quantile early
        # tail > path-mean > endpoint for loss avoidance).
        self.chronos_q10_path_pct = chronos_q10_path_pct
        self.chronos_q90_path_pct = chronos_q90_path_pct
        # The squeeze-signal composite gate flag (1h Donchian extreme with no
        # fresh aligned breakout — the "chasing without confirmation" bucket),
        # recomputed per candidate side by squeeze_signal._set_gate. None = no
        # squeeze data (disabled / fetch failed) -> squeeze_extreme_gate has no
        # opinion and passes. True/False = the flag's actual value; the gate
        # only ever has an opinion on True.
        self.squeeze_extreme_no_breakout = squeeze_extreme_no_breakout
        # The A/B duelist's verdict at entry (LONG / SHORT / PASS / None),
        # carried from research.py's `duelist_at_entry` snapshot via
        # maybe_execute. None = the duelist is disabled or failed — the
        # veto has no opinion and passes.
        self.duelist_verdict = duelist_verdict


def confidence_gate(ctx: GateContext, min_confidence: float) -> GateResult:
    if ctx.confidence >= min_confidence:
        return {"pass": True}
    return {"pass": False, "reason": f"confidence {ctx.confidence:.2f} < {min_confidence}"}


def max_concurrent_positions_gate(ctx: GateContext, max_concurrent: int) -> GateResult:
    if len(ctx.current_positions) < max_concurrent:
        return {"pass": True}
    return {"pass": False, "reason": f"max positions reached ({len(ctx.current_positions)}/{max_concurrent})"}


def per_trade_notional_cap_gate(ctx: GateContext, cap_usd: float) -> GateResult:
    cap = float(cap_usd or 0)
    if cap <= 0:
        return {"pass": True}
    # The executor normalizes the target notional into an exchange-valid coin
    # size before gates. Coin precision can create a few cents/dollars of cap
    # dust, e.g. target $650.00 -> valid size worth $650.05. Treat that as
    # still capped; larger overshoots remain blocked.
    precision_tolerance = max(0.25, cap * 0.005)
    if ctx.trade_notional_usd <= cap + precision_tolerance:
        return {"pass": True}
    return {"pass": False, "reason": f"trade notional ${ctx.trade_notional_usd:.2f} exceeds cap ${cap:.2f}"}


def effective_daily_kill_usd(config: Dict[str, Any], equity: float) -> float:
    """Daily kill threshold in USD (positive magnitude), equity-relative.

    thr = clamp(pct_of_equity * equity, min_usd, cap_usd). The pct form keeps
    the brake proportional as the account grows; the cap is the absolute
    ceiling ("never lose $X in one day no matter the equity"); the min stops
    a small account from halting on ordinary noise.
    Returns 0.0 when the kill switch is disabled (pct unset/0) — the entry
    gate, the halt timer AND the heartbeat's hard flatten then all stand
    down. This is the ENTRY-GATE / HALT threshold; the hard flatten fires
    at `flatten_daily_kill_usd` (a multiple of this) so the book has a
    grace band to recover in."""
    pct = float(_cfg(config, "daily_kill_pct_of_equity", 0) or 0)
    if pct <= 0:
        return 0.0
    thr = pct * max(0.0, float(equity or 0))
    cap = float(_cfg(config, "daily_kill_cap_usd", 0) or 0)
    if cap > 0:
        thr = min(thr, cap)
    floor = float(_cfg(config, "daily_kill_min_usd", 0) or 0)
    if floor > 0:
        thr = max(thr, floor)
    return thr


def flatten_daily_kill_usd(config: Dict[str, Any], equity: float) -> float:
    """Hard flatten threshold for the heartbeat kill-switch: the entry-gate /
    halt threshold times `daily_kill_flatten_mult` (default 1.25).

    The flatten is deliberately HIGHER than the halt so the book has a
    grace band (-T .. -mult*T): once the day breaches -T the halt blocks
    new entries, but the open positions are still alive and can claw the
    day back above the release band (clearing the halt early). A flat
    flatten at T would kill that recovery path — daily PnL is equity-based
    and stays pinned red once the book is empty, so the early release
    would be unreachable and the halt a hard 6h lock. 0 = disabled
    (kill switch off), mirroring effective_daily_kill_usd."""
    base = effective_daily_kill_usd(config, equity)
    if base <= 0:
        return 0.0
    mult = float(_cfg(config, "daily_kill_flatten_mult", 1.25) or 1.25)
    return base * mult


def daily_loss_kill_switch(ctx: GateContext, max_daily_loss: float,
                           halt_remaining_min: float = 0.0) -> GateResult:
    """Block new ENTRIES when the day is deep in the red (or a halt timer is
    active). `max_daily_loss` is the effective USD threshold from
    effective_daily_kill_usd (0 = disabled → this gate passes). Block while
    daily_pnl <= -threshold OR a halt timer from a previous breach is still
    running. The halt is a TIMER (memory.arm_daily_halt) that expires or is
    cleared early on recovery — deliberately NOT a UTC-rollover lock, so a
    genuine regime change back can resume trading the same day."""
    if max_daily_loss <= 0:
        return {"pass": True}
    if halt_remaining_min > 0:
        return {"pass": False,
                "reason": (f"daily loss halt active ({halt_remaining_min:.0f}min "
                           f"remaining; lifts early on PnL recovery)")}
    if ctx.daily_pnl > -max_daily_loss:
        return {"pass": True}
    return {"pass": False,
            "reason": (f"daily loss killswitch triggered (PnL ${ctx.daily_pnl:.2f} "
                       f"<= -${max_daily_loss:.2f}, equity-relative)")}


def daily_giveback_gate(ctx: GateContext, halt_pct: float, min_peak_usd: float) -> GateResult:
    """Lock in a green day: once daily PnL has peaked at >= `min_peak_usd`, block
    NEW positions if it then retraces more than `halt_pct` from that peak. Existing
    positions keep riding their own stops; this only stops opening fresh risk so a
    won day can't fully round-trip. Disabled when halt_pct<=0. Resets at the UTC
    day roll (peak_daily_pnl resets in memory.track_daily_pnl)."""
    if halt_pct <= 0 or ctx.peak_daily_pnl < min_peak_usd:
        return {"pass": True}
    floor = ctx.peak_daily_pnl * (1.0 - halt_pct)
    if ctx.daily_pnl <= floor:
        return {"pass": False,
                "reason": (f"daily give-back halt: PnL ${ctx.daily_pnl:.0f} retraced "
                           f">{halt_pct*100:.0f}% from peak ${ctx.peak_daily_pnl:.0f} "
                           f"(floor ${floor:.0f}) — no new entries until UTC roll")}
    return {"pass": True}


def market_liquidity_floor(
    ctx: GateContext,
    min_volume: float,
    min_volume_hip3: Optional[float] = None,
    gate_config: Optional[dict] = None,
) -> GateResult:
    """Block trades on markets with insufficient 24h notional volume.

    HIP-3 tokenized-equity / commodity perps live on separate dexs and
    naturally carry less volume than BTC/ETH-style native markets (most
    `xyz:*` markets sit in the $1M–$50M range vs $1B+ for BTC). Applying
    the same 5M crypto floor incorrectly blocks adequately-liquid HIP-3
    markets like xyz:CRCL ($4.7M) and km:USTECH ($1.06M). When the coin
    is HIP-3 (colon-namespaced) and a separate `min_volume_hip3` is set,
    use that floor instead.

    High-confidence overrides: when `gate_config.bypass_low_volume=True`
    and confidence >= `gate_config.bypass_low_volume_min_conf`, allow the
    trade despite thin volume.
    """
    is_hip3 = ":" in (ctx.coin or "")
    floor = (min_volume_hip3 if (is_hip3 and min_volume_hip3 is not None) else min_volume)
    if ctx.market_volume_24h_usd >= floor:
        return {"pass": True}

    # High-confidence bypass
    if gate_config:
        if gate_config.get("bypass_low_volume"):
            min_conf = gate_config.get("bypass_low_volume_min_conf", 0.85)
            if ctx.confidence >= min_conf:
                return {
                    "pass": True,
                    "via": "bypass_low_volume",
                    "reason": (
                        f"low-liquidity bypass on {ctx.coin} (conf {ctx.confidence:.2f} >= {min_conf:.2f})"
                    ),
                }

    return {"pass": False, "reason": f"market 24h volume ${ctx.market_volume_24h_usd/1e6:.2f}M below floor ${floor/1e6:.2f}M"}


def short_liquidity_floor(ctx: GateContext, min_short_volume: float) -> GateResult:
    """SHORTS need materially more liquidity than longs — thin markets squeeze.

    Data (72h short segmentation): short BLEEDERS had a median 24h volume of
    ~$13M (XPL 0%/5 win, xyz:LITE -6.7%/10, PUMP, xyz:EWZ) while short WINNERS
    (XMR/TON/DOGE/BTC/ETH + commodities) had ~$223M — a 17x gap. Low-liquidity
    shorts ran to max_loss (the entire short bleed was 14 stopped shorts). Longs
    can tolerate a thin pump; a thin short gets squeezed. Applies ONLY to shorts;
    0/None disables (opt-in, reversible)."""
    if ctx.trade_side != "short" or not min_short_volume:
        return {"pass": True}
    if ctx.market_volume_24h_usd >= min_short_volume:
        return {"pass": True}
    return {"pass": False,
            "reason": (f"short on thin market: 24h vol ${ctx.market_volume_24h_usd/1e6:.1f}M "
                       f"< short floor ${min_short_volume/1e6:.0f}M (squeeze risk)")}


def coin_allowlist_gate(ctx: GateContext, allowlist: List[str], blocklist: List[str]) -> GateResult:
    if blocklist and ctx.coin in blocklist:
        return {"pass": False, "reason": f"{ctx.coin} is on the coin blocklist"}
    if allowlist and ctx.coin not in allowlist:
        return {"pass": False, "reason": f"{ctx.coin} not on the allowlist"}
    return {"pass": True}


def cooldown_gate(ctx: GateContext, last_trade_time: Optional[int], cooldown_min: float) -> GateResult:
    if last_trade_time is None:
        return {"pass": True}
    elapsed = (int(time.time() * 1000) - last_trade_time) / 60_000
    if elapsed >= cooldown_min:
        return {"pass": True}
    return {"pass": False, "reason": f"cooldown active ({int(cooldown_min - elapsed)}min remaining)"}


def opposite_direction_guard(ctx: GateContext) -> GateResult:
    """Block ANY re-entry on a coin we already hold. A held position is managed
    solely by the DSL engine + the periodic AI close-check (CLOSE / HOLD); it is
    never flipped (opposite side = no auto-flip) NOR added to (same side =
    uncontrolled pyramid). The held-coin close-check sometimes returns a fresh
    LONG/SHORT on a strong held name; without this it would try to pyramid in
    (previously only the exchange margin check stopped it)."""
    existing = next((p for p in ctx.current_positions if p["coin"] == ctx.coin), None)
    if not existing:
        return {"pass": True}
    if existing["side"] != ctx.trade_side:
        return {"pass": False, "reason": f"opposite position exists ({ctx.coin} {existing['side']}) — no auto-flip"}
    return {"pass": False, "reason": f"already holding {ctx.coin} {existing['side']} — no pyramid/re-entry"}


# Major crypto coins for correlation cap
_CRYPTO_COINS = frozenset([
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "MATIC", "LINK",
    "DOT", "UNI", "ATOM", "NEAR", "FTM", "APT", "ARB", "OP", "INJ", "TIA",
    "SUI", "SEI", "WIF", "PEPE", "BONK", "FLOKI", "TRX", "LTC", "BCH", "ETC",
    "XLM", "ALGO", "AAVE", "MKR", "SNX", "CRV", "COMP", "YFI", "SUSHI", "1INCH",
])


def correlation_cap(ctx: GateContext, max_crypto_correlated: int) -> GateResult:
    # Only cap long correlation
    if ctx.trade_side != "long":
        return {"pass": True}
    existing_crypto_long = sum(
        1 for p in ctx.current_positions
        if p["coin"] in _CRYPTO_COINS and p["side"] == "long"
    )
    if existing_crypto_long < max_crypto_correlated:
        return {"pass": True}
    return {"pass": False, "reason": f"crypto long correlation cap reached ({existing_crypto_long}/{max_crypto_correlated})"}


def equity_risk_cap(ctx: GateContext, max_total_notional_pct: float) -> GateResult:
    max_notional = ctx.equity * max_total_notional_pct
    projected_notional = ctx.total_open_notional + ctx.trade_notional_usd
    if projected_notional <= max_notional:
        return {"pass": True}
    return {
        "pass": False,
        "reason": f"total notional ${projected_notional:.0f} would exceed {max_total_notional_pct*100:.0f}% of equity (${max_notional:.0f})",
    }


def market_regime_gate(ctx: GateContext, counter_regime_min_conf: float = 0.7,
                       block_counter_trend_bypass: bool = False,
                       crowded_with_min_conf: float = 0.0) -> GateResult:
    """Block counter-regime trades unless conviction OR own-coin signal clears the bar.

      - aligned with regime → pass
      - regime neutral      → pass (subject to funding-regime override below)
      - counter-trend trade → pass if any of:
          * confidence >= counter_regime_min_conf
          * composite_score >= 50
          * momentumBurst fired (large fast move on 5m)
          * slow_burn_fired (1h vol surge or EMA cross — accumulation breakout)
        else block.

    The own-signal bypasses exist because the regime proxy (BTC for crypto,
    SP500 for equity) is slow; a strong individual signal should override
    a stale macro call.

    Funding-regime overlay (added 2026): SYMMETRIC enforcement — when the
    market-wide funding regime is crowded, any trade going AGAINST the crowd
    direction must clear a higher bar. This is direction-agnostic and will
    apply the same way when the regime flips:

      * SHORT_CROWDED + long  → counter-regime, elevated bar
      * LONG_CROWDED  + short → counter-regime, elevated bar
      * SHORT_CROWDED + short → aligned, normal bar (no bias added)
      * LONG_CROWDED  + long  → aligned, normal bar (no bias added)

    Elevated bar = confidence >= max(counter_regime_min_conf, 0.85)
                   OR composite_score >= 60
                   OR any binary trigger (momentumBurst / slow_burn / whale_signal)

    The bypass triggers are preserved on both sides — those are explicit
    "the regime proxy is stale" signals and we never want to hard-block on
    a clear individual setup, just enforce regime discipline by default.
    """
    from hermes_trader.agents.market_regime import detect_regime
    regime = detect_regime(ctx.coin)

    # Pull funding regime (cached) — used as a symmetric overlay on the
    # trend-regime gate. Both directions are treated identically: anything
    # going against the crowded side faces the elevated bar.
    #
    # PER-CLASS LOOKUP: the gate uses the funding regime of THIS coin's
    # asset class (crypto / equity / commodity), not a global crypto-only
    # signal. Without this, a SHORT_CROWDED crypto regime would gate longs
    # on oil (xyz:CL) and semis (xyz:ARM) — those have their own funding
    # markets and shouldn't be evaluated by the crypto crowd.
    try:
        from hermes_trader.agents.hyperfeed import market_get_funding_regime
        from hermes_trader.agents.market_regime import classify_asset
        funding_data = market_get_funding_regime()
        coin_class = classify_asset(ctx.coin)
        by_class = funding_data.get("regimes_by_class") or {}
        funding_regime = by_class.get(coin_class) or funding_data.get("regime", "NEUTRAL")
    except Exception:
        funding_regime = "NEUTRAL"

    # Symmetric counter-funding-regime detection.
    against_funding = (
        (funding_regime == "SHORT_CROWDED" and ctx.trade_side == "long") or
        (funding_regime == "LONG_CROWDED"  and ctx.trade_side == "short")
    )
    # WITH-crowd (squeeze-prone): trading the SAME side the crowd is already on
    # (short into SHORT_CROWDED / long into LONG_CROWDED). These are trend-aligned
    # but are exactly what gets squeezed on a reversal — they round-tripped the
    # 2026-06-06 day. Require elevated conviction so only strong setups join a
    # crowded book. Gated by crowded_with_min_conf (0 = off).
    with_crowd = (
        (funding_regime == "SHORT_CROWDED" and ctx.trade_side == "short") or
        (funding_regime == "LONG_CROWDED"  and ctx.trade_side == "long")
    )

    # Effective thresholds: only elevated when against the funding regime.
    # When aligned with funding regime, use the normal counter_regime_min_conf
    # so we never *raise* the bar for regime-aligned trades.
    effective_min_conf = counter_regime_min_conf
    effective_min_score = 50.0
    if against_funding:
        effective_min_conf = max(counter_regime_min_conf, 0.85)
        effective_min_score = 60.0

    # Context attached to every result so the log reads "why" without
    # re-deriving regime state after the fact.
    base = {"regime": regime, "funding": funding_regime,
            "against_funding": against_funding, "counter_trend": False}

    # Aligned with trend regime AND not against funding regime → easy pass,
    # UNLESS it's a with-crowd (squeeze-prone) entry that fails the elevated
    # conviction bar — those are the crowded shorts/longs that round-trip on a
    # squeeze, so a weak one is blocked here.
    aligned = (regime == "up" and ctx.trade_side == "long") or \
              (regime == "down" and ctx.trade_side == "short")
    if aligned and not against_funding:
        if with_crowd and crowded_with_min_conf > 0 and ctx.confidence < crowded_with_min_conf:
            return {"pass": False, "via": "crowded_squeeze",
                    **{**base, "with_crowd": True},
                    "reason": (f"with-crowd {ctx.trade_side} into {funding_regime} "
                               f"(squeeze risk) — need conf >= {crowded_with_min_conf:.2f}, "
                               f"have {ctx.confidence:.2f}")}
        return {"pass": True, "via": "aligned", **{**base, "with_crowd": with_crowd}}

    # Trend-regime neutral and not against funding regime → pass.
    if regime == "neutral" and not against_funding:
        return {"pass": True, "via": "neutral", **base}

    # Past here the trade is counter-trend and/or against the funding crowd —
    # it must clear the (possibly elevated) bar via conviction or own-signal.
    base["counter_trend"] = not aligned
    if ctx.confidence >= effective_min_conf:
        return {"pass": True, "via": "confidence", **base}
    if ctx.composite_score >= effective_min_score:
        return {"pass": True, "via": "composite", **base}
    # Binary-trigger bypass: a strong own-coin signal (momentum_burst / slow_burn
    # / whale) normally overrides the slow macro-regime call. `block_counter_trend_bypass`
    # (config, default False, reversible) DISABLES this bypass here — i.e. for trades
    # that are already counter-trend and/or against the funding crowd. Data (journal
    # P166-P177, ~-7% drawdown) showed low-conviction LONGS forced through via
    # `trigger:slow_burn` against a DOWN tape (SP500/MU/ORCL longs) and bleeding. With
    # the flag on, a counter-regime trade must clear REAL conviction (conf/score); a
    # lone momentum trigger no longer pushes it through against the regime. Aligned and
    # neutral-regime trades returned earlier (lines above) and are UNAFFECTED, so this
    # does NOT blanket-weaken the bypass — only where it fights a strong directional regime.
    if (ctx.momentum_burst_fired or ctx.slow_burn_fired or ctx.whale_signal_fired) \
            and not block_counter_trend_bypass:
        trig = ("momentum_burst" if ctx.momentum_burst_fired
                else "slow_burn" if ctx.slow_burn_fired else "whale")
        return {"pass": True, "via": f"trigger:{trig}", **base}

    blocked_via = "blocked_bypass" if block_counter_trend_bypass else "blocked"
    return {
        "pass": False,
        "via": blocked_via,
        **base,
        "reason": (f"counter-regime {ctx.trade_side} vs {regime} trend "
                   f"(funding={funding_regime}) — need conf >= {effective_min_conf:.2f} "
                   f"or score >= {effective_min_score:.0f}"
                   f"{'' if block_counter_trend_bypass else ' or own-coin signal'}, "
                   f"have conf {ctx.confidence:.2f}, score {ctx.composite_score:.0f}"),
    }


def news_blackout_gate(ctx: GateContext) -> GateResult:
    if not ctx.has_binary_news_risk:
        return {"pass": True}
    detail = f" — {ctx.binary_news_match}" if ctx.binary_news_match else ""
    return {"pass": False,
            "reason": f"binary news risk (Fed/earnings/hack in recent news){detail} — standing down"}


def _cfg(config: Dict[str, Any], key: str, default: Any) -> Any:
    """Read a config value tolerating snake_case or camelCase keys."""
    if key in config:
        return config[key]
    parts = key.split("_")
    camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
    return config[camel] if camel in config else default


def _chronos_ratio_deadband_rescue(
    ctx: GateContext, fixed_deadband: float, gate_cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Counterfactual: would a ratio-aware deadband have rescued this block?

    The live fixed-deadband rule treats |median| beyond `min_abs_median_pct`
    as a directional claim. The HEMI replay (2026-08-30) showed a median that
    sits inside the model's own p10-p90 band is noise, not a claim — so the
    ratio-aware deadband is `max(fixed, min_conf_ratio * spread)`. Because it
    is ALWAYS >= the fixed deadband, it can only widen the no-opinion zone: it
    rescues blocks, it never creates them. This is the sample we accrue before
    ever flipping the live rule to ratio-aware.

    Called only on the block path (where the fixed rule has already rejected
    the entry, so `abs(med) >= fixed_deadband` holds — asserted defensively
    below). Returns None when the spread is unavailable or the ratio floor
    does not actually widen the no-opinion zone past |median| (i.e. no rescue
    to report).
    """
    med = ctx.chronos_median_pct
    spread = ctx.chronos_spread_pct
    if med is None or spread is None or spread <= 0:
        return None
    if abs(med) < fixed_deadband:
        # Invariant of the block path — a median inside the fixed deadband
        # never reaches the block, so no rescue to report.
        return None
    # Same knob semantics as chronos_signal.resolve_min_conf_ratio: absent ->
    # 0.25, explicit 0 -> 0 (counterfactual inert: ratio_deadband collapses
    # back to the fixed one, so no rescue can ever register).
    try:
        ratio = max(0.0, float(gate_cfg.get("min_conf_ratio", 0.25)))
    except (TypeError, ValueError):
        ratio = 0.25
    ratio_deadband = max(fixed_deadband, ratio * spread)
    if abs(med) < ratio_deadband:
        return {
            "would_pass": True,
            "fixed_deadband_pct": fixed_deadband,
            "ratio_deadband_pct": round(ratio_deadband, 4),
            "min_conf_ratio": ratio,
            "spread_pct": spread,
            "median_pct": med,
        }
    return None


def chronos_mismatch_gate(ctx: GateContext, gate_cfg: Dict[str, Any]) -> GateResult:
    """Chronos direction-mismatch conviction gate (SHADOW by default).

    When the entry side contradicts the cached Chronos-2 forecast (long with
    median_pct < 0, short with median_pct > 0, beyond `min_abs_median_pct`),
    the trade must carry elevated conviction: conf >= `min_conf` (0.90 — the
    same bar as the late-trend-chase bypass) OR composite >= `min_composite`
    (60). A counter-forecast scalp at 0.82 confidence is exactly the PURR
    failure mode this gate is meant to surface.

    SHADOW MODE: with `shadow_mode` true (default) the gate STRUCTURALLY
    returns pass=True and only carries a `shadow_would_block` marker, which
    the executor logs loudly. It cannot alter live execution until the
    operator flips shadow_mode to false in .agent-config.json — one knob.

    Fail-safe: no forecast (cold cache, model not loaded, in_prompt off) or a
    directionally-neutral forecast (within the deadband) = no opinion = pass.

    COUNTERFACTUAL (log-only, never gates): on the block path we also ask what
    a ratio-aware deadband `max(min_abs_median_pct, min_conf_ratio * spread)`
    would have done. It is always wider than the fixed deadband, so it only
    rescues, never adds — a pure over-block sample (HEMI replay, 2026-08-30).
    When it would have rescued, the result carries a `counterfactual_rescue`
    marker the executor logs so we can count them against P/L before deciding
    whether to promote the live rule to ratio-aware. `min_conf_ratio` (default
    0.25) lives in this gate's config block.
    """
    cfg = gate_cfg or {}
    if not bool(cfg.get("enabled", True)):
        return {"pass": True}
    med = ctx.chronos_median_pct
    if med is None:
        return {"pass": True}
    deadband = float(cfg.get("min_abs_median_pct", 0.5) or 0.5)
    if abs(med) < deadband:
        return {"pass": True}
    mismatch = ((ctx.trade_side == "long" and med < 0)
                or (ctx.trade_side == "short" and med > 0))
    if not mismatch:
        return {"pass": True}
    min_conf = float(cfg.get("min_conf", 0.90) or 0.90)
    min_composite = float(cfg.get("min_composite", 60.0) or 60.0)
    if ctx.confidence >= min_conf or ctx.composite_score >= min_composite:
        return {"pass": True}
    reason = (f"chronos_mismatch ({ctx.trade_side} vs forecast {med:+.2f}%; "
              f"conf {ctx.confidence:.2f} < {min_conf:.2f}, "
              f"composite {ctx.composite_score:.1f} < {min_composite:.0f})")
    result: GateResult = {"reason": reason}
    # Log-only counterfactual (never changes the pass/fail below): a
    # ratio-aware deadband is always wider than the fixed one, so it can only
    # rescue a block the fixed rule made — accrue the sample for the
    # fixed-vs-ratio decision.
    cf = _chronos_ratio_deadband_rescue(ctx, deadband, cfg)
    if cf is not None:
        result["counterfactual_rescue"] = cf
    if bool(cfg.get("shadow_mode", True)):
        result["pass"] = True
        result["shadow_would_block"] = True
    else:
        result["pass"] = False
    return result


def chronos_tail_trigger_gate(ctx: GateContext, gate_cfg: Dict[str, Any]) -> GateResult:
    """Chronos tail-trigger conviction gate (SHADOW by default).

    Shape-based counter-forecast veto, validated by the 2026-08-28 60-flag
    replay: when the ADVERSE quantile of the cached Chronos-2 forecast path
    (q10 for longs, q90 for shorts) breaches `-min_adv_path_pct` at ANY of
    the first `window_steps` steps (5m candles each — 6 = 30m ahead), the
    entry is vetoed unless it clears the same elevated-conviction bar as
    `chronos_mismatch` (conf >= min_conf OR composite >= min_composite).

    Why the tail instead of the path mean (the reduction the mismatch gate
    uses): the replay showed the mean/endpoint scalars save ≈$0.96 across 14
    executed flags while the K=6/X=3.0 tail trigger saved ≈$3.21 — it blocks
    the same 4 stop-outs (MOVE, TRUMP, FARTCOIN x2, all within ~25m of entry)
    but RELEASES the small mean-reverting winners the path-mean gate kills.
    The q10 path also overstates realized magnitude ~2x, so the threshold is
    a boolean trip-wire near half the 5% spot stop — not a magnitude
    forecast — and deliberately sits well inside the stop.

    SHADOW MODE: with `shadow_mode` true (default, and how the operator armed
    it 2026-08-28) the gate STRUCTURALLY returns pass=True and only carries a
    `shadow_would_block` marker, which the executor logs loudly next to the
    existing chronos_mismatch shadow line. It cannot alter live execution
    until the operator flips shadow_mode to false in .agent-config.json.

    Fail-safes (no-opinion pass): disabled; no warm cached signal (cold cache
    / model down / disabled); paths missing (pre-change cache entries); path
    shorter than window_steps; tail not breached.
    """
    cfg = gate_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return {"pass": True}
    k = int(cfg.get("window_steps", 6) or 6)
    x = float(cfg.get("min_adv_path_pct", 3.0) or 3.0)
    if k <= 0 or x <= 0:
        return {"pass": True}
    path = (ctx.chronos_q10_path_pct if ctx.trade_side == "long"
            else ctx.chronos_q90_path_pct)
    if not path or len(path) < k:
        return {"pass": True}
    window = path[:k]
    tail = min(window) if ctx.trade_side == "long" else max(window)
    breached = tail <= -x if ctx.trade_side == "long" else tail >= x
    if not breached:
        return {"pass": True}
    min_conf = float(cfg.get("min_conf", 0.90) or 0.90)
    min_composite = float(cfg.get("min_composite", 60.0) or 60.0)
    if ctx.confidence >= min_conf or ctx.composite_score >= min_composite:
        return {"pass": True}
    reason = (f"chronos_tail_trigger ({ctx.trade_side} entry, adverse q-path "
              f"{'min' if ctx.trade_side == 'long' else 'max'} of first {k} steps "
              f"= {tail:+.2f}% beyond {x:.1f}%; conf {ctx.confidence:.2f} < "
              f"{min_conf:.2f}, composite {ctx.composite_score:.1f} < "
              f"{min_composite:.0f})")
    if bool(cfg.get("shadow_mode", True)):
        return {"pass": True, "reason": reason, "shadow_would_block": True}
    return {"pass": False, "reason": reason}


def band_counter_breach_gate(ctx: GateContext, gate_cfg: Dict[str, Any]) -> GateResult:
    """Band counter-trend breach conviction gate (SHADOW by default).

    Deterministic encoding of the shape the research prompt tells the LLM is
    a reversion, not a continuation: the band (same interval/params as the
    band_snapback trigger config) is TRENDING — drift > max_drift_pct — and
    price is >= `min_breach_pct` beyond the OPPOSITE-side edge:

      band drifts DOWN, price above the upper edge  + LONG entry
          -> top of a relief rally = start of the next down-swing
      band drifts UP,   price below the lower edge  + SHORT entry
          -> bottom of a pullback = start of the next up-swing

    The bounce/dip entry fights the drift and the nearest short-term
    reversion is back toward the band — AGAINST the entry. The LLM has been
    rationalizing exactly this as "the drift is already priced in ->
    continuation" (GRASS long 2026-08-26 19:07: 1h band DOWN 6.4% drift,
    px +6.6% vs upper edge, entered long at 0.82 conf -> -9.6% ROE). When
    the breach shape matches the entry side, the trade must clear the
    elevated bar: confidence >= `min_conf` (0.90). A genuinely
    high-conviction continuation still gets through; the 0.82
    rationalization does not. There is deliberately NO composite-score
    bypass on this gate — the composite is a multi-trigger tally and the
    shape it would excuse here is precisely the one this gate encodes.

    SHADOW MODE: with `shadow_mode` true the gate STRUCTURALLY returns
    pass=True and only carries a `shadow_would_block` marker, which the
    executor logs loudly — same pattern as chronos_mismatch_gate. It cannot
    alter live execution until the operator flips shadow_mode to false in
    .agent-config.json. `enabled` false (code default) disables it
    entirely.

    Candles are fetched inside the gate on the band interval (same
    interval/width as the band_snapback trigger config plus the drift
    reference: band_span + drift_ref_span + a few extra bars), so the
    read-side TTL cache in hl_client makes this a hit when perception's band
    trigger already pulled the same series seconds earlier — no extra network
    cost in the common path, and the gate stays self-contained (no
    GateContext plumbing).

    DRIFT REFERENCE (`drift_ref_span`, this gate's config block): the band
    edges stay the trigger's `band_span` MA, but the drift/direction verdict
    samples the SAME edge `drift_ref_span` bars back — the lag knob the
    trigger rework collapsed onto band_span, re-opened here for the gate
    alone. A late-chase entry is late against the LONGER trend, and a 16-bar
    EMA hugs price so tightly its own-window drift is structurally ~4-5x
    smaller (the GRASS 2026-08-26 tick read 6.4% over 48 bars but only 1.1%
    over 16 — the gate slept). Live value 32; absent = band_span = pre-key
    behaviour byte-identical. The drift gate's max_drift_pct stays the same
    1.5% bar.

    Fail-safes: disabled / no band config / candle fetch failure / short
    history / band not trending / breach below `min_breach_pct` / entry on
    the WITH-drift side (chasing the drift, not the bounce) = no opinion = pass.
    """
    cfg = gate_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return {"pass": True}
    from hermes_trader.agents.config_store import read_agent_config
    from hermes_trader.client.hl_client import fetch_hl_candles
    from hermes_trader.indicators.triggers import band_state

    try:
        agent_cfg = read_agent_config() or {}
    except Exception as e:
        logger.debug(f"[gate] band_counter_breach: agent-config read failed: {e}")
        return {"pass": True}
    bs = agent_cfg.get("band_snapback") or {}
    if not bool(bs.get("enabled")):
        return {"pass": True}
    ov = (bs.get("overrides") or {}).get(ctx.coin) or {}
    bs = {**bs, **{k: v for k, v in ov.items() if v is not None}}
    interval = str(bs.get("interval", "1h"))
    span = max(2, int(bs.get("band_span", 16)))
    # Drift-reference lag: absent -> band_span (the trigger's own-window
    # semantics); explicit value -> the longer late-chase reference.
    try:
        drift_ref = int(cfg.get("drift_ref_span", span) or span)
    except (TypeError, ValueError):
        drift_ref = span
    drift_ref = max(1, drift_ref)

    try:
        candles = fetch_hl_candles(ctx.coin, interval=interval,
                                   count=span + drift_ref + 4)
    except Exception as e:
        logger.debug(f"[gate] band_counter_breach: candle fetch failed for {ctx.coin}: {e}")
        return {"pass": True}
    if not candles:
        return {"pass": True}
    try:
        st = band_state(
            candles,
            band_span=span,
            max_drift_pct=float(bs.get("max_drift_pct", 1.5)),
            ma_type=str(bs.get("ma_type", "ema")),
            drift_ref=drift_ref,
        )
    except Exception as e:
        logger.debug(f"[gate] band_counter_breach: band_state failed for {ctx.coin}: {e}")
        return {"pass": True}
    if st is None or not st["trending"]:
        return {"pass": True}

    # Breach = price beyond the OPPOSITE-side edge of the drift: drift DOWN
    # + above the upper edge (bounce) / drift UP + below the lower edge (dip).
    # The entry side must be the bounce/dip side — the counter-trend one.
    min_breach = float(cfg.get("min_breach_pct", 1.0) or 1.0)
    if st["direction"] == "DOWN" and ctx.trade_side == "long":
        breach = st["breach_opposite_pct"]
        shape = f"bounce above the upper edge of a DOWN-drifting {interval} band (drift {st['drift_pct']:.1f}%, breach {breach:.1f}%)"
    elif st["direction"] == "UP" and ctx.trade_side == "short":
        breach = st["breach_opposite_pct"]
        shape = f"dip below the lower edge of an UP-drifting {interval} band (drift {st['drift_pct']:.1f}%, breach {breach:.1f}%)"
    else:
        return {"pass": True}  # WITH the drift (or no breach) — not this shape
    if breach < min_breach:
        return {"pass": True}

    min_conf = float(cfg.get("min_conf", 0.90) or 0.90)
    if ctx.confidence >= min_conf:
        return {"pass": True, "via": "confidence"}
    reason = (
        f"[gate:band_counter_breach] {ctx.coin} {ctx.trade_side} at conf "
        f"{ctx.confidence:.2f} < {min_conf:.2f}: {shape} — the counter-trend "
        f"chase at the top/bottom of a drift needs >= {min_conf:.2f} conviction"
    )
    if bool(cfg.get("shadow_mode", True)):
        logger.warning(
            f"[gate] band_counter_breach would-block {ctx.coin} {ctx.trade_side} "
            f"(conf {ctx.confidence:.2f} < {min_conf:.2f}): {shape} — shadow_mode "
            f"ON, logging only, not blocking. Set "
            f"band_counter_breach_gate.shadow_mode=false to enable."
        )
        return {"pass": True, "reason": reason, "shadow_would_block": True}
    return {"pass": False, "reason": reason}


def squeeze_extreme_gate(ctx: GateContext, gate_cfg: Dict[str, Any]) -> GateResult:
    """Squeeze extreme-without-breakout conviction gate (SHADOW by default).

    The third member of the shadow-gate house pattern, encoding the
    "chasing without confirmation" bucket from the 2026-08-27 15-day
    ledger replay (scratch/research/counterfactual_gate.py): entries on the
    candidate's side while price sits at the extreme of the prior 48h 1h
    Donchian range (top `extreme_pct`% for longs, bottom for shorts) with
    NO fresh aligned breakout confirming the move. The replay: extreme
    WITH fresh breakout confirmation = +$1.96 / 79% win; extreme WITHOUT =
    −$10.73 / 71% win — the flag alone removed ~$10.7 of the window's
    losses. The research verdict was "right shape, overfit threshold":
    the 5% extreme zone was chosen after seeing the data, so this gate
    ships SHADOW and the prospective flag count is the validation.

    The flag itself is computed by squeeze_signal._set_gate (one sync read,
    per-candidate-side, on every _evaluate return point) and fed here via
    GateContext.squeeze_extreme_no_breakout — the gate itself is a pure ctx
    function, no candle fetch, no config read beyond its own block.

    SHADOW MODE: with `shadow_mode` true (default) the gate STRUCTURALLY
    returns pass=True and only carries a `shadow_would_block` marker, which
    the executor logs loudly — same pattern as chronos_mismatch /
    band_counter_breach. It cannot alter live execution until the operator
    flips shadow_mode to false in .agent-config.json.

    Conviction bar: conf >= `min_conf` (0.90) OR composite >=
    `min_composite` (60) — same elevated bar as the chronos conviction
    gates; a genuinely high-conviction late entry still gets through.

    Fail-safes (no-opinion pass): disabled; flag is None (squeeze disabled /
    fetch failed / no data) or False (not at the extreme, or a fresh aligned
    breakout confirms the move); conviction bar met.
    """
    cfg = gate_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return {"pass": True}
    if ctx.squeeze_extreme_no_breakout is not True:
        return {"pass": True}
    min_conf = float(cfg.get("min_conf", 0.90) or 0.90)
    min_composite = float(cfg.get("min_composite", 60.0) or 60.0)
    if ctx.confidence >= min_conf or ctx.composite_score >= min_composite:
        return {"pass": True}
    reason = (f"squeeze_extreme ({ctx.trade_side} entry at the 48h 1h-channel "
              f"extreme with no fresh aligned breakout; conf "
              f"{ctx.confidence:.2f} < {min_conf:.2f}, composite "
              f"{ctx.composite_score:.1f} < {min_composite:.0f})")
    if bool(cfg.get("shadow_mode", True)):
        return {"pass": True, "reason": reason, "shadow_would_block": True}
    return {"pass": False, "reason": reason}


def duelist_veto_gate(ctx: GateContext, gate_cfg: Dict[str, Any]) -> GateResult:
    """Duelist-veto conviction gate (SHADOW by default).

    The fifth member of the shadow-gate house pattern. The strict veto
    validated by the 2026-08-31 48h replay (Aug 29 09:07 → Aug 31 09:07
    UTC, 41 executed trades): when the primary issues a DIRECTIONAL call
    (LONG/SHORT) and the A/B duelist — a second model answering the same
    research prompt — abstains with PASS, the entry is vetoed unless it
    clears the elevated-conviction bar (conf >= `min_conf` OR composite >=
    `min_composite`). The replay: the veto removed 12 of 41 entries
    (+5 losers / −7 winners, net +$1.83 vs the −$2.72 baseline) and caught
    the 2026-08-31 short-cluster's 240-min stale-flat-timeout trap exactly —
    the duelist had said PASS on 5 of the 6 losers that died on the timeout.

    The veto is deliberately STRICT by default (no conviction-escape): the
    48h replay's +$1.83 result was computed with the strict rule, and an
    escape bar (conf >= 0.90) would have changed the count because several
    vetoed entries ran at 0.82. `min_conf` / `min_composite` therefore
    implement the "deliberately conservative" escape — set them high (or
    remove the check) if the duelist's PASS should carry no weight at all.

    SHADOW MODE: with `shadow_mode` true (default, and how it ships) the
    gate STRUCTURALLY returns pass=True and only carries a
    `shadow_would_block` marker, which the executor logs loudly next to the
    other shadow gates (`[gate][SHADOW] duelist_veto WOULD HAVE BLOCKED …`).
    The executed-trade result dict carries the full `gate_results`, so the
    marker lands in the session log's `execute` event and the
    `Trade result:` log line for free — joinable to the ledger (by
    analysis_id/coin/side) and to the duel JSONL (by ts/coin).

    Fail-safes (no-opinion pass): disabled; duelist verdict absent (the
    duelist is disabled or the second LLM call failed); the duelist AGREES
    (its verdict matches the primary's side) or is directionally neutral
    within the deadband — PASS is the only abstention. A data gap can never
    block a trade.
    """
    cfg = gate_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return {"pass": True}
    dl = ctx.duelist_verdict
    if not dl:
        return {"pass": True}
    dl_upper = str(dl).upper()
    # The veto only fires on the primary's directional side being abstained
    # by the duelist. A PASS primary never reaches this gate (route_verdict
    # routes PASS to "none" unless a force-execute hint fires, in which case
    # the force path re-evaluates every gate — including this one — on the
    # upgraded side; a duelist PASS on a force-execute PASS is a genuine
    # disagreement and the veto applies).
    if dl_upper == "PASS":
        pass  # duelist abstained while primary is directional — the veto shape
    elif (ctx.trade_side == "long" and dl_upper == "SHORT") or \
         (ctx.trade_side == "short" and dl_upper == "LONG"):
        # The duelist took the OPPOSITE side — a stronger disagreement than
        # abstention. The same veto applies (it is the "duelist says NO to
        # this entry" family).
        pass
    else:
        # The duelist AGREES with the primary (same side), or the verdict is
        # unrecognised — no opinion, pass.
        return {"pass": True}
    min_conf = float(cfg.get("min_conf", 0.90) or 0.90)
    min_composite = float(cfg.get("min_composite", 60.0) or 60.0)
    if ctx.confidence >= min_conf or ctx.composite_score >= min_composite:
        return {"pass": True}
    reason = (f"duelist_veto ({ctx.trade_side} entry, duelist said "
              f"{dl_upper}; conf {ctx.confidence:.2f} < {min_conf:.2f}, "
              f"composite {ctx.composite_score:.1f} < {min_composite:.0f})")
    if bool(cfg.get("shadow_mode", True)):
        return {"pass": True, "reason": reason, "shadow_would_block": True}
    return {"pass": False, "reason": reason}


def eval_all_gates(
    ctx: GateContext,
    config: Optional[Dict[str, Any]] = None,
    last_trade_time: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate all risk gates and collect results."""
    from hermes_trader.agents.config import get_config
    runtime_config = get_config()
    effective_config = config or runtime_config
    results = {}
    # Regime-aware confidence floor: a WITH-TREND (aligned) trade — long in an up
    # regime, SHORT in a DOWN regime — gets a lower bar (`aligned_min_conf`) than
    # the default `min_ai_confidence`. The 0.78 default was calibrated on the
    # LONG-side 0.70-0.80 leak; applying it to aligned shorts made us sit out
    # selloffs (e.g. SOL SHORT 0.72 / -6.3% / $399M blocked). Demand full
    # conviction only to fight the trend (neutral/counter-trend keep the default).
    min_conf = float(_cfg(effective_config, "min_ai_confidence", 0.8))
    aligned_min_conf = effective_config.get("aligned_min_conf")
    if aligned_min_conf is not None:
        try:
            from hermes_trader.agents.market_regime import detect_regime
            _rg = detect_regime(ctx.coin)  # cached (TTL); market_regime_gate reuses it
            _aligned = (_rg == "up" and ctx.trade_side == "long") or \
                       (_rg == "down" and ctx.trade_side == "short")
            if _aligned:
                min_conf = min(min_conf, float(aligned_min_conf))
        except Exception:
            pass
    results["confidence"] = confidence_gate(ctx, min_conf)
    results["max_concurrent"] = max_concurrent_positions_gate(ctx, _cfg(effective_config, "max_concurrent", 3))
    results["notional_cap"] = per_trade_notional_cap_gate(ctx, _cfg(effective_config, "max_trade_notional_usd", 300))
    # Equity-relative daily kill + halt timer (see daily_loss_kill_switch).
    _kill_thr = effective_daily_kill_usd(effective_config, ctx.equity)
    _halt_min = 0.0
    if _kill_thr > 0:
        try:
            from hermes_trader.agents import memory as _mem
            _halt_min = _mem.memory.daily_halt_remaining_min()
        except Exception:
            pass
    results["daily_loss"] = daily_loss_kill_switch(ctx, _kill_thr, _halt_min)
    results["daily_giveback"] = daily_giveback_gate(
        ctx,
        float(_cfg(effective_config, "daily_giveback_halt_pct", 0.0) or 0.0),
        float(_cfg(effective_config, "daily_giveback_min_peak_usd", 20.0) or 0.0),
    )
    runner_gate_cfg = effective_config.get("runner_entry_gate") or {}
    results["liquidity"] = market_liquidity_floor(
        ctx,
        float(_cfg(effective_config, "min_market_volume_usd", 5_000_000) or 5_000_000),
        float(_cfg(effective_config, "min_hip3_volume_usd", 500_000) or 500_000),
        gate_config=runner_gate_cfg,
    )
    results["short_liquidity"] = short_liquidity_floor(
        ctx, float(_cfg(effective_config, "min_short_volume_usd", 0) or 0)
    )
    results["coin_filter"] = coin_allowlist_gate(
        ctx,
        _cfg(effective_config, "coin_allowlist", []),
        _cfg(effective_config, "coin_blocklist", []),
    )
    results["no_pyramid"] = opposite_direction_guard(ctx)
    results["cooldown"] = cooldown_gate(ctx, last_trade_time, _cfg(effective_config, "cooldown_min", 60))
    results["correlation"] = correlation_cap(ctx, int(_cfg(effective_config, "max_crypto_long_correlated", 2)))
    results["equity_risk"] = equity_risk_cap(ctx, _cfg(effective_config, "max_total_notional_pct", 1.0))
    results["market_regime"] = market_regime_gate(
        ctx, _cfg(effective_config, "counter_regime_min_conf", 0.7),
        bool(_cfg(effective_config, "block_counter_trend_bypass", False)),
        float(_cfg(effective_config, "crowded_with_min_conf", 0.0) or 0.0),
    )
    results["news"] = news_blackout_gate(ctx)
    # Shadow by default (see gate docstring): structurally passes and logs a
    # would-block until shadow_mode is flipped in .agent-config.json.
    results["chronos_mismatch"] = chronos_mismatch_gate(
        ctx, effective_config.get("chronos_mismatch_gate") or {})
    # Tail-trigger (shape-based) counter-forecast veto. Same warm cache and
    # same elevated-conviction bar as chronos_mismatch, but keys off the
    # ADVERSE quantile PATH (min q10 / max q90 over the first `window_steps`
    # 5m steps) rather than the path-mean scalar. SHADOW until shadow_mode
    # is flipped. Passes when the warm cache has no per-step paths.
    results["chronos_tail_trigger"] = chronos_tail_trigger_gate(
        ctx, effective_config.get("chronos_tail_trigger_gate") or {})
    # Band counter-trend breach (GRASS shape): disabled by default in the
    # shadow-gate config block below; when enabled it runs in shadow_mode
    # (would-block marker) until the operator promotes it.
    results["band_counter_breach"] = band_counter_breach_gate(
        ctx, effective_config.get("band_counter_breach_gate") or {})
    # Squeeze extreme-without-breakout (chasing the channel extreme without a
    # fresh aligned breakout — the replay's worst-loss bucket). Pure ctx read:
    # the flag is computed by squeeze_signal on the gate-side sync read in
    # maybe_execute. Shadow until shadow_mode is flipped. Passes when the
    # flag is absent/False (data gap can never block a trade).
    results["squeeze_extreme"] = squeeze_extreme_gate(
        ctx, effective_config.get("squeeze_extreme_gate") or {})
    # Duelist veto: the A/B duelist's abstention (PASS) or opposite-side call
    # on the primary's directional entry. Strict veto (no escape by default);
    # shadow until the operator flips shadow_mode to false. Passes when the
    # duelist has no verdict (disabled / failed) or AGREES with the primary —
    # a data gap can never block a trade.
    results["duelist_veto"] = duelist_veto_gate(
        ctx, effective_config.get("duelist_veto_gate") or {})

    block_reasons = []
    blocked = False
    for key, result in results.items():
        if not result.get("pass"):
            blocked = True
            block_reasons.append(result.get("reason", key))

    return {"results": results, "blocked": blocked, "block_reasons": block_reasons}
