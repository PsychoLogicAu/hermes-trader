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
        chronos_q10_path_pct: Optional[List[float]] = None,
        chronos_q90_path_pct: Optional[List[float]] = None,
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
        # Per-step Chronos-2 quantile paths, % vs last close (same sync read;
        # None on error signals / old shapes). Fed to
        # chronos_tail_trigger_gate — the shape-based counter-forecast veto
        # validated by the 2026-08-28 60-flag replay (adverse-quantile early
        # tail > path-mean > endpoint for loss avoidance).
        self.chronos_q10_path_pct = chronos_q10_path_pct
        self.chronos_q90_path_pct = chronos_q90_path_pct


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


def daily_loss_kill_switch(ctx: GateContext, max_daily_loss: float) -> GateResult:
    if ctx.daily_pnl > max_daily_loss:
        return {"pass": True}
    return {"pass": False, "reason": f"daily loss killswitch triggered (PnL ${ctx.daily_pnl:.0f} <= ${max_daily_loss})"}


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
    if bool(cfg.get("shadow_mode", True)):
        return {"pass": True, "reason": reason, "shadow_would_block": True}
    return {"pass": False, "reason": reason}


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
    interval/window as the band_snapback trigger config), so the read-side
    TTL cache in hl_client makes this a hit when perception's band trigger
    already pulled the same series seconds earlier — no extra network cost
    in the common path, and the gate stays self-contained (no GateContext
    plumbing).

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
    window = int(bs.get("window", 48))

    try:
        candles = fetch_hl_candles(ctx.coin, interval=interval, count=2 * window + 4)
    except Exception as e:
        logger.debug(f"[gate] band_counter_breach: candle fetch failed for {ctx.coin}: {e}")
        return {"pass": True}
    if not candles:
        return {"pass": True}
    try:
        st = band_state(
            candles,
            window=window,
            max_drift_pct=float(bs.get("max_drift_pct", 1.5)),
            ma_type=str(bs.get("ma_type", "ema")),
            band_span=bs.get("band_span"),
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
    results["daily_loss"] = daily_loss_kill_switch(ctx, _cfg(effective_config, "max_daily_loss_usd", -100))
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

    block_reasons = []
    blocked = False
    for key, result in results.items():
        if not result.get("pass"):
            blocked = True
            block_reasons.append(result.get("reason", key))

    return {"results": results, "blocked": blocked, "block_reasons": block_reasons}
