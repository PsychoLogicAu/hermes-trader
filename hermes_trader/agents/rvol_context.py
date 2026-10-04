"""RVOL entry context — relative volume of the most recent closed 5m candle.

r/ai_trading candidate #2 (WATCHLIST B.32, 2026-10-04): a breakout-chasing bot
cut 67 losing trades by feeding the model the relative volume of the signal
candle. Our own counterfactual replay over ALL executed trades
(scratch/_rvol_breakout_replay.py, 422 closed round trips 08-25→10-04):

  * entries on a candle with RVOL < 0.5 (half the prior 20-bar average volume)
    went 116 trades / −$132.91 / PF 0.22, negative in every UTC hour block;
  * but ex-worst-10 that cohort is +$1.67 and the damage concentrated in two
    weeks — tails, not a stable veto (the B.31 fade-veto failure mode again),
    so this ships as PROMPT CONTEXT ONLY, never a gate;
  * the useful shape is monotone up to ~1.5x: [0.5,1) PF 1.24, [1,1.5) PF 1.89
    (best band), then noise at the extreme tail.

The audit that opened this item: the entry prompt carried ZERO volume context —
the OHLC block has no volume column and ``volumeSpike`` only surfaces as a
binary fired-flag. This module gives the model where on the volume spectrum the
entry candle actually sits, as one factual line plus one interpretive framing
drawn from our own replay numbers.

RVOL here = vol(most recent closed 5m bar) / mean(vol of the prior 20 bars),
computed from candles the research cycle ALREADY holds for the 1h timeframe?
No — 5m is the scalp granularity (matches triggers.volume_spike's window and
the replay's definition). The research() pipeline fetches 1h/4h/1d but not 5m,
so this module does its own single cached ``fetch_hl_candles(coin, "5m", 60)``
call (TTL-cached in hl_client; warm cost ~0, cold miss one extra info POST per
coin per TTL). Sparse-market guard mirrors triggers.volume_spike: if >50% of
the baseline bars have zero volume, no line is emitted.

SHIPS DISABLED: absent/false ``rvol_context.enabled`` → returns "" and the
prompt is byte-identical to today (same contract as prior_call_context).

Config (hot, read per call)::

    "rvol_context": {
        "enabled": false,      # master switch (OFF = byte-identical prompt)
        "baseline_bars": 20,   # prior bars averaged for the baseline volume
        "interval": "5m"       # candle interval of the signal bar + baseline
    }

Every path is fail-safe: any fault (fetch failure, short history, sparse tape,
bad config) yields "" — a broken context line must never break prompt assembly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from hermes_trader.agents.config_store import read_agent_config
from hermes_trader.client.hl_client import fetch_hl_candles
from hermes_trader.indicators.math import candle_val

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "baseline_bars": 20,
    "interval": "5m",
}


def _get_config() -> Dict[str, Any]:
    try:
        cfg = read_agent_config().get("rvol_context") or {}
        if not isinstance(cfg, dict):
            return dict(DEFAULTS)
        return {**DEFAULTS, **cfg}
    except Exception:
        return dict(DEFAULTS)


def compute_rvol(coin: str, baseline_bars: int = 20, interval: str = "5m") -> Optional[float]:
    """RVOL of the most recent closed candle vs the mean of the prior bars.

    Returns None when not computable (fetch failure, insufficient history,
    sparse tape — >50% zero-volume baseline bars, or zero-average baseline).
    Note HL's candleSnapshot includes the still-forming bar as the last row;
    "most recent CLOSED" is therefore candles[-2] when present. We keep it
    simple and use the last row (same convention as triggers.volume_spike,
    which scores candle_val(candles[-1]) — consistency with the trigger that
    already feeds the composite score matters more than forming-bar purity).
    """
    try:
        need = baseline_bars + 2
        candles = fetch_hl_candles(coin, interval, max(need, 60))
        if len(candles) < baseline_bars + 1:
            return None
        vols = [candle_val(c, "v") for c in candles]
        current = vols[-1]
        window = vols[-1 - baseline_bars:-1]
        if sum(1 for v in window if v == 0) > len(window) * 0.5:
            return None  # sparse market — ratio meaningless (volume_spike parity)
        avg = sum(window) / len(window)
        if avg <= 0:
            return None
        return current / avg
    except Exception as e:  # never break prompt assembly over a context line
        logger.debug(f"[rvol_context] {coin}: compute failed: {e}")
        return None


def _label(rvol: float) -> str:
    """Coarse trader-vocabulary label for the ratio (replay-banded)."""
    if rvol < 0.5:
        return "very quiet"
    if rvol < 1.0:
        return "below average"
    if rvol < 2.0:
        return "above average"
    return "high"


def build_rvol_block(coin: str) -> str:
    """Render the RVOL context line, or "" when disabled/not computable."""
    cfg = _get_config()
    if not cfg.get("enabled", False):
        return ""
    try:
        rvol = compute_rvol(
            coin,
            baseline_bars=int(cfg.get("baseline_bars", 20)),
            interval=str(cfg.get("interval", "5m")),
        )
        if rvol is None:
            return ""
        return (
            f"Entry-candle volume (RVOL): {rvol:.2f}x the prior "
            f"{int(cfg.get('baseline_bars', 20))}-bar average — {_label(rvol)}. "
            "Historically on this book, entries taken on candles below half of "
            "average volume were the weakest cohort (tail losses on dead-tape "
            "moves); a normal-to-elevated ratio says participants are actually "
            "there. Context only — not a rule to follow mechanically."
        )
    except Exception as e:
        logger.debug(f"[rvol_context] {coin}: block build failed: {e}")
        return ""
