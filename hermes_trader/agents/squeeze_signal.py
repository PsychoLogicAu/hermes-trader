"""Squeeze-breakout SHADOW signal — 1h Donchian channel breakout, logged only.

Wires the researched `squeeze_breakout` entry signal into the live bot in
SHADOW mode: it is LOGGED (and appended to the trade ledger) on every
research candidate, but it is NOT fed to the LLM prompt, NOT used for sizing,
and NOT allowed to gate or trigger any trade. Forward validation decides
whether it ever earns a real execution path.

The rule (identical to scratch/research/signals.py `squeeze_breakout`, the
OOS-verified candidate from the 2026-08 trend-reversal research — see
scratch/research/REPORT.md):

  1h confirmed candles only (the forming bar is dropped, so this is the same
  close convention the harness evaluated):
    - Donchian channel = high/low of the PRIOR `lookback` 1h bars (default 48)
    - ACTIVE LONG  if close > channel high
    - ACTIVE SHORT if close < channel low
    - the breakout bar must carry a real body: body >= 0.5 x bar range
      (a wick-only pierce is not a breakout)
    - FRESHNESS: the signal only lives within `fresh_min` minutes (default 15)
      of the 1h bar CLOSE — the same anti-re-fire gate the research used.
      Outside the window the signal decays to inactive rather than persisting
      for hours.

Config (`.agent-config.json`, hot-reloaded per call):
    squeeze_signal:
        enabled: false            # global toggle — SHADOW until proven
        debug: false              # extra log detail (channel bounds, px)
        lookback: 48              # Donchian lookback in 1h bars
        fresh_min: 15             # minutes after the 1h close the signal lives
        cache_ttl_seconds: 300    # per-coin result TTL (bounds fetch cost)
        extreme_pct: 5.0          # composite-gate zone: top/bottom % of the
                                  # 48h range where "extreme" is defined
                                  # (0 disables the flag, shadow-only either way)

Cost: one 1h candle fetch per coin per TTL (the shared `fetch_hl_candles`
already 90s-caches the raw snapshot, so steady-state cost is ~0). The async
daemon path runs only on the research-candidate hot path and never blocks.

Ledger: every ACTIVE signal appends a `SHADOW` row to trades.jsonl (event type
distinct from OPEN/CLOSE) carrying the would-be entry/stop/tp so the
counterfactual P/L can be replayed against the live DSL exit logic later.

Composite entry-gate observability (2026-08-27 retrospective, scratch/
research/counterfactual_gate.py): the 15-day ledger's worst losses were
same-side re-entries at the EXTREME of the 48h channel with NO fresh breakout
confirming the move (the "chasing without confirmation" bucket). Two fields
are computed on every evaluation and attached to each `Trade result:` line:

  - `chan_pos`: the last confirmed 1h close's 0–1 position in the PRIOR
    48h Donchian range (same window the breakout rule uses; a breakout
    close reads > 1.0 or < 0.0).
  - `extreme_no_breakout`: the composite gate flag — True when the
    candidate's side is at the extreme (`chan_pos` above 1 - extreme_pct /
    below extreme_pct, default 5%) with NO active breakout aligned with the
    candidate. Shadow ONLY: it is logged, never gating. Two weeks of
    prospective data decides whether the chasing bucket is real or a
    15-day artifact.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hermes_trader.agents.config_store import read_agent_config
from hermes_trader.client.hl_client import fetch_hl_candles
from hermes_trader.indicators.math import atr as calc_atr
from hermes_trader.ledger import _append_event
from hermes_trader.models.types import Candle

logger = logging.getLogger(__name__)

_H_MS = 3_600_000

# Live stop/TP ATR multiples (mirrors executor.py so the ledger row carries
# the levels the bot WOULD arm if this entry were real).
_SL_ATR_MULT = 1.5
_TP_ATR_MULT = 1.0


# ── Config ────────────────────────────────────────────────────────────────────
def _get_config() -> Dict[str, Any]:
    """Read the squeeze_signal block fresh from .agent-config.json (hot)."""
    try:
        return read_agent_config().get("squeeze_signal") or {}
    except Exception:
        return {}


# ── Result structure ──────────────────────────────────────────────────────────
@dataclass
class SqueezeSignal:
    coin: str
    verdict_side: str                  # candidate's side ("long"/"short"); alignment only
    active: bool
    side: Optional[str] = None         # breakout direction ("long"/"short")
    score: Optional[float] = None      # conviction 0..1 (sizing hook, reported only)
    close: Optional[float] = None      # breakout 1h close
    chan_high: Optional[float] = None
    chan_low: Optional[float] = None
    ext_pct: Optional[float] = None    # close beyond the channel edge, %
    atr1h: Optional[float] = None      # absolute ATR(14) on 1h
    atr1h_pct: Optional[float] = None
    fresh_age_min: Optional[float] = None  # minutes since the breakout 1h close
    breakout_bar_t: Optional[int] = None   # open time (ms) of the breakout 1h bar
    lookback: int = 48
    chan_pos: Optional[float] = None       # last confirmed close's 0-1 position
    # in the prior 48h Donchian range (>1 above the high, <0 below the low)
    extreme_no_breakout: bool = False      # composite gate: side at the
    # channel extreme with no fresh aligned breakout (shadow flag, never gates)
    logged: bool = False               # a ledger row was written for this breakout
    error: Optional[str] = None


# ── Per-coin cache (TTL) ──────────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}


def _cache_get(coin: str, ttl: float) -> Optional[SqueezeSignal]:
    with _cache_lock:
        entry = _cache.get(coin)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["sig"]
    return None


def _cache_set(coin: str, sig: SqueezeSignal, ttl: float) -> None:
    with _cache_lock:
        _cache[coin] = {"sig": sig, "ts": time.time()}


# ── Composite entry-gate flag (shadow observability) ────────────────────────
def _set_gate(base: SqueezeSignal, cfg: Dict[str, Any]) -> None:
    """Set `extreme_no_breakout` from `chan_pos` + the signal state.

    True  = the candidate's side sits at the channel extreme (top 5% for a
    long, bottom 5% for a short) with NO fresh breakout aligned with it —
    the "chasing without confirmation" bucket from the 15-day retrospective
    (scratch/research/counterfactual_gate.py). False = either not at the
    extreme, or a fresh aligned breakout is confirming the move.

    SHADOW ONLY: this is an observability field, never a gate. Called at every
    `_evaluate` return point (active and inactive) so the flag is recorded
    exactly where the losing pattern lived.
    """
    pos = base.chan_pos
    if pos is None:
        base.extreme_no_breakout = False
        return
    zone = float(cfg.get("extreme_pct", 5.0)) / 100.0
    if zone <= 0:
        base.extreme_no_breakout = False
        return
    at_extreme = pos < zone if base.verdict_side == "short" else pos > 1.0 - zone
    confirmed = base.active and base.side == base.verdict_side
    base.extreme_no_breakout = bool(at_extreme and not confirmed)


# ── Core evaluation (pure; candle list injectable for tests) ─────────────────
def _evaluate(coin: str, verdict_side: str, cfg: Dict[str, Any],
              candles: List[Candle]) -> SqueezeSignal:
    base = SqueezeSignal(
        coin=coin, verdict_side=verdict_side or "long", active=False,
        lookback=int(cfg.get("lookback", 48)),
    )
    # Drop the still-forming 1h bar: the researched convention is confirmed
    # closes only. (A fetch that returns only the forming bar -> insufficient.)
    now_ms = int(time.time() * 1000)
    confirmed = [c for c in candles if c.t + _H_MS <= now_ms]
    if len(confirmed) < base.lookback + 2:
        base.error = f"insufficient 1h history ({len(confirmed)} bars)"
        return base
    lb = base.lookback
    chan = confirmed[-lb - 1:-1]
    last = confirmed[-1]
    hi = max(c.h for c in chan)
    lo = min(c.l for c in chan)
    base.chan_high, base.chan_low = hi, lo
    base.close = last.c
    # Composite-gate input (exact research definition): the last confirmed
    # close's 0-1 position in the PRIOR 48h Donchian range. >1 = above the
    # channel high, <0 = below the channel low.
    if hi > lo:
        base.chan_pos = (last.c - lo) / (hi - lo)
    if lo < last.c < hi:
        base.error = "inside channel"
        _set_gate(base, cfg)
        return base
    side = "long" if last.c > hi else "short"
    body = abs(last.c - last.o)
    rng = last.h - last.l
    if rng > 0 and body < 0.5 * rng:
        base.error = "wick-only pierce (body < 50% of range)"
        _set_gate(base, cfg)
        return base
    # Freshness window: act only within fresh_min of the 1h close.
    age_min = (now_ms - (last.t + _H_MS)) / 60_000
    if age_min < 0 or age_min >= float(cfg.get("fresh_min", 15.0)):
        base.error = f"stale (1h close {age_min:.0f}m ago)"
        _set_gate(base, cfg)
        return base
    vals = [v for v in calc_atr(confirmed, 14) if v == v and v > 0]
    atr1 = vals[-1] if vals else 0.0
    ext_pct = ((last.c - hi) if side == "long" else (lo - last.c)) / last.c * 100.0
    score = 0.6 + min(0.4, ext_pct / max(atr1 / last.c * 100.0, 1e-9)) if atr1 > 0 else 0.7
    base.active = True
    base.side = side
    base.score = min(1.0, score)
    base.ext_pct = ext_pct
    base.atr1h = atr1
    base.atr1h_pct = (atr1 / last.c * 100.0) if last.c > 0 else None
    base.fresh_age_min = age_min
    base.breakout_bar_t = last.t
    _set_gate(base, cfg)
    return base


def _inactive(coin: str, verdict_side: str, error: str,
              lookback: int) -> SqueezeSignal:
    return SqueezeSignal(coin=coin, verdict_side=verdict_side or "long",
                         active=False, lookback=lookback, error=error)


# ── Fetch (sync, cache-aware) ─────────────────────────────────────────────────
def _fetch(coin: str, verdict_side: str) -> SqueezeSignal:
    cfg = _get_config()
    if not cfg.get("enabled", False):
        return _inactive(coin, verdict_side, "disabled",
                         int(cfg.get("lookback", 48)))
    ttl = float(cfg.get("cache_ttl_seconds", 300))
    cached = _cache_get(coin, ttl)
    if cached is not None:
        # Alignment + the composite-gate flag depend on the candidate side —
        # recompute per call from the stored chan_pos (free, no fetch).
        cached.verdict_side = verdict_side or "long"
        _set_gate(cached, cfg)
        return cached
    candles = fetch_hl_candles(coin, "1h", int(cfg.get("lookback", 48)) + 25)
    sig = _evaluate(coin, verdict_side, cfg, candles or [])
    if sig.active and not sig.error:
        _cache_set(coin, sig, ttl)
    # Log once per real compute; cache hits return above silently.
    logger.info(_format_log(sig, bool(cfg.get("debug", False))))
    return sig


# ── Log format ────────────────────────────────────────────────────────────────
def _format_log(sig: SqueezeSignal, debug: bool) -> str:
    if sig.error and not sig.active:
        gate = " extreme_no_breakout" if sig.extreme_no_breakout else ""
        return (f"[squeeze-shadow] {sig.coin} ({sig.verdict_side}): "
                f"inactive ({sig.error}){gate}")
    arrow = "↑" if sig.side == "long" else "↓"
    aligned = (sig.side == sig.verdict_side)
    tag = "ALIGNED" if aligned else "AGAINST"
    gate = " extreme_no_breakout" if sig.extreme_no_breakout else ""
    line = (
        f"[squeeze-shadow] {sig.coin} ({sig.verdict_side}): "
        f"{arrow} {sig.side} score={sig.score:.2f} ext={sig.ext_pct:+.2f}% "
        f"atr1h={sig.atr1h_pct:.2f}% age={sig.fresh_age_min:.0f}m "
        f"lb={sig.lookback} {tag}{gate} (shadow: logged only, no execution)"
    )
    if debug:
        pos = f"{sig.chan_pos:.3f}" if sig.chan_pos is not None else "?"
        line += (f" | close={sig.close:.6g} chan=[{sig.chan_low:.6g}, {sig.chan_high:.6g}] "
                 f"atr={sig.atr1h:.6g} chan_pos={pos}")
    return line


# ── Async daemon entry point (the hot-path call) ──────────────────────────────
def get_squeeze_signal_async(coin: str, verdict_side: str) -> None:
    """Fire-and-forget shadow evaluation on a daemon thread.

    NEVER blocks the caller; every failure is swallowed. Warming the cache here
    is what makes the trade-result attach (sync, cache-only) cheap.
    """
    cfg = _get_config()

    def _worker() -> None:
        try:
            if not cfg.get("enabled", False):
                return
            _fetch(coin, verdict_side)
        except Exception as e:  # noqa: BLE001 — a shadow signal must never break the loop
            logger.debug(f"[squeeze-shadow] {coin} worker failed: {e}")

    threading.Thread(target=_worker, name=f"squeeze-{coin}", daemon=True).start()


# ── Sync wrapper (trade-result attach; cache-first) ───────────────────────────
def get_squeeze_signal_sync(coin: str, verdict_side: str) -> SqueezeSignal:
    """Return the squeeze signal, computing only on a cache miss.

    The async daemon path warms the cache per candidate; by the time the
    trade-result attach runs the fetch is normally a 90s-TTL candle cache hit,
    so this adds ~0 ms steady-state. Wrapped non-fatally by the caller.
    """
    return _fetch(coin, verdict_side)


# ── Ledger persistence (counterfactual P/L later) ─────────────────────────────
# One ledger row per (coin, breakout bar). An active breakout stays active for
# up to `fresh_min` (15m), longer than the 300s result cache TTL — without
# this the same breakout would append a row on every re-evaluation,
# inflating the counterfactual trade count. Bounded; prunes from the front.
_logged_lock = threading.Lock()
_logged_breakouts: List[Any] = []          # (coin, breakout_bar_t) in insertion order
_LOGGED_MAX = 1000


def _already_logged(coin: str, bar_t: Optional[int]) -> bool:
    with _logged_lock:
        return (coin, bar_t) in _logged_breakouts


def record_shadow(coin: str, verdict_side: str, sig: SqueezeSignal,
                  analysis_id: Optional[str] = None, mode: str = "") -> None:
    """Append a `SHADOW` row to trades.jsonl for an ACTIVE squeeze signal.

    Distinct event type (never OPEN/CLOSE) so existing ledger queries are
    unaffected. Carries the would-be entry/stop/tp (live SL/TP ATR multiples)
    so the counterfactual P/L can be replayed against the live DSL exit logic
    — join to the session log via `analysis_id` for the candidate context.

    One row per (coin, breakout_bar_t): the in-process dedup set survives the
    15m active window; a process restart mid-window can re-log a still-active
    breakout once (harmless — a replay can dedup on `breakout_bar_t`).
    """
    try:
        # One row per breakout bar (see _logged_breakouts above).
        if sig.logged or _already_logged(coin, sig.breakout_bar_t):
            sig.logged = True
            return
        entry = sig.close or 0.0
        atr1 = sig.atr1h or 0.0
        if sig.side == "long":
            stop, tp = entry - atr1 * _SL_ATR_MULT, entry + atr1 * _TP_ATR_MULT
        elif sig.side == "short":
            stop, tp = entry + atr1 * _SL_ATR_MULT, entry - atr1 * _TP_ATR_MULT
        else:
            stop, tp = None, None
        _append_event("SHADOW", {
            "signal": "squeeze_breakout",
            "coin": coin,
            "verdict_side": sig.verdict_side,
            "side": sig.side,
            "score": round(sig.score, 4) if sig.score is not None else None,
            "close": entry,
            "stop_px": stop,
            "tp_px": tp,
            "ext_pct": round(sig.ext_pct, 4) if sig.ext_pct is not None else None,
            "atr1h": sig.atr1h,
            "fresh_age_min": round(sig.fresh_age_min, 2) if sig.fresh_age_min is not None else None,
            "chan_high": sig.chan_high,
            "chan_low": sig.chan_low,
            "chan_pos": round(sig.chan_pos, 4) if sig.chan_pos is not None else None,
            "extreme_no_breakout": sig.extreme_no_breakout,
            "breakout_bar_t": sig.breakout_bar_t,
            "lookback": sig.lookback,
            "mode": mode,
            "analysis_id": analysis_id,
        })
        with _logged_lock:
            _logged_breakouts.append((coin, sig.breakout_bar_t))
            if len(_logged_breakouts) > _LOGGED_MAX:
                del _logged_breakouts[: len(_logged_breakouts) - _LOGGED_MAX]
        sig.logged = True
    except Exception as e:  # noqa: BLE001 — logging must never break the loop
        logger.debug(f"[squeeze-shadow] ledger append failed for {coin}: {e}")