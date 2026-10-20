"""Timestral-209M shadow signal module (shadow-mode, logged only).

Pulls Mistral's Timestral-209M (patch-transformer zero-shot forecaster,
Oct 2026) in as a standalone shadow signal next to Chronos-2, TimesFM-3 and
TiRex so it can be measured on the identical 5m context before any decision
integration. The shadow worker runs ASYNC on a daemon thread and logs every
forecast (forward validation); the model is preloaded at app init
(`preload_model`) and the per-coin cache (300s TTL by default) bounds
steady-state cost to one candle fetch + one inference per coin per TTL.

WHY (model card, 2026-10): Timestral-209M ranked #1 among zero-shot models
on all three main forecasting benchmarks at release — GIFT-Eval MASE 0.658
(chronos-2: 0.698), TIME MASE 0.626, fev-bench SQL skill 50.39 — the best
published headline numbers of any candidate in this stack. Whether that
holds on OUR 5m crypto stream at ctx-50 is exactly what the accrual answers
(chronos/timesfm/tirex each earned or lost trust the same way).

LICENSE: Timestral code is Apache-2.0; weights are Modified MIT (a misuse
clause, not a non-commercial one — unlike TimesFM-3's license). Enabling is
a pure perf decision, same posture as TiRex. The module still ships with
`enabled: false` until the operator flips it (first-ever load also downloads
the ~837MB fp32 checkpoint into the hf-cache volume).

Config-driven via `.agent-config.json` under `timestral_signal`:
    timestral_signal:
        enabled: false              # global toggle (default: false; flip to accrue)
        debug: false                # extra log detail; off = summary only
        in_prompt: false            # render the prompt block (see research._timestral_block)
        model_id: "mistralai/Timestral-209M"   # or -5M/-28M/-123M for a cheaper arm
        device: "cpu"               # no GPU in our stack
        context_length: 50          # candles fed as context (prod regime; model max 32768)
        forecast_horizon: 12        # steps ahead (5m bars; match chronos live; model max 1024/pass)
        min_conf_ratio: 0.25        # confidence floor: |median_pct| / spread_pct.
                                    # Below it the median sits inside the model's
                                    # own p10-p90 band, so the log flag reads
                                    # NEUTRAL (not ALIGN/MISMATCH). 0 disables.
        cache_ttl_seconds: 300      # TTL for per-coin cache
        timeout_seconds: 45         # abort deadline per forecast; 0 = no deadline
                                    # (209M params + 21-quantile head — slower than
                                    # tirex on CPU but single-pass with TAA off)

API shape (timestral, github.com/mistralai/timestral):
`TimestralForecaster.from_pretrained(model_id, device=...)` then
`.forecast(context, prediction_length, symmetric_averaging=, clip_nonnegative=)`
returns a `Forecast` with `.quantiles` (B, V, H, Q) — for our 1-D context
that is (1, 1, horizon, 21) — and `.quantile_levels`, the 21 checkpoint-fixed
levels (0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99). We resolve
p10/p50/p90 by VALUE against those levels (same guard as timesfm_signal: a
checkpoint shipping a different fixed set can't silently mislabel columns).
The library's DEFAULT test-time augmentation (clampsym: reflect a negated
context, mirror-average the two passes, clamp non-negative series) is
DISABLED here — symmetric_averaging doubles the forward pass for a gain
irrelevant to shadow logging, and chronos/tirex are also read single-pass.
The model card's headline benchmark numbers use it, so treat the accrual as
"raw single-pass Timestral" and re-measure before quoting card numbers.

timeout_seconds is enforced at the caller level exactly as in chronos/tirex:
the forecast runs on a dedicated worker thread and the CALLER abandons it
(error signal, never cached) once the deadline passes. The forward pass can't
be interrupted mid-inference, so an overrun leaves that thread busy until it
finishes; the point is the CALLER never blocks past the deadline.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from hermes_trader.agents.config_store import read_agent_config
from hermes_trader.client.hl_client import fetch_hl_candles
from hermes_trader.indicators.math import candle_val
from hermes_trader.models.types import Candle

logger = logging.getLogger(__name__)


# ── Singleton forecaster + config cache ───────────────────────────────────────
_timestral_lock = threading.Lock()
_timestral_forecaster: Optional[Any] = None
_timestral_loaded_at: float = 0
_timestral_config_cache: Dict[str, Any] = {}
_model_init_error: Optional[str] = None


def _get_timestral_config() -> Dict[str, Any]:
    """Read timestral config from `.agent-config.json` with sensible defaults."""
    global _timestral_config_cache
    try:
        cfg = read_agent_config()
        if cfg is not _timestral_config_cache:
            _timestral_config_cache = cfg
            return cfg.get("timestral_signal", {})
        return _timestral_config_cache.get("timestral_signal", {})
    except Exception as e:
        logger.debug(f"[timestral] config read error: {e}")
        return {}


def _load_forecaster() -> Any:
    """Build the TimestralForecaster from the checkpoint.

    Standalone load (same ownership as timesfm_signal) — expert_pool does not
    know timestral, so there is no shared singleton to borrow. torch/
    safetensors/huggingface_hub stay OFF the module-import path via the lazy
    import, exactly like the other forecaster modules.
    """
    cfg = _get_timestral_config()
    model_id = cfg.get("model_id", "mistralai/Timestral-209M")
    device = cfg.get("device", "cpu")
    from timestral import TimestralForecaster
    return TimestralForecaster.from_pretrained(model_id, device=device)


def _get_forecaster() -> Any:
    """Lazy-load the TimestralForecaster singleton (thread-safe). Returns
    None if disabled or if loading fails."""
    global _timestral_forecaster, _timestral_loaded_at, _model_init_error
    cfg = _get_timestral_config()
    if not cfg.get("enabled", False):
        return None

    with _timestral_lock:
        if _timestral_forecaster is not None:
            return _timestral_forecaster
        if _model_init_error is not None:
            return None

        model_id = cfg.get("model_id", "mistralai/Timestral-209M")
        device = cfg.get("device", "cpu")
        try:
            logger.info(f"[timestral] loading model {model_id} on {device}")
            start = time.time()
            forecaster = _load_forecaster()
            _timestral_forecaster = forecaster
            _timestral_loaded_at = time.time()
            logger.info(
                f"[timestral] model loaded in {time.time() - start:.1f}s "
                f"(quantiles: {tuple(forecaster.quantile_levels)})"
            )
            return forecaster
        except ImportError as e:
            logger.warning(f"[timestral] import failed (missing dependency?): {e}")
            _model_init_error = str(e)
            return None
        except Exception as e:
            logger.warning(f"[timestral] model load failed: {e}")
            _model_init_error = str(e)
            return None


def preload_model(timeout_s: float = 180.0) -> bool:
    """Preload Timestral on a bounded background thread at app init.

    Pays the one-time checkpoint cost OFF the first-scan critical path (the
    209M fp32 weights are ~837MB — the first-ever load also downloads them
    into the hf-cache volume). Bounded (join with timeout) so a hung load
    can't stall startup; on timeout/exception it falls back to lazy load on
    first use. Returns True if the model is ready when the join completes.
    """
    state = {"done": False, "forecaster": None, "error": None}

    def _run() -> None:
        try:
            state["forecaster"] = _get_forecaster()
        except Exception as e:
            state["error"] = e
        finally:
            state["done"] = True

    t = threading.Thread(target=_run, name="timestral-preload", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        logger.warning(
            f"[timestral] model preload exceeded {timeout_s:.0f}s — continuing; "
            "it will finish in the background or load lazily on first use")
    elif state["error"] is not None:
        logger.warning(f"[timestral] model preload failed (lazy fallback): {state['error']}")
    elif state["forecaster"] is None:
        logger.debug("[timestral] model not loaded (disabled or error cached); lazy fallback")
    else:
        logger.info("[timestral] model preloaded at init")
    return state["forecaster"] is not None


# ── Result structure ──────────────────────────────────────────────────────────
@dataclass
class TimestralSignal:
    coin: str
    side: str
    context_last: float        # last close used as context
    median: Optional[float]    # forecast median (path average over horizon)
    q_low: Optional[float]     # p10 path average
    q_high: Optional[float]    # p90 path average
    median_pct: Optional[float]  # (median - context_last) / context_last * 100
    spread_pct: Optional[float]  # (q_high - q_low) / context_last * 100
    horizon: int               # forecast horizon (in candle steps)
    model_id: str
    inference_ms: float
    error: Optional[str] = None
    # Per-step quantile paths, % vs context_last, across the full horizon.
    # Same contract as ChronosSignal/TimesfmSignal/TirexSignal so a
    # head-to-head gate (or a chronos->timestral swap) consumes any of the
    # four with one code path. None on every failure path.
    q10_path_pct: Optional[List[float]] = None
    q90_path_pct: Optional[List[float]] = None


# ── Per-coin cache (TTL-based) ────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}

# In-flight dedup: coin -> Event set when the owner's compute settles. The
# loop fire (async worker) and the route_verdict/executor attaches can reach
# _fetch_signal for the same cold coin within milliseconds of each other;
# without this, BOTH compute and BOTH log — duplicate accrual lines that skew
# any grep-counted rate (the failure mode fixed in tirex_signal/expert_select).
# One compute per coin per TTL; late callers wait on the owner and read its
# cache.
_inflight_lock = threading.Lock()
_inflight: Dict[str, threading.Event] = {}


def _cache_get(coin: str, ttl: float) -> Optional[TimestralSignal]:
    with _cache_lock:
        entry = _cache.get(coin)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["signal"]
        return None


def _cache_set(coin: str, signal: TimestralSignal, ttl: float) -> None:
    with _cache_lock:
        _cache[coin] = {"signal": signal, "ts": time.time()}


# ── Quantile index helper ─────────────────────────────────────────────────────
def _find_quantile_index(quantile_levels: List[float], target: float) -> int:
    """Find the index of the model's quantile closest to `target`."""
    return min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - target))


# ── Forecast from candles ─────────────────────────────────────────────────────
def _forecast_from_candles(
    coin: str,
    side: str,
    candles: List[Candle],
    cfg: Dict[str, Any],
) -> TimestralSignal:
    def _err(msg: str, last: float = 0.0) -> TimestralSignal:
        return TimestralSignal(
            coin=coin, side=side, context_last=last,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=int(cfg.get("forecast_horizon", 12)),
            model_id=cfg.get("model_id", "mistralai/Timestral-209M"),
            inference_ms=0, error=msg,
        )

    forecaster = _get_forecaster()
    if forecaster is None:
        return _err("timestral disabled or failed to load")
    if not candles:
        return _err("no candles")

    context_length = int(cfg.get("context_length", 50))
    horizon = int(cfg.get("forecast_horizon", 12))

    context = candles[-context_length:] if len(candles) > context_length else candles
    closes = [float(candle_val(c, "c")) for c in context]
    last_close = closes[-1] if closes else 0.0
    if last_close <= 0:
        return _err("invalid last close", last_close)

    # The 21 quantile levels are checkpoint-fixed (config.quantile_knots).
    # Resolve p10/p50/p90 by VALUE so a checkpoint shipping a different knot
    # set can't silently mislabel the columns we read (timesfm guard).
    model_quantiles = [float(q) for q in forecaster.quantile_levels]
    idx_low = _find_quantile_index(model_quantiles, 0.1)
    idx_med = _find_quantile_index(model_quantiles, 0.5)
    idx_high = _find_quantile_index(model_quantiles, 0.9)

    try:
        start = time.time()
        # Univariate close-price forecast: 1-D context -> (1, 1, H, Q).
        # TAA off (see module docstring): single raw pass, no non-negative
        # clamp (prices are clamped by the market, not us).
        fc = forecaster.forecast(
            np.asarray(closes, dtype=np.float64),
            prediction_length=horizon,
            symmetric_averaging=False,
            clip_nonnegative=False,
        )
        inference_ms = (time.time() - start) * 1000

        q = np.asarray(fc.quantiles)[0, 0, :horizon, :]  # (H, Q)
        if q.ndim != 2 or q.shape[0] < horizon or q.shape[-1] < 3:
            return _err(f"unexpected forecast shape {q.shape}", last_close)

        levels = [float(v) for v in fc.quantile_levels]
        if len(levels) != q.shape[-1]:
            # Reported levels must line up with the quantile axis before we
            # name any column p10/p50/p90.
            return _err(f"quantile_levels ({len(levels)}) != quantile axis "
                        f"({q.shape[-1]})", last_close)
        idx_low = _find_quantile_index(levels, 0.1)
        idx_med = _find_quantile_index(levels, 0.5)
        idx_high = _find_quantile_index(levels, 0.9)

        # Path averages — same reduction chronos/timesfm/tirex use, so
        # median_pct / spread_pct are directly comparable across all four.
        median = float(q[:, idx_med].mean())
        q_low = float(q[:, idx_low].mean())
        q_high = float(q[:, idx_high].mean())
        q10_path_pct = [float(v) for v in
                        ((q[:, idx_low] - last_close) / last_close * 100)]
        q90_path_pct = [float(v) for v in
                        ((q[:, idx_high] - last_close) / last_close * 100)]
    except Exception as e:
        return _err(str(e), last_close)

    median_pct = ((median - last_close) / last_close * 100) if last_close > 0 else None
    spread_pct = ((q_high - q_low) / last_close * 100) if last_close > 0 else None

    return TimestralSignal(
        coin=coin,
        side=side,
        context_last=last_close,
        median=median,
        q_low=q_low,
        q_high=q_high,
        median_pct=median_pct,
        spread_pct=spread_pct,
        horizon=horizon,
        model_id=cfg.get("model_id", "mistralai/Timestral-209M"),
        inference_ms=inference_ms,
        q10_path_pct=q10_path_pct,
        q90_path_pct=q90_path_pct,
    )


# ── Signal fetch (sync, cache-aware, still potentially slow) ──────────────────
def _compute_signal(coin: str, side: str, cfg: Dict[str, Any]) -> TimestralSignal:
    """Candle fetch + forecast for one coin. Runs on a dedicated thread (see
    _fetch_signal); errors are captured there, not raised to the caller."""
    context_length = int(cfg.get("context_length", 50))
    # Same interval + same fetch as Chronos/TimesFM/TiRex (5m, shared candle
    # cache), so all four models forecast from an identical context window.
    candles = fetch_hl_candles(coin, "5m", context_length)
    return _forecast_from_candles(coin, side, candles or [], cfg)


def _fetch_signal(coin: str, side: str) -> TimestralSignal:
    cfg = _get_timestral_config()
    if not cfg.get("enabled", False):
        return TimestralSignal(
            coin=coin, side=side, context_last=0.0,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=int(cfg.get("forecast_horizon", 12)),
            model_id=cfg.get("model_id", "mistralai/Timestral-209M"),
            inference_ms=0, error="disabled",
        )

    ttl = float(cfg.get("cache_ttl_seconds", 300))
    cached = _cache_get(coin, ttl)
    if cached is not None:
        return cached

    # Late-arriving caller for a compute already in flight (loop fire vs the
    # PASS/attach sync read): wait for the owner, then serve its cache entry.
    # Otherwise claim ownership race-safely — setdefault loses to a concurrent
    # claimer, which we then simply wait on like a late arrival. Mirrors
    # tirex_signal._fetch_signal exactly.
    timeout_s = float(cfg.get("timeout_seconds", 0) or 0)
    while True:
        my_evt = threading.Event()
        with _inflight_lock:
            evt = _inflight.setdefault(coin, my_evt)
        if evt is my_evt:
            break  # we own the compute for this coin/TTL window
        if evt.wait(timeout_s if timeout_s > 0 else None):
            cached = _cache_get(coin, ttl)
            if cached is not None:
                return cached
        # owner failed/timed out or left nothing cacheable — fall through and
        # compute ourselves (worst case mirrors the pre-dedup behaviour).
        break

    # Run the forecast on a dedicated daemon thread and wait at most
    # timeout_seconds (0/unset = no deadline). The forward pass cannot be
    # interrupted, so on timeout we abandon the thread and return an error
    # signal (never cached) — the CALLER (attach path or fire-and-forget
    # worker) is bounded either way. Each forecast gets its own thread so a
    # hung inference can't delay other coins; abandoned threads die with the
    # process.
    result: Dict[str, Any] = {}
    done = threading.Event()

    def _run():
        try:
            result["signal"] = _compute_signal(coin, side, cfg)
        except Exception as e:  # surfaced as an error signal, mirrors below
            result["signal"] = TimestralSignal(
                coin=coin, side=side, context_last=0.0,
                median=None, q_low=None, q_high=None,
                median_pct=None, spread_pct=None,
                horizon=int(cfg.get("forecast_horizon", 12)),
                model_id=cfg.get("model_id", "mistralai/Timestral-209M"),
                inference_ms=0, error=str(e),
            )
        finally:
            done.set()

    threading.Thread(target=_run, name=f"timestral-compute-{coin}", daemon=True).start()
    try:
        if not done.wait(timeout_s if timeout_s > 0 else None):
            signal = TimestralSignal(
                coin=coin, side=side, context_last=0.0,
                median=None, q_low=None, q_high=None,
                median_pct=None, spread_pct=None,
                horizon=int(cfg.get("forecast_horizon", 12)),
                model_id=cfg.get("model_id", "mistralai/Timestral-209M"),
                inference_ms=0, error=f"timeout after {timeout_s:.0f}s",
            )
            logger.info(_format_signal_log(signal, bool(cfg.get("debug", False))))
            return signal

        signal = result["signal"]
        if not signal.error:
            _cache_set(coin, signal, ttl)
        # Log once per actual compute (cache miss). Cache hits return above
        # without logging, so each line reflects a real forecast.
        logger.info(_format_signal_log(signal, bool(cfg.get("debug", False))))
        return signal
    finally:
        # Release waiters only AFTER the cache is written (or the failure is
        # final), so a woken late caller always sees the settled result.
        with _inflight_lock:
            if _inflight.get(coin) is my_evt:
                del _inflight[coin]
        my_evt.set()


def resolve_min_conf_ratio(cfg: Dict[str, Any]) -> float:
    """The confidence-floor knob, read once per use site.

    Absent/garbage -> 0.25 (default); an explicit 0 DISABLES the floor. A
    falsy 0 must NOT fall back to the default via `or`. Negative clamps to 0.
    Same semantics as chronos_signal/timesfm_signal/tirex_signal.
    """
    try:
        v = float(cfg.get("min_conf_ratio", 0.25))
    except (TypeError, ValueError):
        return 0.25
    return max(0.0, v)


def confidence_ratio(sig: TimestralSignal) -> float:
    """|median_pct| / spread_pct — the median's size vs the model's own
    p10-p90 band. Fail-safe is ZERO confidence: a missing spread or median
    means we have no basis for a directional claim."""
    if sig.median_pct is None or not sig.spread_pct:
        return 0.0
    return abs(sig.median_pct) / sig.spread_pct


# ── Log formatting ────────────────────────────────────────────────────────────
def _format_signal_log(
    signal: TimestralSignal, debug: bool, min_conf_ratio: float = 0.25
) -> str:
    """Human-readable one-line log for the signal, mirroring the chronos /
    timesfm / tirex lines so a grep/tail of all four models side by side
    reads identically.

    The ALIGN/MISMATCH flag only counts when the median is confident
    (|median_pct| >= min_conf_ratio x the p10-p90 spread); below the floor
    the line reads NEUTRAL (a median inside its own uncertainty band is
    noise, not a call).
    """
    if signal.error:
        return f"[timestral] {signal.coin} error: {signal.error}"

    median_pct_str = f"{signal.median_pct:+.2f}%" if signal.median_pct is not None else "?"
    spread_str = f"{signal.spread_pct:.2f}%" if signal.spread_pct is not None else "?"
    ratio = confidence_ratio(signal)
    if signal.median_pct is None:
        direction = "→"
        alignment = "NEUTRAL (no median)"
    elif ratio < min_conf_ratio:
        direction = "→"
        alignment = f"NEUTRAL (ratio {ratio:.2f} < {min_conf_ratio:.2f})"
    else:
        direction = "↑" if signal.median_pct > 0 else "↓"
        alignment = "ALIGN" if signal.median_pct * (1 if signal.side == "long" else -1) > 0 else "MISMATCH"
    base = (
        f"[timestral] {signal.coin} ({signal.side}) "
        f"median={signal.median:.4f} ({median_pct_str}) "
        f"spread={spread_str} horizon={signal.horizon} "
        f"{direction} {alignment}"
    )
    if debug:
        base += (
            f" | q_low={signal.q_low:.4f} q_high={signal.q_high:.4f} "
            f"last={signal.context_last:.4f} "
            f"inference={signal.inference_ms:.0f}ms model={signal.model_id}"
        )
    return base


# ── Async daemon wrapper (the entry point) ────────────────────────────────────
def get_timestral_signal_async(coin: str, side: str) -> None:
    """Fire-and-forget Timestral forecast on a daemon thread.

    NEVER blocks the caller. The daemon thread checks enabled, lazy-loads the
    forecaster, fetches candles, forecasts, logs, and caches with TTL. Same
    pattern as get_chronos_signal_async / get_timesfm_signal_async /
    get_tirex_signal_async.
    """
    cfg = _get_timestral_config()

    def _worker():
        try:
            if not cfg.get("enabled", False):
                logger.debug(f"[timestral] {coin}: signal disabled")
                return
            # _fetch_signal logs on a fresh compute; cache hits are silent.
            _fetch_signal(coin, side)
        except Exception as e:
            logger.debug(f"[timestral] {coin} worker failed: {e}")

    threading.Thread(target=_worker, name=f"timestral-{coin}", daemon=True).start()


def peek_timestral(coin: str) -> Optional[TimestralSignal]:
    """Return the cached forecast if fresh, else None. NEVER computes, never
    blocks. Same never-computes contract as peek_chronos/peek_timesfm/
    peek_tirex."""
    try:
        cfg = _get_timestral_config()
        if not cfg.get("enabled", False):
            return None
        return _cache_get(coin, float(cfg.get("cache_ttl_seconds", 300)))
    except Exception as e:
        logger.debug(f"[timestral] peek failed for {coin}: {e}")
        return None


# ── Synchronous wrapper (attach / testing path) ───────────────────────────────
def get_timestral_signal_sync(coin: str, side: str) -> TimestralSignal:
    """Return the Timestral signal synchronously (cache-first).

    USE ONLY FOR TESTING or when you explicitly want to block for a forecast.
    The main pipeline uses `get_timestral_signal_async()` instead. Logging
    happens in `_fetch_signal` (once per real compute), so this wrapper does
    not log a second time.
    """
    return _fetch_signal(coin, side)
