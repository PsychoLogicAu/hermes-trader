"""TimesFM-3 shadow signal module (shadow-mode, logged only).

Pulls in Google's TimesFM-3 (zero-shot multivariate time-series foundation
model, Aug 2026) as a SHADOW signal alongside Chronos-2 so the two forecasters
can be compared head-to-head on identical inputs before either is promoted.
The shadow worker runs ASYNC on a daemon thread and logs every forecast
(forward validation); the model is optionally preloaded at app init
(`preload_model`) and the per-coin cache (300s TTL by default) bounds
steady-state cost to one candle fetch + one inference per coin per TTL.

SHADOW ONLY: never gates, never sizes, never enters the LLM prompt. The
attach site is `executor._attach_timesfm_to_result` (every `Trade result:`
return path), same ownership as Chronos.

LICENSE NOTE: TimesFM-3.0 pretrained weights ship under the
timesfm-non-commercial-license-v1.0 (non-commercial, non-production use).
The module therefore defaults to `enabled: false`; running it against the
live book is a production use the operator must consciously accept.

Config-driven via `.agent-config.json` under `timesfm_signal`:
    timesfm_signal:
        enabled: false              # global toggle (default: false — see license note)
        debug: false                # extra log detail; off = summary only
        model_id: "google/timesfm-3.0-pytorch"
        device: "cpu"               # no GPU in our stack
        context_length: 50          # candles fed as context (match chronos live)
        forecast_horizon: 12        # steps ahead (5m bars; match chronos live)
        min_conf_ratio: 0.25        # confidence floor: |median_pct| / spread_pct.
                                    # Below it the median sits inside the model's
                                    # own p10-p90 band, so the log flag reads
                                    # NEUTRAL (not ALIGN/MISMATCH). 0 disables.
        cache_ttl_seconds: 300      # TTL for per-coin cache
        timeout_seconds: 60         # abort deadline per forecast; 0 = no deadline
                                    # (330M params — slower than chronos on CPU)

API shape: `TimesFM3Forecaster.predict(context_1d, horizon, return_quantiles=
True)` returns a ForecastOutput with `.forecast` = median-path (horizon,) and
`.quantiles` = (horizon, 9) for quantiles 0.1..0.9 (median at index 4). This
is NumPy, not tensors — the chronos index-lookup dance does not apply; we
still resolve p10/p50/p90 by VALUE against `forecaster.config.quantiles` so
a checkpoint with a different fixed quantile set can't silently mislabel.

timeout_seconds is enforced at the caller level exactly as in chronos_signal:
the forecast runs on a dedicated worker thread and the CALLER abandons it
(error signal, never cached) once the deadline passes. torch predict can't be
interrupted mid-inference, so an overrun leaves that thread busy until it
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


# ── Singleton pipeline + config cache ──────────────────────────────────────────
_timesfm_lock = threading.Lock()
_timesfm_forecaster: Optional[Any] = None
_timesfm_loaded_at: float = 0
_timesfm_config_cache: Dict[str, Any] = {}
_model_init_error: Optional[str] = None


def _get_timesfm_config() -> Dict[str, Any]:
    """Read timesfm config from `.agent-config.json` with sensible defaults."""
    global _timesfm_config_cache
    try:
        cfg = read_agent_config()
        if cfg is not _timesfm_config_cache:
            _timesfm_config_cache = cfg
            return cfg.get("timesfm_signal", {})
        return _timesfm_config_cache.get("timesfm_signal", {})
    except Exception as e:
        logger.debug(f"[timesfm] config read error: {e}")
        return {}


def _get_forecaster() -> Any:
    """Lazy-load the TimesFM3Forecaster singleton (thread-safe). Returns None
    if disabled or if loading fails."""
    global _timesfm_forecaster, _timesfm_loaded_at, _model_init_error
    cfg = _get_timesfm_config()
    if not cfg.get("enabled", False):
        return None

    with _timesfm_lock:
        if _timesfm_forecaster is not None:
            return _timesfm_forecaster
        if _model_init_error is not None:
            return None

        model_id = cfg.get("model_id", "google/timesfm-3.0-pytorch")
        device = cfg.get("device", "cpu")
        try:
            logger.info(f"[timesfm] loading model {model_id} on {device}")
            from timesfm3 import TimesFM3Forecaster
            start = time.time()
            forecaster = TimesFM3Forecaster.from_pretrained(model_id, device=device)
            _timesfm_forecaster = forecaster
            _timesfm_loaded_at = time.time()
            logger.info(
                f"[timesfm] model loaded in {time.time() - start:.1f}s "
                f"(quantiles: {list(forecaster.config.quantiles)})"
            )
            return forecaster
        except ImportError as e:
            logger.warning(f"[timesfm] import failed (missing dependency?): {e}")
            _model_init_error = str(e)
            return None
        except Exception as e:
            logger.warning(f"[timesfm] model load failed: {e}")
            _model_init_error = str(e)
            return None


def preload_model(timeout_s: float = 120.0) -> bool:
    """Preload the TimesFM-3 pipeline on a bounded background thread at app init.

    Pays the one-time model-load cost (weights are ~1.3GB fp32 — first load
    also downloads into the HF cache volume) OFF the first-trade critical
    path. Bounded (join with timeout) so a hung load can't stall startup; on
    timeout/exception it falls back to lazy load on first use. Returns True
    if the model is ready when the join completes.
    """
    state = {"done": False, "forecaster": None, "error": None}

    def _run() -> None:
        try:
            state["forecaster"] = _get_forecaster()
        except Exception as e:
            state["error"] = e
        finally:
            state["done"] = True

    t = threading.Thread(target=_run, name="timesfm-preload", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        logger.warning(
            f"[timesfm] model preload exceeded {timeout_s:.0f}s — continuing; "
            "it will finish in the background or load lazily on first use")
    elif state["error"] is not None:
        logger.warning(f"[timesfm] model preload failed (lazy fallback): {state['error']}")
    elif state["forecaster"] is None:
        logger.debug("[timesfm] model not loaded (disabled or error cached); lazy fallback")
    else:
        logger.info("[timesfm] model preloaded at init")
    return state["forecaster"] is not None


# ── Result structure ──────────────────────────────────────────────────────────
@dataclass
class TimesfmSignal:
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
    # Same contract as ChronosSignal.q10_path_pct/q90_path_pct so a future
    # head-to-head gate (or a chronos/timesfm agreement gate) can consume
    # either signal with one code path. None on every failure path.
    q10_path_pct: Optional[List[float]] = None
    q90_path_pct: Optional[List[float]] = None


# ── Per-coin cache (TTL-based) ────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}


def _cache_get(coin: str, ttl: float) -> Optional[TimesfmSignal]:
    with _cache_lock:
        entry = _cache.get(coin)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["signal"]
        return None


def _cache_set(coin: str, signal: TimesfmSignal, ttl: float) -> None:
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
) -> TimesfmSignal:
    def _err(msg: str, last: float = 0.0) -> TimesfmSignal:
        return TimesfmSignal(
            coin=coin, side=side, context_last=last,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=int(cfg.get("forecast_horizon", 12)),
            model_id=cfg.get("model_id", "google/timesfm-3.0-pytorch"),
            inference_ms=0, error=msg,
        )

    forecaster = _get_forecaster()
    if forecaster is None:
        return _err("timesfm disabled or failed to load")
    if not candles:
        return _err("no candles")

    context_length = int(cfg.get("context_length", 50))
    horizon = int(cfg.get("forecast_horizon", 12))

    context = candles[-context_length:] if len(candles) > context_length else candles
    closes = [float(candle_val(c, "c")) for c in context]
    last_close = closes[-1] if closes else 0.0
    if last_close <= 0:
        return _err("invalid last close", last_close)

    # Quantiles are fixed by the checkpoint (0.1..0.9 for 3.0). Resolve
    # p10/p50/p90 by VALUE against the loaded config so a different fixed
    # set can't silently mislabel the columns we read out.
    model_quantiles = [float(q) for q in forecaster.config.quantiles]
    idx_low = _find_quantile_index(model_quantiles, 0.1)
    idx_med = _find_quantile_index(model_quantiles, 0.5)
    idx_high = _find_quantile_index(model_quantiles, 0.9)

    try:
        start = time.time()
        # Univariate close-price forecast, benchmark-off flags: no symmetric
        # averaging (2x compute for a gain irrelevant to shadow logging), no
        # positivity clamp (prices are clamped by the market, not us).
        out = forecaster.predict(
            np.asarray(closes, dtype=np.float32),
            horizon=horizon,
            return_quantiles=True,
            use_symmetric_averaging=False,
            make_positive=False,
        )
        inference_ms = (time.time() - start) * 1000

        forecast_path = np.asarray(out.forecast, dtype=np.float64)  # (horizon,) median
        q = np.asarray(out.quantiles, dtype=np.float64)             # (horizon, n_quantiles)
        if forecast_path.size < horizon or q.ndim != 2 or q.shape[0] < horizon:
            return _err(f"unexpected forecast shape {forecast_path.shape}/{q.shape}",
                        last_close)
        q = q[:horizon]
        forecast_path = forecast_path[:horizon]

        # Path averages — same reduction chronos uses, so median_pct /
        # spread_pct are directly comparable between the two models.
        median = float(forecast_path.mean())
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

    return TimesfmSignal(
        coin=coin,
        side=side,
        context_last=last_close,
        median=median,
        q_low=q_low,
        q_high=q_high,
        median_pct=median_pct,
        spread_pct=spread_pct,
        horizon=horizon,
        model_id=cfg.get("model_id", "google/timesfm-3.0-pytorch"),
        inference_ms=inference_ms,
        q10_path_pct=q10_path_pct,
        q90_path_pct=q90_path_pct,
    )


# ── Signal fetch (sync, cache-aware, still potentially slow) ──────────────────
def _compute_signal(coin: str, side: str, cfg: Dict[str, Any]) -> TimesfmSignal:
    """Candle fetch + forecast for one coin. Runs on a dedicated thread (see
    _fetch_signal); errors are captured there, not raised to the caller."""
    context_length = int(cfg.get("context_length", 50))
    # Same interval + same fetch as Chronos (5m, 90s shared candle cache), so
    # the two models always forecast from an identical context window.
    candles = fetch_hl_candles(coin, "5m", context_length)
    return _forecast_from_candles(coin, side, candles or [], cfg)


def _fetch_signal(coin: str, side: str) -> TimesfmSignal:
    cfg = _get_timesfm_config()
    if not cfg.get("enabled", False):
        return TimesfmSignal(
            coin=coin, side=side, context_last=0.0,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=int(cfg.get("forecast_horizon", 12)),
            model_id=cfg.get("model_id", "google/timesfm-3.0-pytorch"),
            inference_ms=0, error="disabled",
        )

    ttl = float(cfg.get("cache_ttl_seconds", 300))
    cached = _cache_get(coin, ttl)
    if cached is not None:
        return cached

    # Run the forecast on a dedicated daemon thread and wait at most
    # timeout_seconds (0/unset = no deadline). A torch predict cannot be
    # interrupted, so on timeout we abandon the thread and return an error
    # signal (never cached) — the CALLER (attach path or fire-and-forget
    # worker) is bounded either way. Each forecast gets its own thread so a
    # hung inference can't delay other coins; abandoned threads die with the
    # process. Mirrors chronos_signal._fetch_signal exactly.
    timeout_s = float(cfg.get("timeout_seconds", 0) or 0)
    result: Dict[str, Any] = {}
    done = threading.Event()

    def _run():
        try:
            result["signal"] = _compute_signal(coin, side, cfg)
        except Exception as e:  # surfaced as an error signal, mirrors below
            result["signal"] = TimesfmSignal(
                coin=coin, side=side, context_last=0.0,
                median=None, q_low=None, q_high=None,
                median_pct=None, spread_pct=None,
                horizon=int(cfg.get("forecast_horizon", 12)),
                model_id=cfg.get("model_id", "google/timesfm-3.0-pytorch"),
                inference_ms=0, error=str(e),
            )
        finally:
            done.set()

    threading.Thread(target=_run, name=f"timesfm-compute-{coin}", daemon=True).start()
    if not done.wait(timeout_s if timeout_s > 0 else None):
        signal = TimesfmSignal(
            coin=coin, side=side, context_last=0.0,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=int(cfg.get("forecast_horizon", 12)),
            model_id=cfg.get("model_id", "google/timesfm-3.0-pytorch"),
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


def resolve_min_conf_ratio(cfg: Dict[str, Any]) -> float:
    """The confidence-floor knob, read once per use site.

    Absent/garbage -> 0.25 (default); an explicit 0 DISABLES the floor. A
    falsy 0 must NOT fall back to the default via `or`. Negative clamps to 0.
    Same semantics as chronos_signal.resolve_min_conf_ratio.
    """
    try:
        v = float(cfg.get("min_conf_ratio", 0.25))
    except (TypeError, ValueError):
        return 0.25
    return max(0.0, v)


def confidence_ratio(sig: TimesfmSignal) -> float:
    """|median_pct| / spread_pct — the median's size vs the model's own
    p10-p90 band. Fail-safe is ZERO confidence: a missing spread or median
    means we have no basis for a directional claim."""
    if sig.median_pct is None or not sig.spread_pct:
        return 0.0
    return abs(sig.median_pct) / sig.spread_pct


# ── Log formatting ────────────────────────────────────────────────────────────
def _format_signal_log(
    signal: TimesfmSignal, debug: bool, min_conf_ratio: float = 0.25
) -> str:
    """Human-readable one-line log for the signal, mirroring the chronos line
    so a grep/tail of both models side by side reads identically.

    The ALIGN/MISMATCH flag only counts when the median is confident
    (|median_pct| >= min_conf_ratio x the p10-p90 spread); below the floor
    the line reads NEUTRAL (a median inside its own uncertainty band is
    noise, not a call).
    """
    if signal.error:
        return f"[timesfm] {signal.coin} error: {signal.error}"

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
        f"[timesfm] {signal.coin} ({signal.side}) "
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
def get_timesfm_signal_async(coin: str, side: str) -> None:
    """Fire-and-forget TimesFM-3 forecast on a daemon thread.

    NEVER blocks the caller. The daemon thread checks enabled, lazy-loads the
    forecaster, fetches candles, forecasts, logs, and caches with TTL. Same
    pattern as get_chronos_signal_async.
    """
    cfg = _get_timesfm_config()

    def _worker():
        try:
            if not cfg.get("enabled", False):
                logger.debug(f"[timesfm] {coin}: signal disabled")
                return
            # _fetch_signal logs on a fresh compute; cache hits are silent.
            _fetch_signal(coin, side)
        except Exception as e:
            logger.debug(f"[timesfm] {coin} worker failed: {e}")

    threading.Thread(target=_worker, name=f"timesfm-{coin}", daemon=True).start()


def peek_timesfm(coin: str) -> Optional[TimesfmSignal]:
    """Return the cached forecast if fresh, else None. NEVER computes, never
    blocks. Same never-computes contract as peek_chronos."""
    try:
        cfg = _get_timesfm_config()
        if not cfg.get("enabled", False):
            return None
        return _cache_get(coin, float(cfg.get("cache_ttl_seconds", 300)))
    except Exception as e:
        logger.debug(f"[timesfm] peek failed for {coin}: {e}")
        return None


# ── Synchronous wrapper (attach / testing path) ───────────────────────────────
def get_timesfm_signal_sync(coin: str, side: str) -> TimesfmSignal:
    """Return the TimesFM signal synchronously (cache-first).

    Used by `executor._attach_timesfm_to_result` at every `Trade result:`
    site — the same ownership model as Chronos (executor owns the signal; no
    research.py duplicate shadow call). Logging happens in `_fetch_signal`
    (once per real compute), so this wrapper does not log a second time.
    """
    return _fetch_signal(coin, side)
