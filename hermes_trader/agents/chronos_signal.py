"""Chronos-2 shadow signal module (shadow-mode, logged only).

Pulls in Chronos-2 zero-shot forecasting as a supplementary signal. The
shadow worker runs ASYNC on a daemon thread and logs every forecast (forward
validation); when `in_prompt: true`, the LLM prompt path additionally calls
`get_chronos_signal_sync()` so the line is deterministic per call. The model
is preloaded at app init (`preload_model`) and the per-coin cache (300s TTL)
bounds steady-state cost to ~200ms per coin (mostly the HL candle fetch).

Config-driven via `.agent-config.json` under `chronos_signal`:
    chronos_signal:
        enabled: true|false          # global toggle (default: false)
        debug: true|false            # extra log detail; off = summary only
        model_id: "amazon/chronos-2" # or autogluon/chronos-2-small
        device: "cpu"               # no GPU in our stack
        context_length: 100         # candles fed as context
        forecast_horizon: 48        # steps ahead to predict (in candle interval)
        cache_ttl_seconds: 300      # TTL for per-coin cache
        timeout_seconds: 30         # max seconds per forecast before abort

Chronos2Pipeline.predict() returns tensors of shape (n_variates, n_quantiles, horizon).
Quantiles are fixed by the model (model.quantiles), not configurable per call.
We match our interest quantiles to the closest model-provided ones.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from hermes_trader.agents.config_store import read_agent_config
from hermes_trader.client.hl_client import fetch_hl_candles
from hermes_trader.indicators.math import candle_val
from hermes_trader.models.types import Candle

logger = logging.getLogger(__name__)


# ── Singleton pipeline + config cache ──────────────────────────────────────────
_chronos_lock = threading.Lock()
_chronos_pipeline: Optional[Any] = None
_chronos_loaded_at: float = 0
_chronos_config_cache: Dict[str, Any] = {}
_model_init_error: Optional[str] = None


def _get_chronos_config() -> Dict[str, Any]:
    """Read chronos config from `.agent-config.json` with sensible defaults."""
    global _chronos_config_cache
    try:
        cfg = read_agent_config()
        if cfg is not _chronos_config_cache:
            _chronos_config_cache = cfg
            return cfg.get("chronos_signal", {})
        return _chronos_config_cache.get("chronos_signal", {})
    except Exception as e:
        logger.debug(f"[chronos] config read error: {e}")
        return {}


def _get_pipeline() -> Any:
    """Lazy-load Chronos2Pipeline singleton (thread-safe). Returns None if
    disabled or if loading fails."""
    global _chronos_pipeline, _chronos_loaded_at, _model_init_error
    cfg = _get_chronos_config()
    if not cfg.get("enabled", False):
        return None

    with _chronos_lock:
        if _chronos_pipeline is not None:
            return _chronos_pipeline
        if _model_init_error is not None:
            return None

        model_id = cfg.get("model_id", "amazon/chronos-2")
        device = cfg.get("device", "cpu")
        try:
            logger.info(f"[chronos] loading model {model_id} on {device}")
            from chronos import Chronos2Pipeline
            pipeline = Chronos2Pipeline.from_pretrained(model_id, device_map=device)
            _chronos_pipeline = pipeline
            _chronos_loaded_at = time.time()
            logger.info(
                f"[chronos] model loaded in {time.time() - _chronos_loaded_at:.1f}s "
                f"(quantiles: {pipeline.quantiles})"
            )
            return pipeline
        except ImportError as e:
            logger.warning(f"[chronos] import failed (missing dependency?): {e}")
            _model_init_error = str(e)
            return None
        except Exception as e:
            logger.warning(f"[chronos] model load failed: {e}")
            _model_init_error = str(e)
            return None


def preload_model(timeout_s: float = 60.0) -> bool:
    """Preload the Chronos pipeline on a bounded background thread at app init.

    Pays the one-time model-load cost (~2-4s) OFF the first-scan critical path
    so the prompt-path sync call never eats it. Bounded (join with timeout) so
    a hung load can't stall startup; on timeout/exception it falls back to
    lazy load on first use. Returns True if the model is ready when the join
    completes.
    """
    state = {"done": False, "pipeline": None, "error": None}

    def _run() -> None:
        try:
            state["pipeline"] = _get_pipeline()
        except Exception as e:
            state["error"] = e
        finally:
            state["done"] = True

    t = threading.Thread(target=_run, name="chronos-preload", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        logger.warning(
            f"[chronos] model preload exceeded {timeout_s:.0f}s — continuing; "
            "it will finish in the background or load lazily on first use")
    elif state["error"] is not None:
        logger.warning(f"[chronos] model preload failed (lazy fallback): {state['error']}")
    elif state["pipeline"] is None:
        logger.debug("[chronos] model not loaded (disabled or error cached); lazy fallback")
    else:
        logger.info("[chronos] model preloaded at init")
    return state["pipeline"] is not None


# ── Result structure ──────────────────────────────────────────────────────────
@dataclass
class ChronosSignal:
    coin: str
    side: str
    context_last: float        # last close used as context
    median: Optional[float]    # forecast median
    q_low: Optional[float]     # e.g. 0.1 quantile
    q_high: Optional[float]    # e.g. 0.9 quantile
    median_pct: Optional[float]  # (median - context_last) / context_last * 100
    spread_pct: Optional[float]  # (q_high - q_low) / context_last * 100
    horizon: int               # forecast horizon (in candle steps)
    model_id: str
    inference_ms: float
    error: Optional[str] = None
    # Per-step quantile paths, % vs context_last, across the full horizon.
    # Replay finding (2026-08-28): the forecast's SHAPE — specifically the
    # adverse quantile's early path — carries more veto information than the
    # path-mean scalars above. Kept on the signal so the
    # chronos_tail_trigger gate (and future shape-based gates) can consume
    # it; None on every failure path (old caches, errors, disabled).
    q10_path_pct: Optional[List[float]] = None
    q90_path_pct: Optional[List[float]] = None


# ── Per-coin cache (TTL-based) ────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}


def _cache_get(coin: str, ttl: float) -> Optional[ChronosSignal]:
    with _cache_lock:
        entry = _cache.get(coin)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["signal"]
        return None


def _cache_set(coin: str, signal: ChronosSignal, ttl: float) -> None:
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
) -> ChronosSignal:
    pipeline = _get_pipeline()
    if pipeline is None:
        return ChronosSignal(
            coin=coin, side=side, context_last=0.0,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=cfg.get("forecast_horizon", 48),
            model_id=cfg.get("model_id", "amazon/chronos-2"),
            inference_ms=0, error="chronos disabled or failed to load"
        )

    if not candles:
        return ChronosSignal(
            coin=coin, side=side, context_last=0.0,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=cfg.get("forecast_horizon", 48),
            model_id=cfg.get("model_id", "amazon/chronos-2"),
            inference_ms=0, error="no candles"
        )

    context_length = int(cfg.get("context_length", 100))
    horizon = int(cfg.get("forecast_horizon", 48))

    # Use only as many candles as context_length allows
    context = candles[-context_length:] if len(candles) > context_length else candles
    closes = [float(candle_val(c, "c")) for c in context]
    last_close = closes[-1] if closes else 0.0

    # Chronos-2 quantiles are model-fixed, not configurable per call.
    # Default target quantiles we care about; we'll match to model's.
    model_quantiles = pipeline.quantiles
    q_target_low = 0.1
    q_target_med = 0.5
    q_target_high = 0.9
    idx_low = _find_quantile_index(model_quantiles, q_target_low)
    idx_med = _find_quantile_index(model_quantiles, q_target_med)
    idx_high = _find_quantile_index(model_quantiles, q_target_high)

    try:
        start = time.time()
        # predict() returns list of torch.Tensor, each of shape
        # (n_variates, n_quantiles, prediction_length)
        forecasts = pipeline.predict(
            [torch.tensor(closes, dtype=torch.float32)],
            prediction_length=horizon,
        )
        inference_ms = (time.time() - start) * 1000

        # forecasts[0]: shape (1, n_quantiles, horizon) for univariate input
        f = forecasts[0]
        # Average over horizon to get a single forecast value per quantile
        # f[q, :] is the forecast at quantile q across all horizon steps
        median = float(f[0, idx_med, :].mean().item())
        q_low = float(f[0, idx_low, :].mean().item())
        q_high = float(f[0, idx_high, :].mean().item())
        # Per-step quantile paths, % vs the context's last close. The model
        # does NOT force the path to be linear (replay-verified 2026-08-28:
        # the median path mean-reverts within the horizon, while the adverse
        # quantile's early steps keep their shape). Stored so the
        # chronos_tail_trigger gate can consume the tail directly.
        q10_path_pct = [float(v) for v in
                        ((f[0, idx_low, :] - last_close) / last_close * 100).tolist()]
        q90_path_pct = [float(v) for v in
                        ((f[0, idx_high, :] - last_close) / last_close * 100).tolist()]
    except torch.cuda.OutOfMemoryError as e:
        return ChronosSignal(
            coin=coin, side=side, context_last=last_close,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=horizon,
            model_id=cfg.get("model_id", "amazon/chronos-2"),
            inference_ms=0, error=f"OOM: {e}"
        )
    except Exception as e:
        return ChronosSignal(
            coin=coin, side=side, context_last=last_close,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=horizon,
            model_id=cfg.get("model_id", "amazon/chronos-2"),
            inference_ms=0, error=str(e)
        )

    median_pct = ((median - last_close) / last_close * 100) if last_close > 0 else None
    spread_pct = ((q_high - q_low) / last_close * 100) if last_close > 0 else None

    return ChronosSignal(
        coin=coin,
        side=side,
        context_last=last_close,
        median=median,
        q_low=q_low,
        q_high=q_high,
        median_pct=median_pct,
        spread_pct=spread_pct,
        horizon=horizon,
        model_id=cfg.get("model_id", "amazon/chronos-2"),
        inference_ms=inference_ms,
        q10_path_pct=q10_path_pct,
        q90_path_pct=q90_path_pct,
    )


# ── Signal fetch (sync, cache-aware, still potentially slow) ──────────────────
def _fetch_signal(coin: str, side: str) -> ChronosSignal:
    cfg = _get_chronos_config()
    if not cfg.get("enabled", False):
        return ChronosSignal(
            coin=coin, side=side, context_last=0.0,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=cfg.get("forecast_horizon", 48),
            model_id=cfg.get("model_id", "amazon/chronos-2"),
            inference_ms=0, error="disabled"
        )

    ttl = float(cfg.get("cache_ttl_seconds", 300))
    cached = _cache_get(coin, ttl)
    if cached is not None:
        return cached

    context_length = int(cfg.get("context_length", 100))
    # Fetch on the interval used by perception scan (5m by default).
    # Chronos works best with enough bars; we pull context_length candles.
    candles = fetch_hl_candles(coin, "5m", context_length)

    signal = _forecast_from_candles(coin, side, candles or [], cfg)
    if not signal.error:
        _cache_set(coin, signal, ttl)
    # Log once per actual compute (cache miss). Cache hits return above
    # without logging, so each line reflects a real forecast rather than a
    # wrapper re-reading a warm entry (which made it look like N runs/coin).
    logger.info(_format_signal_log(signal, bool(cfg.get("debug", False))))
    return signal


# ── Async daemon wrapper (the entry point) ────────────────────────────────────
def _format_signal_log(signal: ChronosSignal, debug: bool) -> str:
    """Human-readable one-line log for the signal."""
    if signal.error:
        return f"[chronos] {signal.coin} error: {signal.error}"

    median_pct_str = f"{signal.median_pct:+.2f}%" if signal.median_pct is not None else "?"
    spread_str = f"{signal.spread_pct:.2f}%" if signal.spread_pct is not None else "?"
    direction = "↑" if (signal.median_pct or 0) > 0 else "↓"
    alignment = "ALIGN" if (signal.median_pct or 0) * (1 if signal.side == "long" else -1) > 0 else "MISMATCH"
    base = (
        f"[chronos] {signal.coin} ({signal.side}) "
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


def get_chronos_signal_async(coin: str, side: str) -> None:
    """Fire-and-forget Chronos-2 forecast on a daemon thread.

    NEVER blocks the caller. The daemon thread:
      1. Checks if enabled (config)
      2. Loads the pipeline (singleton, lazy)
      3. Fetches candles, runs forecast
      4. Logs the result (summary or debug)
      5. Caches with TTL

    Call this from `research.py` after collecting the AI verdict — same pattern
    as `run_shadow_async()` for the free signal suite.
    """
    cfg = _get_chronos_config()

    def _worker():
        try:
            if not cfg.get("enabled", False):
                logger.debug(f"[chronos] {coin}: signal disabled")
                return

            # _fetch_signal logs on a fresh compute; cache hits are silent.
            _fetch_signal(coin, side)
        except Exception as e:
            logger.debug(f"[chronos] {coin} worker failed: {e}")

    threading.Thread(target=_worker, name=f"chronos-{coin}", daemon=True).start()


def peek_chronos(coin: str) -> Optional[ChronosSignal]:
    """Return the cached forecast if fresh, else None. NEVER computes, never
    blocks. The LLM-prompt path uses this: it must not pay model-load or
    forecast cost — if the shadow worker has not produced a fresh forecast yet,
    the prompt simply omits the block."""
    try:
        cfg = _get_chronos_config()
        if not cfg.get("enabled", False):
            return None
        return _cache_get(coin, float(cfg.get("cache_ttl_seconds", 300)))
    except Exception as e:
        logger.debug(f"[chronos] peek failed for {coin}: {e}")
        return None


# ── Synchronous wrapper (for testing / explicit use only) ─────────────────────
def get_chronos_signal_sync(coin: str, side: str) -> ChronosSignal:
    """Return the Chronos signal synchronously.

    USE ONLY FOR TESTING or when you explicitly want to block for a forecast.
    The main pipeline uses `get_chronos_signal_async()` instead.
    Logging happens in `_fetch_signal` (once per real compute), so this
    wrapper does not log a second time.
    """
    return _fetch_signal(coin, side)
