"""TiRex 1.1 shadow signal module (shadow-mode, logged only).

Pulls NX-AI's TiRex 1.1 (xLSTM zero-shot forecaster) in as a standalone
shadow signal next to Chronos-2 and TimesFM-3 so it can be measured on the
identical 5m context before any decision integration. The shadow worker runs
ASYNC on a daemon thread and logs every forecast (forward validation); the
model is preloaded at app init (`preload_model`) and the per-coin cache
(300s TTL by default) bounds steady-state cost to one candle fetch + one
inference per coin per TTL.

WHY (2026-09-16 MoiraiAgent/MOE evaluation, scratch/eval/moirai_eval/):
across 2,688 prod-regime anchors (5m / ctx-50 / 12-step) TiRex was the best
single forecaster on BOTH axes — median-MAE 0.626 vs chronos 0.648 /
timesfm 0.703, and tail-AUC ahead of every candidate at each stop threshold
(+0.01..+0.017 over chronos, LOO-stable 15/16 coins). The expert-selection
MOE around it added nothing (mixture ~= best single), so the useful move is
TiRex as its own signal — drop-in replaceable against chronos if the accrual
confirms the offline result.

SINGLETON SHARING: the model object is borrowed from `expert_pool`
(`_ensure_loaded("tirex")`) — expert_select's pool and this module share ONE
loaded TiRex (~284MB checkpoint). Same ownership pattern as expert_pool
borrowing chronos_signal/timesfm_signal singletons, just pointed at tirex.

LICENSE: NX-AI/TiRex weights are Apache-2.0 — no non-commercial caveat
(unlike TimesFM-3); enabling is a pure perf decision.

Config-driven via `.agent-config.json` under `tirex_signal`:
    tirex_signal:
        enabled: false              # global toggle (default: false; flip to accrue)
        debug: false                # extra log detail; off = summary only
        model_id: "NX-AI/TiRex"     # tirex-ts load_model id (1.1 at HEAD)
        device: "cpu"               # no GPU in our stack
        context_length: 50          # candles fed as context (prod regime; xLSTM max 256)
        forecast_horizon: 12        # steps ahead (5m bars; match chronos live)
        min_conf_ratio: 0.25        # confidence floor: |median_pct| / spread_pct.
                                    # Below it the median sits inside the model's
                                    # own p10-p90 band, so the log flag reads
                                    # NEUTRAL (not ALIGN/MISMATCH). 0 disables.
        cache_ttl_seconds: 300      # TTL for per-coin cache
        timeout_seconds: 30         # abort deadline per forecast; 0 = no deadline

API shape (tirex-ts): `model.forecast(context=[1-D arrays],
prediction_length, resample_strategy)` returns `(quantile_forecast, _)` with
shape (batch, horizon, n_quantiles) — TRANSPOSED vs moirai2/chronos. The
checkpoint does not report its quantile levels; the canonical 0.1..0.9 grid
is assumed when it emits exactly 9 columns (same assumption as expert_pool's
adapter), otherwise evenly-spaced interior levels are derived and p10/p50/p90
resolve to the CLOSEST column by value.

timeout_seconds is enforced at the caller level exactly as in chronos_signal:
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


# ── Singleton model (borrowed from expert_pool) + config cache ────────────────
_tirex_lock = threading.Lock()
_tirex_model: Optional[Any] = None
_tirex_loaded_at: float = 0
_tirex_config_cache: Dict[str, Any] = {}
_model_init_error: Optional[str] = None


def _get_tirex_config() -> Dict[str, Any]:
    """Read tirex config from `.agent-config.json` with sensible defaults."""
    global _tirex_config_cache
    try:
        cfg = read_agent_config()
        if cfg is not _tirex_config_cache:
            _tirex_config_cache = cfg
            return cfg.get("tirex_signal", {})
        return _tirex_config_cache.get("tirex_signal", {})
    except Exception as e:
        logger.debug(f"[tirex] config read error: {e}")
        return {}


def _load_model() -> Any:
    """Load the TiRex checkpoint via expert_pool's shared singleton loader.

    expert_pool._ensure_loaded("tirex") owns the load + its own failure cache;
    when expert_select is enabled it has ALREADY loaded the model, so this is
    a dict hit — one ~284MB checkpoint for both features. A private-function
    borrow by design (same as expert_pool borrowing chronos/timesfm internals
    in reverse); if expert_pool ever exposes a public getter, switch to it.
    """
    from hermes_trader.agents import expert_pool
    return expert_pool._ensure_loaded("tirex")


def _get_model() -> Any:
    """Lazy-load the TiRex singleton (thread-safe). Returns None if disabled
    or if loading fails."""
    global _tirex_model, _tirex_loaded_at, _model_init_error
    cfg = _get_tirex_config()
    if not cfg.get("enabled", False):
        return None

    with _tirex_lock:
        if _tirex_model is not None:
            return _tirex_model
        if _model_init_error is not None:
            return None

        model_id = cfg.get("model_id", "NX-AI/TiRex")
        device = cfg.get("device", "cpu")
        try:
            logger.info(f"[tirex] loading model {model_id} on {device}")
            start = time.time()
            model = _load_model()
            if model is None:
                # expert_pool caches its own load failure; mirror it here so
                # we don't re-attempt per call (retries only after restart —
                # same posture as the chronos borrow in expert_pool).
                raise RuntimeError("expert_pool tirex load returned None "
                                   "(see [expert_pool] log lines)")
            _tirex_model = model
            _tirex_loaded_at = time.time()
            logger.info(f"[tirex] model ready in {time.time() - start:.1f}s")
            return model
        except ImportError as e:
            logger.warning(f"[tirex] import failed (missing dependency?): {e}")
            _model_init_error = str(e)
            return None
        except Exception as e:
            logger.warning(f"[tirex] model load failed: {e}")
            _model_init_error = str(e)
            return None


def preload_model(timeout_s: float = 120.0) -> bool:
    """Preload TiRex on a bounded background thread at app init.

    Pays the one-time checkpoint cost OFF the first-scan critical path (warm
    from the hf-cache it is <1s; the first-ever load also downloads ~284MB).
    Bounded (join with timeout) so a hung load can't stall startup; on
    timeout/exception it falls back to lazy load on first use. Returns True
    if the model is ready when the join completes.
    """
    state = {"done": False, "model": None, "error": None}

    def _run() -> None:
        try:
            state["model"] = _get_model()
        except Exception as e:
            state["error"] = e
        finally:
            state["done"] = True

    t = threading.Thread(target=_run, name="tirex-preload", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        logger.warning(
            f"[tirex] model preload exceeded {timeout_s:.0f}s — continuing; "
            "it will finish in the background or load lazily on first use")
    elif state["error"] is not None:
        logger.warning(f"[tirex] model preload failed (lazy fallback): {state['error']}")
    elif state["model"] is None:
        logger.debug("[tirex] model not loaded (disabled or error cached); lazy fallback")
    else:
        logger.info("[tirex] model preloaded at init")
    return state["model"] is not None


# ── Result structure ──────────────────────────────────────────────────────────
@dataclass
class TirexSignal:
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
    # Same contract as ChronosSignal/TimesfmSignal so a head-to-head gate
    # (or a chronos->tirex swap) consumes any of the three with one code
    # path. None on every failure path.
    q10_path_pct: Optional[List[float]] = None
    q90_path_pct: Optional[List[float]] = None


# ── Per-coin cache (TTL-based) ────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}

# In-flight dedup: coin -> Event set when the owner's compute settles. The
# loop fire (async worker) and the route_verdict/executor attaches can reach
# _fetch_signal for the same cold coin within milliseconds of each other;
# without this, BOTH compute and BOTH log — duplicate accrual lines that skew
# any grep-counted rate (the exact failure mode fixed in expert_select). One
# compute per coin per TTL; late callers wait on the owner and read its cache.
_inflight_lock = threading.Lock()
_inflight: Dict[str, threading.Event] = {}


def _cache_get(coin: str, ttl: float) -> Optional[TirexSignal]:
    with _cache_lock:
        entry = _cache.get(coin)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["signal"]
        return None


def _cache_set(coin: str, signal: TirexSignal, ttl: float) -> None:
    with _cache_lock:
        _cache[coin] = {"signal": signal, "ts": time.time()}


# ── Quantile index helper ─────────────────────────────────────────────────────
def _find_quantile_index(quantile_levels: List[float], target: float) -> int:
    """Find the index of the model's quantile closest to `target`."""
    return min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - target))


def _quantile_levels(nq: int) -> List[float]:
    """Quantile levels for an nq-column forecast. The tirex checkpoint does
    not report them; the canonical 0.1..0.9 grid applies when it emits
    exactly 9 (expert_pool adapter makes the same assumption), else evenly-
    spaced interior levels."""
    if nq == 9:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return [round((i + 1) / (nq + 1), 4) for i in range(nq)]


# ── Forecast from candles ─────────────────────────────────────────────────────
def _forecast_from_candles(
    coin: str,
    side: str,
    candles: List[Candle],
    cfg: Dict[str, Any],
) -> TirexSignal:
    def _err(msg: str, last: float = 0.0) -> TirexSignal:
        return TirexSignal(
            coin=coin, side=side, context_last=last,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=int(cfg.get("forecast_horizon", 12)),
            model_id=cfg.get("model_id", "NX-AI/TiRex"),
            inference_ms=0, error=msg,
        )

    model = _get_model()
    if model is None:
        return _err("tirex disabled or failed to load")
    if not candles:
        return _err("no candles")

    context_length = int(cfg.get("context_length", 50))
    horizon = int(cfg.get("forecast_horizon", 12))

    context = candles[-context_length:] if len(candles) > context_length else candles
    closes = [float(candle_val(c, "c")) for c in context]
    last_close = closes[-1] if closes else 0.0
    if last_close <= 0:
        return _err("invalid last close", last_close)

    try:
        start = time.time()
        # tirex-ts contract: context=[1-D arrays] (batch of one), frequency-
        # resampled quantiles — identical call to expert_pool's tirex adapter
        # so the shadow accrual and the MOE pool never diverge on inputs.
        quantile_forecast, _ = model.forecast(
            context=[np.asarray(closes, dtype=float)],
            prediction_length=horizon,
            resample_strategy="frequency",
        )
        inference_ms = (time.time() - start) * 1000

        q = np.asarray(quantile_forecast)[0, :horizon, :]  # (h, n_quantiles)
        if q.ndim != 2 or q.shape[0] < horizon or q.shape[-1] < 3:
            return _err(f"unexpected forecast shape {q.shape}", last_close)

        levels = _quantile_levels(q.shape[-1])
        idx_low = _find_quantile_index(levels, 0.1)
        idx_med = _find_quantile_index(levels, 0.5)
        idx_high = _find_quantile_index(levels, 0.9)

        # Path averages — same reduction chronos/timesfm use, so median_pct /
        # spread_pct are directly comparable across all three models.
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

    return TirexSignal(
        coin=coin,
        side=side,
        context_last=last_close,
        median=median,
        q_low=q_low,
        q_high=q_high,
        median_pct=median_pct,
        spread_pct=spread_pct,
        horizon=horizon,
        model_id=cfg.get("model_id", "NX-AI/TiRex"),
        inference_ms=inference_ms,
        q10_path_pct=q10_path_pct,
        q90_path_pct=q90_path_pct,
    )


# ── Signal fetch (sync, cache-aware, still potentially slow) ──────────────────
def _compute_signal(coin: str, side: str, cfg: Dict[str, Any]) -> TirexSignal:
    """Candle fetch + forecast for one coin. Runs on a dedicated thread (see
    _fetch_signal); errors are captured there, not raised to the caller."""
    context_length = int(cfg.get("context_length", 50))
    # Same interval + same fetch as Chronos/TimesFM (5m, shared candle cache),
    # so all three models forecast from an identical context window.
    candles = fetch_hl_candles(coin, "5m", context_length)
    return _forecast_from_candles(coin, side, candles or [], cfg)


def _fetch_signal(coin: str, side: str) -> TirexSignal:
    cfg = _get_tirex_config()
    if not cfg.get("enabled", False):
        return TirexSignal(
            coin=coin, side=side, context_last=0.0,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=int(cfg.get("forecast_horizon", 12)),
            model_id=cfg.get("model_id", "NX-AI/TiRex"),
            inference_ms=0, error="disabled",
        )

    ttl = float(cfg.get("cache_ttl_seconds", 300))
    cached = _cache_get(coin, ttl)
    if cached is not None:
        return cached

    # Late-arriving caller for a compute already in flight (loop fire vs the
    # PASS/attach sync read): wait for the owner, then serve its cache entry.
    # Otherwise claim ownership race-safely — setdefault loses to a concurrent
    # claimer, which we then simply wait on like a late arrival.
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
    # process. Mirrors chronos_signal/timesfm_signal._fetch_signal exactly.
    result: Dict[str, Any] = {}
    done = threading.Event()

    def _run():
        try:
            result["signal"] = _compute_signal(coin, side, cfg)
        except Exception as e:  # surfaced as an error signal, mirrors below
            result["signal"] = TirexSignal(
                coin=coin, side=side, context_last=0.0,
                median=None, q_low=None, q_high=None,
                median_pct=None, spread_pct=None,
                horizon=int(cfg.get("forecast_horizon", 12)),
                model_id=cfg.get("model_id", "NX-AI/TiRex"),
                inference_ms=0, error=str(e),
            )
        finally:
            done.set()

    threading.Thread(target=_run, name=f"tirex-compute-{coin}", daemon=True).start()
    try:
        if not done.wait(timeout_s if timeout_s > 0 else None):
            signal = TirexSignal(
                coin=coin, side=side, context_last=0.0,
                median=None, q_low=None, q_high=None,
                median_pct=None, spread_pct=None,
                horizon=int(cfg.get("forecast_horizon", 12)),
                model_id=cfg.get("model_id", "NX-AI/TiRex"),
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
    Same semantics as chronos_signal/timesfm_signal.
    """
    try:
        v = float(cfg.get("min_conf_ratio", 0.25))
    except (TypeError, ValueError):
        return 0.25
    return max(0.0, v)


def confidence_ratio(sig: TirexSignal) -> float:
    """|median_pct| / spread_pct — the median's size vs the model's own
    p10-p90 band. Fail-safe is ZERO confidence: a missing spread or median
    means we have no basis for a directional claim."""
    if sig.median_pct is None or not sig.spread_pct:
        return 0.0
    return abs(sig.median_pct) / sig.spread_pct


# ── Log formatting ────────────────────────────────────────────────────────────
def _format_signal_log(
    signal: TirexSignal, debug: bool, min_conf_ratio: float = 0.25
) -> str:
    """Human-readable one-line log for the signal, mirroring the chronos /
    timesfm lines so a grep/tail of all three models side by side reads
    identically.

    The ALIGN/MISMATCH flag only counts when the median is confident
    (|median_pct| >= min_conf_ratio x the p10-p90 spread); below the floor
    the line reads NEUTRAL (a median inside its own uncertainty band is
    noise, not a call).
    """
    if signal.error:
        return f"[tirex] {signal.coin} error: {signal.error}"

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
        f"[tirex] {signal.coin} ({signal.side}) "
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
def get_tirex_signal_async(coin: str, side: str) -> None:
    """Fire-and-forget TiRex forecast on a daemon thread.

    NEVER blocks the caller. The daemon thread checks enabled, lazy-loads the
    model (shared singleton), fetches candles, forecasts, logs, and caches
    with TTL. Same pattern as get_chronos_signal_async /
    get_timesfm_signal_async.
    """
    cfg = _get_tirex_config()

    def _worker():
        try:
            if not cfg.get("enabled", False):
                logger.debug(f"[tirex] {coin}: signal disabled")
                return
            # _fetch_signal logs on a fresh compute; cache hits are silent.
            _fetch_signal(coin, side)
        except Exception as e:
            logger.debug(f"[tirex] {coin} worker failed: {e}")

    threading.Thread(target=_worker, name=f"tirex-{coin}", daemon=True).start()


def peek_tirex(coin: str) -> Optional[TirexSignal]:
    """Return the cached forecast if fresh, else None. NEVER computes, never
    blocks. Same never-computes contract as peek_chronos/peek_timesfm."""
    try:
        cfg = _get_tirex_config()
        if not cfg.get("enabled", False):
            return None
        return _cache_get(coin, float(cfg.get("cache_ttl_seconds", 300)))
    except Exception as e:
        logger.debug(f"[tirex] peek failed for {coin}: {e}")
        return None


# ── Synchronous wrapper (attach / testing path) ───────────────────────────────
def get_tirex_signal_sync(coin: str, side: str) -> TirexSignal:
    """Return the TiRex signal synchronously (cache-first).

    USE ONLY FOR TESTING or when you explicitly want to block for a forecast.
    The main pipeline uses `get_tirex_signal_async()` instead. Logging happens
    in `_fetch_signal` (once per real compute), so this wrapper does not log a
    second time.
    """
    return _fetch_signal(coin, side)
