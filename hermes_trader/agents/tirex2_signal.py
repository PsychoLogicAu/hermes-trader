"""TiRex-2 shadow signal module (shadow-mode, logged only).

Pulls NX-AI's TiRex-2 (xLSTM/Mixture-of-Stacks zero-shot forecaster, 2026
successor of the TiRex 1.1 already running as tirex_signal) in as a parallel
shadow signal so it can be measured at PRODUCTION cadence next to
chronos/timesfm/tirex/moirai2/timestral before any decision integration. The
2026-10-21 offline sweep (scratch/eval/tirex2_sweep/, n=47,400 matched
live-accrual anchors) says TiRex-2 strictly dominates chronos on BOTH legs
(MAE -0.027pp, dir +0.013, bootstrap CIs excluding 0) and ties-or-beats OG
TiRex (MAE tie, dir significantly better) — while C.11's OG-TiRex swap is
still an open owner decision, this accrual prices TiRex-2 as the swap
TARGET instead, at production context timing rather than fetched-candle
replay.

WHY SHADOW-FIRST (owner call C, 2026-10-21): two open risks the offline
sweep cannot settle — (1) the >2%-move dir-acc INVERSION (0.4818 vs OG
0.5007, n=2,856), the exact regime the tail-stop gates care about, and (2)
the live CPU cost/footprint (82.5M params / ~330MB weights; sweep measured
68ms/anchor batched, single-call 64ms — same order as OG but the container
RSS delta is real). Accrual answers both without touching any live gate.

STANDALONE SINGLETON: unlike tirex_signal (which borrows expert_pool's
shared model), TiRex-2 has NO expert_pool candidate yet, so this module
OWNS the load/preload — same direction as timestral_signal (if TiRex-2 is
ever pool-wired via the 1c pattern, the pool will borrow THIS singleton;
never a second copy of the checkpoint).

LICENSE: NX-AI/TiRex-2 weights AND library are Apache-2.0 — verified against
the HF model card front-matter (license: apache-2.0), the checkpoint, and
the tirex-2 0.4.0 dist-info on 2026-10-21. Cleaner than the OG TiRex's
NXAI Community License; no non-commercial caveat.

API shape (tirex-2 0.4.0, git-pinned in requirements.txt):
    from tirex2 import load_model, TimeseriesType
    model = load_model("NX-AI/TiRex-2", device="cpu")   # lazy; first-ever
        load downloads ~330MB into the hf-cache volume
    fcs = model.forecast([TimeseriesType(target=torch_f32(1,T),
                         past_covariates=None, future_covariates=None)],
                         prediction_length=H, output_type="numpy")
    q = fcs[0][0]        # shape (Q, horizon) — Q x H, TRANSPOSED vs
                         # tirex-ts' (batch, horizon, n_quantiles)
The checkpoint REPORTS its quantile levels via model._quantile_levels()
(0.1..0.9 grid, same knots as tirex-ts), falling back to the canonical
9-grid assumption if that private accessor ever disappears.

CHECKPOINT READ FLAGS: tta_sign_flip=False / tta_diff=True are the
checkpoint defaults (ckpt config.json) — the library exposes no per-call
override, and both the 2026-10-21 sweep AND the model-card benchmark numbers
were produced with exactly these, so accrual stays apples-to-apples with
them. Do not compare this line against a TAA-on run.

timeout_seconds is enforced at the caller level exactly as in
chronos/timesfm/tirex signal: the forecast runs on a dedicated worker
thread and the CALLER abandons it (error signal, never cached) once the
deadline passes. The forward pass can't be interrupted mid-inference, so an
overrun leaves that thread busy until it finishes; the point is the CALLER
never blocks past the deadline.

Config-driven via `.agent-config.json` under `tirex2_signal`:
    tirex2_signal:
        enabled: false              # global toggle (ships disabled; flip to accrue)
        debug: false                # extra log detail; off = summary only
        model_id: "NX-AI/TiRex-2"   # tirex2 load_model id (HF or local dir)
        device: "cpu"               # no GPU in our stack
        context_length: 50          # candles fed as context (prod regime; model max 2048)
        forecast_horizon: 12        # steps ahead (5m bars; match the family)
        min_conf_ratio: 0.25        # confidence floor (see resolve_min_conf_ratio)
        cache_ttl_seconds: 300      # TTL for per-coin cache
        timeout_seconds: 30         # abort deadline per forecast; 0 = no deadline
        in_prompt: false            # research._tirex2_block renders only when true
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


# ── Singleton model (owned here) + config cache ───────────────────────────────
_tirex2_lock = threading.Lock()
_tirex2_model: Optional[Any] = None
_tirex2_loaded_at: float = 0
_tirex2_config_cache: Dict[str, Any] = {}
_model_init_error: Optional[str] = None


def _get_tirex2_config() -> Dict[str, Any]:
    """Read tirex2 config from `.agent-config.json` with sensible defaults."""
    global _tirex2_config_cache
    try:
        cfg = read_agent_config()
        if cfg is not _tirex2_config_cache:
            _tirex2_config_cache = cfg
            return cfg.get("tirex2_signal", {})
        return _tirex2_config_cache.get("tirex2_signal", {})
    except Exception as e:
        logger.debug(f"[tirex2] config read error: {e}")
        return {}


def _load_model(model_id: str, device: str) -> Any:
    """Import + load the TiRex-2 checkpoint.

    tirex2 (and its torch tensor plumbing) are imported HERE, inside the
    loader, so this module and its tests import fine in a venv without the
    dep installed (host test posture; pinned by
    test_lazy_import_keeps_module_import_clean). The dep
    is git-pinned in requirements.txt, so in the container it is present and
    the first-ever load also downloads the ~330MB checkpoint into the
    hf-cache volume (warm from cache <1s).

    NO-COMPILE READ PATH (verified 2026-10-21 in the prod image): the
    container image has no C++ compiler, and two tirex2 model components
    carry bare @torch.compile decorators (ResidualBlock, flex-attention)
    that raise InductorError at call time without one. Two knobs keep the
    whole path eager — `use_flex_attention=False` swaps the flex kernel for
    torch SDPA (same attention semantics), and torch._dynamo.config.disable
    neutralizes the decorators. The flag is torch-global and set only here
    (once per process, when this module first loads a model): no other
    torch user in the bot torch.compiles, so the blast radius is nil. The
    2026-10-21 host sweep ran WITH inductor compile at 68ms/anchor; the
    eager path is re-measured in the scratch container and must stay within
    the same order before enabling (see board t_f02823ab addendum).
    """
    import torch  # lazy: keeps the module importable without torch present
    torch._dynamo.config.disable = True
    from tirex2 import load_model
    return load_model(model_id, device=device, use_flex_attention=False)


def _get_model() -> Any:
    """Lazy-load the TiRex-2 singleton (thread-safe). Returns None if
    disabled or if loading fails (failure is cached; retries only after
    restart — same posture as the sibling signal modules)."""
    global _tirex2_model, _tirex2_loaded_at, _model_init_error
    cfg = _get_tirex2_config()
    if not cfg.get("enabled", False):
        return None

    with _tirex2_lock:
        if _tirex2_model is not None:
            return _tirex2_model
        if _model_init_error is not None:
            return None

        model_id = cfg.get("model_id", "NX-AI/TiRex-2")
        device = cfg.get("device", "cpu")
        try:
            logger.info(f"[tirex2] loading model {model_id} on {device}")
            start = time.time()
            model = _load_model(model_id, device)
            if model is None:
                raise RuntimeError("tirex2 load_model returned None")
            _tirex2_model = model
            _tirex2_loaded_at = time.time()
            logger.info(f"[tirex2] model ready in {time.time() - start:.1f}s")
            return model
        except ImportError as e:
            logger.warning(f"[tirex2] import failed (missing dependency?): {e}")
            _model_init_error = str(e)
            return None
        except Exception as e:
            logger.warning(f"[tirex2] model load failed: {e}")
            _model_init_error = str(e)
            return None


def preload_model(timeout_s: float = 180.0) -> bool:
    """Preload TiRex-2 on a bounded background thread at app init.

    Pays the one-time checkpoint cost OFF the first-scan critical path (warm
    from the hf-cache it is <1s; the first-ever load also downloads ~330MB).
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

    t = threading.Thread(target=_run, name="tirex2-preload", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        logger.warning(
            f"[tirex2] model preload exceeded {timeout_s:.0f}s — continuing; "
            "it will finish in the background or load lazily on first use")
    elif state["error"] is not None:
        logger.warning(f"[tirex2] model preload failed (lazy fallback): {state['error']}")
    elif state["model"] is None:
        logger.debug("[tirex2] model not loaded (disabled or error cached); lazy fallback")
    else:
        logger.info("[tirex2] model preloaded at init")
    return state["model"] is not None


# ── Result structure ──────────────────────────────────────────────────────────
@dataclass
class Tirex2Signal:
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
    # head-to-head gate (or a chronos->tirex2 swap) consumes any of the
    # family with one code path. None on every failure path.
    q10_path_pct: Optional[List[float]] = None
    q90_path_pct: Optional[List[float]] = None


# ── Per-coin cache (TTL-based) ────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}

# In-flight dedup: coin -> Event set when the owner's compute settles. The
# loop fire (async worker) and the route_verdict/executor attaches can reach
# _fetch_signal for the same cold coin within milliseconds of each other;
# without this, BOTH compute and BOTH log — duplicate accrual lines that skew
# any grep-counted rate. One compute per coin per TTL; late callers wait on
# the owner and read its cache.
_inflight_lock = threading.Lock()
_inflight: Dict[str, threading.Event] = {}


def _cache_get(coin: str, ttl: float) -> Optional[Tirex2Signal]:
    with _cache_lock:
        entry = _cache.get(coin)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["signal"]
        return None


def _cache_set(coin: str, signal: Tirex2Signal, ttl: float) -> None:
    with _cache_lock:
        _cache[coin] = {"signal": signal, "ts": time.time()}


# ── Quantile index helper ─────────────────────────────────────────────────────
def _find_quantile_index(quantile_levels: List[float], target: float) -> int:
    """Find the index of the model's quantile closest to `target`."""
    return min(range(len(quantile_levels)), key=lambda i: abs(quantile_levels[i] - target))


def _quantile_levels(model: Any, nq: int) -> List[float]:
    """Quantile levels for an nq-column forecast. TiRex-2 REPORTS its levels
    (checkpoint config); read them from the model when the accessor exists,
    fall back to the canonical 0.1..0.9 grid when it emits exactly 9 (the
    tirex-ts assumption), else evenly-spaced interior levels."""
    try:
        levels = list(model._quantile_levels())
        if levels and len(levels) == nq:
            return [float(v) for v in levels]
    except Exception:
        pass
    if nq == 9:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return [round((i + 1) / (nq + 1), 4) for i in range(nq)]


# ── Forecast from candles ─────────────────────────────────────────────────────
def _forecast_from_candles(
    coin: str,
    side: str,
    candles: List[Candle],
    cfg: Dict[str, Any],
) -> Tirex2Signal:
    def _err(msg: str, last: float = 0.0) -> Tirex2Signal:
        return Tirex2Signal(
            coin=coin, side=side, context_last=last,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=int(cfg.get("forecast_horizon", 12)),
            model_id=cfg.get("model_id", "NX-AI/TiRex-2"),
            inference_ms=0, error=msg,
        )

    model = _get_model()
    if model is None:
        return _err("tirex2 disabled or failed to load")
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
        import torch
        from tirex2 import TimeseriesType

        start = time.time()
        # tirex-2 contract (verified host-side against the real checkpoint,
        # 2026-10-21 sweep): a list of TimeseriesType objects in, a list of
        # per-series (Q, horizon) arrays out (output_type="numpy").
        # Univariate mode: target-only, no covariates.
        ts = TimeseriesType(
            target=torch.from_numpy(np.asarray(closes, dtype=np.float32)).unsqueeze(0),
            past_covariates=None,
            future_covariates=None,
        )
        fcs = model.forecast([ts], prediction_length=horizon, output_type="numpy")
        inference_ms = (time.time() - start) * 1000

        q = np.asarray(fcs[0][0])[:, :horizon]  # (n_quantiles, h) — TRANSPOSED vs tirex-ts
        if q.ndim != 2 or q.shape[0] < 3 or q.shape[1] < horizon:
            return _err(f"unexpected forecast shape {q.shape}", last_close)

        levels = _quantile_levels(model, q.shape[0])
        idx_low = _find_quantile_index(levels, 0.1)
        idx_med = _find_quantile_index(levels, 0.5)
        idx_high = _find_quantile_index(levels, 0.9)

        # Path averages — same reduction chronos/timesfm/tirex use, so
        # median_pct / spread_pct are directly comparable across the family.
        median = float(q[idx_med, :].mean())
        q_low = float(q[idx_low, :].mean())
        q_high = float(q[idx_high, :].mean())
        q10_path_pct = [float(v) for v in
                        ((q[idx_low, :] - last_close) / last_close * 100)]
        q90_path_pct = [float(v) for v in
                        ((q[idx_high, :] - last_close) / last_close * 100)]
    except Exception as e:
        return _err(str(e), last_close)

    median_pct = ((median - last_close) / last_close * 100) if last_close > 0 else None
    spread_pct = ((q_high - q_low) / last_close * 100) if last_close > 0 else None

    return Tirex2Signal(
        coin=coin,
        side=side,
        context_last=last_close,
        median=median,
        q_low=q_low,
        q_high=q_high,
        median_pct=median_pct,
        spread_pct=spread_pct,
        horizon=horizon,
        model_id=cfg.get("model_id", "NX-AI/TiRex-2"),
        inference_ms=inference_ms,
        q10_path_pct=q10_path_pct,
        q90_path_pct=q90_path_pct,
    )


# ── Signal fetch (sync, cache-aware, still potentially slow) ──────────────────
def _compute_signal(coin: str, side: str, cfg: Dict[str, Any]) -> Tirex2Signal:
    """Candle fetch + forecast for one coin. Runs on a dedicated thread (see
    _fetch_signal); errors are captured there, not raised to the caller."""
    context_length = int(cfg.get("context_length", 50))
    # Same interval + same fetch as the rest of the family (5m, shared
    # candle cache), so every model forecasts from an identical context.
    candles = fetch_hl_candles(coin, "5m", context_length)
    return _forecast_from_candles(coin, side, candles or [], cfg)


def _fetch_signal(coin: str, side: str) -> Tirex2Signal:
    cfg = _get_tirex2_config()
    if not cfg.get("enabled", False):
        return Tirex2Signal(
            coin=coin, side=side, context_last=0.0,
            median=None, q_low=None, q_high=None,
            median_pct=None, spread_pct=None,
            horizon=int(cfg.get("forecast_horizon", 12)),
            model_id=cfg.get("model_id", "NX-AI/TiRex-2"),
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
    # process. Mirrors the sibling _fetch_signal implementations exactly.
    result: Dict[str, Any] = {}
    done = threading.Event()

    def _run():
        try:
            result["signal"] = _compute_signal(coin, side, cfg)
        except Exception as e:  # surfaced as an error signal, mirrors below
            result["signal"] = Tirex2Signal(
                coin=coin, side=side, context_last=0.0,
                median=None, q_low=None, q_high=None,
                median_pct=None, spread_pct=None,
                horizon=int(cfg.get("forecast_horizon", 12)),
                model_id=cfg.get("model_id", "NX-AI/TiRex-2"),
                inference_ms=0, error=str(e),
            )
        finally:
            done.set()

    threading.Thread(target=_run, name=f"tirex2-compute-{coin}", daemon=True).start()
    try:
        if not done.wait(timeout_s if timeout_s > 0 else None):
            signal = Tirex2Signal(
                coin=coin, side=side, context_last=0.0,
                median=None, q_low=None, q_high=None,
                median_pct=None, spread_pct=None,
                horizon=int(cfg.get("forecast_horizon", 12)),
                model_id=cfg.get("model_id", "NX-AI/TiRex-2"),
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
    Same semantics as the chronos/timesfm/tirex modules.
    """
    try:
        v = float(cfg.get("min_conf_ratio", 0.25))
    except (TypeError, ValueError):
        return 0.25
    return max(0.0, v)


def confidence_ratio(sig: Tirex2Signal) -> float:
    """|median_pct| / spread_pct — the median's size vs the model's own
    p10-p90 band. Fail-safe is ZERO confidence: a missing spread or median
    means we have no basis for a directional claim."""
    if sig.median_pct is None or not sig.spread_pct:
        return 0.0
    return abs(sig.median_pct) / sig.spread_pct


# ── Log formatting ────────────────────────────────────────────────────────────
def _format_signal_log(
    signal: Tirex2Signal, debug: bool, min_conf_ratio: float = 0.25
) -> str:
    """Human-readable one-line log for the signal, mirroring the chronos /
    timesfm / tirex / timestral lines so a grep/tail of all family models
    side by side reads identically.

    The ALIGN/MISMATCH flag only counts when the median is confident
    (|median_pct| >= min_conf_ratio x the p10-p90 spread); below the floor
    the line reads NEUTRAL (a median inside its own uncertainty band is
    noise, not a call).
    """
    if signal.error:
        return f"[tirex2] {signal.coin} error: {signal.error}"

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
        f"[tirex2] {signal.coin} ({signal.side}) "
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
def get_tirex2_signal_async(coin: str, side: str) -> None:
    """Fire-and-forget TiRex-2 forecast on a daemon thread.

    NEVER blocks the caller. The daemon thread checks enabled, lazy-loads
    the model (owned singleton), fetches candles, forecasts, logs, and
    caches with TTL. Same pattern as the rest of the family.
    """
    cfg = _get_tirex2_config()

    def _worker():
        try:
            if not cfg.get("enabled", False):
                logger.debug(f"[tirex2] {coin}: signal disabled")
                return
            # _fetch_signal logs on a fresh compute; cache hits are silent.
            _fetch_signal(coin, side)
        except Exception as e:
            logger.debug(f"[tirex2] {coin} worker failed: {e}")

    threading.Thread(target=_worker, name=f"tirex2-{coin}", daemon=True).start()


def peek_tirex2(coin: str) -> Optional[Tirex2Signal]:
    """Return the cached forecast if fresh, else None. NEVER computes, never
    blocks. Same never-computes contract as peek_chronos/peek_tirex."""
    try:
        cfg = _get_tirex2_config()
        if not cfg.get("enabled", False):
            return None
        return _cache_get(coin, float(cfg.get("cache_ttl_seconds", 300)))
    except Exception as e:
        logger.debug(f"[tirex2] peek failed for {coin}: {e}")
        return None


# ── Synchronous wrapper (attach / testing path) ───────────────────────────────
def get_tirex2_signal_sync(coin: str, side: str) -> Tirex2Signal:
    """Return the TiRex-2 signal synchronously (cache-first).

    USE ONLY FOR TESTING or when you explicitly want to block for a forecast.
    The main pipeline uses `get_tirex2_signal_async()` instead. Logging happens
    in `_fetch_signal` (once per real compute), so this wrapper does not log a
    second time.
    """
    return _fetch_signal(coin, side)
