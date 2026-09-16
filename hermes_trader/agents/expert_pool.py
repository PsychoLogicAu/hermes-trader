"""Expert-selection candidate pool (MoiraiAgent integration, Task 1.0).

Four time-series candidate forecasters behind one uniform interface, for the
`expert_select` shadow signal (salesforce MoiraiAgent expert-selection
agent, vendored selection math in `expert_select.py`):

    chronos  — Chronos-2 (amazon/chronos-2)         — borrows the already-loaded
                 chronos_signal pipeline (no second load)
    timesfm  — TimesFM-3 (google/timesfm-3.0-pytorch) — borrows the already-loaded
                 timesfm_signal forecaster
    tirex    — TiRex 1.1 (NX-AI/TiRex, tirex-ts)    — own lazy singleton
    moirai2  — Moirai-2 (Salesforce/moirai-2.0-R-small) — vendored port
                 (hermes_trader/agents/moirai2_vendor/, D3a-A)

Every adapter forecasts UNIVARIATE closes and returns the SAME contract:

    {"median": [float]*h, "q10": [float]*h, "q90": [float]*h}

in ABSOLUTE price units (percent conversion is the caller's job — the
selection prompt needs raw z-scores, not pct).

Design rules (mirror chronos_signal / timesfm_signal):
- SHADOW ONLY. Nothing here gates, sizes, or enters the LLM verdict prompt.
- Lazy singletons, one lock per adapter; a failed/cold load degrades to
  an error for that adapter only — the pool keeps serving the rest.
- Caller-side timeout per adapter (dedicated worker thread, abandon on
  deadline — torch predict can't be interrupted mid-inference).
- Per-adapter latency + error counters are logged (debug) so the CPU
  budget decision (plan Task 1.0: drop >~2s warm adapters via `pool`
  config) is data-driven, not a guess.

LICENSE NOTES: chronos-2 / timesfm-3 weights carry the licenses accepted
by their existing signal integrations; TiRex weights are NXAI Community
License (Llama-3-style — free unless >€100M consolidated annual revenue
AND commercial product use); moirai-2.0-R-small weights are CC-BY-NC-4.0
(non-commercial). Vendored code is Apache-2.0 (Salesforce uni2ts @
8062ef5). Same posture as the TimesFM-3 integration: shadow-only, and the
pool ships behind `expert_select.enabled: false`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from hermes_trader.agents.config_store import read_agent_config
from hermes_trader.client.hl_client import fetch_hl_candles
from hermes_trader.indicators.math import candle_val
from hermes_trader.models.types import Candle

logger = logging.getLogger(__name__)

MODEL_NAMES = ("chronos", "timesfm", "tirex", "moirai2")

# Adapter-specific context windows (each model sees what it was designed
# for; MoiraiAgent runs every candidate on the same history and lets each
# model truncate internally). Values are candle counts at the signal's
# 5m interval.
_ADAPTER_CTX_DEFAULTS = {
    "chronos": 50,    # match the live chronos_signal context
    "timesfm": 100,   # match the live timesfm_signal context
    "tirex": 256,     # xLSTM context; bounded for CPU
    "moirai2": 1024,  # patch model; its predict() slices to context_length=4000 anyway
}
# Default per-adapter caller timeout (seconds). Cold first inference can
# exceed this (weights load ~10s) — cold runs are expected to degrade to
# an error signal and the next TTL window picks up the warm path.
_ADAPTER_TIMEOUT_DEFAULT = 15.0


@dataclass
class CandidateForecast:
    model: str
    median: Optional[List[float]] = None   # absolute price, len == horizon
    q10: Optional[List[float]] = None
    q90: Optional[List[float]] = None
    horizon: int = 0
    inference_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class _Stats:
    calls: int = 0
    errors: int = 0
    timeouts: int = 0
    last_ms: float = 0.0

    def note(self, ms: float, error: Optional[str] = None) -> None:
        self.calls += 1
        self.last_ms = ms
        if error:
            self.errors += 1
            if "timeout" in error:
                self.timeouts += 1


# ── Config ─────────────────────────────────────────────────────────────────────
_cfg_cache: Dict[str, Any] = {}


def _cfg() -> Dict[str, Any]:
    """expert_select config block (hot-reloaded via config_store)."""
    global _cfg_cache
    try:
        cfg = read_agent_config()
        if cfg is not _cfg_cache:
            _cfg_cache = cfg
        return _cfg_cache.get("expert_select", {})
    except Exception:
        return {}


def _pool() -> List[str]:
    """Enabled candidate list (config `pool` ∩ known names, stable order)."""
    raw = _cfg().get("pool", list(MODEL_NAMES))
    if not isinstance(raw, list):
        return list(MODEL_NAMES)
    pool = [m for m in raw if m in MODEL_NAMES]
    return pool or list(MODEL_NAMES)


def _adapter_timeout(model: str) -> float:
    try:
        v = float(_cfg().get(f"timeout_{model}_seconds", _ADAPTER_TIMEOUT_DEFAULT))
    except (TypeError, ValueError):
        return _ADAPTER_TIMEOUT_DEFAULT
    return v if v > 0 else _ADAPTER_TIMEOUT_DEFAULT


def _adapter_ctx(model: str) -> int:
    try:
        v = int(_cfg().get(f"context_{model}", _ADAPTER_CTX_DEFAULTS[model]))
    except (TypeError, ValueError, KeyError):
        return _ADAPTER_CTX_DEFAULTS[model]
    return max(8, v)


# ── Adapter singletons (lazy, one lock each) ────────────────────────────────────
_locks: Dict[str, threading.Lock] = {m: threading.Lock() for m in MODEL_NAMES}
_resources: Dict[str, Any] = {}      # model -> loaded resource
_load_errors: Dict[str, str] = {}    # model -> cached load error (sticky)
_stats: Dict[str, _Stats] = {m: _Stats() for m in MODEL_NAMES}


def _get_chronos_pipeline():
    """Borrow the chronos_signal pipeline singleton — never a second load."""
    with _locks["chronos"]:
        if "chronos" in _resources:
            return _resources["chronos"]
        if _load_errors.get("chronos"):
            return None
        try:
            from hermes_trader.agents import chronos_signal
            pipeline = chronos_signal._get_pipeline()
            if pipeline is None:
                # chronos_signal is disabled or failed to load — record so we
                # don't re-attempt every TTL (it will retry after config flips
                # because config reload clears nothing here; acceptable: the
                # adapter degrades to an error signal, pool serves the rest).
                _load_errors["chronos"] = "chronos_signal pipeline unavailable (disabled or load error)"
                return None
            _resources["chronos"] = pipeline
            return pipeline
        except Exception as e:
            _load_errors["chronos"] = str(e)
            logger.warning(f"[expert_pool] chronos borrow failed: {e}")
            return None


def _get_timesfm_forecaster():
    """Borrow the timesfm_signal forecaster singleton — never a second load."""
    with _locks["timesfm"]:
        if "timesfm" in _resources:
            return _resources["timesfm"]
        if _load_errors.get("timesfm"):
            return None
        try:
            from hermes_trader.agents import timesfm_signal
            forecaster = timesfm_signal._get_forecaster()
            if forecaster is None:
                _load_errors["timesfm"] = "timesfm_signal forecaster unavailable (disabled or load error)"
                return None
            _resources["timesfm"] = forecaster
            return forecaster
        except Exception as e:
            _load_errors["timesfm"] = str(e)
            logger.warning(f"[expert_pool] timesfm borrow failed: {e}")
            return None


def _get_tirex():
    with _locks["tirex"]:
        if "tirex" in _resources:
            return _resources["tirex"]
        if _load_errors.get("tirex"):
            return None
        try:
            from tirex import load_model
            start = time.time()
            model = load_model("NX-AI/TiRex", device="cpu")
            _resources["tirex"] = model
            logger.info(f"[expert_pool] tirex loaded in {time.time() - start:.1f}s")
            return model
        except Exception as e:
            _load_errors["tirex"] = str(e)
            logger.warning(f"[expert_pool] tirex load failed: {e}")
            return None


def _get_moirai2():
    with _locks["moirai2"]:
        if "moirai2" in _resources:
            return _resources["moirai2"]
        if _load_errors.get("moirai2"):
            return None
        try:
            from hermes_trader.agents.moirai2_vendor import Moirai2Module
            start = time.time()
            module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
            _resources["moirai2"] = module
            logger.info(f"[expert_pool] moirai2 module loaded in {time.time() - start:.1f}s")
            return module
        except Exception as e:
            _load_errors["moirai2"] = str(e)
            logger.warning(f"[expert_pool] moirai2 load failed: {e}")
            return None


def _ensure_loaded(model: str):
    """Load (or borrow) the adapter resource; None on error (logged once)."""
    if model == "chronos":
        return _get_chronos_pipeline()
    if model == "timesfm":
        return _get_timesfm_forecaster()
    if model == "tirex":
        return _get_tirex()
    if model == "moirai2":
        return _get_moirai2()
    return None


def preload_all(models: Optional[List[str]] = None, timeout_s: float = 120.0) -> Dict[str, bool]:
    """Preload every pool adapter on a bounded background thread at app init
    (same pattern as timesfm_signal.preload_model). Returns per-model
    readiness once the join completes; a slow load keeps going in the
    background and is picked up on first use."""
    models = [m for m in (models or _pool()) if m in MODEL_NAMES]

    def _run() -> None:
        for m in models:
            try:
                _ensure_loaded(m)
            except Exception as e:
                logger.warning(f"[expert_pool] preload {m} failed: {e}")

    t = threading.Thread(target=_run, name="expert-pool-preload", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        logger.warning(
            f"[expert_pool] preload exceeded {timeout_s:.0f}s — continuing in background")
    ready = {m: (m in _resources) for m in models}
    logger.info(f"[expert_pool] preload ready={ready}")
    return ready


# ── Per-adapter raw forecast (pure math; caller provides closes) ───────────────
def _forecast_raw(model: str, closes: List[float], horizon: int) -> Dict[str, Any]:
    """Run one adapter on a close-price list. Returns {median,q10,q90} in
    ABSOLUTE units (each len == horizon) or raises on any error."""
    if not closes or horizon <= 0:
        raise ValueError("need closes and horizon > 0")

    if model == "chronos":
        from hermes_trader.agents import chronos_signal
        pipeline = _ensure_loaded("chronos")
        if pipeline is None:
            raise RuntimeError(_load_errors.get("chronos", "unavailable"))
        import torch
        data = [np.asarray(closes, dtype=np.float32)]
        with torch.no_grad():
            quantile_forecast, _ = pipeline.predict_quantiles(
                inputs=data, prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
            )
        q = quantile_forecast[0][0, :horizon, :].cpu().numpy()  # (h, 3)
        return {"median": q[:, 1].tolist(), "q10": q[:, 0].tolist(), "q90": q[:, 2].tolist()}

    if model == "timesfm":
        forecaster = _ensure_loaded("timesfm")
        if forecaster is None:
            raise RuntimeError(_load_errors.get("timesfm", "unavailable"))
        out = forecaster.predict(
            np.asarray(closes, dtype=np.float32),
            horizon=horizon,
            return_quantiles=True,
            use_symmetric_averaging=False,
            make_positive=False,
        )
        median = np.asarray(out.forecast, dtype=np.float64)[:horizon]
        q = np.asarray(out.quantiles, dtype=np.float64)[:horizon]
        # quantile order follows the checkpoint config (0.1..0.9, 3.0)
        qlevels = [float(x) for x in forecaster.config.quantiles]
        i10 = min(range(len(qlevels)), key=lambda i: abs(qlevels[i] - 0.1))
        i90 = min(range(len(qlevels)), key=lambda i: abs(qlevels[i] - 0.9))
        return {"median": median.tolist(), "q10": q[:, i10].tolist(), "q90": q[:, i90].tolist()}

    if model == "tirex":
        m = _ensure_loaded("tirex")
        if m is None:
            raise RuntimeError(_load_errors.get("tirex", "unavailable"))
        # tirex-ts: forecast(context=[1-D arrays], prediction_length,
        # resample_strategy) -> (quantile_forecast, _) with shape
        # (batch, horizon, n_quantiles) — TRANSPOSED vs moirai2/chronos.
        quantile_forecast, _ = m.forecast(
            context=[np.asarray(closes, dtype=float)],
            prediction_length=horizon,
            resample_strategy="frequency",
        )
        q = np.asarray(quantile_forecast)[0, :horizon, :]  # (h, n_quantiles)
        nq = q.shape[-1]
        if nq != 9:
            # resolve by value like timesfm: fixed levels assumed 0.1..0.9
            i10, i50, i90 = int(0.1 * nq) or 0, int(0.5 * nq), int(0.9 * nq)
        else:
            i10, i50, i90 = 0, 4, 8
        return {"median": q[:, i50].tolist(), "q10": q[:, i10].tolist(), "q90": q[:, i90].tolist()}

    if model == "moirai2":
        from hermes_trader.agents.moirai2_vendor import Moirai2Forecast
        module = _ensure_loaded("moirai2")
        if module is None:
            raise RuntimeError(_load_errors.get("moirai2", "unavailable"))
        # Per-call wrapper is cheap (no weights); context_length 4000 matches
        # the MoiraiAgent project's reference usage — predict() slices/pads.
        fc = Moirai2Forecast(
            module=module, prediction_length=horizon, context_length=4000,
            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        )
        out = fc.predict([list(map(float, closes))])  # (1, 9, h)
        out = np.asarray(out)[0, :, :horizon]          # (9, h)
        return {"median": out[4].tolist(), "q10": out[0].tolist(), "q90": out[8].tolist()}

    raise ValueError(f"unknown model {model!r}")


# ── Bounded wrapper (caller-side timeout + stats) ──────────────────────────────
def forecast(model: str, closes: List[float], horizon: int) -> CandidateForecast:
    """One candidate forecast, bounded by the adapter's caller-side timeout.

    NEVER raises: any failure (load error, timeout, shape error) returns a
    CandidateForecast with .error set. The pool degrades to its remaining
    candidates; the signal degrades to mixture (see expert_select)."""
    if model not in MODEL_NAMES:
        return CandidateForecast(model=model, horizon=horizon, error="unknown model")
    if not closes:
        return CandidateForecast(model=model, horizon=horizon, error="no closes")
    if len(closes) > _adapter_ctx(model):
        closes = closes[-_adapter_ctx(model):]

    stats = _stats[model]
    result: Dict[str, Any] = {}
    done = threading.Event()

    def _run() -> None:
        t0 = time.time()
        try:
            result["out"] = _forecast_raw(model, closes, horizon)
            result["ms"] = (time.time() - t0) * 1000
        except Exception as e:
            result["err"] = f"{type(e).__name__}: {e}"
        finally:
            done.set()

    threading.Thread(target=_run, name=f"expert-pool-{model}", daemon=True).start()
    timeout_s = _adapter_timeout(model)
    if not done.wait(timeout_s if timeout_s > 0 else None):
        stats.note(0.0, f"timeout after {timeout_s:.0f}s")
        sig = CandidateForecast(model=model, horizon=horizon,
                                error=f"timeout after {timeout_s:.0f}s")
        _log_result(sig, debug=bool(_cfg().get("debug", False)))
        return sig
    if "err" in result:
        stats.note(0.0, result["err"])
        sig = CandidateForecast(model=model, horizon=horizon, error=result["err"])
        _log_result(sig, debug=bool(_cfg().get("debug", False)))
        return sig

    out = result["out"]
    sig = CandidateForecast(
        model=model,
        median=list(map(float, out["median"])),
        q10=list(map(float, out["q10"])),
        q90=list(map(float, out["q90"])),
        horizon=horizon,
        inference_ms=float(result.get("ms", 0.0)),
    )
    stats.note(sig.inference_ms)
    _log_result(sig, debug=bool(_cfg().get("debug", False)))
    return sig


def _log_result(sig: CandidateForecast, debug: bool) -> None:
    if sig.error:
        logger.info(f"[expert_pool] {sig.model} error: {sig.error}")
        return
    if debug:
        s = _stats[sig.model]
        logger.info(
            f"[expert_pool] {sig.model} median={sig.median[-1]:.4f} "
            f"({sig.inference_ms:.0f}ms; calls={s.calls} errors={s.errors} "
            f"timeouts={s.timeouts} last={s.last_ms:.0f}ms)"
        )
    else:
        logger.debug(f"[expert_pool] {sig.model} ok ({sig.inference_ms:.0f}ms)")


def pool_stats() -> Dict[str, Dict[str, float]]:
    """Per-adapter counters (for the CPU-budget decision and debug logs)."""
    return {
        m: {
            "calls": s.calls, "errors": s.errors, "timeouts": s.timeouts,
            "last_ms": s.last_ms,
        }
        for m, s in _stats.items()
    }


# ── Candle fetch (shared 90s cache via perception, same as chronos/timesfm) ────
def fetch_closes(coin: str, max_bars: int = 520, interval: str = "5m") -> List[Candle]:
    """Candles for pool + CV replay (live context AND a horizon-shifted CV
    window need ~context + horizon + margin bars). Uses the shared
    fetch_hl_candles 90s cache, same as both existing signals."""
    try:
        return fetch_hl_candles(coin, interval, max_bars) or []
    except Exception as e:
        logger.debug(f"[expert_pool] candle fetch failed {coin}: {e}")
        return []


def closes_from_candles(candles: List[Candle]) -> List[float]:
    """Close prices, oldest→newest, skipping non-positive closes."""
    out: List[float] = []
    for c in candles:
        try:
            v = float(candle_val(c, "c"))
        except Exception:
            continue
        if v > 0:
            out.append(v)
    return out
