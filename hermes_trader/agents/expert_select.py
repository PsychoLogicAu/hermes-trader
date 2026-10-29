"""Expert-selection shadow signal (MoiraiAgent integration, Task 1.1).

Runs a POOL of candidate forecasters (see ``expert_pool``), cross-validates
each on the tail of history, and pools them into the upstream "mixture"
distribution — the forecast math of Salesforce's MoiraiAgent expert-selection
agent (``gift_eval/agent.py``, Apache-2.0), ported to our regime.

SELECTION IS THE MIXTURE — the LLM winner-pick was RETIRED 2026-10-29
(t_61d27555). The 2026-10-07 week survey (n=683 diverse anchors, in-container
replay) measured the live selector NEGATIVE on its own cohort: recentered
selector median-MAE 1.220pp vs 1.146pp for the plain mixture on identical
contexts — picking a winner made the forecast worse, and the 1h-CV proxy it
read anti-correlated with realized accuracy (it ranked TiRex worst; TiRex was
best). The leg was pinned off by config on 2026-10-08
(``diversity_threshold: 999.0``) and its code deleted here. The per-model
``candidate_medians`` / q10 / q90 accrual is kept deliberately: it is what
makes the recenter-ORACLE ("what a perfect pick would have scored" — ~0.27pp
better than the mixture in that survey) measurable offline without any LLM.
If the CV proxy is ever fixed, re-enable = git revert the deletion commit +
re-add the ``expert_select.llm`` config block.

SHADOW ONLY. Nothing here gates, sizes, or enters the LLM verdict prompt.
The async worker logs one line per real compute; the sync attach reads the
per-coin cache only (never blocks the exec loop — the same contract as
chronos/timesfm attach).

Decision procedure:
  1. Run every pool candidate on the live context → per-model 9-quantile
     forecasts (absolute price).
  2. Run the SAME pool on the context shifted back by `horizon` bars
     (cross-validation replay over the last `horizon` bars) → per-model MAE
     vs the actual tail → cv_ranking. Kept as accrual/diagnostic: it is the
     scoreboard the retired selector read (and why it mis-picked).
  3. Diversity diagnostic: per-step std of the z-normalized candidate medians
     (downsampled to ≤40 steps) → the ``diverse=`` field on the log line ONLY.
     It gates nothing since the retirement.
  4. Final = the pooled mixture, unshifted → expert_median_pct /
     expert_spread_pct / q10/q90 paths, same scale as the other signals.
     ``_mixture_and_recenter`` still supports their re-center step for the
     offline oracle math; production always passes ``selected="mixture"``.

LICENSE NOTES: selection math Apache-2.0 (Salesforce uni2ts @8062ef5); TiRex
weights are NXAI Community License; moirai-2.0-R-small weights are
CC-BY-NC-4.0. Same posture as TimesFM-3: shadow-only, ships behind
``expert_select.enabled: false``. The retired moirai-agent selector weights
were CC-BY-NC-4.0; its GGUF is deleted from the lemonade model store
(mradermacher/moirai-agent-GGUF re-downloads it if the revert path is taken).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hermes_trader.agents.config_store import read_agent_config

logger = logging.getLogger(__name__)


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
    from hermes_trader.agents.expert_pool import MODEL_NAMES
    raw = _cfg().get("pool", list(MODEL_NAMES))
    if not isinstance(raw, list):
        return list(MODEL_NAMES)
    pool = [m for m in raw if m in MODEL_NAMES]
    return pool or list(MODEL_NAMES)


def _horizon() -> int:
    """Selection horizon in 5m candle steps (default 12 = 1h, apples-to-apples
    with the live chronos/timesfm comparison window)."""
    try:
        return max(1, int(_cfg().get("horizon", 12)))
    except (TypeError, ValueError):
        return 12


def _context_len() -> int:
    try:
        return max(8, int(_cfg().get("context_length", 50)))
    except (TypeError, ValueError):
        return 50


def _diversity_threshold() -> float:
    try:
        return float(_cfg().get("diversity_threshold", 0.2))
    except (TypeError, ValueError):
        return 0.2


def _cache_ttl() -> float:
    try:
        return float(_cfg().get("cache_ttl_seconds", 300))
    except (TypeError, ValueError):
        return 300.0


def _cv_ttl() -> float:
    try:
        return float(_cfg().get("cv_ttl_seconds", 3600))
    except (TypeError, ValueError):
        return 3600.0


def _timeout() -> float:
    try:
        return float(_cfg().get("timeout_seconds", 90))
    except (TypeError, ValueError):
        return 90.0


def _debug() -> bool:
    return bool(_cfg().get("debug", False))


# ── Result structure ──────────────────────────────────────────────────────────
@dataclass
class ExpertSelectSignal:
    coin: str
    side: str
    context_last: float
    horizon: int
    # Selection outcome. Post-retirement (2026-10-29) `selected_model` is
    # always "mixture"; the per-model names stay in the contract because the
    # historical accrual carries them and a revert of the deletion re-enables
    # picking. `diverse` is now a pure candidate-disagreement diagnostic.
    selected_model: Optional[str] = None   # always "mixture" post-retirement
    diverse: Optional[bool] = None
    cv_ranking: Optional[List[Tuple[str, float]]] = None  # [(model, mae)] best-first
    # Final (re-centered mixture) forecast, % vs context_last
    median_pct: Optional[float] = None
    spread_pct: Optional[float] = None
    q10_path_pct: Optional[List[float]] = None
    q90_path_pct: Optional[List[float]] = None
    inference_ms: float = 0.0
    error: Optional[str] = None
    # Per-candidate diagnostics (debug / accrual)
    candidates: Dict[str, str] = field(default_factory=dict)  # model -> ok|error
    # Raw per-model LIVE median paths (absolute price, len == horizon) — the
    # spec's "cache stores raw per-model median paths + CV metrics +
    # selection". Without these, Phase-2 replay could only recompute
    # tirex/moirai2 paths from candles; with them, the accrual alone is a
    # complete record for the per-model median-MAE join. (CV preds live in
    # the CV cache entry under "preds".)
    candidate_medians: Dict[str, List[float]] = field(default_factory=dict)
    # Raw per-model q10/q90 paths (% vs context_last, len == horizon) — the
    # complete record for the favorable-tail accrual (WATCHLIST §C.21): the
    # 2026-10-07 survey found the size-up tilt lives in the PER-MODEL
    # favorable tail (max q90[:6] long / -min q10[:6] short; chronos/timesfm
    # top-decile +1pp end-ret vs +0.10 baseline) and the mixture DILUTES it,
    # so per-model tails are required, not the re-centered mixture's.
    candidate_q10_pct: Dict[str, List[float]] = field(default_factory=dict)
    candidate_q90_pct: Dict[str, List[float]] = field(default_factory=dict)


# ── Per-coin cache (TTL-based; stores the full signal, not just the value) ────
_cache_lock = threading.Lock()
_cache: Dict[str, Dict[str, Any]] = {}


def _cache_get(coin: str, ttl: float) -> Optional[ExpertSelectSignal]:
    with _cache_lock:
        entry = _cache.get(coin)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["signal"]
        return None


def _cache_set(coin: str, signal: ExpertSelectSignal) -> None:
    with _cache_lock:
        _cache[coin] = {"signal": signal, "ts": time.time()}


# ── CV replay cache (longer TTL; the CV window shifts one bar per cycle) ──────
_cv_lock = threading.Lock()
_cv_cache: Dict[str, Dict[str, Any]] = {}


def _cv_cache_get(coin: str, ttl: float) -> Optional[Dict[str, Any]]:
    with _cv_lock:
        entry = _cv_cache.get(coin)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["data"]
        return None


def _cv_cache_set(coin: str, data: Dict[str, Any]) -> None:
    with _cv_lock:
        _cv_cache[coin] = {"data": data, "ts": time.time()}


# ── Normalization helpers (faithful to TimeSeriesProcessor) ───────────────────
def _norm_factor(values: List[float]) -> Tuple[float, float]:
    """mean/std of valid values; std clipped to >=1e-5 (their _get_norm_factor)."""
    arr = np.asarray(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return 0.0, 1.0
    mean = float(np.mean(arr[mask]))
    std = float(np.clip(np.std(arr[mask]), a_min=1e-5, a_max=None))
    return mean, std


def _normalize(values: List[float], mean: float, std: float) -> List[float]:
    """Z-score valid entries, leave NaNs (their _normalize_values)."""
    arr = np.asarray(values, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return arr.tolist()
    out = arr.copy()
    out[mask] = (arr[mask] - mean) / std
    return out.tolist()


def _mae(pred: List[float], target: List[float]) -> float:
    pred_arr = np.asarray(pred, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    mask = ~np.isnan(target_arr)
    if mask.sum() == 0:
        return 0.0
    diff = pred_arr[mask] - target_arr[mask]
    mae = float(np.mean(np.abs(diff)))
    return mae if not np.isnan(mae) else float("inf")


# ── Candidate collection (live + CV) ──────────────────────────────────────────
def _collect_live(closes: List[float], horizon: int) -> Dict[str, Any]:
    """Run every pool candidate on the LIVE context. Returns {model: sig}."""
    from hermes_trader.agents import expert_pool
    out: Dict[str, Any] = {}
    for model in _pool():
        out[model] = expert_pool.forecast(model, closes, horizon)
    return out


def _collect_cv(closes: List[float], horizon: int) -> Dict[str, Any]:
    """Cross-validation replay: forecast the last `horizon` bars using context
    ending `horizon` bars back; MAE vs the actual tail. Cached on cv_ttl."""
    from hermes_trader.agents import expert_pool
    n = len(closes)
    if n < horizon * 2 + 8:
        return {}  # not enough history for a meaningful replay
    cv_ctx = closes[: n - horizon]          # context ending `horizon` bars back
    actual_tail = closes[n - horizon:]       # the bars we're replaying
    cv_preds: Dict[str, List[float]] = {}
    cv_maes: Dict[str, float] = {}
    for model in _pool():
        sig = expert_pool.forecast(model, cv_ctx, horizon)
        if sig.error or not sig.median:
            continue
        cv_preds[model] = sig.median
        cv_maes[model] = _mae(sig.median, actual_tail)
    if not cv_preds:
        return {}
    ranking = sorted(cv_maes.items(), key=lambda kv: kv[1])
    return {"preds": cv_preds, "maes": cv_maes, "ranking": ranking,
            "tail": list(actual_tail)}


def _run_cv(coin: str, closes: List[float], horizon: int) -> Dict[str, Any]:
    """CV replay with the longer TTL cache (the window shifts one bar/cycle)."""
    cached = _cv_cache_get(coin, _cv_ttl())
    if cached is not None:
        return cached
    data = _collect_cv(closes, horizon)
    if data:
        _cv_cache_set(coin, data)
    return data


# ── Diversity gate (their check_preds_diversity) ──────────────────────────────
def _is_diverse(live: Dict[str, Any], closes: List[float],
                threshold: float, max_future_length: int = 40) -> bool:
    """Per-step std of the z-normalized candidate medians (downsampled to
    ≤max_future_length steps). DIAGNOSTIC ONLY since the selector LLM was
    retired (2026-10-29): selection is always the mixture, so this value just
    records whether the candidates disagreed (``diverse=`` on the log line)."""
    if threshold <= 0.0:
        return True
    mean, std = _norm_factor(closes)
    preds = []
    for model in _pool():
        sig = live.get(model)
        if sig and sig.median:
            preds.append(_normalize(list(sig.median), mean, std))
    if len(preds) < 2:
        # Nothing to compare with a single (or zero) candidate — the signal is
        # that lone candidate's own distribution, so report not-diverse.
        return False
    arr = np.array(preds, dtype=float)
    downspl_step = max(1, arr.shape[1] // max_future_length)
    arr = arr[:, ::downspl_step]
    diversity = float(np.std(arr, axis=0).mean())
    return diversity > threshold


# ── Mixture + re-centering (their get_best_pred) ──────────────────────────────
_MIXTURE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _mixture_and_recenter(live: Dict[str, Any], selected: str,
                          horizon: int) -> Dict[str, List[float]]:
    """Pool the candidates exactly like their ``_get_mixture_pred``: stack
    EVERY quantile feature of EVERY candidate (median + 0.1..0.9 — median is
    itself one of their FORECAST_FEATURES) into a (n_cand × 10, h) matrix,
    then take ONE ``np.quantile([0.1..0.9], axis=0)`` across that pooled
    sample. NOT a per-level quantile across candidates — that was an earlier
    (wrong) reading of "per-quantile pooling"; the concatenated stack is what
    their code does and what we match. Then shift the whole distribution so
    its median matches the selected model's median (their ``get_best_pred``).
    Returns {level: [h]} in absolute price.

    Production passes ``selected="mixture"`` (unshifted pool) since the selector
    LLM was retired 2026-10-29; the re-center branch is kept because the offline
    survey harness recomputes recenter-ORACLE variants through it and a revert of
    the deletion restores the selector.
    """
    models = [m for m in _pool() if live.get(m) and live[m].quantiles]
    if not models:
        raise ValueError("no candidate produced quantiles")
    rows: List[List[float]] = []
    for m in models:
        q = live[m].quantiles
        med = q.get("0.5") or (list(live[m].median) if live[m].median else None)
        if med is not None and len(med) >= horizon:
            rows.append(list(med[:horizon]))          # their "median" feature
        for lv in _MIXTURE_LEVELS:
            vals = q.get(f"{lv:.1f}") or q.get(str(lv)) or med
            if vals is not None and len(vals) >= horizon:
                rows.append(list(vals[:horizon]))
    if not rows:
        raise ValueError("mixture median unavailable")
    stacked = np.array(rows, dtype=float)              # (n_cand × 10, h)
    mixture: Dict[str, List[float]] = {}
    for i, lv in enumerate(_MIXTURE_LEVELS):
        mixture[f"{lv:.1f}"] = np.quantile(stacked, lv, axis=0).tolist()
    mix_med = mixture.get("0.5")
    if not mix_med:
        raise ValueError("mixture median unavailable")
    # Re-center: shift every level by (selected_median - mixture_median).
    if selected != "mixture":
        sel_sig = live.get(selected)
        if sel_sig and sel_sig.median and len(sel_sig.median) >= horizon:
            offset = np.asarray(sel_sig.median[:horizon], dtype=float) - \
                np.asarray(mix_med, dtype=float)
            for k in mixture:
                mixture[k] = (np.asarray(mixture[k], dtype=float) + offset).tolist()
    return mixture














# ── Core compute ──────────────────────────────────────────────────────────────
def _compute(coin: str, side: str) -> ExpertSelectSignal:
    from hermes_trader.agents import expert_pool
    horizon = _horizon()
    candles = expert_pool.fetch_closes(coin, max_bars=_context_len() + horizon + 16, interval="5m")
    closes = expert_pool.closes_from_candles(candles)
    if len(closes) < horizon + 8:
        return ExpertSelectSignal(
            coin=coin, side=side, context_last=0.0, horizon=horizon,
            error=f"insufficient candles ({len(closes)})",
        )
    # Live context = the most recent `_context_len()` closes.
    ctx = closes[-_context_len():]
    last_close = ctx[-1]
    t0 = time.time()

    live = _collect_live(ctx, horizon)
    ok_models = [m for m in live if not live[m].error and live[m].median]
    if not ok_models:
        return ExpertSelectSignal(
            coin=coin, side=side, context_last=last_close, horizon=horizon,
            candidates={m: (s.error or "no median") for m, s in live.items()},
            error="all candidates failed",
        )

    cv = _run_cv(coin, closes, horizon)
    # Selection is ALWAYS the pooled mixture — the selector LLM leg was deleted
    # 2026-10-29 (t_61d27555; the week survey measured recentered live-selector
    # median-MAE 1.220 vs 1.146 for the plain mixture on identical contexts, so
    # picking a winner was net-negative). `diverse` stays as a pure diagnostic
    # on the log line (did the candidates disagree?) — it gates nothing.
    diverse = _is_diverse(live, ctx, _diversity_threshold())
    selected = "mixture"

    try:
        mixture = _mixture_and_recenter(live, selected, horizon)
    except Exception as e:  # noqa: BLE001
        return ExpertSelectSignal(
            coin=coin, side=side, context_last=last_close, horizon=horizon,
            selected_model=selected, diverse=diverse,
            cv_ranking=[(m, round(mae, 6)) for m, mae in (cv.get("ranking") or [])],
            candidates={m: (s.error or "ok") for m, s in live.items()},
            error=f"mixture failed: {e}",
        )

    def pct(path: List[float]) -> List[float]:
        return [((v - last_close) / last_close * 100) for v in path]

    med = mixture.get("0.5")
    q10 = mixture.get("0.1")
    q90 = mixture.get("0.9")
    median_pct = ((med[-1] - last_close) / last_close * 100) if med and last_close > 0 else None
    spread_pct = ((q90[-1] - q10[-1]) / last_close * 100) if (q90 and q10 and last_close > 0) else None

    return ExpertSelectSignal(
        coin=coin, side=side, context_last=last_close, horizon=horizon,
        selected_model=selected, diverse=diverse,
        cv_ranking=[(m, round(mae, 6)) for m, mae in (cv.get("ranking") or [])],
        median_pct=median_pct, spread_pct=spread_pct,
        q10_path_pct=pct(q10) if q10 else None,
        q90_path_pct=pct(q90) if q90 else None,
        inference_ms=(time.time() - t0) * 1000,
        candidates={m: (s.error or "ok") for m, s in live.items()},
        candidate_medians={m: list(s.median[:horizon])
                           for m, s in live.items() if s.median},
        candidate_q10_pct={m: pct(list(s.quantiles["0.1"][:horizon]))
                           for m, s in live.items()
                           if s.quantiles and s.quantiles.get("0.1")},
        candidate_q90_pct={m: pct(list(s.quantiles["0.9"][:horizon]))
                           for m, s in live.items()
                           if s.quantiles and s.quantiles.get("0.9")},
        # No selection failure is possible any more (the LLM leg is deleted);
        # the field itself stays — `_format_log` renders `note=` from it and the
        # attach/error paths downstream depend on the shape.
        error=None,
    )


# ── Logging ───────────────────────────────────────────────────────────────────
def _format_log(sig: ExpertSelectSignal) -> str:
    if sig.error and sig.median_pct is None:
        return f"[expert] {sig.coin} ({sig.side}) ERROR: {sig.error}"
    sel = sig.selected_model or "?"
    div = "?" if sig.diverse is None else ("yes" if sig.diverse else "no")
    cv = "<".join(m for m, _ in (sig.cv_ranking or [])) or "-"
    med = f"{sig.median_pct:+.2f}%" if sig.median_pct is not None else "?"
    sp = f"{sig.spread_pct:.2f}%" if sig.spread_pct is not None else "?"
    base = (f"[expert] {sig.coin} ({sig.side}) sel={sel} diverse={div} "
            f"cv={cv} expert_median={med} spread={sp} horizon={sig.horizon} "
            f"inference={sig.inference_ms:.0f}ms")
    if sig.error:
        base += f" note={sig.error}"
    return base


# ── Public API ────────────────────────────────────────────────────────────────
def get_expert_select_async(coin: str, side: str) -> None:
    """Fire-and-forget selection on a daemon thread (NEVER blocks the caller).
    Mirrors get_chronos_signal_async: enabled-check, compute, log, cache."""
    cfg = _cfg()

    def _worker():
        try:
            if not cfg.get("enabled", False):
                logger.debug(f"[expert] {coin}: signal disabled")
                return
            # _fetch logs once per real compute (cache hits silent — a log in
            # this wrapper re-printed warm entries and inflated the mixture-
            # rate counters, same bug chronos fixed by logging inside _fetch).
            _fetch(coin, side)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[expert] {coin} worker failed: {e}")

    threading.Thread(target=_worker, name=f"expert-select-{coin}", daemon=True).start()


def peek_expert_select(coin: str) -> Optional[ExpertSelectSignal]:
    """Return the cached selection if fresh, else None. NEVER computes/blocks
    (the attach path uses this — it must not pay model-load or LLM cost)."""
    try:
        if not _cfg().get("enabled", False):
            return None
        return _cache_get(coin, _cache_ttl())
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[expert] peek failed for {coin}: {e}")
        return None


def _fetch(coin: str, side: str) -> ExpertSelectSignal:
    """Compute (bounded by timeout) and cache. Logs nothing (caller logs once
    per real compute)."""
    if not _cfg().get("enabled", False):
        return ExpertSelectSignal(coin=coin, side=side, context_last=0.0,
                                  horizon=_horizon(), error="disabled")
    cached = _cache_get(coin, _cache_ttl())
    if cached is not None:
        return cached

    timeout_s = _timeout()
    result: Dict[str, Any] = {}
    done = threading.Event()

    def _run():
        try:
            result["signal"] = _compute(coin, side)
        except Exception as e:  # noqa: BLE001
            result["signal"] = ExpertSelectSignal(
                coin=coin, side=side, context_last=0.0, horizon=_horizon(),
                error=str(e),
            )
        finally:
            done.set()

    threading.Thread(target=_run, name=f"expert-compute-{coin}", daemon=True).start()
    if not done.wait(timeout_s if timeout_s > 0 else None):
        sig = ExpertSelectSignal(
            coin=coin, side=side, context_last=0.0, horizon=_horizon(),
            error=f"timeout after {timeout_s:.0f}s",
        )
        logger.info(_format_log(sig))  # a real (failed) compute — log once
        return sig  # never cached

    sig = result["signal"]
    if not sig.error:
        _cache_set(coin, sig)
    # Log once per actual compute (cache misses only — cache hits returned
    # above). Errors are NOT cached but ARE logged, so an outage is visible.
    logger.info(_format_log(sig))
    return sig


def get_expert_select_sync(coin: str, side: str) -> ExpertSelectSignal:
    """Synchronous wrapper for testing / explicit use. USE ONLY FOR TESTING —
    the main pipeline uses get_expert_select_async()."""
    return _fetch(coin, side)


def clear_cache(coin: Optional[str] = None) -> None:
    """Test helper: drop the per-coin caches."""
    with _cache_lock:
        if coin is None:
            _cache.clear()
        else:
            _cache.pop(coin, None)
    with _cv_lock:
        if coin is None:
            _cv_cache.clear()
        else:
            _cv_cache.pop(coin, None)
