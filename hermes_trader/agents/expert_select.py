"""Expert-selection shadow signal (MoiraiAgent integration, Task 1.1).

Runs a POOL of candidate forecasters (see ``expert_pool``), cross-validates
each on the tail of history, and has an LLM pick the winner (or pool them
into a "mixture") — Salesforce's MoiraiAgent expert-selection agent
(``gift_eval/agent.py``, Apache-2.0), ported to our regime. The final
forecast is the pooled "mixture" distribution RE-CENTERED so its median
matches the selected model's median (their ``get_best_pred`` math): a winner
≠ mixture shifts the whole mixture, it does not replace it.

SHADOW ONLY. Nothing here gates, sizes, or enters the LLM verdict prompt.
The async worker logs one line per real compute; the sync attach reads the
per-coin cache only (never blocks the exec loop — the same contract as
chronos/timesfm attach).

Decision procedure (mirrors MoiraiAgentTimeSeriesForecast.__call__):
  1. Run every pool candidate on the live context → per-model 9-quantile
     forecasts (absolute price).
  2. Run the SAME pool on the context shifted back by `horizon` bars
     (cross-validation replay over the last `horizon` bars) → per-model MAE
     vs the actual tail → cv_ranking.
  3. Diversity gate: per-step std of the z-normalized candidate medians
     (downsampled to ≤40 steps); below `diversity_threshold` (0.2) → skip the
     LLM, selection = "mixture".
  4. Otherwise build THEIR exact text prompt (z-normalized history ≤400
     steps, each model's normalized candidate medians, the CV replay medians,
     and the CV MAE ranking string) → one short chat completion → parse the
     winner via ``\\boxed{...}``. A parse failure / timeout / endpoint error
     degrades to "mixture" (their code ASSERTS on bad output; we must not).
  5. Final = mixture quantile pool re-centered to the selected model's
     median → expert_median_pct / expert_spread_pct / q10/q90 paths on the
     same scale as the existing signals.

Selector LLM (D2): default = the running Qwen primary via the existing
LLM_BASE_URL/LLM_API_KEY env + llm.primary.model from config; hot-swappable
to a self-hosted moirai-agent fine-tune by setting explicit base_url/model in
config (no code change). Thinking off via per-request chat_template_kwargs.

LICENSE NOTES: selection code Apache-2.0 (Salesforce uni2ts @8062ef5); the
optional moirai-agent selector weights are CC-BY-NC-4.0; TiRex weights are
NXAI Community License; moirai-2.0-R-small weights are CC-BY-NC-4.0. Same
posture as TimesFM-3: shadow-only, ships behind expert_select.enabled: false.
"""

from __future__ import annotations

import json
import logging
import os
import re
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


def _llm_cfg() -> Dict[str, Any]:
    return _cfg().get("llm", {}) or {}


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
    # Selection outcome
    selected_model: Optional[str] = None   # chronos/timesfm/tirex/moirai2/mixture
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
    ≤max_future_length steps). Below threshold → not diverse → mixture."""
    if threshold <= 0.0:
        return True
    mean, std = _norm_factor(closes)
    preds = []
    for model in _pool():
        sig = live.get(model)
        if sig and sig.median:
            preds.append(_normalize(list(sig.median), mean, std))
    if len(preds) < 2:
        # Nothing to select between with a single (or zero) candidate — the
        # selection is the mixture (the lone candidate's own distribution,
        # unshifted) and the LLM call would be meaningless.
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
    Returns {level: [h]} in absolute price."""
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


# ── Prompt construction (their TimeSeriesProcessor.__call__, faithful) ────────
_INSTRUCTION = (
    "You are given a sequence of history values and several future predictions by candidate models. "
    "Analyze the future predictions by the candidates and their cross-validation errors on the last part of the history values. "
    "Select the optimal future predictions. Enclose the name of the best model by \\boxed{ and }. "
)


def _fmt_ts(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ms / 1000.0))


def _build_prompt(closes: List[float], ts: List[int], live: Dict[str, Any],
                  cv: Dict[str, Any], horizon: int,
                  max_future_length: int = 40,
                  max_history_future_ratio: int = 10) -> str:
    """Build the selection user-prompt in their exact format (three JSON blocks
    + instruction). All values z-normalized to 3 decimals; history truncated to
    ratio*future steps; candidate paths downsampled to ≤max_future_length."""
    mean, std = _norm_factor(closes)
    norm_hist = _normalize(closes, mean, std)
    models = [m for m in _pool() if live.get(m) and live[m].median]

    # Downsample step: they use max(1, pred_length // max_future_length). Our
    # horizon (12) < max_future_length (40) → step 1 (no downsampling).
    downspl_step = max(1, horizon // max_future_length)

    # Candidate medians, normalized, downsampled (forward).
    cand_strs: Dict[str, str] = {}
    for m in models:
        norm = _normalize(list(live[m].median), mean, std)[:horizon]
        ds = norm[0::downspl_step]
        cand_strs[m] = ",".join(f"{v:.3f}" for v in ds)

    # History: downsampled (backward) then truncated to max_ctx_length.
    max_ctx = int(max_history_future_ratio * max_future_length)
    hist_ds = norm_hist[-downspl_step::-downspl_step][::-1]
    hist_ts_ds = ts[-downspl_step::-downspl_step][::-1]
    ctx_len = min(len(hist_ds), max_ctx)
    hist_ds = hist_ds[-ctx_len:]
    hist_ts_ds = hist_ts_ds[-ctx_len:]
    hist_str = ",".join(f"{v:.3f}" for v in hist_ds)

    # Future timestamps (the horizon ahead of the last bar).
    last_ms = ts[-1]
    future_ts = [last_ms + (i + 1) * 300_000 for i in range(horizon)]
    future_ts_ds = future_ts[0::downspl_step]

    # CV section: normalized CV replay medians (downsampled backward) + GT tail.
    cv_preds_strs: Dict[str, str] = {}
    cv_gt_str = ""
    if cv:
        tail = cv.get("tail", [])
        cv_gt = _normalize(list(tail), mean, std)
        cv_gt_str = ",".join(f"{v:.3f}" for v in cv_gt)
        for m, pred in (cv.get("preds") or {}).items():
            norm = _normalize(list(pred), mean, std)[:horizon]
            ds = norm[-downspl_step::-downspl_step][::-1]
            cv_preds_strs[m] = ",".join(f"{v:.3f}" for v in ds)
        ranking = [m for m, _ in cv.get("ranking", [])]
        cv_ranking_str = " < ".join(ranking)
    else:
        cv_ranking_str = ""

    history_info = {
        "history_window": [_fmt_ts(hist_ts_ds[0]), _fmt_ts(hist_ts_ds[-1])] if hist_ts_ds else ["?", "?"],
        "history_values": hist_str,
    }
    pred_info = {
        "future_window": [_fmt_ts(future_ts_ds[0]), _fmt_ts(future_ts_ds[-1])] if future_ts_ds else ["?", "?"],
        "model_names": models,
        "candidate_preds": cand_strs,
    }
    cv_info = {
        "crossval_window": [_fmt_ts(ts[-horizon]), _fmt_ts(ts[-1])] if len(ts) >= horizon else ["?", "?"],
        "crossval_ground_truth": cv_gt_str,
        "crossval_preds": cv_preds_strs,
        "crossval_error_ranking": cv_ranking_str,
    }
    query = (
        f"{json.dumps(history_info, indent=2)}\n\n\n"
        f"{json.dumps(pred_info, indent=2)}\n\n\n"
        f"{json.dumps(cv_info, indent=2)}\n\n\n"
        f"{_INSTRUCTION}"
    )
    return query


# ── LLM selection call (their parse_answer, but fallback instead of assert) ───
_BOXED_RE = re.compile(r"\\boxed\{(.+)\}", re.DOTALL)


def _resolve_llm_params() -> Tuple[str, str, str]:
    """Return (base_url, api_key, model). Explicit config wins; otherwise the
    running primary (LLM_* env + llm.primary.model)."""
    lc = _llm_cfg()
    base_url = lc.get("base_url") or os.environ.get(lc.get("base_url_env", "LLM_BASE_URL"), "")
    api_key = os.environ.get(lc.get("api_key_env", "LLM_API_KEY"), "")
    model = lc.get("model") or ""
    if not model:
        if lc.get("use_primary_model", True):
            try:
                from hermes_trader.agents.duel_store import effective_primary_model
                model = effective_primary_model()
            except Exception:
                model = ""
        if not model:
            model = os.environ.get("LLM_MODEL", "")
    return base_url, api_key, model


def _select_with_llm(prompt: str, model_names: List[str]) -> Tuple[str, str]:
    """One chat completion → parse the boxed winner. Returns (selection, note).
    ANY failure (no endpoint, timeout, HTTP error, empty box, unknown name)
    degrades to ("mixture", note) — never raises (their code asserts; we must
    not block the exec loop)."""
    base_url, api_key, model = _resolve_llm_params()
    if not base_url or not api_key:
        return "mixture", "llm endpoint not configured"
    lc = _llm_cfg()
    timeout_s = float(lc.get("timeout_seconds", 20) or 20)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": int(lc.get("max_tokens", 256)),
        "temperature": float(lc.get("temperature", 0.7)),
        "top_p": float(lc.get("top_p", 0.8)),
    }
    if lc.get("top_k"):
        body["top_k"] = int(lc["top_k"])
    if lc.get("repetition_penalty"):
        body["repetition_penalty"] = float(lc["repetition_penalty"])
    # Thinking off (mirror research.py's per-request chat_template_kwargs).
    if lc.get("enable_thinking") is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        import httpx
        with httpx.Client(timeout=httpx.Timeout(timeout_s)) as client:
            resp = client.post(
                base_url.rstrip("/") + "/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code != 200:
            return "mixture", f"llm http {resp.status_code}"
        data = resp.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        match = _BOXED_RE.search(content or "")
        if not match:
            return "mixture", "llm no boxed answer"
        ans = match.group(1).strip("\n").strip()
        if ans not in model_names:
            return "mixture", f"llm unknown model {ans!r}"
        return ans, "ok"
    except Exception as e:  # noqa: BLE001 — selection must never raise
        return "mixture", f"llm error: {type(e).__name__}: {e}"


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
    diverse = _is_diverse(live, ctx, _diversity_threshold())
    if not diverse:
        selected, note = "mixture", "not diverse"
    else:
        prompt = _build_prompt(ctx, [int(c.t) for c in candles[-len(ctx):]], live, cv, horizon)
        selected, note = _select_with_llm(prompt, ok_models)

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
        # A non-"ok" note (LLM parse-fail / endpoint error) is the diagnostic
        # that the selection degraded to mixture — record it. Only a clean
        # pick ("ok") or a pre-LLM "not diverse" skip is error-free.
        error=None if note in ("ok", "not diverse") else note,
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
