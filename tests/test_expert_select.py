"""expert_select (MoiraiAgent expert-selection shadow) tests — no model, no
network, host-runnable.

Mirrors tests/test_timesfm_signal.py: patch at the SOURCE modules (expert_pool
for the candidate math, httpx.Client for the LLM call, config for the gate)
so no torch checkpoint, no HL fetch, and no real lemonade endpoint is
involved. Also pins the executor wiring (attach def + call sites on every
Trade result path, PASS-branch fields) with AST + route_verdict checks.
"""

import ast
import time
from types import SimpleNamespace

import numpy as np
import pytest

import hermes_trader.agents.expert_select as es
import hermes_trader.agents.expert_pool as ep
import hermes_trader.agents.chronos_signal as cs
import hermes_trader.agents.timesfm_signal as ts_mod
from hermes_trader.agents.executor import (
    _attach_expert_select_to_result,
    route_verdict,
)


# ── helpers ──────────────────────────────────────────────────────────────────
_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
H = 12


def _cfg_block(**over):
    """A full enabled expert_select config block (2-candidate pool by default
    so the fakes never need tirex/moirai2 adapters)."""
    cfg = {
        "enabled": True,
        "pool": ["chronos", "timesfm"],
        "horizon": 12,
        "context_length": 50,
        "diversity_threshold": 0.2,
        "cache_ttl_seconds": 300,
        "cv_ttl_seconds": 3600,
        "timeout_seconds": 10,
        "llm": {"model": "fake-model", "use_primary_model": False,
                "timeout_seconds": 5},
    }
    cfg.update(over)
    return {"expert_select": cfg}


def _patch_cfg(monkeypatch, block):
    """Monkeypatch es._cfg (and bypass its identity cache)."""
    monkeypatch.setattr(es, "_cfg", lambda: block.get("expert_select", {}))


def _mk_candidate(model, closes, horizon, step=0.0, width=2.0,
                  quantiles=None, error=None):
    """Fake CandidateForecast: median rises `step` per bar from the last
    close; quantiles at (level-0.5)*width around the median."""
    last = float(closes[-1])
    med = [last + step * (i + 1) for i in range(horizon)]
    if quantiles is None:
        quantiles = {f"{lv:.1f}": [v + (lv - 0.5) * width for v in med]
                     for lv in _LEVELS}
    return ep.CandidateForecast(
        model=model, median=med if not error else None,
        q10=quantiles.get("0.1") if quantiles else None,
        q90=quantiles.get("0.9") if quantiles else None,
        quantiles=quantiles, horizon=horizon,
        inference_ms=1.0, error=error,
    )


def _fake_forecast(steps, errors=None):
    """expert_pool.forecast stand-in. steps: {model: per-bar step}; errors:
    {model: str} — those adapters return an error (pool degrade path)."""
    errors = errors or {}

    def _forecast(model, closes, horizon):
        if model in errors:
            return ep.CandidateForecast(model=model, horizon=horizon,
                                        error=errors[model])
        return _mk_candidate(model, closes, horizon,
                             step=steps.get(model, 0.0))
    return _forecast


def _fake_candles(n=78):
    """Rising closes (100 + 0.5*i) with ms timestamps, as Candle objects
    (expert_select._compute reads `c.t` — dicts have no attributes)."""
    from hermes_trader.models.types import Candle
    return [Candle(t=1_700_000_000_000 + i * 300_000, o=100.0, h=100.0,
                   l=100.0, c=100.0 + 0.5 * i, v=1.0) for i in range(n)]


class _FakeResp:
    def __init__(self, content, status_code=200):
        self.content = content
        self.status_code = status_code

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class _FakeClient:
    """httpx.Client stand-in; `post` returns _FAKE_RESP (set per test) or
    raises it when it is an Exception instance (the network-error path)."""
    _FAKE_RESP: object = _FakeResp("")

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        if isinstance(self._FAKE_RESP, Exception):
            raise self._FAKE_RESP
        return self._FAKE_RESP


@pytest.fixture(autouse=True)
def _clean_caches():
    es.clear_cache()
    yield
    es.clear_cache()


# ── config gate ──────────────────────────────────────────────────────────────
def test_disabled_by_default_returns_error_signal(monkeypatch):
    """No expert_select block at all (test conftest points the config at an
    empty tmp file) → _fetch degrades to a disabled error signal, never
    computes."""
    monkeypatch.setattr(es, "_cfg", lambda: {})
    real_compute = es._compute

    def _boom(*a, **k):
        raise AssertionError("disabled signal must never compute")

    monkeypatch.setattr(es, "_compute", _boom)
    try:
        sig = es._fetch("X", "long")
        assert sig.error == "disabled"
        assert sig.median_pct is None
    finally:
        monkeypatch.setattr(es, "_compute", real_compute)


def test_peek_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(es, "_cfg", lambda: {})
    assert es.peek_expert_select("X") is None


def test_peek_never_computes(monkeypatch):
    """The sync attach path must pay zero model/LLM cost: peek reads the
    cache only, even with the signal enabled and nothing cached."""
    monkeypatch.setattr(es, "_cfg", lambda: _cfg_block()["expert_select"])

    def _boom(*a, **k):
        raise AssertionError("peek must never compute")

    real = es._compute
    monkeypatch.setattr(es, "_compute", _boom)
    try:
        assert es.peek_expert_select("NEVER-CACHED-COIN") is None
    finally:
        monkeypatch.setattr(es, "_compute", real)


# ── pure math (normalization / MAE) ─────────────────────────────────────────
def test_mae_basic_nan_and_empty():
    assert es._mae([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(1.0 / 3)
    # all-NaN target → their _compute_mae returns 0 (mask empty), not inf
    assert es._mae([1.0, 2.0], [float("nan"), float("nan")]) == 0.0
    # NaN in the prediction propagates → inf (unrankable, sorts last)
    assert es._mae([float("nan"), 2.0], [1.0, 2.0]) == float("inf")


def test_norm_factor_all_nan_and_clipped_std():
    assert es._norm_factor([float("nan"), float("nan")]) == (0.0, 1.0)
    # constant series: std 0 clipped to the 1e-5 floor (their _get_norm_factor)
    mean, std = es._norm_factor([5.0, 5.0, 5.0])
    assert mean == 5.0 and std == 1e-5


# ── diversity gate (their check_preds_diversity) ────────────────────────────
def _live_from_steps(steps, closes, horizon=H):
    return {m: _mk_candidate(m, closes, horizon, step=s)
            for m, s in steps.items()}


def test_diversity_identical_candidates_not_diverse(monkeypatch):
    monkeypatch.setattr(es, "_pool", lambda: ["chronos", "timesfm"])
    closes = [100.0 + i for i in range(50)]
    live = _live_from_steps({"chronos": 0.3, "timesfm": 0.3}, closes)
    assert es._is_diverse(live, closes, 0.2) is False


def test_diversity_different_candidates_diverse(monkeypatch):
    monkeypatch.setattr(es, "_pool", lambda: ["chronos", "timesfm"])
    closes = [100.0 + i for i in range(50)]
    live = _live_from_steps({"chronos": 0.3, "timesfm": 1.5}, closes)
    assert es._is_diverse(live, closes, 0.2) is True


def test_diversity_single_candidate_skips_llm(monkeypatch):
    """With <2 live candidates there is nothing to select between →
    not-diverse → mixture, and the LLM call is skipped (a selection over one
    candidate would be meaningless)."""
    monkeypatch.setattr(es, "_pool", lambda: ["chronos"])
    closes = [100.0 + i for i in range(50)]
    live = _live_from_steps({"chronos": 0.3}, closes)
    assert es._is_diverse(live, closes, 0.2) is False
    # zero live candidates likewise
    assert es._is_diverse({}, closes, 0.2) is False


def test_diversity_threshold_zero_forces_diverse(monkeypatch):
    monkeypatch.setattr(es, "_pool", lambda: ["chronos", "timesfm"])
    closes = [100.0 + i for i in range(50)]
    live = _live_from_steps({"chronos": 0.3, "timesfm": 0.3}, closes)
    assert es._is_diverse(live, closes, 0.0) is True


# ── mixture pooling + re-centering (their get_best_pred math) ───────────────
def test_mixture_pooling_matches_upstream_math(monkeypatch):
    """Upstream `_get_mixture_pred` stacks EVERY quantile feature of every
    candidate — median + 0.1..0.9 (median is itself one of their
    FORECAST_FEATURES) — into a (n_cand × 10, h) matrix and takes ONE
    np.quantile([0.1..0.9], axis=0) across that pooled sample. Two candidates
    with medians last+1+i / last+2+i (both width-2 fans): the pooled q10/median/
    q90 at step i are last+0.58+i / last+1.5+i / last+2.42+i (pinned below).
    This is NOT a per-level quantile across candidates — that reading was an
    error; the concatenated stack is what their code does."""
    monkeypatch.setattr(es, "_pool", lambda: ["chronos", "timesfm"])
    last = 100.0
    live = {
        "chronos": _mk_candidate("chronos", [last], H, step=1.0),
        "timesfm": _mk_candidate("timesfm", [last + 1.0], H, step=1.0),
    }
    mix = es._mixture_and_recenter(live, "mixture", H)
    assert np.allclose(mix["0.5"], [last + 1.5 + i for i in range(H)])
    # pooled empirical quantiles of the 20-row fan (median rows count twice):
    assert np.allclose(mix["0.1"], [last + 0.58 + i for i in range(H)])
    assert np.allclose(mix["0.9"], [last + 2.42 + i for i in range(H)])


def test_recentering_winner_shifts_mixture_exactly(monkeypatch):
    """Winner ≠ mixture: the whole mixture distribution shifts so its median
    equals the winner's median (every level by the same per-step offset)."""
    monkeypatch.setattr(es, "_pool", lambda: ["chronos", "timesfm"])
    last = 100.0
    live = {
        "chronos": _mk_candidate("chronos", [last], H, step=1.0),
        "timesfm": _mk_candidate("timesfm", [last + 1.0], H, step=1.0),
    }
    mix = es._mixture_and_recenter(live, "mixture", H)
    sel = es._mixture_and_recenter(live, "timesfm", H)
    # timesfm's median is [last+2+i ...] — the re-centered mixture matches it
    assert np.allclose(sel["0.5"], [last + 2.0 + i for i in range(H)])
    # the re-centering shift is exactly the (winner − mixture) offset = 0.5
    for lv, path in sel.items():
        assert np.allclose(path, np.asarray(mix[lv]) + 0.5), f"level {lv}"


def test_mixture_missing_levels_fall_back_to_median(monkeypatch):
    """A candidate that exposes only its median path still participates:
    absent levels are pooled from the median (the adapter contract)."""
    monkeypatch.setattr(es, "_pool", lambda: ["chronos", "timesfm"])
    last = 100.0
    # shared median path (both step 1.0 from last): [last+1, last+2, ...]
    med_c = [last + 1.0 + i for i in range(H)]
    live = {
        # chronos exposes ONLY its median ("0.5") — its 0.1/0.9 fall back to it
        "chronos": _mk_candidate("chronos", [last], H, step=1.0,
                                 quantiles={"0.5": med_c}),
        "timesfm": _mk_candidate("timesfm", [last], H, step=1.0),
    }
    mix = es._mixture_and_recenter(live, "mixture", H)
    # both medians identical → mixture median == the shared median
    assert "0.5" in mix and np.allclose(mix["0.5"], med_c)
    # timesfm's real q10 still pooled at level 0.1 (chronos falls back to med)
    assert "0.1" in mix and len(mix["0.1"]) == H


def test_mixture_no_candidates_raises(monkeypatch):
    monkeypatch.setattr(es, "_pool", lambda: [])
    with pytest.raises(ValueError):
        es._mixture_and_recenter({}, "mixture", H)


# ── CV replay ───────────────────────────────────────────────────────────────
def test_cv_mae_ranking_orders_models(monkeypatch):
    """CV replay: forecast the last H bars from the context ending H bars
    back; MAE vs the actual tail ranks the models (chronos closer → first)."""
    monkeypatch.setattr(es, "_pool", lambda: ["chronos", "timesfm"])
    closes = [100.0 + 0.5 * i for i in range(78)]
    # chronos step 0.3 tracks the actual 0.5/bar rise better than 0.9
    monkeypatch.setattr(ep, "forecast", _fake_forecast(
        {"chronos": 0.3, "timesfm": 0.9}))
    cv = es._collect_cv(closes, H)
    assert cv and len(cv["ranking"]) == 2
    assert cv["ranking"][0][0] == "chronos"
    assert cv["ranking"][1][0] == "timesfm"
    assert cv["ranking"][0][1] < cv["ranking"][1][1]
    assert len(cv["tail"]) == H and len(cv["preds"]["chronos"]) == H


def test_cv_insufficient_history_returns_empty(monkeypatch):
    monkeypatch.setattr(es, "_pool", lambda: ["chronos"])
    closes = [100.0 + i for i in range(20)]  # < horizon*2 + 8
    monkeypatch.setattr(ep, "forecast",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("no history → no CV inference")))
    assert es._collect_cv(closes, H) == {}


def test_cv_all_failing_returns_empty(monkeypatch):
    """Every adapter erroring on the CV replay → no CV section (the prompt
    degrades to an empty cv_info; the signal still computes the mixture)."""
    monkeypatch.setattr(es, "_pool", lambda: ["chronos", "timesfm"])
    closes = [100.0 + 0.5 * i for i in range(78)]
    monkeypatch.setattr(ep, "forecast", _fake_forecast(
        {}, errors={"chronos": "boom", "timesfm": "boom"}))
    assert es._collect_cv(closes, H) == {}


# ── full _compute (fake pool + fake LLM) ────────────────────────────────────
def _patch_compute_inputs(monkeypatch, steps, candles=None, errors=None):
    monkeypatch.setattr(ep, "fetch_closes",
                        lambda coin, max_bars=520, interval="5m": candles
                        if candles is not None else _fake_candles())
    monkeypatch.setattr(ep, "closes_from_candles",
                        lambda cl: [float(c["c"]) for c in cl])
    monkeypatch.setattr(ep, "forecast",
                        _fake_forecast(steps, errors=errors))


def test_compute_end_to_end_diverse_llm_selects(monkeypatch):
    """Diverse candidates → prompt built → LLM picks timesfm → the re-centered
    mixture's median lands exactly on timesfm's median, and the CV ranking
    (chronos had the better CV MAE) is still recorded alongside."""
    _patch_cfg(monkeypatch, _cfg_block())
    monkeypatch.setenv("LLM_BASE_URL", "http://fake:13305/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    # chronos tracks the 0.5/bar rise (0.3) better than timesfm (0.9) on the
    # CV replay, but the LLM (faked) still picks timesfm — selection ≠ CV.
    _patch_compute_inputs(monkeypatch, {"chronos": 0.3, "timesfm": 0.9})
    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _FakeClient._FAKE_RESP = _FakeResp("The best model is \\boxed{timesfm}.")

    sig = es._compute("X", "long")
    assert sig.error is None
    assert sig.selected_model == "timesfm"
    assert sig.diverse is True
    assert sig.cv_ranking and sig.cv_ranking[0][0] == "chronos"
    # re-centered: final median == timesfm's own median (0.9/bar × 12 bars)
    last = 100.0 + 0.5 * 77  # 138.5 — last close of the 78-bar context
    expected = (0.9 * H / last) * 100
    assert sig.median_pct == pytest.approx(expected, rel=1e-9)
    assert sig.spread_pct is not None and sig.spread_pct > 0
    assert len(sig.q10_path_pct) == H and len(sig.q90_path_pct) == H
    assert all(a < b for a, b in zip(sig.q10_path_pct, sig.q90_path_pct))


def test_compute_not_diverse_skips_llm(monkeypatch):
    """Identical candidates → diversity below threshold → mixture, and the
    LLM call must not happen (saves the lemonade call; the kill-signal rate)."""
    _patch_cfg(monkeypatch, _cfg_block())
    _patch_compute_inputs(monkeypatch, {"chronos": 0.3, "timesfm": 0.3})
    import httpx

    def _no_llm(*a, **k):
        raise AssertionError("LLM must not be called below diversity")

    monkeypatch.setattr(httpx, "Client", _no_llm)

    sig = es._compute("X", "long")
    assert sig.error is None
    assert sig.selected_model == "mixture"
    assert sig.diverse is False


def test_compute_llm_parse_fail_degrades_to_mixture(monkeypatch):
    """Their code ASSERTS on an unboxed answer; we must degrade to the
    mixture with the error noted, never raise into the exec loop."""
    _patch_cfg(monkeypatch, _cfg_block())
    monkeypatch.setenv("LLM_BASE_URL", "http://fake:13305/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    _patch_compute_inputs(monkeypatch, {"chronos": 0.3, "timesfm": 0.9})
    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _FakeClient._FAKE_RESP = _FakeResp("I'm not sure which is best.")

    sig = es._compute("X", "long")
    assert sig.error and "no boxed" in sig.error
    assert sig.selected_model == "mixture"
    assert sig.median_pct is not None  # mixture still computed


def test_compute_llm_unknown_model_degrades_to_mixture(monkeypatch):
    _patch_cfg(monkeypatch, _cfg_block())
    monkeypatch.setenv("LLM_BASE_URL", "http://fake:13305/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    _patch_compute_inputs(monkeypatch, {"chronos": 0.3, "timesfm": 0.9})
    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _FakeClient._FAKE_RESP = _FakeResp("\\boxed{gpt5}")

    sig = es._compute("X", "long")
    assert sig.error and "unknown model" in sig.error
    assert sig.selected_model == "mixture"


def test_compute_llm_endpoint_error_degrades_to_mixture(monkeypatch):
    """HTTP 500 → mixture (single endpoint; no fallback chain)."""
    _patch_cfg(monkeypatch, _cfg_block())
    monkeypatch.setenv("LLM_BASE_URL", "http://fake:13305/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    _patch_compute_inputs(monkeypatch, {"chronos": 0.3, "timesfm": 0.9})
    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _FakeClient._FAKE_RESP = _FakeResp("server error", status_code=500)

    sig = es._compute("X", "long")
    assert sig.error and "http 500" in sig.error
    assert sig.selected_model == "mixture"


def test_compute_llm_network_exception_degrades_to_mixture(monkeypatch):
    _patch_cfg(monkeypatch, _cfg_block())
    monkeypatch.setenv("LLM_BASE_URL", "http://fake:13305/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    _patch_compute_inputs(monkeypatch, {"chronos": 0.3, "timesfm": 0.9})
    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClient)
    _FakeClient._FAKE_RESP = ConnectionError("lemonade saturated")

    sig = es._compute("X", "long")
    assert sig.error and "ConnectionError" in sig.error
    assert sig.selected_model == "mixture"
    assert sig.median_pct is not None


def test_compute_pool_degrades_to_remaining_candidates(monkeypatch):
    """One adapter erroring degrades the pool to the rest (the signal never
    dies with one candidate); the per-candidate diagnostic records it."""
    _patch_cfg(monkeypatch, _cfg_block())
    _patch_compute_inputs(monkeypatch, {"chronos": 0.3},
                          errors={"timesfm": "tirex down"})
    import httpx

    def _no_llm(*a, **k):
        raise AssertionError("1 live candidate → no LLM call")

    monkeypatch.setattr(httpx, "Client", _no_llm)

    sig = es._compute("X", "long")
    assert sig.error is None
    assert sig.selected_model == "mixture"
    assert sig.candidates["timesfm"] == "tirex down"
    assert sig.candidates["chronos"] == "ok"


def test_compute_all_candidates_failed_error(monkeypatch):
    _patch_cfg(monkeypatch, _cfg_block())
    _patch_compute_inputs(monkeypatch, {}, errors={
        "chronos": "boom-c", "timesfm": "boom-t"})
    sig = es._compute("X", "long")
    assert sig.error == "all candidates failed"
    assert sig.median_pct is None


def test_compute_insufficient_candles_error(monkeypatch):
    _patch_cfg(monkeypatch, _cfg_block())
    monkeypatch.setattr(ep, "fetch_closes",
                        lambda *a, **k: _fake_candles(15))
    monkeypatch.setattr(ep, "closes_from_candles",
                        lambda cl: [float(c["c"]) for c in cl])
    sig = es._compute("X", "long")
    assert sig.error and "insufficient candles" in sig.error


# ── prompt construction ─────────────────────────────────────────────────────
def test_prompt_carries_their_three_blocks_and_instruction(monkeypatch):
    """The selection prompt must stay in their exact format: three JSON
    blocks (history_info / pred_info / cv_info), z-normalized to 3 decimals,
    the CV MAE ranking string, and the verbatim \\boxed instruction."""
    monkeypatch.setattr(es, "_pool", lambda: ["chronos", "timesfm"])
    closes = [100.0 + 0.5 * i for i in range(50)]
    ts_list = [1_700_000_000_000 + i * 300_000 for i in range(50)]
    live = _live_from_steps({"chronos": 0.3, "timesfm": 0.9}, closes)
    cv = {"preds": {m: [100.0 + 0.5 * i for i in range(12)]
                    for m in ("chronos", "timesfm")},
          "maes": {"chronos": 1.0, "timesfm": 2.0},
          "ranking": [("chronos", 1.0), ("timesfm", 2.0)],
          "tail": [140.0 + 0.5 * i for i in range(12)]}
    p = es._build_prompt(closes, ts_list, live, cv, H)
    for key in ("history_window", "history_values", "future_window",
                "model_names", "candidate_preds", "crossval_window",
                "crossval_ground_truth", "crossval_preds",
                "crossval_error_ranking"):
        assert f'"{key}"' in p, f"missing {key} block key"
    assert "chronos < timesfm" in p  # ranking string
    assert "\\boxed{" in p  # the parse target
    # z-normalized to 3 decimals
    import re as _re
    vals = _re.findall(r"-?\d+\.\d{3}", p)
    assert len(vals) > 50


def test_prompt_without_cv_still_builds(monkeypatch):
    """A failed CV replay (cv={}) degrades the cv_info section, not the
    whole prompt."""
    monkeypatch.setattr(es, "_pool", lambda: ["chronos", "timesfm"])
    closes = [100.0 + i for i in range(50)]
    ts_list = [1_700_000_000_000 + i * 300_000 for i in range(50)]
    live = _live_from_steps({"chronos": 0.3, "timesfm": 0.9}, closes)
    p = es._build_prompt(closes, ts_list, live, {}, H)
    assert '"crossval_error_ranking": ""' in p
    assert "\\boxed{" in p


# ── LLM param resolution (D2: primary by default, explicit wins) ────────────
def test_resolve_llm_explicit_config_wins(monkeypatch):
    _patch_cfg(monkeypatch, _cfg_block())
    monkeypatch.setenv("LLM_BASE_URL", "http://primary:13305/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "primary-key")
    monkeypatch.setenv("LLM_MODEL", "primary-model")
    base, key, model = es._resolve_llm_params()
    assert base == "http://primary:13305/api/v1"
    assert key == "primary-key"
    assert model == "fake-model"  # explicit llm.model beats the primary


def test_resolve_llm_use_primary_model(monkeypatch):
    """use_primary_model (default) with no explicit model resolves the
    primary model — the selector tracks whatever the primary is."""
    cfg = _cfg_block()
    cfg["expert_select"]["llm"] = {}  # no explicit model
    _patch_cfg(monkeypatch, cfg)
    monkeypatch.setenv("LLM_BASE_URL", "http://primary:13305/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "primary-key")
    monkeypatch.setenv("LLM_MODEL", "qwen-primary")
    base, key, model = es._resolve_llm_params()
    assert model == "qwen-primary"
    assert base == "http://primary:13305/api/v1"


def test_select_llm_no_endpoint_degrades(monkeypatch):
    """No base_url/api_key → mixture, never a connection attempt."""
    _patch_cfg(monkeypatch, _cfg_block())
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    sel, note = es._select_with_llm("prompt", ["chronos"])
    assert sel == "mixture" and "endpoint" in note


# ── fetch / timeout / caching (mirrors the chronos/timesfm tests) ───────────
def _mk_sig(coin: str = "X", median_pct: float | None = None,
            error: str | None = None, selected: str | None = "mixture",
            side: str = "long"):
    return es.ExpertSelectSignal(
        coin=coin, side=side, context_last=100.0, horizon=H,
        selected_model=selected, diverse=False, median_pct=median_pct,
        spread_pct=2.0, error=error,
    )


def test_fetch_timeout_never_caches(monkeypatch):
    """A compute that blows the caller-side deadline returns a timeout error
    and is NEVER written to the cache (a slow selection must not pin a stale
    value)."""
    _patch_cfg(monkeypatch, _cfg_block(timeout_seconds=0.2))

    def _slow(*a, **k):
        time.sleep(1.5)
        return _mk_sig()

    stored = []
    monkeypatch.setattr(es, "_compute", _slow)
    real_set = es._cache_set
    monkeypatch.setattr(es, "_cache_set",
                        lambda coin, sig: stored.append((coin, sig)))
    try:
        start = time.time()
        sig = es._fetch("X", "long")
        elapsed = time.time() - start
        assert elapsed < 1.2, f"caller blocked {elapsed:.1f}s past 0.2s"
        assert sig.error and "timeout" in sig.error
        assert sig.median_pct is None
        assert stored == [], "timed-out signal must never be cached"
    finally:
        monkeypatch.setattr(es, "_cache_set", real_set)


def test_fetch_success_caches_and_hits(monkeypatch):
    _patch_cfg(monkeypatch, _cfg_block())
    monkeypatch.setattr(es, "_compute", lambda *a, **k: _mk_sig(median_pct=0.5))
    sig1 = es._fetch("X", "long")
    assert sig1.error is None and sig1.median_pct == 0.5
    assert es.peek_expert_select("X") is sig1  # cache hit, same object
    # a second fetch within TTL must not recompute
    monkeypatch.setattr(es, "_compute", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("cache hit must not recompute")))
    assert es._fetch("X", "short").median_pct == 0.5


def test_compute_crash_surfaces_as_error_signal(monkeypatch):
    """An exception inside the worker thread becomes an error signal, not a
    raise into the caller (the attach path must never see a traceback)."""
    _patch_cfg(monkeypatch, _cfg_block())

    def _boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(es, "_compute", _boom)
    sig = es._fetch("X", "long")
    assert sig.error and "kaboom" in sig.error
    assert sig.median_pct is None
    assert es.peek_expert_select("X") is None  # error not cached


# ── async entry point ───────────────────────────────────────────────────────
def test_async_disabled_is_noop(monkeypatch):
    """get_expert_select_async with the signal disabled must not compute."""
    monkeypatch.setattr(es, "_cfg", lambda: {})

    def _boom(*a, **k):
        raise AssertionError("disabled async must not compute")

    monkeypatch.setattr(es, "_fetch", _boom)
    es.get_expert_select_async("X", "long")  # must not raise
    time.sleep(0.05)  # let the daemon worker run its enabled-check


def test_async_enabled_computes_and_logs(monkeypatch, caplog):
    _patch_cfg(monkeypatch, _cfg_block())
    # Patch _compute (not _fetch): logging lives INSIDE _fetch now, so the
    # real fetch must run to prove the async path logs one line per compute.
    monkeypatch.setattr(es, "_compute",
                        lambda coin, side: _mk_sig(coin=coin, side=side))
    import logging
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.expert_select"):
        es.get_expert_select_async("BTC", "long")
        time.sleep(0.05)
    lines = [r.message for r in caplog.records if "[expert] BTC (long)" in r.message]
    assert len(lines) == 1


def test_cache_hit_is_silent(monkeypatch, caplog):
    """Spec: cache hits log nothing — the wrapper firing again inside the TTL
    must not re-print the cached line (double counts polluted the mixture-
    rate kill-signal)."""
    _patch_cfg(monkeypatch, _cfg_block())
    calls = []

    def _compute_once(coin, side):
        calls.append(coin)
        return _mk_sig(coin=coin, side=side)

    monkeypatch.setattr(es, "_compute", _compute_once)
    import logging
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.expert_select"):
        es._fetch("BTC", "long")   # fresh compute → logs
        es._fetch("BTC", "short")  # cache hit → silent, no recompute
    assert len(calls) == 1
    assert len([r for r in caplog.records if "[expert] BTC" in r.message]) == 1


# ── pool config ─────────────────────────────────────────────────────────────
def test_pool_config_filters_unknown_names(monkeypatch):
    monkeypatch.setattr(es, "_cfg", lambda: {
        "pool": ["bogus", "timesfm", "moirai2"]})
    assert es._pool() == ["timesfm", "moirai2"]
    # non-list / empty → the full MODEL_NAMES order (stable, greppable)
    monkeypatch.setattr(es, "_cfg", lambda: {"pool": "junk"})
    assert es._pool() == list(ep.MODEL_NAMES)
    monkeypatch.setattr(es, "_cfg", lambda: {"pool": ["bogus"]})
    assert es._pool() == list(ep.MODEL_NAMES)


def test_pool_single_entry_mixture_only(monkeypatch):
    """A 1-entry pool config → the diversity gate can't fire (needs ≥2
    candidates) → mixture-only path, no LLM call, signal still live."""
    _patch_cfg(monkeypatch, _cfg_block(pool=["chronos"]))
    _patch_compute_inputs(monkeypatch, {"chronos": 0.3})
    import httpx

    def _no_llm(*a, **k):
        raise AssertionError("single-candidate pool → no LLM call")

    monkeypatch.setattr(httpx, "Client", _no_llm)
    sig = es._compute("X", "long")
    assert sig.error is None
    assert sig.selected_model == "mixture"
    assert sig.median_pct is not None


# ── log formatting ──────────────────────────────────────────────────────────
def test_log_line_shape():
    sig = _mk_sig(median_pct=0.78, selected="timesfm")
    sig.diverse = True
    sig.cv_ranking = [("chronos", 1.3), ("timesfm", 2.6)]
    line = es._format_log(sig)
    assert line.startswith("[expert] X (long)")
    assert "sel=timesfm" in line and "diverse=yes" in line
    assert "chronos<timesfm" in line and "expert_median=+0.78%" in line


def test_log_line_error_shape():
    sig = _mk_sig(median_pct=None, error="all candidates failed")
    line = es._format_log(sig)
    assert "ERROR" in line and "all candidates failed" in line


# ── executor wiring (AST — host-side, no container deps) ───────────────────
def test_executor_attach_wired_on_all_paths():
    """1 def + ≥4 call sites (runner-gate-blocked, gate-blocked, shadow-mode,
    executed) — same pin as the chronos/timesfm attaches so a new return
    path can't drop the field silently."""
    import hermes_trader.agents.executor as ex_mod
    src = open(ex_mod.__file__).read()
    tree = ast.parse(src)
    defs = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "_attach_expert_select_to_result"]
    assert len(defs) == 1, "expected exactly one attach def"
    calls = sum(1 for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_attach_expert_select_to_result")
    assert calls >= 4, f"expected >=4 call sites, found {calls}"
    # PASS branch: the route_verdict no-action line carries the expert fields
    assert 'expert_median_pct' in src and 'expert_aligned_if_long' in src


def test_attach_disabled_fills_error_fields(monkeypatch):
    """Disabled (default) attach: all four fields present with the error
    shape — never raises into the trade path."""
    monkeypatch.setattr(es, "_cfg", lambda: {})
    result = {}
    _attach_expert_select_to_result(result, "X", "long")
    assert result["expert_median_pct"] is None
    assert result["expert_aligned"] is None
    assert result["expert_selected_model"] is None
    assert result["expert_error"] == "no cached selection"


def test_attach_cached_hit_fills_fields(monkeypatch):
    monkeypatch.setattr(es, "_cfg", lambda: _cfg_block()["expert_select"])
    es._cache["X"] = {"signal": _mk_sig(median_pct=1.0, selected="timesfm"),
                      "ts": time.time()}
    result = {}
    _attach_expert_select_to_result(result, "X", "long")
    assert result["expert_median_pct"] == 1.0
    assert result["expert_aligned"] is True
    assert result["expert_selected_model"] == "timesfm"
    assert result["expert_error"] is None


def test_attach_cache_hit_side_flips_alignment(monkeypatch):
    """The alignment flag is side-dependent: a cache hit with a DIFFERENT
    verdict side must re-derive it from the stored median, not echo the
    previous candidate's side (the cache-hit pitfall)."""
    monkeypatch.setattr(es, "_cfg", lambda: _cfg_block()["expert_select"])
    es._cache["X"] = {"signal": _mk_sig(median_pct=1.0), "ts": time.time()}
    r_long, r_short = {}, {}
    _attach_expert_select_to_result(r_long, "X", "long")
    _attach_expert_select_to_result(r_short, "X", "short")
    assert r_long["expert_aligned"] is True
    assert r_short["expert_aligned"] is False
    # same cached median on both (direction-agnostic signal)
    assert r_long["expert_median_pct"] == r_short["expert_median_pct"]


def test_attach_error_signal_shape(monkeypatch):
    """A cached error signal (all candidates failed) still populates the
    error field with the selected_model absent (nothing was selected)."""
    monkeypatch.setattr(es, "_cfg", lambda: _cfg_block()["expert_select"])
    es._cache["X"] = {"signal": _mk_sig(median_pct=None, selected=None,
                                        error="all candidates failed"),
                      "ts": time.time()}
    result = {}
    _attach_expert_select_to_result(result, "X", "long")
    assert result["expert_median_pct"] is None
    assert result["expert_selected_model"] is None
    assert result["expert_error"] == "all candidates failed"


def test_attach_never_raises(monkeypatch):
    """Any exception inside the attach (e.g. a broken peek) degrades to the
    error field shape — the trade path must never see a traceback."""
    monkeypatch.setattr(es, "peek_expert_select",
                        lambda coin: (_ for _ in ()).throw(
                            RuntimeError("peek exploded")))
    result = {}
    _attach_expert_select_to_result(result, "X", "long")
    assert result["expert_median_pct"] is None
    assert result["expert_error"] and "peek exploded" in result["expert_error"]


# ── PASS verdict (route_verdict) ────────────────────────────────────────────
def test_route_verdict_pass_error_shape_keeps_chronos_fields(monkeypatch):
    """A disabled/expert failure must never blank the chronos/timesfm PASS
    fields (separate try/except) — and the expert error shape renders."""
    real_cs = cs.get_chronos_signal_sync
    real_ts = ts_mod.get_timesfm_signal_sync
    monkeypatch.setattr(cs, "get_chronos_signal_sync",
                        lambda c, s: (_ for _ in ()).throw(
                            RuntimeError("chronos down")))
    monkeypatch.setattr(ts_mod, "get_timesfm_signal_sync",
                        lambda c, s: (_ for _ in ()).throw(
                            RuntimeError("timesfm down")))
    monkeypatch.setattr(es, "_cfg", lambda: {})  # disabled → peek None
    try:
        routed = route_verdict({"verdict": "PASS", "coin": "X",
                                "confidence": 0.0})
    finally:
        monkeypatch.setattr(cs, "get_chronos_signal_sync", real_cs)
        monkeypatch.setattr(ts_mod, "get_timesfm_signal_sync", real_ts)
    assert routed["action"] == "none"
    # expert error shape present
    assert routed["expert_median_pct"] is None
    assert routed["expert_aligned_if_long"] is False
    assert routed["expert_aligned_if_short"] is False
    assert routed["expert_selected_model"] is None
    # chronos/timesfm keys intact (their own except arms)
    assert "chronos_median_pct" in routed and routed["chronos_median_pct"] is None
    assert routed["timesfm_median_pct"] is None


def test_route_verdict_pass_renders_enabled_expert(monkeypatch):
    """Enabled + warm cache → the re-centered-mixture median + both-side
    alignment flags render next to the chronos/timesfm PASS fields."""
    real_cs = cs.get_chronos_signal_sync
    monkeypatch.setattr(cs, "get_chronos_signal_sync",
                        lambda c, s: (_ for _ in ()).throw(
                            RuntimeError("chronos down")))
    monkeypatch.setattr(es, "_cfg", lambda: _cfg_block()["expert_select"])
    es._cache["EXP-PASS"] = {"signal": _mk_sig(median_pct=0.5,
                                               selected="mixture"),
                             "ts": time.time()}
    try:
        routed = route_verdict({"verdict": "PASS", "coin": "EXP-PASS",
                                "confidence": 0.0})
    finally:
        monkeypatch.setattr(cs, "get_chronos_signal_sync", real_cs)
    assert routed["expert_median_pct"] == 0.5
    assert routed["expert_aligned_if_long"] is True
    assert routed["expert_aligned_if_short"] is False
    assert routed["expert_selected_model"] == "mixture"
