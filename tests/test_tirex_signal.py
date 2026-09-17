"""tirex_signal shadow module tests (no model, no network).

Mirrors tests/test_timesfm_signal.py: patch _compute_signal / _get_model so
no tirex checkpoint or HL fetch is involved. Also pins the executor wiring
(attach def + call sites on every Trade result path) with an AST check.
"""

import ast
import threading
import time
from types import SimpleNamespace

import numpy as np

import hermes_trader.agents.tirex_signal as tx


def _mk_signal(median_pct=1.0):
    return tx.TirexSignal(
        coin="X", side="long", context_last=100.0,
        median=101.0, q_low=100.0, q_high=102.0,
        median_pct=median_pct, spread_pct=2.0,
        horizon=12, model_id="NX-AI/TiRex", inference_ms=1.0,
    )


def _with_cfg(cfg):
    real = tx._get_tirex_config
    tx._get_tirex_config = lambda: cfg
    return real


def _closes(n=50, start=100.0):
    return [{"o": start, "h": start, "l": start, "c": start + i, "v": 1.0}
            for i in range(n)]


class _FakeTirex:
    """Stand-in for the tirex-ts model: p50 path rises 1% per step from the
    last close, p10/p90 at +/-2 fixed %. Returns the (quantile_forecast, _)
    tuple with shape (batch, horizon, n_quantiles) — TRANSPOSED layout, same
    as the real tirex-ts contract."""

    def forecast(self, context, prediction_length, resample_strategy=None):
        last = float(context[0][-1])
        med = last * (1.0 + 0.01 * np.arange(1, prediction_length + 1))
        q = np.zeros((1, prediction_length, 9))
        q[0, :, 4] = med
        q[0, :, 0] = med * 0.98
        q[0, :, 8] = med * 1.02
        for i in (1, 2, 3, 5, 6, 7):
            q[0, :, i] = med
        return q, None


# ── config gate ───────────────────────────────────────────────────────────────
def test_disabled_by_default_returns_error_signal():
    real = _with_cfg({})  # no tirex_signal block at all
    try:
        sig = tx._fetch_signal("X", "long")
        assert sig.error == "disabled"
        assert sig.median is None
    finally:
        tx._get_tirex_config = real


def test_config_absent_defaults():
    real = _with_cfg({})
    try:
        cfg = tx._get_tirex_config()
        assert tx.resolve_min_conf_ratio(cfg) == 0.25
        assert tx.resolve_min_conf_ratio({"min_conf_ratio": 0}) == 0.0
        assert tx.resolve_min_conf_ratio({"min_conf_ratio": -1}) == 0.0
        assert tx.resolve_min_conf_ratio({"min_conf_ratio": "junk"}) == 0.25
    finally:
        tx._get_tirex_config = real


# ── quantile level assumption ─────────────────────────────────────────────────
def test_quantile_levels_canonical_for_nine():
    assert tx._quantile_levels(9) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_quantile_levels_evenly_spaced_otherwise():
    lv = tx._quantile_levels(4)
    assert lv == [0.2, 0.4, 0.6, 0.8]
    # p10/p50/p90 resolve to the CLOSEST column without crashing
    assert tx._find_quantile_index(lv, 0.1) == 0
    assert tx._find_quantile_index(lv, 0.5) in (1, 2)
    assert tx._find_quantile_index(lv, 0.9) == 3


# ── forecast math ─────────────────────────────────────────────────────────────
def test_forecast_paths_and_scalars():
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gm = tx._get_model
    tx._get_model = lambda: _FakeTirex()
    try:
        sig = tx._forecast_from_candles("X", "long", _closes(50), tx._get_tirex_config())
        assert sig.error is None, sig.error
        assert sig.context_last == 149.0
        assert sig.median is not None and sig.median > sig.context_last
        # q10 < median < q90 at every step of the path
        assert len(sig.q10_path_pct) == 12 and len(sig.q90_path_pct) == 12
        assert all(a < b for a, b in zip(sig.q10_path_pct, sig.q90_path_pct))
        assert sig.spread_pct is not None and sig.spread_pct > 0
    finally:
        tx._get_tirex_config = real
        tx._get_model = real_gm


def test_forecast_overlong_horizon_truncates():
    """If the checkpoint ever pads past the requested horizon, the signal must
    carry exactly `horizon` steps."""
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gm = tx._get_model

    class _Padded(_FakeTirex):
        def forecast(self, context, prediction_length, resample_strategy=None):
            return super().forecast(context, 64)  # padded output

    tx._get_model = lambda: _Padded()
    try:
        sig = tx._forecast_from_candles("X", "long", _closes(50), tx._get_tirex_config())
        assert sig.error is None, sig.error
        assert len(sig.q10_path_pct) == 12
        assert len(sig.q90_path_pct) == 12
    finally:
        tx._get_tirex_config = real
        tx._get_model = real_gm


def test_model_error_surfaces_as_signal():
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gm = tx._get_model

    class _Boom:
        def forecast(self, **kwargs):
            raise RuntimeError("boom")

    tx._get_model = lambda: _Boom()
    try:
        sig = tx._forecast_from_candles("X", "long", _closes(50), tx._get_tirex_config())
        assert sig.error is not None and "boom" in sig.error
        assert sig.median is None
    finally:
        tx._get_tirex_config = real
        tx._get_model = real_gm


def test_no_candles_error():
    real = _with_cfg({"enabled": True})
    real_gm = tx._get_model
    tx._get_model = lambda: _FakeTirex()
    try:
        sig = tx._forecast_from_candles("X", "long", [], tx._get_tirex_config())
        assert sig.error == "no candles"
    finally:
        tx._get_tirex_config = real
        tx._get_model = real_gm


# ── caller-side timeout (never caches, never raises) ─────────────────────────
def test_fetch_signal_times_out_and_never_caches():
    real = _with_cfg({"enabled": True, "timeout_seconds": 0.2,
                      "cache_ttl_seconds": 300})
    real_compute = tx._compute_signal

    def _slow(coin, side, cfg):
        time.sleep(1.0)
        return _mk_signal()

    tx._compute_signal = _slow
    tx._cache.pop("TIMEOUT-TX", None)
    try:
        t0 = time.time()
        sig = tx._fetch_signal("TIMEOUT-TX", "long")
        elapsed = time.time() - t0
        assert sig.error and "timeout" in sig.error
        assert elapsed < 0.9  # returned at the deadline, not after the work
        # never cached: a follow-up with a fast compute must recompute
        tx._compute_signal = lambda coin, side, cfg: _mk_signal(median_pct=7.0)
        sig2 = tx._fetch_signal("TIMEOUT-TX", "long")
        assert sig2.median_pct == 7.0
    finally:
        tx._compute_signal = real_compute
        tx._get_tirex_config = real
        tx._cache.pop("TIMEOUT-TX", None)


def test_fetch_signal_within_deadline_caches_normally():
    real = _with_cfg({"enabled": True, "timeout_seconds": 5,
                      "cache_ttl_seconds": 300})
    real_compute = tx._compute_signal
    calls = {"n": 0}

    def _counting(coin, side, cfg):
        calls["n"] += 1
        return _mk_signal()

    tx._compute_signal = _counting
    tx._cache.pop("CACHE-TX", None)
    try:
        sig = tx._fetch_signal("CACHE-TX", "long")
        assert sig.error is None and calls["n"] == 1
        sig2 = tx._fetch_signal("CACHE-TX", "long")
        assert calls["n"] == 1  # second read served from cache
    finally:
        tx._compute_signal = real_compute
        tx._get_tirex_config = real
        tx._cache.pop("CACHE-TX", None)


def test_compute_crash_surfaces_as_error_signal():
    real = _with_cfg({"enabled": True, "timeout_seconds": 5})
    real_compute = tx._compute_signal

    def _boom(coin, side, cfg):
        raise RuntimeError("crashed")

    tx._compute_signal = _boom
    try:
        sig = tx._fetch_signal("X", "long")
        assert sig.error == "crashed"
    finally:
        tx._compute_signal = real_compute
        tx._get_tirex_config = real


# ── in-flight dedup (one compute per cold coin) ──────────────────────────────
def test_concurrent_callers_compute_once():
    """Loop fire + PASS attach can hit a cold coin within ms of each other.
    One owner computes and logs; the late caller waits and reads its cache —
    duplicate lines would skew any grep-counted accrual rate."""
    real = _with_cfg({"enabled": True, "timeout_seconds": 5,
                      "cache_ttl_seconds": 300})
    real_compute = tx._compute_signal
    calls = {"n": 0}

    def _slow(coin, side, cfg):
        calls["n"] += 1
        time.sleep(0.3)
        return _mk_signal()

    tx._compute_signal = _slow
    tx._cache.pop("DEDUP-TX", None)
    results = []
    try:
        threads = [threading.Thread(target=lambda: results.append(
                       tx._fetch_signal("DEDUP-TX", "long")))
                   for _ in range(3)]
        for t in threads:
            t.start()
        time.sleep(0.05)  # ensure the first thread claims ownership first
        for t in threads:
            t.join(timeout=5)
        assert calls["n"] == 1, f"expected 1 compute, got {calls['n']}"
        assert len(results) == 3 and all(r.error is None for r in results)
    finally:
        tx._compute_signal = real_compute
        tx._get_tirex_config = real
        tx._cache.pop("DEDUP-TX", None)


def test_owner_failure_does_not_strand_waiters():
    """If the owner's compute errors (nothing cached), a late caller must not
    hang — it falls through and computes itself."""
    real = _with_cfg({"enabled": True, "timeout_seconds": 5,
                      "cache_ttl_seconds": 300})
    real_compute = tx._compute_signal
    state = {"n": 0}

    def _flaky(coin, side, cfg):
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(0.2)
            raise RuntimeError("owner boom")
        return _mk_signal(median_pct=3.0)

    tx._compute_signal = _flaky
    tx._cache.pop("STRAND-TX", None)
    out = {}
    try:
        owner = threading.Thread(
            target=lambda: out.setdefault("owner", tx._fetch_signal("STRAND-TX", "long")))
        owner.start()
        time.sleep(0.05)  # late caller arrives while the owner is mid-boom
        late = tx._fetch_signal("STRAND-TX", "long")
        owner.join(timeout=5)
        assert late.error is None and late.median_pct == 3.0
        assert out["owner"].error == "owner boom"
    finally:
        tx._compute_signal = real_compute
        tx._get_tirex_config = real
        tx._cache.pop("STRAND-TX", None)

def test_peek_never_computes():
    real = _with_cfg({"enabled": True, "cache_ttl_seconds": 300})
    real_compute = tx._compute_signal

    def _boom(*a, **k):
        raise AssertionError("peek must never compute")

    tx._compute_signal = _boom
    try:
        assert tx.peek_tirex("ABSENT-TX") is None
    finally:
        tx._compute_signal = real_compute
        tx._get_tirex_config = real


# ── log line ──────────────────────────────────────────────────────────────────
def test_log_neutral_below_confidence_floor():
    sig = _mk_signal(median_pct=0.05)  # ratio 0.025 < 0.25 floor
    line = tx._format_signal_log(sig, debug=False)
    assert "NEUTRAL" in line and "MISMATCH" not in line


def test_log_align_and_mismatch():
    sig = _mk_signal(median_pct=1.0)
    assert "ALIGN" in tx._format_signal_log(sig, debug=False)
    sig2 = _mk_signal(median_pct=-1.0)
    assert "MISMATCH" in tx._format_signal_log(sig2, debug=False)
    assert "ALIGN" in tx._format_signal_log(
        tx.TirexSignal(**{**sig2.__dict__, "side": "short"}), debug=False)


def test_confidence_ratio_failsafe_zero():
    sig = _mk_signal()
    sig.spread_pct = None
    assert tx.confidence_ratio(sig) == 0.0
    sig2 = _mk_signal()
    sig2.median_pct = None
    assert tx.confidence_ratio(sig2) == 0.0


# ── singleton sharing (expert_pool owns the load) ─────────────────────────────
def test_load_borrows_expert_pool_singleton():
    """_load_model must go through expert_pool._ensure_loaded('tirex') — one
    loaded checkpoint shared with the MOE pool, never a private second copy."""
    from hermes_trader.agents import expert_pool
    real = expert_pool._ensure_loaded
    seen = {}

    def _fake(name):
        seen["name"] = name
        return object()

    expert_pool._ensure_loaded = _fake
    try:
        m = tx._load_model()
        assert m is not None
        assert seen["name"] == "tirex"
    finally:
        expert_pool._ensure_loaded = real


def test_load_none_from_pool_caches_init_error():
    """A None from expert_pool (its own failure cache) must poison our init so
    we don't re-attempt per call."""
    real_gm_target = tx._load_model
    cfg_real = _with_cfg({"enabled": True})
    tx._load_model = lambda: None
    saved_err = tx._model_init_error
    saved_model = tx._tirex_model
    try:
        tx._model_init_error = None
        tx._tirex_model = None
        assert tx._get_model() is None
        assert tx._model_init_error is not None  # error cached
        # second call short-circuits without touching _load_model again
        calls = {"n": 0}

        def _counting():
            calls["n"] += 1
            return None
        tx._load_model = _counting
        assert tx._get_model() is None
        assert calls["n"] == 0
    finally:
        tx._load_model = real_gm_target
        tx._get_tirex_config = cfg_real
        tx._model_init_error = saved_err
        tx._tirex_model = saved_model


# ── executor wiring (AST — runs on host, no container deps) ──────────────────
def test_executor_attach_wired_on_all_paths():
    import hermes_trader.agents.executor as ex_mod
    src = open(ex_mod.__file__).read()
    tree = ast.parse(src)
    defs = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_attach_tirex_to_result"]
    assert len(defs) == 1, "expected exactly one _attach_tirex_to_result def"
    calls = sum(1 for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_attach_tirex_to_result")
    # 4 trade-result return paths: runner-gate-blocked, blocked-by-gates,
    # shadow-mode, executed.
    assert calls >= 4, f"expected >=4 call sites, found {calls}"


def test_attach_disabled_fills_error_fields():
    """With the module disabled (default config), the attach must add the
    three fields with an error, never raise into the trade path."""
    from hermes_trader.agents.executor import _attach_tirex_to_result
    result = {}
    real = tx._get_tirex_config
    tx._get_tirex_config = lambda: {}
    try:
        _attach_tirex_to_result(result, "X", "long")
    finally:
        tx._get_tirex_config = real
    assert result["tirex_median_pct"] is None
    assert result["tirex_aligned"] is None
    assert result["tirex_error"] == "disabled"


def test_route_verdict_pass_carries_tirex_fields():
    """PASS verdicts log via route_verdict's own forecast block (not the
    executor attach path), so tirex fields must be added there too — and a
    disabled/failed tirex read must never blank the chronos fields."""
    from hermes_trader.agents import executor as ex
    import hermes_trader.agents.chronos_signal as cs
    real_cs = cs.get_chronos_signal_sync

    def _boom(coin, side):
        raise RuntimeError("chronos down")

    cs.get_chronos_signal_sync = _boom  # chronos into its except arm; hermetic
    try:
        routed = ex.route_verdict({"verdict": "PASS", "coin": "X", "confidence": 0.0})
    finally:
        cs.get_chronos_signal_sync = real_cs
    assert routed["action"] == "none"
    assert "tirex_median_pct" in routed
    assert "tirex_aligned_if_long" in routed
    assert "tirex_aligned_if_short" in routed
    # disabled default -> error field, no median
    assert routed["tirex_median_pct"] is None
    assert routed["tirex_error"] == "disabled"
    # chronos keys exist regardless (error shape) — tirex never blanks them
    assert "chronos_median_pct" in routed


def test_route_verdict_pass_renders_enabled_tirex():
    """Enabled + warm signal -> median and alignment flags render on PASS."""
    import hermes_trader.agents.chronos_signal as cs
    from hermes_trader.agents import executor as ex
    real_cs = cs.get_chronos_signal_sync

    def _fake_chronos(coin, side):
        return cs.ChronosSignal(
            coin=coin, side=side, context_last=100.0,
            median=101.0, q_low=100.0, q_high=102.0,
            median_pct=0.5, spread_pct=2.0,
            horizon=12, model_id="amazon/chronos-2", inference_ms=1.0)

    cs.get_chronos_signal_sync = _fake_chronos
    real = _with_cfg({"enabled": True, "cache_ttl_seconds": 300})
    real_compute = tx._compute_signal
    real_cache_get = tx._cache_get
    try:
        tx._cache_get = lambda coin, ttl: None
        tx._compute_signal = lambda coin, side, cfg: _mk_signal(median_pct=1.0)
        routed = ex.route_verdict({"verdict": "PASS", "coin": "ROUTE-TX",
                                   "confidence": 0.0})
    finally:
        tx._cache_get = real_cache_get
        tx._get_tirex_config = real
        tx._compute_signal = real_compute
        tx._cache.pop("ROUTE-TX", None)
    assert routed["tirex_median_pct"] == 1.0
    assert routed["tirex_aligned_if_long"] is True
    assert routed["tirex_aligned_if_short"] is False
