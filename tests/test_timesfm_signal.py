"""timesfm_signal shadow module tests (no model, no network).

Mirrors tests/test_chronos_timeout.py: patch _compute_signal / _get_forecaster
so no torch checkpoint or HL fetch is involved. Also pins the executor wiring
(attach def + call sites on every Trade result path) with an AST check.
"""

import ast
import time
from types import SimpleNamespace

import numpy as np

import hermes_trader.agents.timesfm_signal as ts


def _mk_signal(median_pct=1.0):
    return ts.TimesfmSignal(
        coin="X", side="long", context_last=100.0,
        median=101.0, q_low=100.0, q_high=102.0,
        median_pct=median_pct, spread_pct=2.0,
        horizon=12, model_id="google/timesfm-3.0-pytorch", inference_ms=1.0,
    )


def _with_cfg(cfg):
    real = ts._get_timesfm_config
    ts._get_timesfm_config = lambda: cfg
    return real


def _closes(n=50, start=100.0):
    return [{"o": start, "h": start, "l": start, "c": start + i, "v": 1.0}
            for i in range(n)]


class _FakeForecaster:
    """Stand-in for TimesFM3Forecaster: p50 path rises 1% per step from the
    last close, p10/p90 at +/-2 fixed %."""

    def __init__(self):
        self.config = SimpleNamespace(
            quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def predict(self, context, horizon, **kwargs):
        last = float(context[-1])
        med = last * (1.0 + 0.01 * np.arange(1, horizon + 1))
        q = np.zeros((horizon, 9))
        q[:, 4] = med
        q[:, 0] = med * 0.98
        q[:, 8] = med * 1.02
        for i in (1, 2, 3, 5, 6, 7):
            q[:, i] = med
        return SimpleNamespace(forecast=med, quantiles=q)


# ── config gate ───────────────────────────────────────────────────────────────
def test_disabled_by_default_returns_error_signal():
    real = _with_cfg({})  # no timesfm_signal block at all
    try:
        sig = ts._fetch_signal("X", "long")
        assert sig.error == "disabled"
        assert sig.median is None
    finally:
        ts._get_timesfm_config = real


def test_config_absent_defaults():
    real = _with_cfg({})
    try:
        cfg = ts._get_timesfm_config()
        assert ts.resolve_min_conf_ratio(cfg) == 0.25
        assert ts.resolve_min_conf_ratio({"min_conf_ratio": 0}) == 0.0
        assert ts.resolve_min_conf_ratio({"min_conf_ratio": -1}) == 0.0
        assert ts.resolve_min_conf_ratio({"min_conf_ratio": "junk"}) == 0.25
    finally:
        ts._get_timesfm_config = real


# ── forecast math ─────────────────────────────────────────────────────────────
def test_forecast_paths_and_scalars():
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gf = ts._get_forecaster
    ts._get_forecaster = lambda: _FakeForecaster()
    try:
        sig = ts._forecast_from_candles("X", "long", _closes(50), ts._get_timesfm_config())
        assert sig.error is None, sig.error
        assert sig.context_last == 149.0
        assert sig.median is not None and sig.median > sig.context_last
        # q10 < median < q90 at every step of the path
        assert len(sig.q10_path_pct) == 12 and len(sig.q90_path_pct) == 12
        assert all(a < b for a, b in zip(sig.q10_path_pct, sig.q90_path_pct))
        assert sig.spread_pct is not None and sig.spread_pct > 0
    finally:
        ts._get_timesfm_config = real
        ts._get_forecaster = real_gf


def test_forecast_short_horizon_truncates():
    """The forecaster always pads the horizon to a 64-patch multiple; the
    signal must carry exactly `horizon` steps."""
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gf = ts._get_forecaster

    class _Padded(_FakeForecaster):
        def predict(self, context, horizon, **kwargs):
            out = super().predict(context, 64, **kwargs)  # library pads to 64
            return out

    ts._get_forecaster = lambda: _Padded()
    try:
        sig = ts._forecast_from_candles("X", "long", _closes(50), ts._get_timesfm_config())
        assert sig.error is None, sig.error
        assert len(sig.q10_path_pct) == 12
        assert len(sig.q90_path_pct) == 12
    finally:
        ts._get_timesfm_config = real
        ts._get_forecaster = real_gf


def test_forecaster_error_surfaces_as_signal():
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gf = ts._get_forecaster

    class _Boom(_FakeForecaster):
        def predict(self, context, horizon, **kwargs):
            raise RuntimeError("kaboom-infer")

    ts._get_forecaster = lambda: _Boom()
    try:
        sig = ts._forecast_from_candles("X", "long", _closes(50), ts._get_timesfm_config())
        assert sig.error and "kaboom-infer" in sig.error
        assert sig.median is None
    finally:
        ts._get_timesfm_config = real
        ts._get_forecaster = real_gf


def test_no_candles_error():
    real = _with_cfg({"enabled": True})
    real_gf = ts._get_forecaster
    ts._get_forecaster = lambda: _FakeForecaster()
    try:
        sig = ts._forecast_from_candles("X", "long", [], ts._get_timesfm_config())
        assert sig.error == "no candles"
    finally:
        ts._get_timesfm_config = real
        ts._get_forecaster = real_gf


# ── fetch/timeout/caching (mirrors chronos tests) ─────────────────────────────
def test_fetch_signal_times_out_and_never_caches():
    real_cfg = _with_cfg({
        "enabled": True, "cache_ttl_seconds": 300, "timeout_seconds": 0.2,
        "context_length": 10, "forecast_horizon": 12,
    })
    real_compute = ts._compute_signal
    real_cache_get, real_cache_set = ts._cache_get, ts._cache_set
    try:
        ts._cache_get = lambda coin, ttl: None  # force a miss
        ts._compute_signal = lambda coin, side, cfg: (
            time.sleep(2.0), _mk_signal())[1]
        stored = []
        ts._cache_set = lambda coin, sig, ttl: stored.append((coin, sig))

        start = time.time()
        sig = ts._fetch_signal("X", "long")
        elapsed = time.time() - start

        assert elapsed < 1.5, f"caller blocked {elapsed:.1f}s past 0.2s deadline"
        assert sig.error is not None and "timeout" in sig.error
        assert sig.median is None
        assert stored == [], "timed-out signal must never be cached"
    finally:
        ts._get_timesfm_config = real_cfg
        ts._compute_signal = real_compute
        ts._cache_get, ts._cache_set = real_cache_get, real_cache_set


def test_fetch_signal_within_deadline_caches_normally():
    real_cfg = _with_cfg({
        "enabled": True, "cache_ttl_seconds": 300, "timeout_seconds": 5,
        "context_length": 10, "forecast_horizon": 12,
    })
    real_compute = ts._compute_signal
    real_cache_get, real_cache_set = ts._cache_get, ts._cache_set
    try:
        ts._cache_get = lambda coin, ttl: None
        ts._compute_signal = lambda coin, side, cfg: _mk_signal()
        stored = []
        ts._cache_set = lambda coin, sig, ttl: stored.append((coin, sig))

        sig = ts._fetch_signal("X", "long")
        assert sig.error is None and sig.median_pct == 1.0
        assert stored and stored[0][0] == "X"
    finally:
        ts._get_timesfm_config = real_cfg
        ts._compute_signal = real_compute
        ts._cache_get, ts._cache_set = real_cache_get, real_cache_set


def test_compute_crash_surfaces_as_error_signal():
    """An exception inside the worker thread becomes an error TimesfmSignal,
    not a raise into the caller (the attach path must never see a traceback)."""
    real_cfg = _with_cfg({
        "enabled": True, "cache_ttl_seconds": 300, "timeout_seconds": 5,
        "context_length": 10, "forecast_horizon": 12,
    })
    real_compute = ts._compute_signal
    real_cache_get = ts._cache_get
    try:
        ts._cache_get = lambda coin, ttl: None

        def _boom(coin, side, cfg):
            raise RuntimeError("kaboom")

        ts._compute_signal = _boom
        sig = ts._fetch_signal("X", "long")
        assert sig.error is not None and "kaboom" in sig.error
    finally:
        ts._get_timesfm_config = real_cfg
        ts._compute_signal = real_compute
        ts._cache_get = real_cache_get


def test_peek_never_computes():
    real_cfg = _with_cfg({"enabled": True, "cache_ttl_seconds": 300})
    real_compute = ts._compute_signal
    try:
        ts._compute_signal = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("peek must never compute"))
        assert ts.peek_timesfm("NEVER-CACHED-COIN") is None
    finally:
        ts._get_timesfm_config = real_cfg
        ts._compute_signal = real_compute


# ── log formatting / confidence floor ─────────────────────────────────────────
def test_log_neutral_below_confidence_floor():
    sig = _mk_signal(median_pct=0.05)  # ratio 0.05/2.0 = 0.025 < 0.25
    line = ts._format_signal_log(sig, debug=False)
    assert "NEUTRAL" in line and "MISMATCH" not in line


def test_log_align_and_mismatch():
    long_up = _mk_signal(median_pct=1.0)
    assert "ALIGN" in ts._format_signal_log(long_up, debug=False)
    short_up = _mk_signal(median_pct=1.0)
    short_up.side = "short"
    assert "MISMATCH" in ts._format_signal_log(short_up, debug=False)


def test_confidence_ratio_failsafe_zero():
    sig = _mk_signal()
    sig.spread_pct = None
    assert ts.confidence_ratio(sig) == 0.0
    sig2 = _mk_signal()
    sig2.median_pct = None
    assert ts.confidence_ratio(sig2) == 0.0


# ── executor wiring (AST — runs on host, no container deps) ──────────────────
def test_executor_attach_wired_on_all_paths():
    import hermes_trader.agents.executor as ex_mod
    src = open(ex_mod.__file__).read()
    tree = ast.parse(src)
    defs = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_attach_timesfm_to_result"]
    assert len(defs) == 1, "expected exactly one _attach_timesfm_to_result def"
    calls = sum(1 for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_attach_timesfm_to_result")
    # 4 trade-result return paths: runner-gate-blocked, blocked-by-gates,
    # shadow-mode, executed.
    assert calls >= 4, f"expected >=4 call sites, found {calls}"


def test_attach_disabled_fills_error_fields():
    """With the module disabled (default config), the attach must add the
    three fields with an error, never raise into the trade path."""
    from hermes_trader.agents.executor import _attach_timesfm_to_result
    result = {}
    real = ts._get_timesfm_config
    ts._get_timesfm_config = lambda: {}
    try:
        _attach_timesfm_to_result(result, "X", "long")
    finally:
        ts._get_timesfm_config = real
    assert result["timesfm_median_pct"] is None
    assert result["timesfm_aligned"] is None
    assert result["timesfm_error"] == "disabled"


def test_route_verdict_pass_carries_timesfm_fields():
    """PASS verdicts log via route_verdict's own forecast block (not the
    executor attach path), so timesfm fields must be added there too — and a
    disabled/failed timesfm read must never blank the chronos fields."""
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
    assert "timesfm_median_pct" in routed
    assert "timesfm_aligned_if_long" in routed
    assert "timesfm_aligned_if_short" in routed
    # disabled default -> error field, no median
    assert routed["timesfm_median_pct"] is None
    assert routed["timesfm_error"] == "disabled"
    # chronos keys exist regardless (error shape) — timesfm never blanks them
    assert "chronos_median_pct" in routed


def test_route_verdict_pass_renders_enabled_timesfm():
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
    real_compute = ts._compute_signal
    try:
        ts._cache_get = lambda coin, ttl: None
        ts._compute_signal = lambda coin, side, cfg: _mk_signal(median_pct=1.0)
        routed = ex.route_verdict({"verdict": "PASS", "coin": "ROUTE-TFM",
                                   "confidence": 0.0})
    finally:
        ts._get_timesfm_config = real
        ts._compute_signal = real_compute
        cs.get_chronos_signal_sync = real_cs
        ts._cache.pop("ROUTE-TFM", None)
    assert routed["timesfm_median_pct"] == 1.0
    assert routed["timesfm_aligned_if_long"] is True
    assert routed["timesfm_aligned_if_short"] is False
    assert routed["chronos_median_pct"] == 0.5  # chronos block intact
