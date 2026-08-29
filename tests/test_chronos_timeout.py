"""chronos_signal.timeout_seconds enforcement (chore/audit-followups).

The config key was documented-but-dead; _fetch_signal now runs the forecast
on a dedicated daemon thread and abandons it (error signal, never cached)
once the deadline passes. These tests patch _compute_signal — the candle
fetch + forecast unit — so no model or network is involved.
"""

import time

import hermes_trader.agents.chronos_signal as cs


def _mk_signal(median_pct=1.0):
    return cs.ChronosSignal(
        coin="X", side="long", context_last=100.0,
        median=101.0, q_low=100.0, q_high=102.0,
        median_pct=median_pct, spread_pct=2.0,
        horizon=12, model_id="amazon/chronos-2", inference_ms=1.0,
    )


def _with_cfg(cfg):
    real = cs._get_chronos_config
    cs._get_chronos_config = lambda: cfg
    return real


def test_fetch_signal_times_out_and_never_caches():
    real_cfg = _with_cfg({
        "enabled": True, "cache_ttl_seconds": 300, "timeout_seconds": 0.2,
        "context_length": 10, "forecast_horizon": 12,
    })
    real_compute = cs._compute_signal
    real_cache_get, real_cache_set = cs._cache_get, cs._cache_set
    try:
        cs._cache_get = lambda coin, ttl: None  # force a miss
        cs._compute_signal = lambda coin, side, cfg: (
            time.sleep(2.0), _mk_signal())[1]
        stored = []
        cs._cache_set = lambda coin, sig, ttl: stored.append((coin, sig))

        start = time.time()
        sig = cs._fetch_signal("X", "long")
        elapsed = time.time() - start

        assert elapsed < 1.5, f"caller blocked {elapsed:.1f}s past 0.2s deadline"
        assert sig.error is not None and "timeout" in sig.error
        assert sig.median is None
        assert stored == [], "timed-out signal must never be cached"
    finally:
        cs._get_chronos_config = real_cfg
        cs._compute_signal = real_compute
        cs._cache_get, cs._cache_set = real_cache_get, real_cache_set


def test_fetch_signal_within_deadline_caches_normally():
    real_cfg = _with_cfg({
        "enabled": True, "cache_ttl_seconds": 300, "timeout_seconds": 5,
        "context_length": 10, "forecast_horizon": 12,
    })
    real_compute = cs._compute_signal
    real_cache_get, real_cache_set = cs._cache_get, cs._cache_set
    try:
        cs._cache_get = lambda coin, ttl: None
        cs._compute_signal = lambda coin, side, cfg: _mk_signal()
        stored = []
        cs._cache_set = lambda coin, sig, ttl: stored.append((coin, sig))

        sig = cs._fetch_signal("X", "long")
        assert sig.error is None and sig.median_pct == 1.0
        assert stored and stored[0][0] == "X"
    finally:
        cs._get_chronos_config = real_cfg
        cs._compute_signal = real_compute
        cs._cache_get, cs._cache_set = real_cache_get, real_cache_set


def test_compute_crash_surfaces_as_error_signal():
    """An exception inside the worker thread becomes an error ChronosSignal,
    not a raise into the caller (sync gate path must never see a traceback)."""
    real_cfg = _with_cfg({
        "enabled": True, "cache_ttl_seconds": 300, "timeout_seconds": 5,
        "context_length": 10, "forecast_horizon": 12,
    })
    real_compute = cs._compute_signal
    real_cache_get = cs._cache_get
    try:
        cs._cache_get = lambda coin, ttl: None

        def _boom(coin, side, cfg):
            raise RuntimeError("kaboom")

        cs._compute_signal = _boom
        sig = cs._fetch_signal("X", "long")
        assert sig.error is not None and "kaboom" in sig.error
    finally:
        cs._get_chronos_config = real_cfg
        cs._compute_signal = real_compute
        cs._cache_get = real_cache_get
