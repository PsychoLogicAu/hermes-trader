"""timestral_signal shadow module tests (no model, no network).

Mirrors tests/test_tirex_signal.py: patch _compute_signal / _get_forecaster
so no timestral checkpoint or HL fetch is involved. Also pins the executor
wiring (attach def + call sites on every Trade result path) with an AST
check.
"""

import ast
import threading
import time
from types import SimpleNamespace

import numpy as np

import hermes_trader.agents.timestral_signal as tms

# The checkpoint-fixed 21 knots from the model card
# (0.01, 0.05, 0.10, 0.15, ..., 0.90, 0.95, 0.99).
KNOTS = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
         0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)


def _mk_signal(median_pct=1.0):
    return tms.TimestralSignal(
        coin="X", side="long", context_last=100.0,
        median=101.0, q_low=100.0, q_high=102.0,
        median_pct=median_pct, spread_pct=2.0,
        horizon=12, model_id="mistralai/Timestral-209M", inference_ms=1.0,
    )


def _with_cfg(cfg):
    real = tms._get_timestral_config
    tms._get_timestral_config = lambda: cfg
    return real


def _closes(n=50, start=100.0):
    return [{"o": start, "h": start, "l": start, "c": start + i, "v": 1.0}
            for i in range(n)]


class _FakeTimestral:
    """Stand-in for TimestralForecaster. p50 path rises 1% per step from the
    last close, p10/p90 at +/-2 fixed %. Returns a Forecast-shaped object:
    .quantiles (1, 1, H, 21) — univariate 1-D context per the timestral
    contract — plus .quantile_levels (the 21 knots). Records the forecast
    kwargs so the single-pass (TAA off) decision is pinned."""

    def __init__(self):
        self.kwargs_seen = {}

    @property
    def quantile_levels(self):
        return KNOTS

    def forecast(self, context, prediction_length, **kwargs):
        self.kwargs_seen = dict(kwargs)
        last = float(context[-1])
        med = last * (1.0 + 0.01 * np.arange(1, prediction_length + 1))
        q = np.zeros((1, 1, prediction_length, len(KNOTS)))
        q[0, 0, :, KNOTS.index(0.50)] = med
        q[0, 0, :, KNOTS.index(0.10)] = med * 0.98
        q[0, 0, :, KNOTS.index(0.90)] = med * 1.02
        for i, kv in enumerate(KNOTS):
            if kv not in (0.10, 0.50, 0.90):
                q[0, 0, :, i] = med
        return SimpleNamespace(quantiles=q, quantile_levels=KNOTS)


# ── config gate ───────────────────────────────────────────────────────────────
def test_disabled_by_default_returns_error_signal():
    real = _with_cfg({})  # no timestral_signal block at all
    try:
        sig = tms._fetch_signal("X", "long")
        assert sig.error == "disabled"
        assert sig.median is None
    finally:
        tms._get_timestral_config = real


def test_config_absent_defaults():
    real = _with_cfg({})
    try:
        cfg = tms._get_timestral_config()
        assert tms.resolve_min_conf_ratio(cfg) == 0.25
        assert tms.resolve_min_conf_ratio({"min_conf_ratio": 0}) == 0.0
        assert tms.resolve_min_conf_ratio({"min_conf_ratio": -1}) == 0.0
        assert tms.resolve_min_conf_ratio({"min_conf_ratio": "junk"}) == 0.25
    finally:
        tms._get_timestral_config = real


# ── quantile resolution ───────────────────────────────────────────────────────
def test_knots_resolve_p10_p50_p90_exactly():
    assert tms._find_quantile_index(list(KNOTS), 0.1) == KNOTS.index(0.10)
    assert tms._find_quantile_index(list(KNOTS), 0.5) == KNOTS.index(0.50)
    assert tms._find_quantile_index(list(KNOTS), 0.9) == KNOTS.index(0.90)


def test_offgrid_levels_resolve_to_closest():
    lv = [0.2, 0.4, 0.6, 0.8]
    assert tms._find_quantile_index(lv, 0.1) == 0
    assert tms._find_quantile_index(lv, 0.5) in (1, 2)
    assert tms._find_quantile_index(lv, 0.9) == 3


# ── forecast math ─────────────────────────────────────────────────────────────
def test_forecast_paths_and_scalars():
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gf = tms._get_forecaster
    tms._get_forecaster = lambda: _FakeTimestral()
    try:
        sig = tms._forecast_from_candles("X", "long", _closes(50), tms._get_timestral_config())
        assert sig.error is None, sig.error
        assert sig.context_last == 149.0
        assert sig.median is not None and sig.median > sig.context_last
        # q10 < median < q90 at every step of the path
        assert len(sig.q10_path_pct) == 12 and len(sig.q90_path_pct) == 12
        assert all(a < b for a, b in zip(sig.q10_path_pct, sig.q90_path_pct))
        assert sig.spread_pct is not None and sig.spread_pct > 0
    finally:
        tms._get_timestral_config = real
        tms._get_forecaster = real_gf


def test_forecast_runs_single_pass_taa_off():
    """The shadow read must NOT pay the library's default clampsym TAA
    (symmetric averaging = 2x compute) — same decision as timesfm, so all
    four models are read single-pass and stay comparable."""
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    fake = _FakeTimestral()
    real_gf = tms._get_forecaster
    tms._get_forecaster = lambda: fake
    try:
        sig = tms._forecast_from_candles("X", "long", _closes(50), tms._get_timestral_config())
        assert sig.error is None, sig.error
        assert fake.kwargs_seen.get("symmetric_averaging") is False
        assert fake.kwargs_seen.get("clip_nonnegative") is False
    finally:
        tms._get_timestral_config = real
        tms._get_forecaster = real_gf


def test_forecast_levels_mismatch_axis_fails_clean():
    """Reported quantile_levels must line up with the quantile axis before
    any column is named p10/p50/p90."""
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gf = tms._get_forecaster

    class _BadLevels(_FakeTimestral):
        def forecast(self, context, prediction_length, **kwargs):
            fc = super().forecast(context, prediction_length, **kwargs)
            return SimpleNamespace(quantiles=fc.quantiles,
                                   quantile_levels=KNOTS[:10])

    tms._get_forecaster = lambda: _BadLevels()
    try:
        sig = tms._forecast_from_candles("X", "long", _closes(50), tms._get_timestral_config())
        assert sig.error is not None and "quantile_levels" in sig.error
        assert sig.median is None
    finally:
        tms._get_timestral_config = real
        tms._get_forecaster = real_gf


def test_forecast_overlong_horizon_truncates():
    """If the model ever pads past the requested horizon, the signal must
    carry exactly `horizon` steps."""
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gf = tms._get_forecaster

    class _Padded(_FakeTimestral):
        def forecast(self, context, prediction_length, **kwargs):
            return super().forecast(context, 64, **kwargs)  # padded output

    tms._get_forecaster = lambda: _Padded()
    try:
        sig = tms._forecast_from_candles("X", "long", _closes(50), tms._get_timestral_config())
        assert sig.error is None, sig.error
        assert len(sig.q10_path_pct) == 12
        assert len(sig.q90_path_pct) == 12
    finally:
        tms._get_timestral_config = real
        tms._get_forecaster = real_gf


def test_model_error_surfaces_as_signal():
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gf = tms._get_forecaster

    class _Boom:
        quantile_levels = KNOTS

        def forecast(self, context, **kwargs):
            raise RuntimeError("boom")

    tms._get_forecaster = lambda: _Boom()
    try:
        sig = tms._forecast_from_candles("X", "long", _closes(50), tms._get_timestral_config())
        assert sig.error is not None and "boom" in sig.error
        assert sig.median is None
    finally:
        tms._get_timestral_config = real
        tms._get_forecaster = real_gf


def test_no_candles_error():
    real = _with_cfg({"enabled": True})
    real_gf = tms._get_forecaster
    tms._get_forecaster = lambda: _FakeTimestral()
    try:
        sig = tms._forecast_from_candles("X", "long", [], tms._get_timestral_config())
        assert sig.error == "no candles"
    finally:
        tms._get_timestral_config = real
        tms._get_forecaster = real_gf


# ── caller-side timeout (never caches, never raises) ─────────────────────────
def test_fetch_signal_times_out_and_never_caches():
    real = _with_cfg({"enabled": True, "timeout_seconds": 0.2,
                      "cache_ttl_seconds": 300})
    real_compute = tms._compute_signal

    def _slow(coin, side, cfg):
        time.sleep(1.0)
        return _mk_signal()

    tms._compute_signal = _slow
    tms._cache.pop("TIMEOUT-TMS", None)
    try:
        t0 = time.time()
        sig = tms._fetch_signal("TIMEOUT-TMS", "long")
        elapsed = time.time() - t0
        assert sig.error and "timeout" in sig.error
        assert elapsed < 0.9  # returned at the deadline, not after the work
        # never cached: a follow-up with a fast compute must recompute
        tms._compute_signal = lambda coin, side, cfg: _mk_signal(median_pct=7.0)
        sig2 = tms._fetch_signal("TIMEOUT-TMS", "long")
        assert sig2.median_pct == 7.0
    finally:
        tms._compute_signal = real_compute
        tms._get_timestral_config = real
        tms._cache.pop("TIMEOUT-TMS", None)


def test_fetch_signal_within_deadline_caches_normally():
    real = _with_cfg({"enabled": True, "timeout_seconds": 5,
                      "cache_ttl_seconds": 300})
    real_compute = tms._compute_signal
    calls = {"n": 0}

    def _counting(coin, side, cfg):
        calls["n"] += 1
        return _mk_signal()

    tms._compute_signal = _counting
    tms._cache.pop("CACHE-TMS", None)
    try:
        sig = tms._fetch_signal("CACHE-TMS", "long")
        assert sig.error is None and calls["n"] == 1
        sig2 = tms._fetch_signal("CACHE-TMS", "long")
        assert calls["n"] == 1  # second read served from cache
    finally:
        tms._compute_signal = real_compute
        tms._get_timestral_config = real
        tms._cache.pop("CACHE-TMS", None)


def test_compute_crash_surfaces_as_error_signal():
    real = _with_cfg({"enabled": True, "timeout_seconds": 5})
    real_compute = tms._compute_signal

    def _boom(coin, side, cfg):
        raise RuntimeError("crashed")

    tms._compute_signal = _boom
    try:
        sig = tms._fetch_signal("X", "long")
        assert sig.error == "crashed"
    finally:
        tms._compute_signal = real_compute
        tms._get_timestral_config = real


# ── in-flight dedup (one compute per cold coin) ──────────────────────────────
def test_concurrent_callers_compute_once():
    """Loop fire + PASS attach can hit a cold coin within ms of each other.
    One owner computes and logs; the late caller waits and reads its cache —
    duplicate lines would skew any grep-counted accrual rate."""
    real = _with_cfg({"enabled": True, "timeout_seconds": 5,
                      "cache_ttl_seconds": 300})
    real_compute = tms._compute_signal
    calls = {"n": 0}

    def _slow(coin, side, cfg):
        calls["n"] += 1
        time.sleep(0.3)
        return _mk_signal()

    tms._compute_signal = _slow
    tms._cache.pop("DEDUP-TMS", None)
    results = []
    try:
        threads = [threading.Thread(target=lambda: results.append(
                       tms._fetch_signal("DEDUP-TMS", "long")))
                   for _ in range(3)]
        for t in threads:
            t.start()
        time.sleep(0.05)  # ensure the first thread claims ownership first
        for t in threads:
            t.join(timeout=5)
        assert calls["n"] == 1, f"expected 1 compute, got {calls['n']}"
        assert len(results) == 3 and all(r.error is None for r in results)
    finally:
        tms._compute_signal = real_compute
        tms._get_timestral_config = real
        tms._cache.pop("DEDUP-TMS", None)


def test_owner_failure_does_not_strand_waiters():
    """If the owner's compute errors (nothing cached), a late caller must not
    hang — it falls through and computes itself."""
    real = _with_cfg({"enabled": True, "timeout_seconds": 5,
                      "cache_ttl_seconds": 300})
    real_compute = tms._compute_signal
    state = {"n": 0}

    def _flaky(coin, side, cfg):
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(0.2)
            raise RuntimeError("owner boom")
        return _mk_signal(median_pct=3.0)

    tms._compute_signal = _flaky
    tms._cache.pop("STRAND-TMS", None)
    out = {}
    try:
        owner = threading.Thread(
            target=lambda: out.setdefault("owner", tms._fetch_signal("STRAND-TMS", "long")))
        owner.start()
        time.sleep(0.05)  # late caller arrives while the owner is mid-boom
        late = tms._fetch_signal("STRAND-TMS", "long")
        owner.join(timeout=5)
        assert late.error is None and late.median_pct == 3.0
        assert out["owner"].error == "owner boom"
    finally:
        tms._compute_signal = real_compute
        tms._get_timestral_config = real
        tms._cache.pop("STRAND-TMS", None)


def test_peek_never_computes():
    real = _with_cfg({"enabled": True, "cache_ttl_seconds": 300})
    real_compute = tms._compute_signal

    def _boom(*a, **k):
        raise AssertionError("peek must never compute")

    tms._compute_signal = _boom
    try:
        assert tms.peek_timestral("ABSENT-TMS") is None
    finally:
        tms._compute_signal = real_compute
        tms._get_timestral_config = real


# ── log line ──────────────────────────────────────────────────────────────────
def test_log_neutral_below_confidence_floor():
    sig = _mk_signal(median_pct=0.05)  # ratio 0.025 < 0.25 floor
    line = tms._format_signal_log(sig, debug=False)
    assert "NEUTRAL" in line and "MISMATCH" not in line


def test_log_align_and_mismatch():
    sig = _mk_signal(median_pct=1.0)
    assert "ALIGN" in tms._format_signal_log(sig, debug=False)
    sig2 = _mk_signal(median_pct=-1.0)
    assert "MISMATCH" in tms._format_signal_log(sig2, debug=False)
    assert "ALIGN" in tms._format_signal_log(
        tms.TimestralSignal(**{**sig2.__dict__, "side": "short"}), debug=False)


def test_confidence_ratio_failsafe_zero():
    sig = _mk_signal()
    sig.spread_pct = None
    assert tms.confidence_ratio(sig) == 0.0
    sig2 = _mk_signal()
    sig2.median_pct = None
    assert tms.confidence_ratio(sig2) == 0.0


# ── lazy import (torch/timestral stay off the module-import path) ────────────
def test_module_import_does_not_pull_timestral():
    """`timestral` (torch + safetensors + hub) must only be imported inside
    the loader — same posture as the timesfm/tirex modules, so a host or CI
    without the package still imports and tests this module."""
    import subprocess
    import sys
    code = (
        "import sys; import hermes_trader.agents.timestral_signal; "
        "print('timestral' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("False")


# ── executor wiring (AST — runs on host, no container deps) ──────────────────
def test_executor_attach_wired_on_all_paths():
    import hermes_trader.agents.executor as ex_mod
    src = open(ex_mod.__file__).read()
    tree = ast.parse(src)
    defs = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_attach_timestral_to_result"]
    assert len(defs) == 1, "expected exactly one _attach_timestral_to_result def"
    calls = sum(1 for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_attach_timestral_to_result")
    # 4 trade-result return paths: runner-gate-blocked, blocked-by-gates,
    # shadow-mode, executed.
    assert calls >= 4, f"expected >=4 call sites, found {calls}"


def test_attach_disabled_fills_error_fields():
    """With the module disabled (default config), the attach must add the
    three fields with an error, never raise into the trade path."""
    from hermes_trader.agents.executor import _attach_timestral_to_result
    result = {}
    real = tms._get_timestral_config
    tms._get_timestral_config = lambda: {}
    try:
        _attach_timestral_to_result(result, "X", "long")
    finally:
        tms._get_timestral_config = real
    assert result["timestral_median_pct"] is None
    assert result["timestral_aligned"] is None
    assert result["timestral_error"] == "disabled"


def test_route_verdict_pass_carries_timestral_fields():
    """PASS verdicts log via route_verdict's own forecast block (not the
    executor attach path), so timestral fields must be added there too — and
    a disabled/failed timestral read must never blank the chronos fields."""
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
    assert "timestral_median_pct" in routed
    assert "timestral_aligned_if_long" in routed
    assert "timestral_aligned_if_short" in routed
    # disabled default -> error field, no median
    assert routed["timestral_median_pct"] is None
    assert routed["timestral_error"] == "disabled"
    # chronos keys exist regardless (error shape) — timestral never blanks them
    assert "chronos_median_pct" in routed


def test_route_verdict_pass_renders_enabled_timestral():
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
    real_compute = tms._compute_signal
    real_cache_get = tms._cache_get
    try:
        tms._cache_get = lambda coin, ttl: None
        tms._compute_signal = lambda coin, side, cfg: _mk_signal(median_pct=1.0)
        routed = ex.route_verdict({"verdict": "PASS", "coin": "ROUTE-TMS",
                                   "confidence": 0.0})
    finally:
        tms._cache_get = real_cache_get
        tms._get_timestral_config = real
        tms._compute_signal = real_compute
        tms._cache.pop("ROUTE-TMS", None)
    assert routed["timestral_median_pct"] == 1.0
    assert routed["timestral_aligned_if_long"] is True
    assert routed["timestral_aligned_if_short"] is False
