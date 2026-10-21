"""tirex2_signal shadow module tests (no model, no network).

Mirrors tests/test_tirex_signal.py: patch _compute_signal / _get_model so
no TiRex-2 checkpoint or HL fetch is involved. Differences vs the tirex
sibling tests:
  * the fake speaks the tirex-2 API (TimeseriesType list in, per-series
    (Q, horizon) arrays out — TRANSPOSED vs tirex-ts) — cloned fakes must
    match the NEW module's call convention, not the sibling's;
  * the lib is not installed in host venvs, so _forecast_from_candles'
    lazy `from tirex2 import TimeseriesType` is stubbed via sys.modules
    (and a subprocess test PINS that merely importing our module never
    pulls tirex2 in);
  * no expert_pool borrow — the module OWNS its load (timestral
    direction), pinned by test_load_uses_tirex2_load_model.
Also pins the executor wiring (attach def + call sites on every Trade
result path) with an AST check.
"""

import ast
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

import hermes_trader.agents.tirex2_signal as tx2


def _mk_signal(median_pct=1.0):
    return tx2.Tirex2Signal(
        coin="X", side="long", context_last=100.0,
        median=101.0, q_low=100.0, q_high=102.0,
        median_pct=median_pct, spread_pct=2.0,
        horizon=12, model_id="NX-AI/TiRex-2", inference_ms=1.0,
    )


def _with_cfg(cfg):
    real = tx2._get_tirex2_config
    tx2._get_tirex2_config = lambda: cfg
    return real


def _closes(n=50, start=100.0):
    return [{"o": start, "h": start, "l": start, "c": start + i, "v": 1.0}
            for i in range(n)]


def _stub_tirex2():
    """Stand-in module for the uninstalled tirex-2 lib: just enough for
    _forecast_from_candles' lazy import. Restored by the caller."""
    prev = sys.modules.get("tirex2")
    mod = types.ModuleType("tirex2")

    class TimeseriesType:
        def __init__(self, target=None, past_covariates=None, future_covariates=None):
            self.target = target
            self.past_covariates = past_covariates
            self.future_covariates = future_covariates

    mod.TimeseriesType = TimeseriesType
    sys.modules["tirex2"] = mod
    return prev


class _FakeTirex2:
    """Stand-in for the tirex-2 model: p50 path rises 1% per step from the
    last close, p10/p90 at +/-2 fixed %. Returns a per-series (n_quantiles,
    horizon) array inside a list — the tirex-2 output_type="numpy" contract,
    TRANSPOSED vs tirex-ts' (batch, horizon, n_quantiles)."""

    def _quantile_levels(self):
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def forecast(self, series, prediction_length=None, output_type="numpy"):
        last = float(series[0].target[0, -1])
        med = last * (1.0 + 0.01 * np.arange(1, prediction_length + 1))
        q = np.zeros((9, prediction_length))
        q[4, :] = med
        q[0, :] = med * 0.98
        q[8, :] = med * 1.02
        for i in (1, 2, 3, 5, 6, 7):
            q[i, :] = med
        # real nesting: [series][variate] -> (Q, H); univariate = one variate.
        return [[q]]


# ── config gate ───────────────────────────────────────────────────────────────
def test_disabled_by_default_returns_error_signal():
    real = _with_cfg({})  # no tirex2_signal block at all
    try:
        sig = tx2._fetch_signal("X", "long")
    finally:
        tx2._get_tirex2_config = real
    assert sig.error == "disabled"
    assert sig.median_pct is None


def test_config_absent_defaults():
    """The real config reader swallows a read error to {} (module must stay
    importable/usable with no config file at all)."""
    real = tx2.read_agent_config
    try:
        tx2.read_agent_config = lambda: (_ for _ in ()).throw(RuntimeError("no cfg file"))
        assert tx2._get_tirex2_config() == {}
    finally:
        tx2.read_agent_config = real


# ── quantile level mapping ────────────────────────────────────────────────────
def test_quantile_levels_reported_by_model():
    class _M:
        def _quantile_levels(self):
            return [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
    lv = tx2._quantile_levels(_M(), 7)
    assert lv == [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95]


def test_quantile_levels_canonical_for_nine_without_accessor():
    class _M:
        def _quantile_levels(self):
            raise AttributeError("gone")
    lv = tx2._quantile_levels(_M(), 9)
    assert lv == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_quantile_levels_evenly_spaced_otherwise():
    class _M:
        def _quantile_levels(self):
            raise AttributeError("gone")
    lv = tx2._quantile_levels(_M(), 4)
    assert lv == [0.2, 0.4, 0.6, 0.8]


# ── forecast core (fake model, fake candles) ──────────────────────────────────
def test_forecast_paths_and_scalars():
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 12})
    real_gm = tx2._get_model
    prev = _stub_tirex2()
    tx2._get_model = lambda: _FakeTirex2()
    try:
        sig = tx2._forecast_from_candles("X", "long", _closes(50), tx2._get_tirex2_config())
        assert sig.error is None, sig.error
        assert sig.context_last == 149.0
        assert sig.median is not None and sig.median > sig.context_last
        assert len(sig.q10_path_pct) == 12 and len(sig.q90_path_pct) == 12
        assert all(a < b for a, b in zip(sig.q10_path_pct, sig.q90_path_pct))
        assert sig.spread_pct is not None and sig.spread_pct > 0
        assert sig.model_id == "NX-AI/TiRex-2"
    finally:
        tx2._get_tirex2_config = real
        tx2._get_model = real_gm
        if prev is None:
            sys.modules.pop("tirex2", None)
        else:
            sys.modules["tirex2"] = prev


def test_forecast_overlong_horizon_truncates():
    """A checkpoint that emits MORE than horizon steps per quantile row must
    be truncated to horizon, not crash the scalar math."""
    real = _with_cfg({"enabled": True, "context_length": 50, "forecast_horizon": 6})
    real_gm = tx2._get_model
    prev = _stub_tirex2()
    tx2._get_model = lambda: _FakeTirex2()
    try:
        sig = tx2._forecast_from_candles("X", "short", _closes(50), tx2._get_tirex2_config())
        assert sig.error is None, sig.error
        assert len(sig.q10_path_pct) == 6 and len(sig.q90_path_pct) == 6
    finally:
        tx2._get_tirex2_config = real
        tx2._get_model = real_gm
        if prev is None:
            sys.modules.pop("tirex2", None)
        else:
            sys.modules["tirex2"] = prev


def test_model_error_surfaces_as_signal():
    class _Boom:
        def forecast(self, *a, **k):
            raise RuntimeError("cuda OOM")
    real = _with_cfg({"enabled": True})
    real_gm = tx2._get_model
    prev = _stub_tirex2()
    tx2._get_model = lambda: _Boom()
    try:
        sig = tx2._forecast_from_candles("X", "long", _closes(50), tx2._get_tirex2_config())
        assert "cuda OOM" in sig.error
        assert sig.median_pct is None and sig.q10_path_pct is None
    finally:
        tx2._get_tirex2_config = real
        tx2._get_model = real_gm
        if prev is None:
            sys.modules.pop("tirex2", None)
        else:
            sys.modules["tirex2"] = prev


def test_no_candles_error():
    real = _with_cfg({"enabled": True})
    real_gm = tx2._get_model
    tx2._get_model = lambda: _FakeTirex2()
    try:
        sig = tx2._forecast_from_candles("X", "long", [], tx2._get_tirex2_config())
        assert sig.error == "no candles"
    finally:
        tx2._get_tirex2_config = real
        tx2._get_model = real_gm


# ── _fetch_signal: timeout + cache posture ───────────────────────────────────
def test_fetch_signal_times_out_and_never_caches():
    real = _with_cfg({"enabled": True, "timeout_seconds": 0.2, "cache_ttl_seconds": 300})
    real_compute = tx2._compute_signal
    try:
        def _slow(coin, side, cfg):
            time.sleep(1.0)
            return _mk_signal()
        tx2._compute_signal = _slow
        sig = tx2._fetch_signal("TIMEOUT-X", "long")
        assert sig.error and sig.error.startswith("timeout")
        # never cached: a second (fast) call must compute again, not read it
        sig2 = tx2._cache_get("TIMEOUT-X", 300)
        assert sig2 is None
    finally:
        tx2._compute_signal = real_compute
        tx2._get_tirex2_config = real
        tx2._cache.pop("TIMEOUT-X", None)


def test_fetch_signal_within_deadline_caches_normally():
    real = _with_cfg({"enabled": True, "timeout_seconds": 5, "cache_ttl_seconds": 300})
    real_compute = tx2._compute_signal
    try:
        tx2._compute_signal = lambda coin, side, cfg: _mk_signal()
        sig = tx2._fetch_signal("FRESH-X", "long")
        assert sig.error is None and sig.median_pct == 1.0
        cached = tx2._cache_get("FRESH-X", 300)
        assert cached is sig
    finally:
        tx2._compute_signal = real_compute
        tx2._get_tirex2_config = real
        tx2._cache.pop("FRESH-X", None)


def test_compute_crash_surfaces_as_error_signal():
    real = _with_cfg({"enabled": True, "timeout_seconds": 5})
    real_compute = tx2._compute_signal
    try:
        def _boom(coin, side, cfg):
            raise RuntimeError("kernel exploded")
        tx2._compute_signal = _boom
        sig = tx2._fetch_signal("CRASH-X", "long")
        assert "kernel exploded" in sig.error
    finally:
        tx2._compute_signal = real_compute
        tx2._get_tirex2_config = real
        tx2._cache.pop("CRASH-X", None)


# ── in-flight dedup ───────────────────────────────────────────────────────────
def test_concurrent_callers_compute_once():
    """Loop fire vs attach racing the same cold coin: exactly ONE compute,
    both callers see a signal (owner computes, late caller waits + reads)."""
    real = _with_cfg({"enabled": True, "timeout_seconds": 5, "cache_ttl_seconds": 300})
    real_compute = tx2._compute_signal
    computes = {"n": 0}
    try:
        def _counted(coin, side, cfg):
            computes["n"] += 1
            time.sleep(0.3)
            return _mk_signal()
        tx2._compute_signal = _counted
        sigs = []
        ts = [threading.Thread(target=lambda: sigs.append(tx2._fetch_signal("DEDUP-X", "long")))
              for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(5)
        assert computes["n"] == 1, f"expected 1 compute, got {computes['n']}"
        assert len(sigs) == 4
        assert all(s.error is None for s in sigs)
    finally:
        tx2._compute_signal = real_compute
        tx2._get_tirex2_config = real
        tx2._cache.pop("DEDUP-X", None)


def test_owner_failure_does_not_strand_waiters():
    """Owner raises -> its Event is set (waiters wake), a waiter may compute
    itself but must never hang past its deadline."""
    real = _with_cfg({"enabled": True, "timeout_seconds": 5, "cache_ttl_seconds": 300})
    real_compute = tx2._compute_signal
    try:
        def _boom(coin, side, cfg):
            time.sleep(0.1)
            raise RuntimeError("down")
        tx2._compute_signal = _boom
        out = {}

        def _late():
            out["sig"] = tx2._fetch_signal("STRAND-X", "long")

        t1 = threading.Thread(target=lambda: tx2._fetch_signal("STRAND-X", "long"))
        t2 = threading.Thread(target=_late)
        t1.start()
        time.sleep(0.02)  # ensure t1 owns the in-flight slot
        t2.start()
        t1.join(5)
        t2.join(5)
        assert not t1.is_alive() and not t2.is_alive()
        assert out["sig"].error  # error, not a hang
    finally:
        tx2._compute_signal = real_compute
        tx2._get_tirex2_config = real
        tx2._cache.pop("STRAND-X", None)


def test_peek_never_computes():
    real = _with_cfg({"enabled": True, "cache_ttl_seconds": 300})
    real_compute = tx2._compute_signal
    try:
        def _boom(coin, side, cfg):
            raise AssertionError("peek must not compute")
        tx2._compute_signal = _boom
        assert tx2.peek_tirex2("PEEK-X") is None
    finally:
        tx2._compute_signal = real_compute
        tx2._get_tirex2_config = real


# ── log formatting + confidence floor ─────────────────────────────────────────
def test_log_neutral_below_confidence_floor():
    sig = _mk_signal(median_pct=0.1)  # spread 2.0 -> ratio 0.05 < 0.25
    line = tx2._format_signal_log(sig, debug=False)
    assert line.startswith("[tirex2] X (long)")
    assert "NEUTRAL" in line


def test_log_align_and_mismatch():
    sig = _mk_signal(median_pct=1.0)  # ratio 0.5 >= 0.25
    assert "ALIGN" in tx2._format_signal_log(sig, debug=False)
    sig2 = _mk_signal(median_pct=1.0)
    sig2.side = "short"
    assert "MISMATCH" in tx2._format_signal_log(sig2, debug=False)


def test_confidence_ratio_failsafe_zero():
    sig = _mk_signal()
    sig.median_pct = None
    assert tx2.confidence_ratio(sig) == 0.0
    sig = _mk_signal()
    sig.spread_pct = 0.0
    assert tx2.confidence_ratio(sig) == 0.0


def test_resolve_min_conf_ratio_semantics():
    assert tx2.resolve_min_conf_ratio({}) == 0.25
    assert tx2.resolve_min_conf_ratio({"min_conf_ratio": 0}) == 0.0  # explicit 0 DISABLES
    assert tx2.resolve_min_conf_ratio({"min_conf_ratio": "garbage"}) == 0.25
    assert tx2.resolve_min_conf_ratio({"min_conf_ratio": -1}) == 0.0


# ── model load: OWNS its load via tirex2.load_model (no pool borrow) ─────────
def test_load_uses_tirex2_load_model():
    """_load_model must import the tirex-2 lib here (lazy) and call
    load_model(model_id, device) — standalone singleton, no expert_pool."""
    prev = sys.modules.get("tirex2")
    mod = types.ModuleType("tirex2")
    seen = {}

    def _fake_load(model_id, device="cpu", use_flex_attention=None):
        seen["args"] = (model_id, device, use_flex_attention)
        return object()

    mod.load_model = _fake_load
    sys.modules["tirex2"] = mod
    try:
        m = tx2._load_model("NX-AI/TiRex-2", "cpu")
        assert m is not None
        # eager read path: flex kernel OFF (prod image has no C++ compiler;
        # see _load_model docstring) — the kwarg must actually land.
        assert seen["args"] == ("NX-AI/TiRex-2", "cpu", False)
    finally:
        if prev is None:
            sys.modules.pop("tirex2", None)
        else:
            sys.modules["tirex2"] = prev


def test_lazy_import_keeps_module_import_clean():
    """Importing tirex2_signal must NOT import tirex2 (host venvs lack it —
    the module and its tests must import anywhere)."""
    root = Path(__file__).parent.parent
    code = (
        "import sys; import hermes_trader.agents.tirex2_signal; "
        "print('tirex2' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=root, timeout=60)
    assert out.stdout.strip().endswith("False"), out.stdout + out.stderr


def test_load_failure_caches_init_error():
    """A load exception poisons the init so we don't re-attempt per call."""
    real_gm_target = tx2._load_model
    cfg_real = _with_cfg({"enabled": True})
    tx2._load_model = lambda mid, dev: (_ for _ in ()).throw(RuntimeError("404 weights"))
    saved_err = tx2._model_init_error
    saved_model = tx2._tirex2_model
    try:
        tx2._model_init_error = None
        tx2._tirex2_model = None
        assert tx2._get_model() is None
        assert tx2._model_init_error is not None  # error cached
        calls = {"n": 0}

        def _counting(mid, dev):
            calls["n"] += 1
            return None
        tx2._load_model = _counting
        assert tx2._get_model() is None
        assert calls["n"] == 0
    finally:
        tx2._load_model = real_gm_target
        tx2._get_tirex2_config = cfg_real
        tx2._model_init_error = saved_err
        tx2._tirex2_model = saved_model


# ── executor wiring (AST — runs on host, no container deps) ──────────────────
def test_executor_attach_wired_on_all_paths():
    import hermes_trader.agents.executor as ex_mod
    src = open(ex_mod.__file__).read()
    tree = ast.parse(src)
    defs = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_attach_tirex2_to_result"]
    assert len(defs) == 1, "expected exactly one _attach_tirex2_to_result def"
    calls = sum(1 for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_attach_tirex2_to_result")
    # 5 trade-result return paths: runner-gate-blocked, dv-trap-blocked,
    # blocked-by-gates, shadow-mode, executed.
    assert calls >= 5, f"expected >=5 call sites, found {calls}"


def test_attach_disabled_fills_error_fields():
    """With the module disabled (default config), the attach must add the
    three fields with an error, never raise into the trade path."""
    from hermes_trader.agents.executor import _attach_tirex2_to_result
    result = {}
    real = tx2._get_tirex2_config
    tx2._get_tirex2_config = lambda: {}
    try:
        _attach_tirex2_to_result(result, "X", "long")
    finally:
        tx2._get_tirex2_config = real
    assert result["tirex2_median_pct"] is None
    assert result["tirex2_aligned"] is None
    assert result["tirex2_error"] == "disabled"


def test_route_verdict_pass_carries_tirex2_fields():
    """PASS verdicts log via route_verdict's own forecast block, so tirex2
    fields must be added there too — and a disabled/failed read must never
    blank the other models' fields."""
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
    assert "tirex2_median_pct" in routed
    assert "tirex2_aligned_if_long" in routed
    assert "tirex2_aligned_if_short" in routed
    # disabled default -> error field, no median
    assert routed["tirex2_median_pct"] is None
    assert routed["tirex2_error"] == "disabled"
    # chronos keys exist regardless (error shape) — tirex2 never blanks them
    assert "chronos_median_pct" in routed


def test_route_verdict_pass_renders_enabled_tirex2():
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
    real_compute = tx2._compute_signal
    real_cache_get = tx2._cache_get
    try:
        tx2._cache_get = lambda coin, ttl: None
        tx2._compute_signal = lambda coin, side, cfg: _mk_signal(median_pct=1.0)
        routed = ex.route_verdict({"verdict": "PASS", "coin": "ROUTE-TX2",
                                   "confidence": 0.0})
    finally:
        tx2._cache_get = real_cache_get
        tx2._get_tirex2_config = real
        tx2._compute_signal = real_compute
        tx2._cache.pop("ROUTE-TX2", None)
    assert routed["tirex2_median_pct"] == 1.0
    assert routed["tirex2_aligned_if_long"] is True
    assert routed["tirex2_aligned_if_short"] is False
