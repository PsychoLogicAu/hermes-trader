"""P10: TimesFM mirror gates — the timesfm-alone per-forecaster counterfactual
legs (chronos-alone is already live; the AND leg is forecast_agreement_veto).

Two new gates, both SHADOW-ONLY BY CONSTRUCTION:

  * ``timesfm_mismatch_gate``    — mirror of ``chronos_mismatch_gate`` on the
    TimesFM **median** (``ctx.timesfm_median_pct``).
  * ``timesfm_tail_trigger_gate`` — mirror of ``chronos_tail_trigger_gate`` on
    the TimesFM **adverse q-path** (min q10 / max q90 over the first
    ``window_steps``).

Unlike the chronos pair there is NO ``shadow_mode`` config key and NO code
path that returns ``pass: False`` — each gate is a pure GateContext function
(no log calls in-gate) that ALWAYS returns ``pass: True``; when its blocking
condition would hold it instead carries ``shadow_would_block`` plus the join
variables, and the EXECUTOR logs the loud accrual line. ``enabled`` (default
True) only controls whether the always-passing gate runs/accrues.

These tests pin:
  (a) shape-fire   -> pass True + shadow_would_block True + the join vars;
  (b) no-shape     -> pass True, no marker;
  (c) missing/None -> pass True, no marker, NO raise;
  (d) escape bar   -> adverse shape but conf >= 0.90 / composite >= 60 -> no
                      marker;
  (e) STRUCTURAL GUARANTEE: across the full input grid, the return dict NEVER
      contains ``pass: False`` (shadow-only is an invariant, not a convention);
  (f) ``eval_all_gates`` returns BOTH keys;
  (g) the two executor shadow-log lines fire via caplog when
      ``shadow_would_block`` is set (driven end-to-end through
      ``maybe_execute`` with a stubbed TimesFM warm-cache read).

Hermetic: GateContext is constructed directly; the only I/O is the stubbed
``get_timesfm_signal_sync`` in (g). No live LLM / model / network.
"""
from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    eval_all_gates,
    timesfm_mismatch_gate,
    timesfm_tail_trigger_gate,
)

# Code defaults (the shipped config — the .agent-config.json keys are
# deliberately ABSENT; an empty/{} cfg must reproduce exactly these).
MM_CFG = {
    "enabled": True,
    "min_abs_median_pct": 0.5,
    "min_conf": 0.90,
    "min_composite": 60.0,
}
TT_CFG = {
    "enabled": True,
    "window_steps": 12,
    "min_adv_path_pct": 2.0,
    "min_conf": 0.90,
    "min_composite": 60.0,
}

# A deep adverse q10 tail for a LONG (min over the first 12 steps is -3.5%,
# beyond the 2.0 code default). 12 steps so the default window (12) is met.
LONG_Q10_DEEP = [-0.8, -1.4, -2.1, -2.6, -3.0, -3.3, -3.5, -3.4,
                 -3.1, -2.8, -2.5, -2.2]
# A shallow q10 for a LONG (min -1.2%, inside the 2.0 threshold) — no shape.
LONG_Q10_SHALLOW = [-0.2, -0.5, -0.9, -1.2, -1.1, -0.8,
                    -0.5, -0.3, -0.1, 0.1, 0.2, 0.3]
# A deep adverse q90 tail for a SHORT (max over the first 12 steps is +3.4%).
SHORT_Q90_DEEP = [0.6, 1.3, 1.9, 2.3, 2.6, 2.8, 2.9, 3.0,
                  3.2, 3.4, 3.1, 2.8]
# A shallow q90 for a SHORT (max +1.2%, inside threshold) — no shape.
SHORT_Q90_SHALLOW = [0.2, 0.5, 0.9, 1.2, 1.1, 0.8,
                     0.5, 0.3, 0.1, -0.1, -0.2, -0.3]


def _ctx(
    side: str = "long",
    conf: float = 0.70,
    composite: float = 30.0,
    tmed=None,
    tq10=None,
    tq90=None,
    coin: str = "TON",
) -> GateContext:
    return GateContext(
        confidence=conf,
        current_positions=[],
        trade_notional_usd=50.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin=coin,
        trade_side=side,
        has_binary_news_risk=False,
        equity=1000.0,
        total_open_notional=50.0,
        composite_score=composite,
        timesfm_median_pct=tmed,
        timesfm_q10_path_pct=tq10,
        timesfm_q90_path_pct=tq90,
    )


def _cfg(base: dict, **over) -> dict:
    base = dict(base)
    base.update(over)
    return base


# ===========================================================================
# timesfm_mismatch_gate — (a) shape-fire
# ===========================================================================


def test_mismatch_long_adverse_median_fires():
    r = timesfm_mismatch_gate(_ctx("long", 0.70, 30.0, tmed=-1.5), _cfg(MM_CFG))
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert r["median_pct"] == -1.5
    assert r["confidence"] == 0.70
    assert r["composite_score"] == 30.0
    assert "timesfm_mismatch" in r["reason"]
    assert "long" in r["reason"] and "-1.50" in r["reason"]


def test_mismatch_short_adverse_median_fires():
    r = timesfm_mismatch_gate(_ctx("short", 0.80, 40.0, tmed=2.1), _cfg(MM_CFG))
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert r["median_pct"] == 2.1
    assert "short" in r["reason"]


def test_mismatch_fires_on_code_defaults_only():
    """An empty/{} cfg reproduces the shipped defaults (deadband 0.5)."""
    r = timesfm_mismatch_gate(_ctx("long", 0.70, 30.0, tmed=-0.9), {})
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert r["median_pct"] == -0.9


def test_mismatch_disabled_never_fires():
    r = timesfm_mismatch_gate(_ctx("long", 0.70, 30.0, tmed=-3.0),
                              _cfg(MM_CFG, enabled=False))
    assert r == {"pass": True}
    assert "shadow_would_block" not in r


# ===========================================================================
# timesfm_mismatch_gate — (b) no-shape
# ===========================================================================


def test_mismatch_aligned_median_no_marker():
    """Long with a POSITIVE median (the model agrees) -> no opinion."""
    r = timesfm_mismatch_gate(_ctx("long", 0.70, 30.0, tmed=1.5), _cfg(MM_CFG))
    assert r == {"pass": True}


def test_mismatch_within_deadband_no_marker():
    """|median| < 0.5 is directionally neutral -> no opinion."""
    r = timesfm_mismatch_gate(_ctx("long", 0.70, 30.0, tmed=-0.3), _cfg(MM_CFG))
    assert r == {"pass": True}


# ===========================================================================
# timesfm_mismatch_gate — (c) missing/None signal
# ===========================================================================


def test_mismatch_none_median_no_marker_no_raise():
    r = timesfm_mismatch_gate(_ctx("long", 0.70, 30.0, tmed=None), _cfg(MM_CFG))
    assert r == {"pass": True}


# ===========================================================================
# timesfm_mismatch_gate — (d) escape bar
# ===========================================================================


def test_mismatch_adverse_but_high_conf_no_marker():
    r = timesfm_mismatch_gate(_ctx("long", 0.90, 30.0, tmed=-1.5), _cfg(MM_CFG))
    assert r == {"pass": True}


def test_mismatch_adverse_but_high_composite_no_marker():
    r = timesfm_mismatch_gate(_ctx("long", 0.70, 60.0, tmed=-1.5), _cfg(MM_CFG))
    assert r == {"pass": True}


# ===========================================================================
# timesfm_tail_trigger_gate — (a) shape-fire
# ===========================================================================


def test_tail_long_deep_q10_fires():
    r = timesfm_tail_trigger_gate(
        _ctx("long", 0.70, 30.0, tq10=LONG_Q10_DEEP), _cfg(TT_CFG))
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    # min of the first 12 steps is -3.5
    assert r["tail_pct"] == pytest.approx(-3.5)
    assert r["window_steps"] == 12
    assert "timesfm_tail_trigger" in r["reason"]
    assert "long entry" in r["reason"] and "min" in r["reason"]


def test_tail_short_deep_q90_fires():
    r = timesfm_tail_trigger_gate(
        _ctx("short", 0.70, 30.0, tq90=SHORT_Q90_DEEP), _cfg(TT_CFG))
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert r["tail_pct"] == pytest.approx(3.4)
    assert "short entry" in r["reason"] and "max" in r["reason"]


def test_tail_fires_on_code_defaults_window12_x2():
    """An empty cfg uses window_steps 12 / min_adv_path_pct 2.0 (the sweep
    timesfm window — NOT the chronos 6 / 3.0)."""
    r = timesfm_tail_trigger_gate(
        _ctx("long", 0.70, 30.0, tq10=LONG_Q10_DEEP), {})
    assert r["pass"] is True
    assert r.get("shadow_would_block") is True
    assert r["window_steps"] == 12


def test_tail_disabled_never_fires():
    r = timesfm_tail_trigger_gate(
        _ctx("long", 0.70, 30.0, tq10=LONG_Q10_DEEP), _cfg(TT_CFG, enabled=False))
    assert r == {"pass": True}
    assert "shadow_would_block" not in r


# ===========================================================================
# timesfm_tail_trigger_gate — (b) no-shape
# ===========================================================================


def test_tail_long_shallow_q10_no_marker():
    r = timesfm_tail_trigger_gate(
        _ctx("long", 0.70, 30.0, tq10=LONG_Q10_SHALLOW), _cfg(TT_CFG))
    assert r == {"pass": True}


def test_tail_long_ignores_q90_path():
    """A LONG keys off q10 (its downside), not q90 — a climbing q90 is the
    long's friend, not its risk. q10 shallow -> no shape."""
    r = timesfm_tail_trigger_gate(
        _ctx("long", 0.70, 30.0, tq10=LONG_Q10_SHALLOW, tq90=SHORT_Q90_DEEP),
        _cfg(TT_CFG))
    assert r == {"pass": True}


def test_tail_short_ignores_q10_path():
    """A SHORT keys off q90 (its upside risk), not q10 — a deep q10 dip is the
    short's friend. q90 shallow -> no shape."""
    r = timesfm_tail_trigger_gate(
        _ctx("short", 0.70, 30.0, tq10=LONG_Q10_DEEP, tq90=SHORT_Q90_SHALLOW),
        _cfg(TT_CFG))
    assert r == {"pass": True}


# ===========================================================================
# timesfm_tail_trigger_gate — (c) missing/None / short path
# ===========================================================================


def test_tail_missing_path_no_marker_no_raise():
    r = timesfm_tail_trigger_gate(_ctx("long", 0.70, 30.0), _cfg(TT_CFG))
    assert r == {"pass": True}


def test_tail_short_path_no_marker_no_raise():
    """A path shorter than the 12-step window cannot be evaluated."""
    r = timesfm_tail_trigger_gate(
        _ctx("long", 0.70, 30.0, tq10=LONG_Q10_DEEP[:5]), _cfg(TT_CFG))
    assert r == {"pass": True}


def test_tail_dip_after_window_ignored():
    """A deep dip that only appears AFTER the 12-step window must not arm."""
    shallow12 = [0.2, 0.1, -0.3, -0.5, -0.4, -0.6,
                 -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
    deep_later = shallow12 + [-4.0, -5.0, -5.5]
    r = timesfm_tail_trigger_gate(
        _ctx("long", 0.70, 30.0, tq10=deep_later), _cfg(TT_CFG))
    assert r == {"pass": True}


# ===========================================================================
# timesfm_tail_trigger_gate — (d) escape bar
# ===========================================================================


def test_tail_adverse_but_high_conf_no_marker():
    r = timesfm_tail_trigger_gate(
        _ctx("long", 0.90, 30.0, tq10=LONG_Q10_DEEP), _cfg(TT_CFG))
    assert r == {"pass": True}


def test_tail_adverse_but_high_composite_no_marker():
    r = timesfm_tail_trigger_gate(
        _ctx("long", 0.70, 60.0, tq10=LONG_Q10_DEEP), _cfg(TT_CFG))
    assert r == {"pass": True}


# ===========================================================================
# (e) THE STRUCTURAL GUARANTEE: pass is NEVER False, anywhere on the grid
# ===========================================================================


def test_never_returns_pass_false_across_full_grid():
    """Shadow-only is an INVARIANT: iterate both gates over a grid of sides,
    signal values, conviction extremes, disabled, and None signals, and assert
    ``result['pass'] is True`` in EVERY case. A single pass: False would fail
    the suite — pinning that the pass: False branch is never written."""
    sides = ["long", "short"]
    medians = [None, -5.0, -2.0, -0.5, -0.49, 0.0, 0.5, 0.51, 2.0, 5.0]
    confs = [0.0, 0.5, 0.89, 0.90, 0.95, 1.0]
    composites = [0.0, 59.0, 60.0, 80.0]
    # tail paths: missing, short (below window), adverse-deep, shallow
    t10_paths = [None, LONG_Q10_DEEP[:4], LONG_Q10_DEEP, LONG_Q10_SHALLOW,
                 [0.1, 0.2] * 6]
    t90_paths = [None, SHORT_Q90_DEEP[:4], SHORT_Q90_DEEP, SHORT_Q90_SHALLOW,
                 [-0.1, -0.2] * 6]
    enableds = [True, False]
    cfgs = [None, {}, dict(MM_CFG), _cfg(MM_CFG, enabled=False),
            _cfg(MM_CFG, min_abs_median_pct=1.0, min_conf=0.5, min_composite=10)]
    tcfgs = [None, {}, dict(TT_CFG), _cfg(TT_CFG, enabled=False),
             _cfg(TT_CFG, window_steps=1, min_adv_path_pct=0.5)]

    checked = 0
    for side, med, conf, comp, t10, t90, en in itertools.product(
            sides, medians, confs, composites, t10_paths, t90_paths, enableds):
        ctx = _ctx(side, conf, comp, tmed=med, tq10=t10, tq90=t90)
        for cfg in cfgs:
            r = timesfm_mismatch_gate(ctx, cfg)
            assert r["pass"] is True, f"mismatch pass=False: {side} med={med} " \
                                      f"conf={conf} comp={comp} cfg={cfg}"
            checked += 1

    for side, conf, comp, t10, t90, en in itertools.product(
            sides, confs, composites, t10_paths, t90_paths, enableds):
        ctx = _ctx(side, conf, comp, tmed=-3.0, tq10=t10, tq90=t90)
        for cfg in tcfgs:
            r = timesfm_tail_trigger_gate(ctx, cfg)
            assert r["pass"] is True, f"tail pass=False: {side} " \
                                      f"conf={conf} comp={comp} cfg={cfg}"
            checked += 1
    assert checked > 0


# ===========================================================================
# (f) eval_all_gates returns BOTH keys
# ===========================================================================


def test_eval_all_gates_returns_both_timesfm_keys():
    # A ctx where BOTH legs fire (adverse median + adverse deep q10 tail) so
    # the keys carry the shadow marker; config keys ABSENT -> code defaults.
    ctx = _ctx("long", 0.70, 30.0, tmed=-1.5, tq10=LONG_Q10_DEEP)
    cfg = {
        "min_ai_confidence": 0.5, "max_concurrent": 5,
        "max_trade_notional_usd": 300, "min_market_volume_usd": 1_000_000,
        "coin_allowlist": [], "coin_blocklist": [], "cooldown_min": 0,
        "max_crypto_long_correlated": 2, "max_total_notional_pct": 1.0,
        "counter_regime_min_conf": 0.0, "block_counter_trend_bypass": False,
    }
    out = eval_all_gates(ctx, cfg, last_trade_time=None)
    res = out["results"]
    assert "timesfm_mismatch" in res
    assert "timesfm_tail_trigger" in res
    # Both fire (adverse + below the escape bar) and both are structurally
    # shadow-only: pass True + marker, and they must NOT block the stack.
    assert res["timesfm_mismatch"]["pass"] is True
    assert res["timesfm_mismatch"].get("shadow_would_block") is True
    assert res["timesfm_tail_trigger"]["pass"] is True
    assert res["timesfm_tail_trigger"].get("shadow_would_block") is True
    # Neither may appear as a block reason — the whole point of shadow-only.
    assert "timesfm_mismatch" not in out["block_reasons"]
    assert "timesfm_tail_trigger" not in out["block_reasons"]


def test_eval_all_gates_timesfm_keys_pass_when_no_signal():
    """Data gap (no median, no paths) -> both keys present and passing clean."""
    ctx = _ctx("long", 0.70, 30.0, tmed=None, tq10=None, tq90=None)
    cfg = {
        "min_ai_confidence": 0.5, "max_concurrent": 5,
        "max_trade_notional_usd": 300, "min_market_volume_usd": 1_000_000,
        "coin_allowlist": [], "coin_blocklist": [], "cooldown_min": 0,
        "max_crypto_long_correlated": 2, "max_total_notional_pct": 1.0,
        "counter_regime_min_conf": 0.0, "block_counter_trend_bypass": False,
    }
    out = eval_all_gates(ctx, cfg, last_trade_time=None)
    assert out["results"]["timesfm_mismatch"] == {"pass": True}
    assert out["results"]["timesfm_tail_trigger"] == {"pass": True}


# ===========================================================================
# (g) executor shadow-log lines fire via caplog (end-to-end through
#     maybe_execute with a stubbed TimesFM warm-cache read)
# ===========================================================================


def test_maybe_execute_accrues_both_timesfm_shadow_lines(monkeypatch, caplog):
    """The ACCRUAL query greps exactly these anchored strings. Drive
    maybe_execute end-to-end with an adverse TimesFM signal (median -1.5% +
    deep early q10 tail) on a low-conviction LONG (conf 0.70 < 0.90,
    composite 30 < 60): both mirror gates fire and the executor logs both
    would-block lines — while the trade itself is NOT blocked (it executes)."""
    from test_cleanup import _analysis, _exec_baseline
    from hermes_trader.agents import executor

    ex, captured, _ = _exec_baseline(
        monkeypatch,
        cfg_overrides={
            "timesfm_signal": {"enabled": True},  # arm the gate-side fetch
            "min_ai_confidence": 0.30,
        })

    class _FakeTsig:
        # Adverse on both legs: median below the 0.5 deadband AND a deep
        # early q10 tail (min of first 12 steps = -2.5 <= -2.0).
        median_pct = -1.5
        q10_path_pct = [-0.5, -1.0, -2.4, -2.5, -2.3, -2.1,
                        -1.8, -1.5, -1.2, -0.9, -0.7, -0.5]
        q90_path_pct = None

    monkeypatch.setattr(
        "hermes_trader.agents.timesfm_signal.get_timesfm_signal_sync",
        lambda coin, side: _FakeTsig())
    monkeypatch.setattr(ex, "_get_market_volume_24h", lambda c: 5e7)

    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        res = ex.maybe_execute(_analysis())

    # Shadow-only: the accruals MUST NOT have blocked the trade.
    assert res["executed"] is True, res
    msgs = [r.getMessage() for r in caplog.records]
    assert any("timesfm_mismatch WOULD HAVE BLOCKED" in m for m in msgs), \
        caplog.text
    assert any("timesfm_tail_trigger WOULD HAVE BLOCKED" in m for m in msgs), \
        caplog.text
    # The anchored signature the counterfactual join greps (both legs).
    assert any("[gate][SHADOW] timesfm_mismatch WOULD HAVE BLOCKED" in m
               for m in msgs), caplog.text
    assert any("[gate][SHADOW] timesfm_tail_trigger WOULD HAVE BLOCKED" in m
               for m in msgs), caplog.text
    assert "NOT blocking (shadow-only)" in [
        m for m in msgs if "timesfm_mismatch WOULD HAVE BLOCKED" in m][0]


def test_maybe_execute_timesfm_aligned_signal_is_silent(monkeypatch, caplog):
    """Control: an ALIGNED TimesFM signal (positive median, shallow q10) on a
    LONG produces NO accrual line — the shape does not fire."""
    from test_cleanup import _analysis, _exec_baseline
    from hermes_trader.agents import executor

    ex, _captured, _ = _exec_baseline(
        monkeypatch,
        cfg_overrides={
            "timesfm_signal": {"enabled": True},
            "min_ai_confidence": 0.30,
        })

    class _AlignedTsig:
        median_pct = 1.5  # model agrees with the long
        q10_path_pct = [-0.2, -0.5, -0.9, -1.2, -1.1, -0.8,
                        -0.5, -0.3, -0.1, 0.1, 0.2, 0.3]
        q90_path_pct = None

    monkeypatch.setattr(
        "hermes_trader.agents.timesfm_signal.get_timesfm_signal_sync",
        lambda coin, side: _AlignedTsig())
    monkeypatch.setattr(ex, "_get_market_volume_24h", lambda c: 5e7)

    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        res = ex.maybe_execute(_analysis())
    assert res["executed"] is True, res
    assert not any("timesfm_mismatch WOULD HAVE BLOCKED" in r.getMessage()
                   for r in caplog.records), caplog.text
    assert not any("timesfm_tail_trigger WOULD HAVE BLOCKED" in r.getMessage()
                   for r in caplog.records), caplog.text
