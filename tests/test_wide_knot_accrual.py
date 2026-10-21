"""Wide-knot (0.05/0.95) path accrual — timestral_signal + chronos_signal
(2026-10-21). Both 21-knot checkpoints already compute these columns; the
tail gates have only ever consumed 0.10/0.90. Shadow-only accrual: fields
on the cached signal + a compact `wide_tail6(q05/q95)` suffix on the
existing accrual line. Nothing gates or sizes on them.

Pins:
  (a) wide paths populated from the 21-knot grid, value-correct, ordered
      q05 < q10 < median < q90 < q95 on the low/high sides;
  (b) narrow-grid checkpoints (9-level 0.1..0.9) -> fields None, no
      mislabelling of neighbouring columns (the value guard);
  (c) suffix renders exactly how the tail gates judge tails (min/max of
      the first 6 steps) and is absent for error/None/short paths;
  (d) accrual line still parses the old way (prefix contract intact).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import chronos_signal as cs  # noqa: E402
from hermes_trader.agents import timestral_signal as tms  # noqa: E402

KNOTS_21 = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
GRID_9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CFG = {"enabled": True, "context_length": 50, "forecast_horizon": 12}


def _closes(n=50, start=100.0):
    return [{"o": start, "h": start, "l": start, "c": start + i, "v": 1.0}
            for i in range(n)]


def _tmfake(levels):
    """Timestral stand-in: p50 +1%/step, band multipliers per knot so the
    wide columns are distinguishable from p10/p90."""
    nq = len(levels)

    def forecast(context, prediction_length, **kwargs):
        last = float(context[-1])
        med = last * (1.0 + 0.01 * np.arange(1, prediction_length + 1))
        q = np.zeros((1, 1, prediction_length, nq))
        for i, lv in enumerate(levels):
            mult = {0.05: 0.94, 0.10: 0.97, 0.50: 1.0,
                    0.90: 1.03, 0.95: 1.06}.get(lv, 1.0)
            q[0, 0, :, i] = med * mult
        return SimpleNamespace(quantiles=q, quantile_levels=list(levels))

    return SimpleNamespace(quantile_levels=list(levels), forecast=forecast)


def _chronos_fake(levels):
    """Chronos-2 pipeline stand-in: predict() -> [tensor (1, nq, H)]."""
    nq = len(levels)

    def predict(batch, prediction_length):
        last = float(batch[0][-1])
        med = torch.tensor([last * (1.0 + 0.01 * (i + 1))
                            for i in range(prediction_length)])
        rows = []
        for lv in levels:
            mult = {0.05: 0.94, 0.10: 0.97, 0.50: 1.0,
                    0.90: 1.03, 0.95: 1.06}.get(lv, 1.0)
            rows.append(med * mult)
        return [torch.stack(rows, dim=0).unsqueeze(0)]

    return SimpleNamespace(quantiles=list(levels), predict=predict)


# ── (a) wide paths populated on the 21-knot grid ─────────────────────────────
def test_timestral_wide_paths_populated():
    gf = tms._get_forecaster
    tms._get_forecaster = lambda: _tmfake(KNOTS_21)
    try:
        sig = tms._forecast_from_candles("X", "long", _closes(50), CFG)
        assert sig.error is None, sig.error
        assert sig.q05_path_pct and sig.q95_path_pct
        assert len(sig.q05_path_pct) == 12
        # value-correct: 0.05 sits DEEPER than 0.10 on the low side
        assert all(a < b for a, b in zip(sig.q05_path_pct, sig.q10_path_pct))
        assert all(a > b for a, b in zip(sig.q95_path_pct, sig.q90_path_pct))
    finally:
        tms._get_forecaster = gf


def test_chronos_wide_paths_populated(monkeypatch):
    monkeypatch.setattr(cs, "_get_pipeline", lambda: _chronos_fake(KNOTS_21))
    sig = cs._forecast_from_candles("X", "long", _closes(50), CFG)
    assert sig.error is None, sig.error
    assert sig.q05_path_pct and sig.q95_path_pct
    assert all(a < b for a, b in zip(sig.q05_path_pct, sig.q10_path_pct))
    assert all(a > b for a, b in zip(sig.q95_path_pct, sig.q90_path_pct))


# ── (b) value guard: coarser grids get None, never mislabelled columns ───────
def test_timestral_narrow_grid_wide_none():
    gf = tms._get_forecaster
    tms._get_forecaster = lambda: _tmfake(GRID_9)
    try:
        sig = tms._forecast_from_candles("X", "long", _closes(50), CFG)
        assert sig.error is None, sig.error
        assert sig.q10_path_pct and sig.q90_path_pct   # p10/p90 fine
        assert sig.q05_path_pct is None and sig.q95_path_pct is None
    finally:
        tms._get_forecaster = gf


def test_chronos_narrow_grid_wide_none(monkeypatch):
    monkeypatch.setattr(cs, "_get_pipeline", lambda: _chronos_fake(GRID_9))
    sig = cs._forecast_from_candles("X", "long", _closes(50), CFG)
    assert sig.error is None, sig.error
    assert sig.q10_path_pct and sig.q90_path_pct
    assert sig.q05_path_pct is None and sig.q95_path_pct is None


# ── (c) suffix contract ───────────────────────────────────────────────────────
def _tmsig(q05=None, q95=None):
    return tms.TimestralSignal(
        coin="X", side="long", context_last=100.0, median=101.0,
        q_low=100.0, q_high=102.0, median_pct=1.0, spread_pct=2.0,
        horizon=12, model_id="m", inference_ms=1.0,
        q10_path_pct=[-1.0] * 12, q90_path_pct=[1.0] * 12,
        q05_path_pct=q05, q95_path_pct=q95)


def _csig(q05=None, q95=None):
    return cs.ChronosSignal(
        coin="X", side="long", context_last=100.0, median=101.0,
        q_low=100.0, q_high=102.0, median_pct=1.0, spread_pct=2.0,
        horizon=12, model_id="m", inference_ms=1.0,
        q10_path_pct=[-1.0] * 12, q90_path_pct=[1.0] * 12,
        q05_path_pct=q05, q95_path_pct=q95)


def test_suffix_uses_gate_semantics_first6_min_max():
    q05 = [-1.0, -2.0, -3.5, -2.0, -1.0, -0.5, -9.9]  # min of FIRST 6 = -3.5
    q95 = [1.0, 2.0, 3.2, 2.0, 1.0, 0.5, 9.9]         # max of FIRST 6 = +3.2
    for sig, mod in ((_tmsig(q05, q95), tms), (_csig(q05, q95), cs)):
        s = mod._wide_tail_suffix(sig)
        assert s == " | wide_tail6(q05/q95)=-3.50%/+3.20%", s


def test_suffix_absent_without_wide_paths():
    for mod in (tms, cs):
        sig = _tmsig() if mod is tms else _csig()
        assert mod._wide_tail_suffix(sig) == ""
        short = _tmsig([-1.0] * 5, [1.0] * 5) if mod is tms else _csig([-1.0] * 5, [1.0] * 5)
        assert mod._wide_tail_suffix(short) == ""


def test_accrual_line_contract_preserved():
    # old parsers regex-SEARCH the existing fields; the suffix appends AFTER
    # them and the error line carries no suffix at all.
    line = tms._format_signal_log(_tmsig([-2.0] * 12, [2.5] * 12),
                                  debug=False, min_conf_ratio=0.25)
    assert "NEUTRAL (ratio 0.50 < 0.25)" not in line  # ratio 0.5 >= floor:
    assert "ALIGN" in line and "| wide_tail6(q05/q95)=-2.00%/+2.50%" in line
    err = tms.TimestralSignal(coin="X", side="long", context_last=0.0,
                              median=None, q_low=None, q_high=None,
                              median_pct=None, spread_pct=None, horizon=12,
                              model_id="m", inference_ms=0.0, error="boom")
    assert "wide_tail6" not in tms._format_signal_log(err, debug=False)
