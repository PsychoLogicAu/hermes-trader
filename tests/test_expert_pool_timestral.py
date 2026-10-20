"""Expert-pool adapter tests for the Timestral candidate (Task: timestral
into expert_select). No model, no network: the forecaster singleton is
patched at its SOURCE (timestral_signal._get_forecaster), mirroring
tests/test_expert_select.py's source-patch idiom."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import hermes_trader.agents.expert_pool as ep
import hermes_trader.agents.timestral_signal as ts

CONFIG_PATH = Path(__file__).resolve().parent.parent / ".agent-config.json"


def _levels_21():
    lv = [0.01] + [round(0.05 + i * 0.05, 2) for i in range(19)]  # 0.01,0.05,0.10,...,0.95
    return lv + [0.99]


class _FakeForecast:
    def __init__(self, h: int):
        self.quantile_levels = _levels_21()
        base = np.linspace(100.0, 101.0, h)
        # (1, 1, H, 21) non-crossing, matching the real Forecast contract
        grid = np.stack([base + (i - 10) * 0.05 for i in range(21)], axis=-1)
        self.quantiles = grid.reshape(1, 1, h, 21)


class _FakeForecaster:
    def __init__(self):
        self.calls = []

    def forecast(self, context, prediction_length, symmetric_averaging=None,
                 clip_nonnegative=None, **kwargs):
        self.calls.append({
            "n": len(context),
            "prediction_length": prediction_length,
            "symmetric_averaging": symmetric_averaging,
            "clip_nonnegative": clip_nonnegative,
        })
        return _FakeForecast(prediction_length)


@pytest.fixture
def clean_pool():
    ep._resources.pop("timestral", None)
    ep._load_errors.pop("timestral", None)
    yield
    ep._resources.pop("timestral", None)
    ep._load_errors.pop("timestral", None)


def test_timestral_is_a_named_candidate():
    assert "timestral" in ep.MODEL_NAMES
    assert ep._adapter_ctx("timestral") == 50  # matches live timestral_signal ctx


def test_borrows_timestral_signal_singleton(clean_pool, monkeypatch):
    fake = _FakeForecaster()
    monkeypatch.setattr(ts, "_get_forecaster", lambda: fake)
    got = ep._get_timestral_forecaster()
    assert got is fake  # borrow, not a second load
    assert ep._resources.get("timestral") is fake


def test_unavailable_singleton_is_sticky(clean_pool, monkeypatch):
    monkeypatch.setattr(ts, "_get_forecaster", lambda: None)
    assert ep._get_timestral_forecaster() is None
    assert "timestral" in ep._load_errors
    monkeypatch.setattr(ts, "_get_forecaster", lambda: _FakeForecaster())
    assert ep._get_timestral_forecaster() is None  # sticky until cleared


def test_forecast_raw_contract(clean_pool, monkeypatch):
    fake = _FakeForecaster()
    monkeypatch.setattr(ts, "_get_forecaster", lambda: fake)
    ep._resources["timestral"] = fake
    out = ep._forecast_raw("timestral", [10.0] * 60, 12)
    q = out["quantiles"]
    # single-pass read (clampsym TAA off) — pool candidate == shadow line
    assert fake.calls[0]["symmetric_averaging"] is False
    assert fake.calls[0]["clip_nonnegative"] is False
    assert fake.calls[0]["prediction_length"] == 12
    # raw contract: float level keys (the forecast() wrapper stringifies)
    assert len(q) == 21
    for lv in (0.1, 0.5, 0.9):
        assert lv in q and len(q[lv]) == 12
    med = np.array(q[0.5])
    assert (med <= np.array(q[0.9])).all() and (np.array(q[0.1]) <= med).all()


def test_wrapper_surfaces_median_and_q10_q90(clean_pool, monkeypatch):
    fake = _FakeForecaster()
    monkeypatch.setattr(ts, "_get_forecaster", lambda: fake)
    ep._resources["timestral"] = fake
    fc = ep.forecast("timestral", [10.0] * 60, horizon=12)
    assert not fc.error
    # wrapper stringifies levels; mixture levels resolve exactly
    assert fc.quantiles and all(
        lv in fc.quantiles for lv in ("0.1", "0.5", "0.9"))
    assert len(fc.median) == 12
    q10 = fc.q10
    q90 = fc.q90
    assert q10 is not None and q90 is not None
    assert len(q10) == 12
    assert (np.array(q10) <= np.array(fc.median)).all()


def test_default_config_ships_timestral_in_pool():
    cfg = json.loads(CONFIG_PATH.read_text())
    sel = cfg.get("expert_select", {})
    assert "timestral" in sel.get("pool", [])


def test_pool_filter_accepts_timestral(clean_pool, monkeypatch):
    cfg = dict(ep._cfg())
    cfg["pool"] = ["chronos", "timestral"]
    monkeypatch.setattr(ep, "_cfg", lambda: cfg)
    assert ep._pool() == ["chronos", "timestral"]
