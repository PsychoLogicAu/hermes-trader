"""Scan composite-gate tuning knob (t_b1d64fe2).

The scan gate (`scan.minCompositeScore`) used to be startup-static:
trading_loop read the TRIGGER_CONFIG value (54) once at boot and the
operator-visible key in .agent-config.json (25) was a DEAD knob — flipping
it changed nothing silently. Post-nil-2026-10-23 post-mortem:

  * `resolve_scan_gate` — pure per-tick resolution. The .agent-config.json
    value wins when it parses to a positive number; missing / garbage /
    <=0 / NaN fall back to the STATIC default (never the old dead-knob
    value — a config hiccup must not silently move the live gate).
  * Composite-gate accrual — with .agent-config.json `scan_gate_shadow_log`
    truthy, every below-bar drop logs a [scan][gate-shadow] DROPPED line
    (coin, composite, bar, fired hits). Log-only; accrues the population
    the bar acts on so the TS-era 54 can be re-derived on evidence.

Hermetic: no network, no LLM — `_fetch_candles_sync` patched to flat candles
and one volume trigger monkeypatched to a low-score fired hit, so the ONLY
path to research is the composite gate itself.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import perception  # noqa: E402
from hermes_trader.agents.config import get_config  # noqa: E402
from hermes_trader.models.types import Candle  # noqa: E402

_T0 = 1_700_000_000_000
_STEP = 300_000

# Flat 5m candles: no real trigger fires; the monkeypatched volume_spike hit
# below is the single fired trigger (weight 0.25, score 1.0 -> composite
# 1*0.25/1.85*10 ~= 1.35, far under any sane bar).
FLAT_5M = [Candle(t=_T0 + i * _STEP, o=100.0, h=100.05, l=99.95, c=100.0, v=1000.0)
           for i in range(100)]

_MARKET = {"coin": "TEST", "type": "perp"}


# ---------------------------------------------------------------------------
# Pure helper — resolve_scan_gate resolution matrix.
# ---------------------------------------------------------------------------

def test_gate_reads_agent_value():
    assert perception.resolve_scan_gate({"scan": {"minCompositeScore": 40}}, 54.0) == 40.0
    assert perception.resolve_scan_gate({"scan": {"minCompositeScore": "30"}}, 54.0) == 30.0


def test_gate_missing_key_falls_back_to_static():
    assert perception.resolve_scan_gate({}, 54.0) == 54.0
    assert perception.resolve_scan_gate({"scan": {}}, 54.0) == 54.0
    assert perception.resolve_scan_gate(None, 54.0) == 54.0
    assert perception.resolve_scan_gate({"scan": None}, 54.0) == 54.0


def test_gate_garbage_nonpositive_nan_fall_back():
    for bad in ("abc", 0, -5, float("nan"), [], {}):
        assert perception.resolve_scan_gate({"scan": {"minCompositeScore": bad}}, 54.0) == 54.0


# ---------------------------------------------------------------------------
# Integration — composite-gate accrual line via _scan_single_market.
# ---------------------------------------------------------------------------

def _low_score_hit(name="volumeSpike"):
    return {"name": name, "score": 1.0, "reason": "test spike", "fired": True}


def _scan(monkeypatch, cfg):
    monkeypatch.setattr(perception, "_fetch_candles_sync",
                        lambda coin, interval, count, cache_ttl_ms, **kw: FLAT_5M)
    monkeypatch.setattr(perception.trigger_mod, "volume_spike",
                        lambda *a, **k: _low_score_hit())
    ok, res = perception._scan_single_market(_MARKET, 100.0, cfg, 54.0, None, False, True)
    assert ok is True
    return res


def _base_cfg(**extra):
    return {**get_config(), **extra}


def test_below_bar_dropped_with_accrual_line(monkeypatch, caplog):
    """Flag on: the drop still happens (log-only), AND the accrual line lands
    at INFO with coin, composite, bar and the fired-hit list."""
    cfg = _base_cfg(scan_gate_shadow_log=True)
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.perception"):
        res = _scan(monkeypatch, cfg)
    assert res is None  # log-only: behavior unchanged
    lines = [r.getMessage() for r in caplog.records
             if "[scan][gate-shadow] DROPPED" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "TEST" in line and "volumeSpike" in line and "< bar 54" in line


def test_below_bar_dropped_silent_when_flag_off(monkeypatch, caplog):
    """Flag absent/false: drop remains silent at INFO (pre-change bytes)."""
    for cfg in (_base_cfg(), _base_cfg(scan_gate_shadow_log=False)):
        with caplog.at_level(logging.INFO, logger="hermes_trader.agents.perception"):
            res = _scan(monkeypatch, cfg)
        assert res is None
    assert not [r for r in caplog.records if "gate-shadow" in r.getMessage()]


def test_above_bar_surfaces_regardless_of_flag(monkeypatch):
    """Bar BELOW the composite (tunable direction): the coin surfaces — this is
    what makes the knob real for tuning, flag or no flag."""
    for cfg in (_base_cfg(scan_gate_shadow_log=True), _base_cfg()):
        monkeypatch.setattr(perception, "_fetch_candles_sync",
                            lambda coin, interval, count, cache_ttl_ms, **kw: FLAT_5M)
        monkeypatch.setattr(perception.trigger_mod, "volume_spike",
                            lambda *a, **k: _low_score_hit())
        ok, res = perception._scan_single_market(_MARKET, 100.0, cfg, 1.0, None, False, True)
        assert ok is True
        assert res is not None and res["coin"] == "TEST"
