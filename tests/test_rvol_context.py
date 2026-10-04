"""Tests for the RVOL entry-context block (WATCHLIST B.32 item 2 —
hermes_trader/agents/rvol_context.py) and its wiring into _build_user_message.

Pins:
  * disabled flag -> "" and prompt byte-identical to pre-feature — THE
    shipping state;
  * compute_rvol = last-candle volume / mean(prior baseline_bars volumes),
    matching triggers.volume_spike's sparse-market guard (>50% zero bars) and
    zero-average guard;
  * enabled -> one line with the ratio, a coarse label, and the replay-derived
    framing; labels band at 0.5/1.0/2.0;
  * every failure path (fetch raise, short history, sparse tape, bad config)
    -> "" never raises into prompt assembly;
  * _build_user_message omits the section entirely when rvol_block="" and
    injects it after the prior-call block when non-empty.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import research  # noqa: E402
from hermes_trader.agents import rvol_context  # noqa: E402


class _Cfg:
    def __init__(self, d): self._d = d
    def get(self, k, default=None): return self._d.get(k, default)


def _patch_cfg(monkeypatch, cfg=None):
    monkeypatch.setattr(rvol_context, "read_agent_config", lambda: _Cfg(cfg or {}))


def _bars(vols):
    return [{"t": i * 300_000, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": v}
            for i, v in enumerate(vols)]


def _patch_fetch(monkeypatch, bars):
    monkeypatch.setattr(rvol_context, "fetch_hl_candles",
                        lambda coin, interval="5m", count=100, fresh=False: list(bars))


ENABLED = {"rvol_context": {"enabled": True}}


# ── compute_rvol ──────────────────────────────────────────────────────

def test_compute_basic_ratio(monkeypatch):
    # 20 baseline bars of volume 10, last bar volume 25 -> RVOL 2.5
    _patch_fetch(monkeypatch, _bars([10.0] * 20 + [25.0]))
    assert rvol_context.compute_rvol("BTC", baseline_bars=20) == 2.5


def test_compute_uses_baseline_window_only(monkeypatch):
    # Old bars far from the signal candle must not leak into the baseline.
    vols = [999.0] * 10 + [10.0] * 20 + [10.0]
    _patch_fetch(monkeypatch, _bars(vols))
    assert rvol_context.compute_rvol("BTC", baseline_bars=20) == 1.0


def test_compute_short_history_returns_none(monkeypatch):
    _patch_fetch(monkeypatch, _bars([10.0] * 15))
    assert rvol_context.compute_rvol("BTC", baseline_bars=20) is None


def test_compute_sparse_tape_returns_none(monkeypatch):
    # >50% of the BASELINE bars zero -> meaningless ratio (volume_spike parity).
    vols = [10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0] + \
           [10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0, 0.0, 0.0] + [5.0]
    # zeros in baseline = 13/20 -> sparse
    _patch_fetch(monkeypatch, _bars(vols))
    assert rvol_context.compute_rvol("BTC", baseline_bars=20) is None


def test_compute_zero_average_returns_none(monkeypatch):
    _patch_fetch(monkeypatch, _bars([0.0] * 20 + [5.0]))
    assert rvol_context.compute_rvol("BTC", baseline_bars=20) is None


def test_compute_fetch_raises_returns_none(monkeypatch):
    def boom(*a, **k): raise RuntimeError("network")
    monkeypatch.setattr(rvol_context, "fetch_hl_candles", boom)
    assert rvol_context.compute_rvol("BTC") is None


# ── build_rvol_block ──────────────────────────────────────────────────

def test_disabled_flag_renders_empty(monkeypatch):
    _patch_cfg(monkeypatch, {})  # absent block = disabled (the shipping default)
    _patch_fetch(monkeypatch, _bars([10.0] * 21 + [25.0]))
    assert rvol_context.build_rvol_block("BTC") == ""

    monkeypatch.setattr(rvol_context, "read_agent_config",
                        lambda: _Cfg({"rvol_context": {"enabled": False}}))
    assert rvol_context.build_rvol_block("BTC") == ""


def test_enabled_renders_ratio_line(monkeypatch):
    _patch_cfg(monkeypatch, ENABLED)
    _patch_fetch(monkeypatch, _bars([10.0] * 20 + [25.0]))
    block = rvol_context.build_rvol_block("BTC")
    assert "RVOL" in block
    assert "2.50x" in block
    assert "high" in block
    assert "Context only" in block


def test_labels_band():
    assert rvol_context._label(0.3) == "very quiet"
    assert rvol_context._label(0.7) == "below average"
    assert rvol_context._label(1.5) == "above average"
    assert rvol_context._label(4.0) == "high"


def test_enabled_but_uncomputable_renders_empty(monkeypatch):
    _patch_cfg(monkeypatch, ENABLED)
    _patch_fetch(monkeypatch, [])  # fetch failure -> empty list
    assert rvol_context.build_rvol_block("BTC") == ""


def test_bad_config_type_is_safe(monkeypatch):
    monkeypatch.setattr(rvol_context, "read_agent_config",
                        lambda: _Cfg({"rvol_context": "nonsense"}))
    _patch_fetch(monkeypatch, _bars([10.0] * 20 + [25.0]))
    # config garbage -> DEFAULTS (disabled) -> ""
    assert rvol_context.build_rvol_block("BTC") == ""


def test_config_read_raises_is_safe(monkeypatch):
    def boom(): raise RuntimeError("config store down")
    monkeypatch.setattr(rvol_context, "read_agent_config", boom)
    assert rvol_context.build_rvol_block("BTC") == ""


def test_baseline_bars_config_respected(monkeypatch):
    _patch_cfg(monkeypatch, {"rvol_context": {"enabled": True, "baseline_bars": 5}})
    # baseline = last 5 before signal (all 10), older bars irrelevant
    _patch_fetch(monkeypatch, _bars([999.0] * 20 + [10.0] * 5 + [30.0]))
    block = rvol_context.build_rvol_block("BTC")
    assert "3.00x" in block
    assert "prior 5-bar average" in block


# ── _build_user_message wiring ────────────────────────────────────────

def _msg(**kw):
    return research._build_user_message(
        "BTC", {"mid": 100.0, "composite_score": 40, "triggers": []},
        {}, {}, {}, "0.01%", "no news", 100.0, [], "OFF", **kw)


def test_prompt_identical_when_rvol_block_empty():
    base = _msg()
    withempty = _msg(rvol_block="")
    assert base == withempty  # shipping state: byte-identical


def test_prompt_injects_rvol_block():
    line = "Entry-candle volume (RVOL): 0.42x the prior 20-bar average — very quiet."
    m = _msg(rvol_block=line)
    assert line in m
    # sits before the Candidate: header (context preamble, like prior_block)
    assert m.index(line) < m.index("Candidate: BTC")


def test_prompt_orders_prior_then_rvol():
    p = "PRIORBLOCK"
    r = "RVOLBLOCK"
    m = _msg(prior_block=p, rvol_block=r)
    assert m.index(p) < m.index(r) < m.index("Candidate: BTC")
