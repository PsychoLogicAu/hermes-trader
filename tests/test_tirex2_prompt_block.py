"""Tests for the TiRex-2 prompt block (2026-10-21 mirror of _tirex_block).

Ships DISABLED (`tirex2_signal.in_prompt` absent/false): added so a
chronos->tirex2 prompt swap is a config flip once the tirex2 shadow accrual
and the C.11 decision settle. Pins:

  * disabled flag -> '' (prompt shape unchanged) — THE shipping state;
  * enabled + in_prompt + signal -> lean render (median, band, early tail
    from the first 6 steps, confidence-floor note);
  * same HEMI min_conf_ratio semantics as the rest of the family;
  * every failure path -> '' never raises into prompt assembly;
  * the block is REGISTERED in _build_user_message (a defined block not in
    the list renders nothing).
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import research  # noqa: E402
from hermes_trader.agents import tirex2_signal as tx2  # noqa: E402


def _sig(median_pct=-1.5, spread_pct=4.0, err=None, q_paths=True):
    base = 100.0
    lo = base * (1 + (-spread_pct / 2) / 100)
    hi = base * (1 + (spread_pct / 2) / 100)
    return tx2.Tirex2Signal(
        coin="TEST", side="long", context_last=base,
        median=base * (1 + median_pct / 100), q_low=lo, q_high=hi,
        median_pct=median_pct, spread_pct=spread_pct, horizon=12,
        model_id="NX-AI/TiRex-2", inference_ms=50.0, error=err,
        q10_path_pct=[-0.5, -1.2, -3.1, -2.0, -1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5] if q_paths else None,
        q90_path_pct=[0.6, 1.1, 2.2, 1.5, 0.9, 0.4, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3] if q_paths else None,
    )


class _Cfg:
    def __init__(self, d): self._d = d
    def get(self, k, default=None): return self._d.get(k, default)


def _patch(monkeypatch, cfg=None, sig=None, boom=False):
    monkeypatch.setattr(research, "read_agent_config", lambda: _Cfg(cfg or {}))
    if boom:
        def raiser(coin, side): raise RuntimeError("boom")
        monkeypatch.setattr(tx2, "get_tirex2_signal_sync", raiser)
    else:
        monkeypatch.setattr(tx2, "get_tirex2_signal_sync", lambda coin, side: sig)


def test_disabled_absent_block_renders_empty(monkeypatch):
    _patch(monkeypatch, cfg={}, sig=_sig())
    assert research._tirex2_block("TEST") == ""


def test_enabled_but_not_in_prompt_renders_empty(monkeypatch):
    _patch(monkeypatch, cfg={"tirex2_signal": {"enabled": True}}, sig=_sig())
    assert research._tirex2_block("TEST") == ""


def test_in_prompt_but_disabled_renders_empty(monkeypatch):
    _patch(monkeypatch, cfg={"tirex2_signal": {"enabled": False, "in_prompt": True}}, sig=_sig())
    assert research._tirex2_block("TEST") == ""


def test_enabled_in_prompt_renders_lean_block(monkeypatch):
    _patch(monkeypatch, cfg={"tirex2_signal": {"enabled": True, "in_prompt": True}}, sig=_sig())
    out = research._tirex2_block("TEST")
    assert "TiRex-2 forecast" in out
    assert "-1.50%" in out
    assert "p10 min -3.1%" in out and "p90 max +2.2%" in out  # first 6 steps only
    assert "FADE within ~1h" in out


def test_positive_median_renders_continuation(monkeypatch):
    _patch(monkeypatch, cfg={"tirex2_signal": {"enabled": True, "in_prompt": True}},
           sig=_sig(median_pct=1.5, spread_pct=4.0))
    assert "continuation" in research._tirex2_block("TEST")


def test_confidence_floor_note(monkeypatch):
    _patch(monkeypatch, cfg={"tirex2_signal": {"enabled": True, "in_prompt": True}},
           sig=_sig(median_pct=0.08, spread_pct=4.98))
    out = research._tirex2_block("TEST")
    assert "no confident direction" in out


def test_min_conf_ratio_respected(monkeypatch):
    _patch(monkeypatch,
           cfg={"tirex2_signal": {"enabled": True, "in_prompt": True, "min_conf_ratio": 0.9}},
           sig=_sig(median_pct=-1.5, spread_pct=4.0))
    assert "no confident direction" in research._tirex2_block("TEST")


def test_signal_none_renders_empty(monkeypatch):
    _patch(monkeypatch, cfg={"tirex2_signal": {"enabled": True, "in_prompt": True}}, sig=None)
    assert research._tirex2_block("TEST") == ""


def test_signal_error_renders_empty(monkeypatch):
    _patch(monkeypatch, cfg={"tirex2_signal": {"enabled": True, "in_prompt": True}},
           sig=_sig(err="timeout after 30s"))
    assert research._tirex2_block("TEST") == ""


def test_reader_boom_never_raises(monkeypatch):
    _patch(monkeypatch, cfg={"tirex2_signal": {"enabled": True, "in_prompt": True}}, boom=True)
    assert research._tirex2_block("TEST") == ""


def test_build_user_message_includes_tirex2_block():
    src = inspect.getsource(research._build_user_message)
    assert "_tirex2_block(coin)" in src, "block defined but never registered"
