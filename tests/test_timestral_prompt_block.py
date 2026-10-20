"""Tests for the Timestral prompt block (2026-10 mirror of _tirex_block).

Ships DISABLED (`timestral_signal.in_prompt` absent/false): added so a
chronos->timestral prompt swap is a config flip once the shadow accrual
settles. Pins:

  * disabled flag -> '' (prompt shape unchanged) — THE shipping state;
  * enabled + signal -> lean render (median, band, early tail from the
    first 6 steps, confidence-floor note);
  * same HEMI min_conf_ratio semantics as chronos/timesfm/tirex;
  * every failure path -> '' never raises into prompt assembly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import research  # noqa: E402
from hermes_trader.agents import timestral_signal as tms  # noqa: E402


def _sig(median_pct=-1.5, spread_pct=4.0, err=None, q_paths=True):
    base = 100.0
    lo = base * (1 + (-spread_pct / 2) / 100)
    hi = base * (1 + (spread_pct / 2) / 100)
    return tms.TimestralSignal(
        coin="TEST", side="long", context_last=base,
        median=base * (1 + median_pct / 100), q_low=lo, q_high=hi,
        median_pct=median_pct, spread_pct=spread_pct, horizon=12,
        model_id="mistralai/Timestral-209M", inference_ms=50.0, error=err,
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
        monkeypatch.setattr(tms, "get_timestral_signal_sync", raiser)
    else:
        monkeypatch.setattr(tms, "get_timestral_signal_sync", lambda coin, side: sig)


# ── shipping state: disabled -> empty prompt contribution ───────────────────

def test_disabled_by_default_renders_empty(monkeypatch):
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True}}, sig=_sig())
    assert research._timestral_block("TEST") == ""


def test_missing_block_config_renders_empty(monkeypatch):
    _patch(monkeypatch, cfg={}, sig=_sig())
    assert research._timestral_block("TEST") == ""


def test_signal_disabled_renders_empty_even_with_in_prompt(monkeypatch):
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": False, "in_prompt": True}}, sig=_sig())
    assert research._timestral_block("TEST") == ""


# ── enabled render ───────────────────────────────────────────────────────────

def test_enabled_renders_median_band_and_tail(monkeypatch):
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True, "in_prompt": True}}, sig=_sig())
    out = research._timestral_block("TEST")
    assert "Timestral forecast" in out
    assert "-1.50%" in out                       # path-average median
    assert "p10 avg -2.0% / p90 avg +2.0%" in out
    assert "p10 min -3.1%" in out                # early tail = first 6 steps only
    assert "p90 max +2.2%" in out
    assert "-3.1%" in out and "0.5%" not in out.split("early tail")[1].split("\n")[0]


def test_fade_note_above_confidence_floor(monkeypatch):
    # |median| 1.5 / spread 4.0 = 0.375 >= 0.25 floor -> interpretive note
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True, "in_prompt": True}}, sig=_sig(median_pct=-1.5, spread_pct=4.0))
    out = research._timestral_block("TEST")
    assert "FADE" in out


def test_continuation_note_when_median_positive(monkeypatch):
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True, "in_prompt": True}}, sig=_sig(median_pct=1.5, spread_pct=4.0))
    assert "continuation" in research._timestral_block("TEST")


def test_neutral_note_below_confidence_floor(monkeypatch):
    # HEMI semantics: median inside its own band -> no fade claim
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True, "in_prompt": True}}, sig=_sig(median_pct=0.08, spread_pct=4.98))
    out = research._timestral_block("TEST")
    assert "no confident direction" in out
    assert "FADE" not in out


def test_custom_min_conf_ratio_honoured(monkeypatch):
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True, "in_prompt": True, "min_conf_ratio": 0.9}}, sig=_sig(median_pct=-1.5, spread_pct=4.0))
    out = research._timestral_block("TEST")
    assert "no confident direction" in out and "FADE" not in out


# ── failure paths: never raise into prompt assembly ─────────────────────────

def test_none_signal_renders_empty(monkeypatch):
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True, "in_prompt": True}}, sig=None)
    assert research._timestral_block("TEST") == ""


def test_error_signal_renders_empty(monkeypatch):
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True, "in_prompt": True}}, sig=_sig(err="timeout"))
    assert research._timestral_block("TEST") == ""


def test_missing_median_renders_empty(monkeypatch):
    s = _sig(); s.median_pct = None
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True, "in_prompt": True}}, sig=s)
    assert research._timestral_block("TEST") == ""


def test_raising_wrapper_renders_empty(monkeypatch):
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True, "in_prompt": True}}, boom=True)
    assert research._timestral_block("TEST") == ""


def test_no_paths_still_renders_without_tail_segment(monkeypatch):
    _patch(monkeypatch, cfg={"timestral_signal": {"enabled": True, "in_prompt": True}}, sig=_sig(q_paths=False))
    out = research._timestral_block("TEST")
    assert "Timestral forecast" in out and "early tail" not in out


# ── wiring: block is part of the prompt assembly list ───────────────────────

def test_build_user_message_includes_timestral_block():
    import inspect
    src = inspect.getsource(research._build_user_message)
    assert "_timestral_block(coin)" in src
