"""Tests for the Chronos confidence floor (the 2026-08-30 HEMI replay fix).

A median that sits inside the model's own p10-p90 band is noise, not a
directional claim. ``confidence_ratio`` = |median_pct| / spread_pct; when it
is below ``chronos_signal.min_conf_ratio`` (default 0.25):

  * the LLM prompt note renders "no confident direction" instead of the full
    FADE/continuation warning (``research._chronos_block``);
  * the log line reads NEUTRAL instead of ALIGN/MISMATCH
    (``chronos_signal._format_signal_log``).

Fail-safe is ZERO confidence: a missing spread or median never opens the
floor — we must never claim a fade (or continuation) we have no basis for.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import chronos_signal as cs  # noqa: E402
from hermes_trader.agents import research  # noqa: E402


def _sig(median_pct, spread_pct, med=None, side="long"):
    base = 0.011
    return cs.ChronosSignal(
        coin="HEMI",
        side=side,
        context_last=base,
        median=base if median_pct is None else base * (1 + median_pct / 100),
        q_low=None,
        q_high=None,
        median_pct=median_pct,
        spread_pct=spread_pct,
        horizon=12,
        model_id="amazon/chronos-2",
        inference_ms=100,
    )


# ── confidence_ratio ─────────────────────────────────────────────────────────


def test_ratio_is_abs_median_over_spread():
    assert cs.confidence_ratio(_sig(-0.08, 4.98)) == (0.08 / 4.98)
    assert cs.confidence_ratio(_sig(1.67, 6.45)) == (1.67 / 6.45)


def test_ratio_is_zero_when_spread_or_median_missing():
    assert cs.confidence_ratio(_sig(-0.08, None)) == 0.0
    assert cs.confidence_ratio(_sig(None, 4.98)) == 0.0
    assert cs.confidence_ratio(_sig(None, None)) == 0.0


# ── log flag: NEUTRAL below the floor ────────────────────────────────────────


def test_log_neutral_below_floor():
    # The 18:00 HEMI anchor: -0.08% median inside a 4.98% band.
    line = cs._format_signal_log(_sig(-0.08, 4.98), debug=False)
    assert "NEUTRAL (ratio 0.02 < 0.25)" in line
    assert "MISMATCH" not in line and "ALIGN" not in line


def test_log_mismatch_at_or_above_floor():
    line = cs._format_signal_log(_sig(-0.75, 2.0), debug=False)  # ratio 0.375
    assert "MISMATCH" in line


def test_log_align_at_or_above_floor_positive():
    line = cs._format_signal_log(_sig(1.67, 6.45), debug=False)  # ratio 0.259
    assert "ALIGN" in line


def test_log_neutral_when_median_missing():
    line = cs._format_signal_log(_sig(None, 4.98), debug=False)
    assert "NEUTRAL (no median)" in line


def test_log_floor_is_configurable_and_zero_disables():
    # ratio 0.016; at a 0.01 floor it is confident, at the default it is not.
    s = _sig(-0.08, 4.98)
    assert "MISMATCH" in cs._format_signal_log(s, False, min_conf_ratio=0.01)
    assert "NEUTRAL" in cs._format_signal_log(s, False, min_conf_ratio=0.25)


def test_resolve_min_conf_ratio_zero_disables_not_default():
    """Documented contract: 0 disables the floor (docs/CONFIG.md + the module
    header). An `or 0.25` fallback would silently re-arm it — pinned here."""
    assert cs.resolve_min_conf_ratio({"min_conf_ratio": 0}) == 0.0
    assert cs.resolve_min_conf_ratio({"min_conf_ratio": 0.0}) == 0.0
    assert cs.resolve_min_conf_ratio({}) == 0.25
    assert cs.resolve_min_conf_ratio({"min_conf_ratio": 0.4}) == 0.4
    assert cs.resolve_min_conf_ratio({"min_conf_ratio": -1}) == 0.0   # clamps
    assert cs.resolve_min_conf_ratio({"min_conf_ratio": "junk"}) == 0.25
    # end-to-end: floor 0 in the cfg never reads NEUTRAL (ratio 0.02 signal)
    line = cs._format_signal_log(_sig(-0.08, 4.98), False,
                                 cs.resolve_min_conf_ratio({"min_conf_ratio": 0}))
    assert "NEUTRAL" not in line


# ── prompt note: the interpretive claim only renders above the floor ────────


def _block(monkeypatch, cfg_over, sig):
    # research imports read_agent_config BY NAME — patch the consumer's
    # reference, not the config_store module attribute.
    cfg = {
        "chronos_signal": {
            "enabled": True,
            "in_prompt": True,
            "min_conf_ratio": 0.25,
        }
    }
    cfg["chronos_signal"].update(cfg_over)
    monkeypatch.setattr(research, "read_agent_config", lambda: cfg)
    monkeypatch.setattr(cs, "get_chronos_signal_sync", lambda coin, side: sig)
    return research._chronos_block("HEMI")


def test_prompt_note_neutral_below_floor(monkeypatch):
    """The 19:00 HEMI anchor: -1.70% inside an 8.34% band (ratio 0.20).
    Old code rendered the full FADE warning; new code says noise."""
    out = _block(monkeypatch, {}, _sig(-1.70, 8.34))
    assert "no confident direction" in out
    assert "FADE" not in out
    # the data still renders so the LLM can weigh it itself
    assert "-1.70%" in out and "8.3%" in out


def test_prompt_note_fade_above_floor(monkeypatch):
    out = _block(monkeypatch, {}, _sig(-2.50, 8.0))  # ratio 0.31
    assert "FADE" in out
    assert "no confident direction" not in out


def test_prompt_note_continuation_above_floor(monkeypatch):
    # The 22:00 HEMI anchor: +1.67% inside a 6.45% band (ratio 0.259) —
    # the one call that was right (actual next hour +4.6%) still renders.
    out = _block(monkeypatch, {}, _sig(1.67, 6.45))
    assert "continuation" in out
    assert "no confident direction" not in out


def test_prompt_floor_is_configurable(monkeypatch):
    # ratio 0.31: FADE at the 0.25 floor, neutral at a 0.5 floor.
    assert "FADE" in _block(monkeypatch, {"min_conf_ratio": 0.25}, _sig(-2.50, 8.0))
    assert "no confident direction" in _block(
        monkeypatch, {"min_conf_ratio": 0.5}, _sig(-2.50, 8.0))


def test_prompt_no_spread_means_neutral(monkeypatch):
    """Fail-safe: a signal with no spread must never claim a direction."""
    out = _block(monkeypatch, {}, _sig(-3.0, None))
    assert "no confident direction" in out
    assert "FADE" not in out