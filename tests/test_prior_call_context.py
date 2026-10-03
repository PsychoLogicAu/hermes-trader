"""Tests for the prior-call context block (statefulness experiment,
WATCHLIST B.32 item 1 — hermes_trader/agents/prior_calls.py) and its wiring
into _build_user_message.

Pins:
  * disabled flag -> "" and prompt byte-identical to pre-feature — THE
    shipping state;
  * enabled + prior analyses -> verdict/confidence/reasoning/price/age render,
    ai_down + close_guard_downgraded rows EXCLUDED, other coins excluded,
    lookback respected, max_calls caps (newest kept);
  * last completed round trip renders from memory.last_close_for with its own
    lookback;
  * model-swap annotation only when the prior model differs;
  * every failure path -> "" never raises into prompt assembly.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import prior_calls  # noqa: E402
from hermes_trader.agents import research  # noqa: E402
from hermes_trader.agents.memory import memory  # noqa: E402


class _Cfg:
    def __init__(self, d): self._d = d
    def get(self, k, default=None): return self._d.get(k, default)


def _patch_cfg(monkeypatch, cfg=None):
    monkeypatch.setattr(prior_calls, "read_agent_config", lambda: _Cfg(cfg or {}))


def _analysis(coin="BTC", verdict="PASS", conf=0.3, reasoning="choppy tape",
              entry_px=100.0, mins_ago=10, model="model-a", **extra):
    a = {
        "id": f"{coin}-{mins_ago}-{verdict}", "coin": coin, "verdict": verdict,
        "confidence": conf, "side": None, "entry_px": entry_px,
        "reasoning": reasoning, "created_at": int((time.time() - mins_ago * 60) * 1000),
        "primary_model": model,
    }
    a.update(extra)
    return a


def _seed(monkeypatch, analyses, close=None):
    """Swap the memory singleton's read methods (conftest already isolates
    HERMES_AGENT_MEMORY_FILE to a temp dir)."""
    monkeypatch.setattr(memory, "get_recent_analyses",
                        lambda limit=20: list(analyses), raising=False)
    monkeypatch.setattr(memory, "last_close_for", lambda c: close, raising=False)


ENABLED = {"prior_call_context": {"enabled": True}}


def test_disabled_flag_renders_empty(monkeypatch):
    _patch_cfg(monkeypatch, {})  # absent block = disabled (the shipping default)
    _seed(monkeypatch, [_analysis()])
    assert prior_calls.build_prior_call_block("BTC") == ""

    monkeypatch.setattr(prior_calls, "read_agent_config",
                        lambda: _Cfg({"prior_call_context": {"enabled": False}}))
    assert prior_calls.build_prior_call_block("BTC") == ""


def test_enabled_renders_prior_decision(monkeypatch):
    _patch_cfg(monkeypatch, ENABLED)
    _seed(monkeypatch, [
        _analysis(verdict="LONG", conf=0.72, reasoning="fresh breakout with volume",
                  entry_px=100.5, mins_ago=35),
    ])
    block = prior_calls.build_prior_call_block("BTC")
    assert "Your recent history on BTC" in block
    assert "LONG conf 0.72" in block
    assert "fresh breakout with volume" in block
    assert "$100.5000" in block
    assert "35min ago" in block


def test_excludes_ai_down_and_guarded_closes(monkeypatch):
    _patch_cfg(monkeypatch, ENABLED)
    _seed(monkeypatch, [
        _analysis(verdict="PASS", ai_down=True, mins_ago=5),
        _analysis(verdict="PASS", close_guard_downgraded=True, mins_ago=6),
    ])
    # Only failure/guarded rows -> nothing to show and no campaign -> "".
    assert prior_calls.build_prior_call_block("BTC") == ""


def test_excludes_other_coins_and_old_rows(monkeypatch):
    _patch_cfg(monkeypatch, ENABLED)
    _seed(monkeypatch, [
        _analysis(coin="ETH", mins_ago=5),
        _analysis(mins_ago=9999),  # outside the 240min default lookback
    ])
    assert prior_calls.build_prior_call_block("BTC") == ""


def test_max_calls_keeps_newest(monkeypatch):
    _patch_cfg(monkeypatch, {"prior_call_context": {"enabled": True, "max_calls": 2}})
    rows = [_analysis(reasoning=f"r{i}", mins_ago=10 * (i + 1)) for i in range(5)]
    _seed(monkeypatch, rows)
    block = prior_calls.build_prior_call_block("BTC")
    assert "r4" in block and "r3" in block   # newest two
    assert "r2" not in block                 # older ones dropped


def test_model_swap_annotation(monkeypatch):
    _patch_cfg(monkeypatch, ENABLED)
    _seed(monkeypatch, [_analysis(model="old-model", mins_ago=10)])
    same = prior_calls.build_prior_call_block("BTC", primary_model="old-model")
    assert "via" not in same
    diff = prior_calls.build_prior_call_block("BTC", primary_model="new-model")
    assert "via old-model" in diff


def test_campaign_line(monkeypatch):
    _patch_cfg(monkeypatch, ENABLED)
    close = {"coin": "BTC", "side": "long", "entry_px": 100.0, "exit_px": 95.0,
             "realized_pnl_pct": -5.2, "realized_pnl_usd": -2.6,
             "hold_minutes": 47.0, "exit_reason": "max_loss (3.1% spot)",
             "closed_at": int((time.time() - 90 * 60) * 1000)}
    _seed(monkeypatch, [], close=close)
    # No prior decisions either -> the campaign line alone still renders.
    block = prior_calls.build_prior_call_block("BTC")
    assert "Last completed round trip on BTC" in block
    assert "LONG entry $100.0000" in block
    assert "-5.20%" in block and "-2.60 USD" in block
    assert "held 47min" in block
    assert "max_loss" in block


def test_campaign_out_of_window_dropped(monkeypatch):
    _patch_cfg(monkeypatch, ENABLED)
    close = {"coin": "BTC", "side": "long", "entry_px": 1.0, "exit_px": 1.0,
             "closed_at": int((time.time() - 3000 * 60) * 1000)}
    _seed(monkeypatch, [], close=close)
    assert prior_calls.build_prior_call_block("BTC") == ""


def test_memory_fault_returns_empty(monkeypatch):
    _patch_cfg(monkeypatch, ENABLED)

    def boom(limit=20):
        raise RuntimeError("memory on fire")
    monkeypatch.setattr(memory, "get_recent_analyses", boom, raising=False)
    assert prior_calls.build_prior_call_block("BTC") == ""


def test_prompt_wiring_byte_identical_when_empty():
    """prior_block='' (the disabled default) must leave _build_user_message's
    output byte-identical to the pre-feature prompt."""
    perception = {"id": "p1", "type": "perp", "mid": 1.0, "composite_score": 10,
                  "triggers": []}
    tf = {"ema8": None, "ema21": None, "slope_up": None, "rsi14": None,
          "atr14": None, "adx14": None, "last_close": 1.0, "last_time": 0}
    # _build_user_message reads read_agent_config (band_snapback / max_concurrent)
    import hermes_trader.agents.research as R
    base = R._build_user_message("BTC", perception, tf, tf, tf, "N/A", "no news",
                                 0.0, [], "OFF")
    with_empty = R._build_user_message("BTC", perception, tf, tf, tf, "N/A",
                                       "no news", 0.0, [], "OFF", prior_block="")
    assert base == with_empty
    with_block = R._build_user_message("BTC", perception, tf, tf, tf, "N/A",
                                       "no news", 0.0, [], "OFF",
                                       prior_block="PRIOR HISTORY HERE")
    assert with_block.startswith("PRIOR HISTORY HERE\n\n" + base)
