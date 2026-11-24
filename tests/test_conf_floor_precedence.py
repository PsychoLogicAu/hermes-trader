"""Conf-floor precedence: general bar < config-level side bar < seat profile side bar.

Owner ruling 2026-11-24 (kanban t_5cc67821 / t_40bcf04d): "The presence of a
general bar config-level side-agnostic bar is fine. A config-level side specific
bar must override it. And if specified, a seat-level side-specific bar must be
the highest precedence. There should be no ambiguity. mammoth's JSON must win."

Why it was ambiguous in practice: the classic confidence gate in
`risk_gates.eval_all_gates` read ONLY the general bar (`min_ai_confidence`), so
it re-blocked candidates the answering seat's `entry_min_short_confidence` was
set to admit. Measured on the mammoth-1 seat era (9 days): 1,210 deduped
`confidence 0.6x < 0.65` refusals against 0 `short confidence` refusals — the
profile's 0.60 short bar never bound outside the cohort-admission lane.
"""

import json
import logging

import pytest

from hermes_trader.agents import config_store, executor, risk_gates
from hermes_trader.agents.risk_gates import GateContext


@pytest.fixture
def seat_dir(tmp_path, monkeypatch):
    """Isolated MODEL_CONFIG_DIR (same seam as tests/test_seat_conf_bar.py)."""
    monkeypatch.setattr(config_store, "MODEL_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _write_profile(tmp_path, fname, **keys):
    (tmp_path / fname).write_text(json.dumps(keys))


def _cfg(short=None, long_=None, general=0.65):
    gate = {"enabled": True, "min_confidence": general, "min_composite": 30.0,
            "allow_shorts": True}
    if short is not None:
        gate["min_short_confidence"] = short
    if long_ is not None:
        gate["min_long_confidence"] = long_
    return {"min_ai_confidence": general, "runner_entry_gate": gate}


# ── resolver: the three tiers, in order ──────────────────────────────────────

def test_general_only_when_nothing_else_set():
    assert config_store.effective_conf_floor(_cfg(), "short", "") == (0.65, "general")


def test_config_side_bar_overrides_general():
    assert config_store.effective_conf_floor(_cfg(short=0.58), "short", "") == (0.58, "config_side")
    # the side key is side-specific: longs keep the general bar
    assert config_store.effective_conf_floor(_cfg(short=0.58), "long", "") == (0.65, "general")


def test_long_side_config_bar(seat_dir):
    assert (config_store.effective_conf_floor(_cfg(long_=0.60), "long", "")
            == (0.60, "config_side"))


def test_seat_profile_side_bar_beats_config_side_bar(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_short_confidence=0.60)
    assert (config_store.effective_conf_floor(_cfg(short=0.58), "short", "seat-x")
            == (0.60, "seat_profile"))


def test_seat_profile_bar_wins_upward_too(seat_dir):
    # precedence, not min(): a seat bar ABOVE the general bar still binds
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_short_confidence=0.72)
    assert (config_store.effective_conf_floor(_cfg(short=0.58), "short", "seat-x")
            == (0.72, "seat_profile"))


def test_side_specific_beats_a_profile_general_gate_bar(seat_dir):
    # gate_bars may set the GENERAL bar for this seat; a side-specific seat bar
    # is still the highest tier for that side.
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   gate_bars={"min_ai_confidence": 0.70,
                              "runner_entry_gate": {"min_short_confidence": 0.62}},
                   entry_min_short_confidence=0.60)
    merged = config_store.apply_seat_gate_bars(_cfg(), "seat-x")
    assert config_store.effective_conf_floor(merged, "short", "seat-x") == (0.60, "seat_profile")
    # with the seat's side key removed, the overlaid config-level side bar wins
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   gate_bars={"min_ai_confidence": 0.70,
                              "runner_entry_gate": {"min_short_confidence": 0.62}})
    merged = config_store.apply_seat_gate_bars(_cfg(), "seat-x")
    assert config_store.effective_conf_floor(merged, "short", "seat-x") == (0.62, "config_side")


def test_malformed_or_out_of_range_values_fall_through(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_short_confidence="banana")
    assert (config_store.effective_conf_floor(_cfg(short=0.58), "short", "seat-x")
            == (0.58, "config_side"))
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_short_confidence=0.0)
    assert (config_store.effective_conf_floor(_cfg(), "short", "seat-x")
            == (0.65, "general"))


def test_unknown_side_uses_general(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_short_confidence=0.60)
    assert config_store.effective_conf_floor(_cfg(short=0.58), "", "seat-x") == (0.65, "general")


def test_default_general_when_config_has_no_bar():
    assert config_store.effective_conf_floor({}, "long", "") == (0.8, "general")


# ── wiring: the runner gate honours the config-level side bar ────────────────

def _gate(**over):
    g = {"enabled": True, "min_confidence": 0.65, "min_composite": 30.0, "allow_shorts": True}
    g.update(over)
    return {"runner_entry_gate": g}


def _short(conf, **over):
    a = {"coin": "PUMP", "side": "short", "confidence": conf, "composite_score": 40.0,
         "volume_spike_fired": True, "breakout_fired": True, "downtrend_momentum_fired": True,
         "primary_model": "seat-x"}
    a.update(over)
    return a


def test_config_side_bar_admits_short_below_general(seat_dir, caplog):
    # No profile bars at all: config min_short_confidence 0.60 must beat the
    # general 0.65 (before the ruling this refused at 0.65 and never reached the
    # side floor).
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    cfg = _gate(min_short_confidence=0.60)
    assert executor._runner_entry_block_reason(_short(0.62), cfg) == ""
    assert "[gate][SIDE-BAR] ADMIT PUMP SHORT" in caplog.text
    assert "min_short_confidence 0.60" in caplog.text


def test_config_side_bar_is_the_bar_reported_when_it_refuses(seat_dir):
    cfg = _gate(min_short_confidence=0.60)
    reason = executor._runner_entry_block_reason(_short(0.58), cfg)
    assert reason == "runner_gate_blocked (confidence 0.58 < 0.60)"


def test_seat_short_bar_still_beats_config_side_bar(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_short_confidence=0.60)
    cfg = _gate(min_short_confidence=0.55)
    reason = executor._runner_entry_block_reason(_short(0.58), cfg)
    assert reason == "runner_gate_blocked (confidence 0.58 < 0.60)"


def test_malformed_config_side_bar_fails_open(seat_dir):
    cfg = _gate(min_short_confidence="banana")
    reason = executor._runner_entry_block_reason(_short(0.62), cfg)
    assert reason == "runner_gate_blocked (confidence 0.62 < 0.65)"


# ── wiring: the CLASSIC gate no longer reads the general bar alone ───────────

def _ctx(side, conf, primary_model="seat-x"):
    return GateContext(confidence=conf, current_positions=[], trade_notional_usd=50.0,
                       daily_pnl=0.0, market_volume_24h_usd=1_000_000_000.0, coin="PUMP",
                       trade_side=side, has_binary_news_risk=False, equity=1000.0,
                       total_open_notional=50.0, primary_model=primary_model)


def _spy_confidence_gate(monkeypatch):
    seen = []
    real = risk_gates.confidence_gate

    def spy(ctx, min_confidence):
        seen.append(min_confidence)
        return real(ctx, min_confidence)

    monkeypatch.setattr(risk_gates, "confidence_gate", spy)
    return seen


def test_classic_gate_uses_the_seat_side_bar(seat_dir, monkeypatch):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_short_confidence=0.60)
    seen = _spy_confidence_gate(monkeypatch)
    cfg = {"min_ai_confidence": 0.65, "max_concurrent": 10,
           "runner_entry_gate": {"min_confidence": 0.65, "min_short_confidence": 0.58}}
    try:
        risk_gates.eval_all_gates(_ctx("short", 0.62), config=cfg)
    except Exception:
        pass  # sibling gates may need I/O this test does not stub
    assert seen and seen[0] == 0.60


def test_classic_gate_uses_config_side_bar_without_profile(seat_dir, monkeypatch):
    seen = _spy_confidence_gate(monkeypatch)
    cfg = {"min_ai_confidence": 0.65, "max_concurrent": 10,
           "runner_entry_gate": {"min_confidence": 0.65, "min_short_confidence": 0.58}}
    try:
        risk_gates.eval_all_gates(_ctx("short", 0.62, primary_model=""), config=cfg)
    except Exception:
        pass
    assert seen and seen[0] == 0.58


def test_classic_gate_longs_keep_the_general_bar(seat_dir, monkeypatch):
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_short_confidence=0.60)
    seen = _spy_confidence_gate(monkeypatch)
    cfg = {"min_ai_confidence": 0.65, "max_concurrent": 10,
           "runner_entry_gate": {"min_confidence": 0.65, "min_short_confidence": 0.58}}
    try:
        risk_gates.eval_all_gates(_ctx("long", 0.62), config=cfg)
    except Exception:
        pass
    assert seen and seen[0] == 0.65
