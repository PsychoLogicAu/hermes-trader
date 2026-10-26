"""Seat-profile gate bars (t_623a4dcf): config-models/<seat>.json "gate_bars".

Conf bars calibrated to one seat's conf scale mis-scale the next (the
d3-27b SHORT max 0.698 vs the 0.65/0.85/0.9 bars, 2026-10-26). A profile
may now carry a ``gate_bars`` block keyed EXACTLY like the main config's
gate keys; the profile OVERRIDES the config block for that seat's
decisions, merged at sub-key level for dict blocks. The config block stays
the fallback for unprofiled seats — byte-identical behavior when nothing
matches (house fail-open contract, hot per-decision read like the
state_admission clause).

Pins: precedence (profile beats inline), byte-identical fallback (no
profile / no gate_bars / corrupt profile = SAME config object), sub-key
merge keeping non-bar siblings (allow_shorts, cohort_admissions,
shadow_mode...), the EFFECTIVE bar inside reason strings (flips/audits are
dated from log lines), and coexistence with the older entry_min_* seat
bars (gate_bars sets the EFFECTIVE bar; entry_min_* stays the sub-floor
admission below it).

Distinction from entry_min_confidence (tests/test_seat_conf_bar.py) and
the ownership rule: a bar compared against a seat's conf belongs to THAT
seat's profile, keyed by the ANSWERING model — duelist_veto_gate's
conviction escape compares the PRIMARY's conf, so its bar lives in the
primary-engine profile.
"""

import json

import pytest

from hermes_trader.agents import config_store, executor
from hermes_trader.agents.config_store import (
    apply_seat_gate_bars,
    read_model_profile_for_model,
)
from hermes_trader.agents.risk_gates import GateContext, eval_all_gates


@pytest.fixture
def seat_dir(tmp_path, monkeypatch):
    """Isolated MODEL_CONFIG_DIR."""
    monkeypatch.setattr(config_store, "MODEL_CONFIG_DIR", str(tmp_path))
    return tmp_path


def _write_profile(tmp_path, fname, **keys):
    (tmp_path / fname).write_text(json.dumps(keys))


# ── helper unit: merge contract ─────────────────────────────────────────────

def test_no_profile_returns_same_object(seat_dir):
    cfg = {"min_ai_confidence": 0.65}
    assert apply_seat_gate_bars(cfg, "seat-x") is cfg
    assert apply_seat_gate_bars(cfg, "") is cfg
    assert apply_seat_gate_bars(cfg, None) is cfg  # type: ignore[arg-type]


def test_profile_without_gate_bars_same_object(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x", conf_multiplier=1.0)
    cfg = {"min_ai_confidence": 0.65}
    assert apply_seat_gate_bars(cfg, "seat-x") is cfg


def test_corrupt_profile_fails_open_same_object(seat_dir):
    (seat_dir / "seat-x.json").write_text("{not json")
    cfg = {"min_ai_confidence": 0.65}
    assert apply_seat_gate_bars(cfg, "seat-x") is cfg


def test_scalar_override_and_subkey_merge(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x", gate_bars={
        "min_ai_confidence": 0.55,
        "band_counter_breach_gate": {"min_conf": 0.8},
    })
    cfg = {
        "min_ai_confidence": 0.65,
        "band_counter_breach_gate": {
            "enabled": True, "shadow_mode": True, "min_conf": 0.9,
            "drift_confirmed_release": {"enabled": True},
        },
        "unrelated": {"keep": True},
    }
    out = apply_seat_gate_bars(cfg, "seat-x")
    assert out is not cfg  # merged view is a copy; input untouched
    assert cfg["min_ai_confidence"] == 0.65  # input NOT mutated
    assert out["min_ai_confidence"] == 0.55
    # dict merge is SUB-KEY: siblings survive
    g = out["band_counter_breach_gate"]
    assert g["min_conf"] == 0.8
    assert g["enabled"] is True and g["shadow_mode"] is True
    assert "drift_confirmed_release" in g
    assert out["unrelated"] == {"keep": True}


def test_lookup_by_model_key_not_filename(seat_dir):
    # profile FILE name differs from the answering model name (real pattern:
    # d3-27b -> d3-27b-artemis.json)
    _write_profile(seat_dir, "seat-x-profile.json", model="Seat-X",
                   gate_bars={"min_ai_confidence": 0.5})
    out = apply_seat_gate_bars({"min_ai_confidence": 0.65}, "seat-x")
    assert out["min_ai_confidence"] == 0.5


# ── runner gate end-to-end (executor read site) ─────────────────────────────

def _gate(**over):
    g = {
        "enabled": True,
        "min_confidence": 0.65,
        "min_short_confidence": 0.65,
        "min_composite": 30.0,
        "allow_shorts": True,
    }
    g.update(over)
    return {"runner_entry_gate": g}


def _long_analysis(conf=0.50, **over):
    a = {
        "coin": "PUMP",
        "side": "long",
        "confidence": conf,
        "composite_score": 40.0,
        "volume_spike_fired": True,
        "breakout_fired": True,
        "primary_model": "seat-x",
    }
    a.update(over)
    return a


def test_runner_fallback_byte_identical(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x")
    reason = executor._runner_entry_block_reason(_long_analysis(), _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.65)"


def test_runner_profile_bar_admits(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x", gate_bars={
        "runner_entry_gate": {"min_confidence": 0.45}})
    assert executor._runner_entry_block_reason(_long_analysis(0.50), _gate()) == ""


def test_runner_reason_string_prints_effective_bar(seat_dir):
    # rule 2: the printed bar is the MERGED (effective) one — flips/audits
    # are dated from this number in the block lines.
    _write_profile(seat_dir, "seat-x.json", model="seat-x", gate_bars={
        "runner_entry_gate": {"min_confidence": 0.55}})
    reason = executor._runner_entry_block_reason(_long_analysis(0.50), _gate())
    assert reason == "runner_gate_blocked (confidence 0.50 < 0.55)"


def test_runner_sibling_keys_survive_merge(seat_dir):
    # allow_shorts is a non-conf sibling: after the sub-key merge the short
    # must still be subject to it and a short still admits at the seat bar.
    _write_profile(seat_dir, "seat-x.json", model="seat-x", gate_bars={
        "runner_entry_gate": {"min_confidence": 0.40,
                              "min_short_confidence": 0.40}})
    a = _long_analysis(0.45, side="short", downtrend_momentum_fired=True)
    assert executor._runner_entry_block_reason(a, _gate()) == ""
    off = _gate(allow_shorts=False)
    reason = executor._runner_entry_block_reason(a, off)
    assert reason == "runner_gate_blocked (shorts disabled)"


def test_gate_bars_and_entry_min_coexistence(seat_dir):
    # gate_bars lowers the EFFECTIVE general bar to 0.60; entry_min_confidence
    # remains the sub-floor ADMISSION (0.50) below it. conf 0.55 admits
    # (seat-bar path), conf 0.45 blocks against the 0.50 seat floor.
    _write_profile(seat_dir, "seat-x.json", model="seat-x",
                   entry_min_confidence=0.50, gate_bars={
                       "runner_entry_gate": {"min_confidence": 0.60}})
    assert executor._runner_entry_block_reason(_long_analysis(0.55), _gate()) == ""
    reason = executor._runner_entry_block_reason(_long_analysis(0.45), _gate())
    assert reason == "runner_gate_blocked (confidence 0.45 < 0.50)"


# ── eval_all_gates end-to-end (risk_gates choke) ────────────────────────────

def _veto_ctx(confidence: float) -> GateContext:
    # opposite-side duelist call on a long entry -> veto shape fires.
    return GateContext(
        confidence=confidence,
        current_positions=[],
        trade_notional_usd=10.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin="TEST",
        trade_side="long",
        has_binary_news_risk=False,
        equity=100.0,
        total_open_notional=0.0,
        composite_score=40.0,
        duelist_verdict="SHORT",
        primary_model="seat-x",
    )


def _veto_cfg():
    # House isolation pattern (mirrors tests/test_duelist_veto_gate.py::
    # test_eval_all_gates_carries_duelist_veto): relax every OTHER gate so
    # the duelist veto is the only thing under test.
    return {
        "min_ai_confidence": 0.5,
        "max_concurrent": 5,
        "max_trade_notional_usd": 300,
        "min_market_volume_usd": 1_000_000,
        "coin_allowlist": [], "coin_blocklist": [],
        "cooldown_min": 0,
        "max_crypto_long_correlated": 2,
        "max_total_notional_pct": 1.0,
        "counter_regime_min_conf": 0.0,
        "block_counter_trend_bypass": False,
        "duelist_veto_gate": {"enabled": True, "shadow_mode": True,
                              "min_conf": 0.9, "min_composite": 60},
    }


def test_eval_gates_fallback_bar_from_config(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x")
    out = eval_all_gates(_veto_ctx(0.75), _veto_cfg())
    v = out["results"]["duelist_veto"]
    assert v["pass"] is True and v.get("shadow_would_block") is True
    assert "conf 0.75 < 0.90" in v["reason"]


def test_eval_gates_profile_escapes_at_seat_bar(seat_dir):
    # Owner call (t_623a4dcf): the veto's conviction-escape bar lives in the
    # PRIMARY-engine profile as an independent gate_bars key — its conf is
    # the primary's, not the duelist's.
    _write_profile(seat_dir, "seat-x.json", model="seat-x", gate_bars={
        "duelist_veto_gate": {"min_conf": 0.70}})
    out = eval_all_gates(_veto_ctx(0.75), _veto_cfg())
    v = out["results"]["duelist_veto"]
    assert v["pass"] is True and not v.get("shadow_would_block")


def test_eval_gates_profile_bar_prints_effective_in_reason(seat_dir):
    _write_profile(seat_dir, "seat-x.json", model="seat-x", gate_bars={
        "duelist_veto_gate": {"min_conf": 0.70}})
    out = eval_all_gates(_veto_ctx(0.60), _veto_cfg())
    v = out["results"]["duelist_veto"]
    assert v.get("shadow_would_block") is True
    assert "conf 0.60 < 0.70" in v["reason"]
    # gate_results dict shape unchanged (counterfactual parsers, rule 3)
    assert set(v) <= {"pass", "reason", "shadow_would_block"}
