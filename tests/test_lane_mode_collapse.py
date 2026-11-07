"""One-key tri-state collapse for the three bypass lanes (owner ask 2026-11-07,
t_71c6d8ab). The bool master + `*_shadow_mode` sibling trio composed THREE
different rules across the lanes; `bypass_late_trend_chase: true` + a stale
shadow sibling logged floor-refusals with "(shadow mode)" text while the lane
was demonstrably live. Collapse: the master key itself takes the string
"live" | "shadow" | "off"; legacy bools keep each lane's historical
composition (byte-identical) so code deploy and config swap can interleave
safely. Invalid string = fail-safe off + loud warning.
"""
import logging
import types

from hermes_trader.agents import executor
from hermes_trader.agents.risk_gates import (GateContext, lane_mode,
                                             market_liquidity_floor)

# ── helper matrix ─────────────────────────────────────────────────────────────

def test_mode_strings_case_insensitive():
    for raw, want in (("live", "live"), (" Shadow ", "shadow"), ("OFF", "off")):
        assert lane_mode({"x": raw}, "x") == want


def test_invalid_string_fails_safe_off(caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        assert lane_mode({"x": "livelike"}, "x") == "off"
    assert any("fail-safe 'off'" in r.getMessage() for r in caplog.records)


def test_legacy_bool_authoritative_matrix():
    assert lane_mode({"x": True, "x_shadow": True}, "x", "x_shadow") == "live"
    assert lane_mode({"x": True}, "x", "x_shadow") == "live"
    assert lane_mode({"x": False, "x_shadow": True}, "x", "x_shadow") == "shadow"
    assert lane_mode({"x": False}, "x", "x_shadow") == "off"
    assert lane_mode({}, "x", "x_shadow") == "off"


def test_legacy_shadow_wins_base_required_matrix():
    assert lane_mode({"x": True}, "x", "x_shadow",
                     legacy="shadow_wins_base_required") == "live"
    assert lane_mode({"x": True, "x_shadow": True}, "x", "x_shadow",
                     legacy="shadow_wins_base_required") == "shadow"
    # base bool required: false + shadow sibling stays SILENT off (matches
    # the low-liquidity gate's historical nesting)
    assert lane_mode({"x": False, "x_shadow": True}, "x", "x_shadow",
                     legacy="shadow_wins_base_required") == "off"


def test_legacy_structural_shadow_only_matrix():
    assert lane_mode({}, "x", "x_shadow",
                     legacy="structural_shadow_only") == "live"
    assert lane_mode({"x_shadow": True}, "x", "x_shadow",
                     legacy="structural_shadow_only") == "shadow"


# ── site 1: late-chase lane ───────────────────────────────────────────────────

def _gate(**over):
    g = {
        "enabled": True,
        "min_confidence": 0.70,
        "min_composite": 30.0,
        "min_hip3_composite": 50.0,
        "bypass_late_trend_chase": True,
        "bypass_late_trend_chase_min_conf": 0.90,
        "late_chase_dynamic_per_signal_drop": 0.10,
        "late_chase_timesfm_vote": False,
        "late_chase_timesfm_drop": 0.0,
    }
    g.update(over)
    return {"runner_entry_gate": g}


def _analysis(conf=0.78, **over):
    a = {
        "coin": "SKR",
        "side": "long",
        "confidence": conf,
        "composite_score": 0.0,
        "uptrend_momentum_fired": True,
    }
    a.update(over)
    return a


def _chronos_aligned(monkeypatch):
    sig = types.SimpleNamespace(median_pct=0.2, spread_pct=0.5, error=None)
    monkeypatch.setattr(executor, "get_chronos_signal_sync", lambda c, s: sig)


def test_site1_mode_live_releases_without_sibling(monkeypatch):
    # string form, sibling keys gone: lane live, conf 0.82 >= 0.80 bar
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_trend_chase="live")
    assert executor._runner_entry_block_reason(_analysis(conf=0.82), g) == ""


def test_site1_mode_shadow_old_text_byte_identical(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_trend_chase="shadow")
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(_analysis(conf=0.82), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    accr = [r.getMessage() for r in caplog.records
            if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert len(accr) == 1
    assert "NOT bypassing (shadow mode), live rule stands" in accr[0]


def test_site1_mode_off_silent_and_reason_matches_legacy_off(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    legacy = executor._runner_entry_block_reason(
        _analysis(conf=0.82), _gate(bypass_late_trend_chase=False))
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(
            _analysis(conf=0.82), _gate(bypass_late_trend_chase="off"))
    assert reason == legacy
    assert not [r for r in caplog.records
                if "WOULD HAVE BYPASSED" in r.getMessage()]


def test_site1_invalid_string_off_plus_warning(monkeypatch, caplog):
    _chronos_aligned(monkeypatch)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.risk_gates"):
        reason = executor._runner_entry_block_reason(
            _analysis(conf=0.82), _gate(bypass_late_trend_chase="LIVE-ish"))
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert any("fail-safe 'off'" in r.getMessage() for r in caplog.records)


def test_site1_live_floor_refusal_logs_floor_refused_not_shadow(monkeypatch, caplog):
    # lane LIVE (legacy bool true, sibling absent) + fuel floor denies:
    # refusal accrual must name the REAL reason, never claim shadow mode.
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_chase_require_fuel=True)  # bool stays True
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(
            _analysis(conf=0.82, breakout_fired=True), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    accr = [r.getMessage() for r in caplog.records
            if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert len(accr) == 1
    assert "FUEL=deny" in accr[0]
    assert "floor refused" in accr[0]
    assert "shadow mode" not in accr[0]


def test_site1_legacy_bool_shadow_true_still_logs_refusal_live_style(monkeypatch,
                                                                     caplog):
    # live lane + stale-true sibling + floor deny: same new suffix (the
    # confusing "(shadow mode)" suffix on refusals is gone for live lanes).
    _chronos_aligned(monkeypatch)
    g = _gate(bypass_late_trend_chase_shadow_mode=True,
              bypass_late_chase_require_fuel=True)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(
            _analysis(conf=0.82, breakout_fired=True), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    accr = [r.getMessage() for r in caplog.records
            if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert len(accr) == 1
    assert "floor refused" in accr[0]


# ── site 2: mover lane ────────────────────────────────────────────────────────

def _cfg(**over):
    g = {
        "enabled": True,
        "min_confidence": 0.65,
        "min_composite": 30.0,
        "allow_shorts": True,
        "mover_min_confidence": 0.72,
        "mover_min_composite": 30.0,
    }
    g.update(over)
    return {"runner_entry_gate": g}


def _mover_long(**over):
    a = {
        "coin": "TEST",
        "side": "long",
        "confidence": 0.75,
        "composite_score": 35.0,
        "primary_model": "test-model",
        "daily_mover_fired": True,
        "volume_spike_fired": True,
        "breakout_fired": False,
        "momentum_burst_fired": False,
        "uptrend_momentum_fired": False,
        "downtrend_momentum_fired": False,
        "slow_burn_count": 0,
        "whale_signal": False,
        "reasoning": "",
    }
    a.update(over)
    return a


BLOCK_REASON = ("runner_gate_blocked (needs volume+breakout/burst "
                "and structure; score=35, slow=0)")


def test_site2_mode_off_blocks_silently_like_term_absent(caplog):
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(
            _mover_long(), _cfg(mover_bypass="off"))
    assert reason == BLOCK_REASON
    assert not [r.getMessage() for r in caplog.records
                if "mover_bypass WOULD HAVE ADMITTED" in r.getMessage()]


def test_site2_mode_shadow_matches_legacy_text(monkeypatch, caplog):
    legacy = _cfg(mover_bypass_shadow_mode=True)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        r1 = executor._runner_entry_block_reason(_mover_long(), legacy)
        r2 = executor._runner_entry_block_reason(
            _mover_long(), _cfg(mover_bypass="shadow"))
    assert r1 == r2 == BLOCK_REASON
    accr = [r.getMessage() for r in caplog.records
            if "mover_bypass WOULD HAVE ADMITTED" in r.getMessage()]
    assert len(accr) == 2  # both legacy and string form accrue identically
    assert all("NOT admitting (shadow mode; live rule blocks)" in m
               for m in accr)


def test_site2_mode_string_live_admits():
    assert executor._runner_entry_block_reason(
        _mover_long(), _cfg(mover_bypass="live")) == ""


# ── site 3: low-liquidity bypass ──────────────────────────────────────────────

NATIVE_FLOOR = 1_000_000.0
HIP3_FLOOR = 5_000_000.0


def _ctx(confidence, vol_usd, coin="STBL", side="long"):
    return GateContext(
        confidence=confidence,
        current_positions=[],
        trade_notional_usd=25.0,
        daily_pnl=0.0,
        market_volume_24h_usd=vol_usd,
        coin=coin,
        trade_side=side,
        has_binary_news_risk=False,
        equity=1000.0,
        total_open_notional=0.0,
    )


def test_site3_mode_strings():
    cfg = {"bypass_low_volume": "live", "bypass_low_volume_min_conf": 0.85}
    r = market_liquidity_floor(_ctx(0.86, 450_000), NATIVE_FLOOR, HIP3_FLOOR,
                               gate_config=cfg)
    assert r["pass"] is True and r["via"] == "bypass_low_volume"

    cfg = {"bypass_low_volume": "off", "bypass_low_volume_min_conf": 0.85}
    r = market_liquidity_floor(_ctx(0.86, 450_000), NATIVE_FLOOR, HIP3_FLOOR,
                               gate_config=cfg)
    assert r["pass"] is False and "via" not in r


def test_site3_mode_shadow_logs_would_have_released(caplog):
    cfg = {"bypass_low_volume": "shadow", "bypass_low_volume_min_conf": 0.85}
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.risk_gates"):
        r = market_liquidity_floor(_ctx(0.86, 450_000), NATIVE_FLOOR,
                                   HIP3_FLOOR, gate_config=cfg)
    assert r["pass"] is False
    accr = [rec.getMessage() for rec in caplog.records
            if "WOULD HAVE RELEASED" in rec.getMessage()]
    assert len(accr) == 1
    assert "NOT releasing (shadow mode; live rule blocks)" in accr[0]
