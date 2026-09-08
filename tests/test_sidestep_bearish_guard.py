"""Falling-knife guard for the long TA-sidestep (port of upstream 51bc23b).

The sidestep upgrades an AI-PASS -> LONG when a magnitude-based composite/burst
setup clears. But those triggers (pctMoveSpike / volumeSpike / shockDay /
momentumBurst) key on `abs(move)`, so a violent SELLOFF — a big red candle on
huge volume — fires them exactly like a breakout and the bot would buy the open
of a red breakdown bar (the xyz:SMSN 2026-06-29, -9.3% ROE case). This restores
the symmetric bullish-direction requirement the short runner gate already has.

Hermetic: in-memory analysis dicts + monkeypatched config; no network. Clause
(b) (24h move) is covered by setting `daily_move_pct` directly on the analysis
dict; with no 24h move available the clause degrades to a no-op and the
downtrend clause (a) is the guaranteed protection.
"""
import pytest

from hermes_trader.agents import executor


CFG = {
    "mode": "LIVE",
    "enable_crypto": True,
    "ta_sidestep_force_execute": True,
    "force_execute_composite": 20,
    "min_ai_confidence": 0.70,
    "runner_entry_gate": {
        "sidestep_require_bullish": True,
        "sidestep_bearish_move_pct": -3.0,
    },
}

CFG_FLAG_OFF = {**CFG, "runner_entry_gate": {**CFG["runner_entry_gate"],
                                             "sidestep_require_bullish": False}}


def _a(**kw):
    base = {
        "id": "sidestep-test",
        "coin": "SMSN",
        "verdict": "PASS",
        "confidence": 0.60,
        "composite_score": 50.0,  # clears force_execute_composite → ta_sidestep_strong
        "uptrend_momentum_fired": False,
        "downtrend_momentum_fired": False,
        "daily_move_pct": None,
    }
    base.update(kw)
    return base


def _patch_cfg(monkeypatch, cfg):
    monkeypatch.setattr(executor, "read_agent_config", lambda: cfg)
    # Downstream plumbing — keep the "proceeds" tests hermetic up to the equity
    # read (the natural marker that the PASS->LONG upgrade happened).
    monkeypatch.setattr(executor, "resolve_user_address", lambda: "0xUSER")
    monkeypatch.setattr(executor, "fetch_account_state", lambda u, **kw: {"equity": 0})


# ── (1) downtrend momentum fired, no uptrend → NOT upgraded (the SMSN case) ──
def test_blocks_downtrend_momentum_long(monkeypatch):
    _patch_cfg(monkeypatch, CFG)
    res = executor.maybe_execute(_a(downtrend_momentum_fired=True))
    assert res["executed"] is False
    assert res["reason"].startswith("sidestep_bearish_blocked")
    assert "downtrend momentum fired" in res["reason"]
    assert res["analysis_id"] == "sidestep-test"


# ── (2) explicit uptrend momentum → sidestep upgrade proceeds ────────────────
def test_allows_uptrend_momentum_proceeds(monkeypatch):
    _patch_cfg(monkeypatch, CFG)
    # Explicit bullish momentum = a real upside setup; the guard must not block.
    res = executor.maybe_execute(
        _a(uptrend_momentum_fired=True, downtrend_momentum_fired=True,
           daily_move_pct=-9.0))
    # Upgraded to LONG and ran past the guard to the equity read (not blocked).
    assert "sidestep_bearish_blocked" not in res["reason"]
    assert "equity_unavailable" in res["reason"]


# ── (3) 24h move clearly negative, no uptrend → blocked (clause b) ───────────
def test_blocks_clearly_negative_24h_move(monkeypatch):
    _patch_cfg(monkeypatch, CFG)
    # -7.5% < -3.0 with no uptrend: a selloff the magnitude triggers still fired.
    res = executor.maybe_execute(_a(daily_move_pct=-7.5))
    assert res["executed"] is False
    assert res["reason"].startswith("sidestep_bearish_blocked")
    assert "-7.5%" in res["reason"]


def test_small_negative_move_not_blocked(monkeypatch):
    # -1% is noise, not a selloff — threshold is -3%; don't over-block.
    _patch_cfg(monkeypatch, CFG)
    res = executor.maybe_execute(_a(daily_move_pct=-1.0))
    assert "sidestep_bearish_blocked" not in res["reason"]
    assert "equity_unavailable" in res["reason"]


def test_missing_24h_move_degrades_to_downtrend_only(monkeypatch):
    # Documented degradation: with no 24h move AND no downtrend, clause (b) is a
    # no-op and nothing blocks (the downtrend clause is the guaranteed guard).
    _patch_cfg(monkeypatch, CFG)
    res = executor.maybe_execute(_a(daily_move_pct=None,
                                    downtrend_momentum_fired=False))
    assert "sidestep_bearish_blocked" not in res["reason"]
    assert "equity_unavailable" in res["reason"]


# ── (4) sidestep_require_bullish: false → guard disabled (reversible) ────────
def test_flag_off_disables_guard(monkeypatch):
    _patch_cfg(monkeypatch, CFG_FLAG_OFF)
    # Bearish setup, but the flag is off → upgrade proceeds (guard reversible).
    res = executor.maybe_execute(_a(downtrend_momentum_fired=True,
                                    daily_move_pct=-9.0))
    assert "sidestep_bearish_blocked" not in res["reason"]
    assert "equity_unavailable" in res["reason"]


# ── (5) non-sidestep override paths are unaffected ───────────────────────────
def test_whale_override_on_bearish_coin_still_upgrades(monkeypatch):
    # Whale force-execute (NOT ta_sidestep) on a bearish coin: the guard is
    # scoped to the sidestep path, so it must NOT be consulted here.
    _patch_cfg(monkeypatch, {
        "mode": "LIVE", "enable_crypto": True,
        "whale_force_execute": True, "min_ai_confidence": 0.70,
        # no ta_sidestep_force_execute → ta_sidestep_strong False
    })
    res = executor.maybe_execute(_a(whale_signal=True,
                                    downtrend_momentum_fired=True,
                                    daily_move_pct=-9.0,
                                    composite_score=0.0))
    assert "sidestep_bearish_blocked" not in res["reason"]
    assert "equity_unavailable" in res["reason"]


# ── helper-level unit checks (pure function, no maybe_execute) ───────────────
def _gate_cfg(**over):
    g = {"sidestep_require_bullish": True, "sidestep_bearish_move_pct": -3.0}
    g.update(over)
    return {"runner_entry_gate": g}


def test_helper_blocks_downtrend_no_uptrend():
    r = executor._sidestep_bearish_block_reason(_a(downtrend_momentum_fired=True),
                                                _gate_cfg())
    assert r and "downtrend momentum fired" in r


def test_helper_blocks_negative_24h_move():
    r = executor._sidestep_bearish_block_reason(_a(daily_move_pct=-7.5), _gate_cfg())
    assert r and "-7.5%" in r


def test_helper_uptrend_wins_even_with_downtrend_and_negative_move():
    r = executor._sidestep_bearish_block_reason(
        _a(uptrend_momentum_fired=True, downtrend_momentum_fired=True,
           daily_move_pct=-9.0), _gate_cfg())
    assert r == ""


def test_helper_allows_neutral_setup():
    # No downtrend, no 24h move: the sidestep's real purpose (flat/up setups)
    # sails through.
    assert executor._sidestep_bearish_block_reason(
        _a(daily_move_pct=2.0), _gate_cfg()) == ""


def test_helper_missing_move_degrades_to_no_op():
    assert executor._sidestep_bearish_block_reason(_a(), _gate_cfg()) == ""


def test_helper_flag_off_returns_empty():
    r = executor._sidestep_bearish_block_reason(
        _a(downtrend_momentum_fired=True, daily_move_pct=-9.0),
        _gate_cfg(sidestep_require_bullish=False))
    assert r == ""
