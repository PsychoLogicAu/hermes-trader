"""Tests for the VETO first-class verdict (2026-09-03).

VETO = the model's ACTIVE rejection of a setup ("this is a trap — avoid like
the plague"), categorically stronger than PASS (neutral abstention). Covers:

1. parse_verdict accepts the token (JSON path; regex fallback stays PASS —
   an unparseable reply must never invent a VETO).
2. route_verdict: VETO → action="none" and NEVER routed to maybe_execute,
   even when every structural-override hint that would route a PASS is on.
3. maybe_execute's independent top-level guard (direct callers — server
   /execute, MCP — bypass the router; without the guard an unknown token
   falls through the PASS equality-checks to `side or "long"` and blind-LONGs).
4. The duel plumbing carries VETO as just another verdict token.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.research import parse_verdict  # noqa: E402


# ── parse_verdict ───────────────────────────────────────────────────────────

def _parse(text: str):
    return parse_verdict(text, "TEST", {"mid": 1.0})


def test_parse_veto_token():
    p = _parse('{"verdict":"VETO","confidence":0.9,"side":null,'
               '"entryPx":0,"stopPx":0,"tpPx":0,'
               '"reasoning":"late chase into reversion, avoid"}')
    assert p["verdict"] == "VETO"
    assert p["confidence"] == 0.9
    assert p["side"] is None


def test_parse_veto_lowercase():
    p = _parse('{"verdict":"veto","confidence":0.8}')
    assert p["verdict"] == "VETO"


def test_parse_unknown_token_stays_pass():
    # Anything outside the five known tokens must fall back to PASS — never a
    # VETO invented from garbage, never a directional side derived from it.
    p = _parse('{"verdict":"HODL","confidence":0.9,"side":"long"}')
    assert p["verdict"] == "PASS"


def test_parse_veto_regex_fallback_stays_pass():
    # JSON-decode failure path: the first-line regex only knows LONG/SHORT/
    # CLOSE. A VETO mention in unparseable text must NOT arm a veto — it
    # degrades to PASS exactly like any other malformed reply.
    p = _parse("VETO this trade\n{broken json")
    assert p["verdict"] == "PASS"


def test_parse_veto_no_side_derivation():
    # The side-derivation block maps LONG→long / SHORT→short only; a VETO
    # with an omitted side must stay None (a derived side would let a VETO
    # sneak a direction into downstream code).
    p = _parse('{"verdict":"VETO","confidence":0.7}')
    assert p["side"] is None


# ── route_verdict: never routed, never upgraded ─────────────────────────────

def _analysis(verdict="VETO", **over):
    a = {
        "id": "a-1", "perception_id": "p-1", "coin": "TEST",
        "verdict": verdict, "confidence": 0.82, "side": None,
        "entry_px": 1.0, "stop_px": 0.9, "tp_px": 1.1,
        "reasoning": "trap shape", "news_context": "", "news_risk": "none",
        "ai_down": False, "duelist_at_entry": None,
        "created_at": 0, "composite_score": 50.0,
        "momentum_burst_fired": True, "slow_burn_fired": True,
        "slow_burn_count": 3, "breakout_fired": True,
        "volume_spike_fired": True, "whale_signal": {"confidence": 0.9},
    }
    a.update(over)
    return a


def test_route_veto_is_noop_even_with_every_force_hint(monkeypatch):
    # Every structural-override hint armed: whale, composite, breakout,
    # slow-burn, sidestep. A PASS would route to maybe_execute here; a VETO
    # must NOT — it is categorically stronger than a hedged PASS.
    from hermes_trader.agents import executor

    calls = []
    monkeypatch.setattr(executor, "maybe_execute",
                        lambda a: calls.append(a) or {"executed": False})
    monkeypatch.setattr(executor, "read_agent_config", lambda: {
        "force_execute_composite": 40,
        "whale_force_execute": True,
        "composite_force_execute": True,
        "breakout_force_execute": True,
        "ta_sidestep_force_execute": True,
    })
    out = executor.route_verdict(_analysis("VETO"))
    assert out["action"] == "none"
    assert out["verdict"] == "VETO"
    assert calls == []  # never reached the executor


def test_route_pass_still_routes_with_hints(monkeypatch):
    # Control: the same hints DO route a PASS (the exclusion is VETO-only).
    from hermes_trader.agents import executor

    calls = []
    monkeypatch.setattr(executor, "maybe_execute",
                        lambda a: calls.append(a) or {"executed": False})
    monkeypatch.setattr(executor, "read_agent_config", lambda: {
        "force_execute_composite": 40,
        "whale_force_execute": True,
        "composite_force_execute": False,
        "breakout_force_execute": False,
    })
    out = executor.route_verdict(_analysis("PASS"))
    assert out["action"] == "execute"
    assert len(calls) == 1


def test_route_veto_attaches_shadow_forecasts(monkeypatch):
    # VETO rides the PASS no-action path's forecast attach (Trade result
    # line keeps its chronos/timesfm columns). Stub both sync readers.
    from hermes_trader.agents import executor

    class _Sig:
        median_pct = 0.4
        error = None

    monkeypatch.setattr("hermes_trader.agents.chronos_signal.get_chronos_signal_sync",
                        lambda c, s: _Sig())
    monkeypatch.setattr("hermes_trader.agents.timesfm_signal.get_timesfm_signal_sync",
                        lambda c, s: _Sig())
    monkeypatch.setattr(executor, "read_agent_config", lambda: {})
    out = executor.route_verdict(_analysis("VETO"))
    assert out["action"] == "none"
    assert out["verdict"] == "VETO"  # real token, not laundered into PASS
    # Forecast fields ride the top level of the result dict (PASS shape).
    assert out["chronos_median_pct"] == 0.4
    assert out["timesfm_median_pct"] == 0.4


# ── maybe_execute top-level guard (direct-call hole) ────────────────────────

def test_maybe_execute_refuses_veto_direct(monkeypatch):
    # server.py /execute and the MCP tool call maybe_execute with any
    # remembered analysis, bypassing route_verdict. Without the guard the
    # VETO falls through every `verdict == "PASS"` equality check to
    # trade_side = side or "long" → a blind LONG on an active rejection.
    from hermes_trader.agents import executor

    def _boom(*a, **k):
        raise AssertionError("VETO reached downstream execution code")

    monkeypatch.setattr(executor, "read_agent_config", lambda: {"mode": "LIVE"})
    monkeypatch.setattr(executor, "_get_market_volume_24h", _boom)
    out = executor.maybe_execute(_analysis("VETO"))
    assert out["executed"] is False
    assert "ai_veto" in (out.get("reason") or "")


def test_maybe_execute_veto_guard_precedes_mode_checks_after_off(monkeypatch):
    # mode OFF short-circuits first (unchanged behavior); the VETO guard is
    # the FIRST thing after it.
    from hermes_trader.agents import executor

    monkeypatch.setattr(executor, "read_agent_config", lambda: {"mode": "OFF"})
    out = executor.maybe_execute(_analysis("VETO"))
    assert out["reason"] == "mode_off"
