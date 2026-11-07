"""The SHADOW DEFAULT of abstention_veto_gate is the fence (t_3348d1d3).

`tests/test_abstention_veto_gate.py` pins the shadow behaviour thoroughly but
always passes `shadow_mode: True` EXPLICITLY (CFG_SHADOW at the top of that
file), and every fail-open test there stops at `enabled: False` before the
shadow branch is reached. So nothing in the suite pinned the shipped posture
itself: `shadow_mode` ABSENT — the state a config block lands in when an
operator arms the gate with `{"enabled": true}` and forgets the rest — must
still be shadow.

Mutation-checked 2026-11-07 (board t_3348d1d3): flipping the default in
`abstention_veto_gate` from `cfg.get("shadow_mode", True)` to `... False)`
left all 31 tests in that file GREEN while the gate started returning
`pass: False` into `block_reasons` — i.e. an armed-with-defaults config would
have cut real entries. These two pins are what make that edit fail.

No live flip is implied: the shipped posture is shadow-only and the LIVE
`.agent-config.json` carries no `abstention_veto_gate` block at all, so the
gate is inert in the running container as well.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import risk_gates  # noqa: E402
from hermes_trader.agents.risk_gates import GateContext  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_flow_cap():
    """Module state keyed by UTC date — clear it so test order can never
    decide whether the per-day cap has been reached."""
    risk_gates._abstention_veto_blocks.clear()
    yield
    risk_gates._abstention_veto_blocks.clear()


def _ctx() -> GateContext:
    return GateContext(
        confidence=0.70, current_positions=[], trade_notional_usd=10.0,
        daily_pnl=0.0, market_volume_24h_usd=1_000_000_000.0, coin="TEST",
        trade_side="long", has_binary_news_risk=False, equity=100.0,
        total_open_notional=0.0, composite_score=40.0,
        duelist_verdict="PASS", duelist_confidence=0.45,
    )


# An ARMED block with no `shadow_mode` key — the operator-forgets-the-rest
# shape, and exactly what an accrual-only block looks like in the repo's
# house style (the other shadow gates are armed the same way).
CFG_ARMED_DEFAULT = {"enabled": True, "max_confidence": 0.55,
                     "escape_min_conf": 0.78, "escape_min_composite": 60}

_BASE_CFG = {
    "min_ai_confidence": 0.5, "max_concurrent": 5,
    "max_trade_notional_usd": 300, "min_market_volume_usd": 1_000_000,
    "coin_allowlist": [], "coin_blocklist": [], "cooldown_min": 0,
    "max_crypto_long_correlated": 2, "max_total_notional_pct": 1.0,
    "counter_regime_min_conf": 0.0, "block_counter_trend_bypass": False,
}


def test_shadow_mode_absent_still_would_block_not_block():
    r = risk_gates.abstention_veto_gate(_ctx(), CFG_ARMED_DEFAULT)
    assert r["pass"] is True, r
    assert r.get("shadow_would_block") is True, r
    assert "abstention_veto" in r.get("reason", ""), r


def test_default_shadow_never_reaches_block_reasons():
    out = risk_gates.eval_all_gates(
        _ctx(), dict(_BASE_CFG, abstention_veto_gate=dict(CFG_ARMED_DEFAULT)),
        last_trade_time=None)
    r = out["results"]["abstention_veto"]
    assert r["pass"] is True and r.get("shadow_would_block") is True, r
    assert out["blocked"] is False, out["block_reasons"]
    assert not any("abstention_veto" in b for b in out["block_reasons"]), out["block_reasons"]
