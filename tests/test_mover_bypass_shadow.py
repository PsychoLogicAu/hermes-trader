"""Tests for the runner gate's mover-bypass shadow (kanban t_b7876025).

`runner_entry_gate.mover_bypass_shadow_mode: true` turns the
structured_daily_mover admission into a shadow accrual: the mover-lane entry
is blocked with the byte-identical "needs volume+breakout/burst and
structure" reason (parsers keep working) while a loud
`[gate][SHADOW] mover_bypass WOULD HAVE ADMITTED` line accrues the
would-admit cohort. Key absent/false = live admission, byte-identical.
Fires only where the mover term is DECISIVE (not fresh, not bypassed):
fresh-impulse admits and the late-chase bypass lane are untouched.
"""

import logging

from hermes_trader.agents import executor


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
    """dailyMover long, no fresh impulse, no uptrend: admission hinges
    ENTIRELY on structured_daily_mover (conf>=0.72, comp>=30, volume
    structure)."""
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


def _shadow_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if "mover_bypass WOULD HAVE ADMITTED" in r.getMessage()]


def test_key_absent_mover_admitted_byte_identical():
    # live admission path unchanged when the key is absent
    assert executor._runner_entry_block_reason(_mover_long(), _cfg()) == ""


def test_shadow_false_mover_admitted():
    cfg = _cfg(mover_bypass_shadow_mode=False)
    assert executor._runner_entry_block_reason(_mover_long(), cfg) == ""


def test_shadow_true_blocks_with_identical_reason_and_accrues(caplog):
    cfg = _cfg(mover_bypass_shadow_mode=True)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(_mover_long(), cfg)
    assert reason == BLOCK_REASON  # byte-identical to the no-admit block
    lines = _shadow_lines(caplog)
    assert len(lines) == 1
    assert "mover_bypass WOULD HAVE ADMITTED TEST LONG: conf 0.75 >= bar 0.72" in lines[0]
    assert "score 35" in lines[0]


def test_shadow_true_fresh_impulse_untouched(caplog):
    # volume_spike + breakout -> fresh_impulse: admitted on the fresh path,
    # the mover term is not decisive, no accrual line.
    a = _mover_long(breakout_fired=True)
    cfg = _cfg(mover_bypass_shadow_mode=True)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        assert executor._runner_entry_block_reason(a, cfg) == ""
    assert _shadow_lines(caplog) == []


def test_shadow_true_sub_mover_bar_no_accrual(caplog):
    # conf below the mover bar: the mover term never fires, the ordinary
    # block stands byte-identically and NOTHING accrues.
    a = _mover_long(confidence=0.70)
    cfg = _cfg(mover_bypass_shadow_mode=True)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(a, cfg)
    assert reason == BLOCK_REASON
    assert _shadow_lines(caplog) == []


def test_shadow_true_shorts_untouched(caplog):
    # shorts exit the gate earlier (structured_short path); the mover
    # accrual is long-only.
    a = _mover_long(side="short", downtrend_momentum_fired=True)
    cfg = _cfg(mover_bypass_shadow_mode=True)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        assert executor._runner_entry_block_reason(a, cfg) == ""
    assert _shadow_lines(caplog) == []


def test_shadow_true_bypassed_mover_long_no_double_accrual(caplog):
    # uptrend mover long past the late-chase bar is admitted by the BYPASS
    # (bypassed=True) — the mover term is not decisive, no accrual here
    # (the late-chase lane's own shadow/bypass machinery owns that cohort).
    a = _mover_long(uptrend_momentum_fired=True)
    cfg = _cfg(mover_bypass_shadow_mode=True,
               bypass_late_trend_chase=True,
               bypass_late_trend_chase_min_conf=0.65,
               late_chase_tape_gate=False)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        assert executor._runner_entry_block_reason(a, cfg) == ""
    assert _shadow_lines(caplog) == []
