"""B.37-B deep-negative stale-flat SHADOW clause (2026-10-07).

The contention-floor gate (stale_flat_min_positions) means an uncontended
never-armed deep bleeder rides to the max_loss stop (the ADA 2026-10-06
shape). This clause LOGS what a deep-negative stale-flat cut would have done
— shadow accrual only, never exits (replay n=36 net-negative at every depth;
clause A went live instead, B accrues for the revisit).

Tests pin: fires only when enabled+deep+aged+uncontended+never-armed, and
NEVER returns an exit verdict.
"""
from __future__ import annotations

import time

from hermes_trader.agents.dsl_exit import (DSLTracker, ExitPolicy,
                                           _active_positions)


def _policy(deep=None, stale_min=3, stale_min_age=240.0):
    return ExitPolicy(protect_pct=1.0, max_loss_pct=5.0,
                      hard_timeout_minutes=1800.0,
                      stale_flat_timeout_minutes=stale_min_age,
                      stale_flat_min_positions=stale_min,
                      stale_flat_deep_release=deep if deep is not None else {})


def _tracker(policy, entry=100.0, age_min=300.0, side="long"):
    t = DSLTracker("FOO", side, entry,
                   entry_time=time.time() - age_min * 60, policy=policy)
    return t


def _check(caplog, mark, t):
    v = t.check(mark)
    fired = any("stale_flat_deep_release WOULD HAVE EXITED" in r.getMessage()
                for r in caplog.records)
    return v, fired


def test_shadow_fires_on_deep_bleeder_uncontended(caplog):
    _active_positions.clear()
    _active_positions["FOO_long"] = _tracker(
        _policy(deep={"enabled": True, "depth_pct": 2.0}))
    v, fired = _check(caplog, 97.0, _active_positions["FOO_long"])
    assert fired
    assert not v.exit  # shadow: NEVER exits
    _active_positions.clear()


def test_inert_without_config(caplog):
    _active_positions.clear()
    _active_positions["FOO_long"] = _tracker(_policy())
    v, fired = _check(caplog, 97.0, _active_positions["FOO_long"])
    assert not fired
    assert not v.exit
    _active_positions.clear()


def test_shallow_mark_does_not_fire(caplog):
    _active_positions.clear()
    _active_positions["FOO_long"] = _tracker(
        _policy(deep={"enabled": True, "depth_pct": 2.0}))
    v, fired = _check(caplog, 99.0, _active_positions["FOO_long"])
    assert not fired
    assert not v.exit
    _active_positions.clear()


def test_young_position_does_not_fire(caplog):
    _active_positions.clear()
    _active_positions["FOO_long"] = _tracker(
        _policy(deep={"enabled": True, "depth_pct": 2.0}), age_min=60.0)
    v, fired = _check(caplog, 97.0, _active_positions["FOO_long"])
    assert not fired
    assert not v.exit
    _active_positions.clear()


def test_contended_book_does_not_fire(caplog):
    """At/above the contention floor the REAL stale-flat timeout owns this
    case; the shadow clause must not double-log."""
    _active_positions.clear()
    pol = _policy(deep={"enabled": True, "depth_pct": 2.0})
    _active_positions["FOO_long"] = _tracker(pol)
    _active_positions["A_long"] = _tracker(pol)
    _active_positions["B_long"] = _tracker(pol)
    v, fired = _check(caplog, 97.0, _active_positions["FOO_long"])
    assert not fired          # shadow clause silent (contended)
    assert v.exit             # the real stale_flat timeout fires instead
    assert "stale_flat_timeout" in v.reason
    _active_positions.clear()


def test_armed_position_does_not_fire(caplog):
    _active_positions.clear()
    pol = _policy(deep={"enabled": True, "depth_pct": 2.0})
    t = _tracker(pol)
    t.check(101.5)            # mark arms phase-2 (>= protect 1.0)
    t.check(97.0)             # then bleeds deep
    v, fired = _check(caplog, 96.0, t)
    assert not fired
    _active_positions.clear()


def test_short_side_deep(caplog):
    _active_positions.clear()
    _active_positions["FOO_short"] = _tracker(
        _policy(deep={"enabled": True, "depth_pct": 2.0}), side="short")
    v, fired = _check(caplog, 102.5, _active_positions["FOO_short"])
    assert fired
    assert not v.exit
    _active_positions.clear()
