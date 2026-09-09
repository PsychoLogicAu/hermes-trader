"""Tests for momentum-continuation re-entry (loss-cooldown bypass). Pure fn:
returns a MomentumReentryDecision (allowed / suppressed / no-opinion)."""

from hermes_trader.agents.executor import momentum_reentry_allowed

ON = {"momentum_reentry": {"enabled": True, "reclaim_pct": 1.0,
                           "min_composite": 30}}


def test_disabled():
    cfg = {"momentum_reentry": {"enabled": False}}
    d1 = momentum_reentry_allowed(100, "long", 105, 50, cfg)
    assert d1.is_allowed() is False and d1.suppressed is False
    d2 = momentum_reentry_allowed(100, "long", 105, 50, {})
    assert d2.is_allowed() is False and d2.suppressed is False


def test_reclaim_above_stop_strong_composite_allows():
    d = momentum_reentry_allowed(100.0, "long", 101.5, 40, ON)  # +1.5% > 1%, comp 40
    assert d.is_allowed() and "reclaimed" in d.reason
    assert d.suppressed is False


def test_no_reclaim_blocks():
    # price still below the stop (falling knife) -> stay in cooldown
    d = momentum_reentry_allowed(100.0, "long", 98.0, 50, ON)
    assert d.is_allowed() is False and d.suppressed is False


def test_reclaim_below_margin_blocks():
    # only +0.5% above stop, < 1% reclaim margin -> not a real reversal
    d = momentum_reentry_allowed(100.0, "long", 100.5, 50, ON)
    assert d.is_allowed() is False and d.suppressed is False


def test_weak_composite_blocks():
    # reclaimed but composite too weak -> not strong enough momentum
    d = momentum_reentry_allowed(100.0, "long", 102.0, 20, ON)
    assert d.is_allowed() is False and d.suppressed is False


def test_short_side_never():
    d = momentum_reentry_allowed(100.0, "short", 102.0, 50, ON)
    assert d.is_allowed() is False and d.suppressed is False


def test_bad_inputs_safe():
    # (None, bad mid), (100, 0), ("x", bad px) — each must fail closed.
    assert momentum_reentry_allowed(None, "long", 102, 50, ON).is_allowed() is False
    assert momentum_reentry_allowed(100, "long", 0, 50, ON).is_allowed() is False
    assert momentum_reentry_allowed("x", "long", 102, 50, ON).is_allowed() is False
    assert momentum_reentry_allowed(None, "long", 102, 50, ON).suppressed is False
    assert momentum_reentry_allowed(100, "long", 0, 50, ON).suppressed is False
    assert momentum_reentry_allowed("x", "long", 102, 50, ON).suppressed is False
