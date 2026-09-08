"""Contention gate for the stale-flat timeout (port of upstream 43fcd9581781,
LIT post-mortem): the timeout's only purpose is freeing capital for QUEUED
candidates, which exist when the book is full. Below `stale_flat_min_positions`
(default 3) active trackers a stale-flat-eligible position keeps its
protect/stop/hard-timeout exits and rides. 0 restores the old unconditional
behavior. Hermetic: in-memory trackers, `_active_positions` manipulated
directly, no network, no sleeps (elapsed is faked via entry_time).
"""

from __future__ import annotations

import time

import pytest

from hermes_trader.agents import dsl_exit as dx
from hermes_trader.agents.dsl_exit import DSLTracker, ExitPolicy


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    """Point the DSL state file at a temp path; empty registry per test."""
    monkeypatch.setattr(dx, "DSL_STATE_FILE", str(tmp_path / "dsl_state.json"))
    with dx._registry_lock:
        dx._active_positions.clear()
    dx._loaded_from_disk = False
    yield
    with dx._registry_lock:
        dx._active_positions.clear()
    dx._loaded_from_disk = False


def _stale_pol(min_positions: int = 3, **kw) -> ExitPolicy:
    """500min-old positions breach the 480min stale window; 10_000min hard
    timeout and 20% loss caps stay out of the way unless a test wants them."""
    base: dict = dict(max_loss_pct=20.0, max_loss_roe_pct=20.0, protect_pct=1.25,
                      retrace_threshold=0.3, hard_timeout_minutes=10_000.0,
                      stale_flat_timeout_minutes=480.0,
                      stale_flat_min_positions=min_positions)
    base.update(kw)
    return ExitPolicy(**base)


def _stale_tracker(coin: str, pol: ExitPolicy, minutes_ago: float = 500.0) -> DSLTracker:
    """A long that never peaked: entry == peak, mark just below protect."""
    return DSLTracker(coin, "long", 100.0, time.time() - minutes_ago * 60, pol,
                      leverage=1)


def _fill_book(n: int, pol: ExitPolicy) -> None:
    for i in range(n):
        c = f"BK{i}"
        dx._active_positions[f"{c}_long"] = _stale_tracker(c, pol)


def test_single_tracker_below_floor_rides():
    """1 active tracker < floor 3: the stale-flat branch is suppressed and the
    position rides (no exit)."""
    pol = _stale_pol()
    _fill_book(1, pol)
    v = dx._active_positions["BK0_long"].check(100.1)
    assert not v.exit, v.reason
    assert "stale_flat" not in (v.reason or "")


def test_contended_book_fires_stale_flat():
    """3 active trackers >= floor 3: the same eligible position exits via
    stale_flat_timeout."""
    pol = _stale_pol()
    _fill_book(3, pol)
    v = dx._active_positions["BK0_long"].check(100.1)
    assert v.exit and "stale_flat" in v.reason, v.reason


def test_min_positions_zero_restores_unconditional():
    """stale_flat_min_positions=0: fires even with a single tracker (old
    pre-gate behavior)."""
    pol = _stale_pol(min_positions=0)
    _fill_book(1, pol)
    v = dx._active_positions["BK0_long"].check(100.1)
    assert v.exit and "stale_flat" in v.reason, v.reason


def test_gate_suppresses_only_stale_flat_branch():
    """With a book below the floor, the other exits still fire: max_loss on a
    breach and hard_timeout on an over-ager."""
    # max_loss: 3% spot loss >= 2% cap, 1 tracker only.
    pol = _stale_pol(max_loss_pct=2.0)
    _fill_book(1, pol)
    v = dx._active_positions["BK0_long"].check(97.0)
    assert v.exit and "max_loss" in v.reason, v.reason
    # hard_timeout: elapsed 500min >= 300min cap, 1 tracker only.
    pol = _stale_pol(hard_timeout_minutes=300.0)
    _fill_book(1, pol)
    v = dx._active_positions["BK0_long"].check(100.1)
    assert v.exit and "hard_timeout" in v.reason, v.reason


def test_min_positions_survives_state_roundtrip():
    """_tracker_to_dict/_tracker_from_dict carry the new key; a legacy state
    record without it rehydrates to the default 3."""
    pol = _stale_pol(min_positions=5)
    t = _stale_tracker("RT", pol)
    t2 = dx._tracker_from_dict(dx._tracker_to_dict(t))
    assert t2.policy.stale_flat_min_positions == 5
    legacy = dx._tracker_to_dict(t)
    legacy["policy"].pop("stale_flat_min_positions")
    t3 = dx._tracker_from_dict(legacy)
    assert t3.policy.stale_flat_min_positions == 3


def test_policy_from_config_threads_min_positions(monkeypatch):
    """_policy_from_config reads the key out of the dsl_exit config block:
    explicit value, absent -> default 3, explicit 0 preserved."""
    import hermes_trader.agents.config_store as cs
    monkeypatch.setattr(cs, "read_agent_config",
                        lambda: {"dsl_exit": {"stale_flat_min_positions": 7}})
    assert dx._policy_from_config().stale_flat_min_positions == 7
    monkeypatch.setattr(cs, "read_agent_config", lambda: {"dsl_exit": {}})
    assert dx._policy_from_config().stale_flat_min_positions == 3
    monkeypatch.setattr(cs, "read_agent_config",
                        lambda: {"dsl_exit": {"stale_flat_min_positions": 0}})
    assert dx._policy_from_config().stale_flat_min_positions == 0
