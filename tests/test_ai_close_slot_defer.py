"""ai_close slot-pressure jurisdiction guard (2026-10-05, scan #5 follow-up).

The LLM's ai_close dead-bag cohort cost ~$3.06 vs letting the deterministic
stale-flat machinery run (n=15 replay): every cut fired at 1-of-5 slots where
the "free the capital" rationale is false. The guard gives the LLM the same
jurisdiction limit the code-level stale-flat timeout already respects
(stale_flat_min_positions contention floor): a CLOSE for a NEVER-ARMED
position on an UNCONTENDED book is deferred (shadow: logged only).

route_verdict is pure routing with injected side effects, so the guard is
tested end-to-end through it.
"""
from __future__ import annotations

import pytest

from hermes_trader.agents import executor
from hermes_trader.agents.dsl_exit import DSLTracker, _active_positions


BASE_CFG = {
    "max_concurrent": 5,
    "dsl_exit": {"stale_flat_min_positions": 3},
    "ai_close_slot_defer": {"enabled": True, "shadow_mode": True},
}


@pytest.fixture
def cfg(monkeypatch):
    def _set(**over):
        c = {k: dict(v) if isinstance(v, dict) else v for k, v in BASE_CFG.items()}
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(c.get(k), dict):
                c[k].update(v)
            else:
                c[k] = v
        monkeypatch.setattr(executor, "read_agent_config", lambda: c)
    return _set


@pytest.fixture(autouse=True)
def clear_book():
    _active_positions.clear()
    yield
    _active_positions.clear()


def _tracker(coin, entry=100.0, peak=100.0, side="long", protect=1.5):
    from hermes_trader.agents.dsl_exit import ExitPolicy
    t = DSLTracker(coin, side, entry, entry_time=0.0,
                   policy=ExitPolicy(protect_pct=protect))
    t.peak_px = peak
    return t


def _close_analysis(coin="FOO"):
    return {"verdict": "CLOSE", "coin": coin, "reasoning": "dead bag"}


def _calls(close_fn_rec):
    return close_fn_rec


def route(coin="FOO"):
    calls = []
    res = executor.route_verdict(
        _close_analysis(coin), close_fn=lambda c, r="": calls.append(c) or {"ok": True})
    return res, calls


def test_shadow_logs_but_still_closes(cfg):
    cfg()
    _active_positions["FOO_long"] = _tracker("FOO", peak=100.2)  # never armed
    res, calls = route()
    assert res["action"] == "close"
    assert calls == ["FOO"]


def test_enforced_defers_never_armed_uncontended(cfg):
    cfg(ai_close_slot_defer={"shadow_mode": False})
    _active_positions["FOO_long"] = _tracker("FOO", peak=100.2)
    res, calls = route()
    assert res["action"] == "none"
    assert "deferred_reason" in res
    assert calls == []  # close never ran


def test_armed_position_closes_even_when_enforced(cfg):
    cfg(ai_close_slot_defer={"shadow_mode": False})
    _active_positions["FOO_long"] = _tracker("FOO", peak=102.0)  # armed (>=1.5%)
    res, calls = route()
    assert res["action"] == "close"
    assert calls == ["FOO"]


def test_contended_book_closes_even_when_enforced(cfg):
    cfg(ai_close_slot_defer={"shadow_mode": False})
    _active_positions["FOO_long"] = _tracker("FOO", peak=100.1)
    _active_positions["A_long"] = _tracker("A")
    _active_positions["B_long"] = _tracker("B")  # 3 of 5 -> contention floor
    res, calls = route()
    assert res["action"] == "close"
    assert calls == ["FOO"]


def test_no_tracker_closes_normally(cfg):
    cfg(ai_close_slot_defer={"shadow_mode": False})
    res, calls = route()  # book empty, no tracker
    assert res["action"] == "close"
    assert calls == ["FOO"]


def test_guard_disabled(cfg):
    cfg(ai_close_slot_defer={"enabled": False, "shadow_mode": False})
    _active_positions["FOO_long"] = _tracker("FOO", peak=100.0)
    res, calls = route()
    assert res["action"] == "close"
    assert calls == ["FOO"]


def test_short_never_armed_defers(cfg):
    cfg(ai_close_slot_defer={"shadow_mode": False})
    # short: peak BELOW entry means profit; peak==entry -> 0% < protect
    _active_positions["FOO_short"] = _tracker("FOO", side="short", peak=100.0)
    res, calls = route()
    assert res["action"] == "none"
    assert calls == []


def test_fail_safe_on_config_read_failure(monkeypatch):
    # config read blowing up must NOT block a close (fail-safe -> close)
    def boom():
        raise RuntimeError("config unreadable")
    monkeypatch.setattr(executor, "read_agent_config", boom)
    _active_positions["FOO_long"] = _tracker("FOO", peak=100.0)
    res, calls = route()
    assert res["action"] == "close"
    assert calls == ["FOO"]
