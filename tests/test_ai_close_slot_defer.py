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


def _tracker(coin, entry=100.0, peak=100.0, side="long", protect=1.5,
             mark_peak=None):
    from hermes_trader.agents.dsl_exit import ExitPolicy
    t = DSLTracker(coin, side, entry, entry_time=0.0,
                   policy=ExitPolicy(protect_pct=protect))
    t.peak_px = peak
    # mark_peak_px = best SAMPLED mark (what the guard tests); defaults to
    # peak so existing peak-based tests keep their intent.
    t.mark_peak_px = peak if mark_peak is None else mark_peak
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


def test_wick_armed_but_mark_never_armed_defers(cfg):
    """CHIP 2026-10-06 seam: candle-ratcheted peak_px >= protect (wick) while
    the best sampled mark stayed below protect — the phase-2 floor never
    armed, so the guard must treat the position as never-armed and defer."""
    cfg(ai_close_slot_defer={"shadow_mode": False})
    # protect 1.0: wick to exactly +1.0% (peak_px), best mark only +0.77%
    _active_positions["FOO_long"] = _tracker(
        "FOO", peak=101.0, mark_peak=100.77, protect=1.0)
    res, calls = route()
    assert res["action"] == "none"
    assert "mark peak" in res["deferred_reason"]
    assert calls == []


def test_mark_armed_closes_even_when_enforced(cfg):
    """A sampled mark >= protect means the floor really armed — LLM
    jurisdiction stands even if mark_peak and peak_px agree."""
    cfg(ai_close_slot_defer={"shadow_mode": False})
    _active_positions["FOO_long"] = _tracker(
        "FOO", peak=101.2, mark_peak=101.0, protect=1.0)
    res, calls = route()
    assert res["action"] == "close"
    assert calls == ["FOO"]


def test_short_wick_seam_defers(cfg):
    cfg(ai_close_slot_defer={"shadow_mode": False})
    # short: wick down to -1.0% (peak_px=99.0) but best mark only 99.25
    _active_positions["FOO_short"] = _tracker(
        "FOO", side="short", peak=99.0, mark_peak=99.25, protect=1.0)
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


# ── B.37-A trend-release escape (2026-10-07, ADA die-trigger follow-up) ──

def _tracker_marked(coin, mark, **kw):
    t = _tracker(coin, **kw)
    t.last_mark_px = mark
    return t


def test_trend_release_disabled_by_default_defers_deep_bleeder(cfg):
    """No trend_release key = byte-identical pre-change guard: a deep
    bleeder still defers (the ADA shape stays deferred until the clause is
    armed in config)."""
    cfg(ai_close_slot_defer={"shadow_mode": False})
    _active_positions["FOO_long"] = _tracker_marked("FOO", 97.0)  # -3% mark
    res, calls = route()
    assert res["action"] == "none"
    assert calls == []


def test_trend_release_fires_on_deep_bleeder(cfg):
    """enabled + mark <= -depth: LLM jurisdiction restored, close proceeds."""
    cfg(ai_close_slot_defer={"shadow_mode": False,
                             "trend_release": {"enabled": True,
                                               "depth_pct": 2.0}})
    _active_positions["FOO_long"] = _tracker_marked("FOO", 97.5)  # -2.5%
    res, calls = route()
    assert res["action"] == "close"
    assert calls == ["FOO"]


def test_trend_release_shallow_bag_still_defers(cfg):
    """Mark inside the depth bar (-1.0% > -2.0%): guard keeps ownership —
    the ZRO-type shallow drifter protection is untouched."""
    cfg(ai_close_slot_defer={"shadow_mode": False,
                             "trend_release": {"enabled": True,
                                               "depth_pct": 2.0}})
    _active_positions["FOO_long"] = _tracker_marked("FOO", 99.0)  # -1.0%
    res, calls = route()
    assert res["action"] == "none"
    assert calls == []


def test_trend_release_short_side(cfg):
    cfg(ai_close_slot_defer={"shadow_mode": False,
                             "trend_release": {"enabled": True,
                                               "depth_pct": 2.0}})
    # short bleeding = mark ABOVE entry
    _active_positions["FOO_short"] = _tracker_marked(
        "FOO", 102.5, side="short")
    res, calls = route()
    assert res["action"] == "close"
    assert calls == ["FOO"]


def test_trend_release_no_fresh_mark_fails_safe(cfg):
    """last_mark_px never set (tracker never checked) -> clause inert,
    guard stands."""
    cfg(ai_close_slot_defer={"shadow_mode": False,
                             "trend_release": {"enabled": True,
                                               "depth_pct": 2.0}})
    _active_positions["FOO_long"] = _tracker("FOO", peak=100.0)
    res, calls = route()
    assert res["action"] == "none"
    assert calls == []


def test_trend_release_depth_zero_inert(cfg):
    cfg(ai_close_slot_defer={"shadow_mode": False,
                             "trend_release": {"enabled": True,
                                               "depth_pct": 0}})
    _active_positions["FOO_long"] = _tracker_marked("FOO", 95.0)
    res, calls = route()
    assert res["action"] == "none"
    assert calls == []
