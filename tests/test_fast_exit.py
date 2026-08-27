"""Fast-exit feature tests: ratchet_peak + executor.fast_exit_pass.

Covers the 2026-08-26 GRASS incident class: a long whose price spiked above
the DSL protect threshold inside a window the ~3-min main-loop cadence never
sampled. The mid-only peak (updated only in check()) registered +0.73% while
the true 5m spike was +1.37%, phase-2's floor never ratcheted, and the
position rode out to the stale-flat timeout at a loss.

Two mitigations under test here:
  * DSLTracker.ratchet_peak — monotonic peak ratchet from a candle's
    high (longs) / low (shorts), so an intrabar wick can no longer be missed.
  * executor.fast_exit_pass — the lighter-cadence pass (run by the
    hermes-fast-exit daemon every ~15s) that ratchets peaks from fresh 1m
    candles and re-checks the DSL floors on a fresh mid.

How the close actually fires (two ticks, GRASS-shaped):
  1. A tick whose fresh mid is >= protect_pct: check() computes the phase-2
     floor from the RATCHETED peak (candle high), and the never-decrease
     floor ratchet stores it in _last_floor.
  2. The next tick after the fade: the mark is back below protect, the
     fresh-computed floor would relax to the phase-1 stop, but the ratchet
     keeps it at the phase-2 level -> mark < floor -> floor_breach exit.

scripts/trading_loop.py itself is NOT importable (top-level `while True:`
main loop; see test_cooldown_research_skip.py), so the daemon wiring is
verified statically, matching the repo's existing convention.

All network is monkeypatched; DSL_STATE_FILE is pointed at tmp so no live
state file is ever written.
"""

from __future__ import annotations

import re
import time
import types

import pytest

from hermes_trader.agents import dsl_exit, executor
from hermes_trader.agents.dsl_exit import DSLTracker, ExitPolicy, RetraceTier


def _policy(**kw) -> ExitPolicy:
    base = dict(
        max_loss_pct=5.0, max_loss_roe_pct=100.0, protect_pct=1.0,
        retrace_threshold=0.25, hard_timeout_minutes=99999.0,
        phase2_tiers=[RetraceTier(0.0, 0.25)],
    )
    base.update(kw)
    return ExitPolicy(**base)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Point the DSL state file at a temp path; clean registry per test."""
    monkeypatch.setattr(dsl_exit, "DSL_STATE_FILE", str(tmp_path / "dsl_state.json"))
    with dsl_exit._registry_lock:
        dsl_exit._active_positions.clear()
    dsl_exit._loaded_from_disk = False
    yield
    with dsl_exit._registry_lock:
        dsl_exit._active_positions.clear()
    dsl_exit._loaded_from_disk = False


# ── ratchet_peak: pure tracker behaviour ─────────────────────────────────────

def test_ratchet_long_arms_from_candle_high():
    t = DSLTracker("SPK", "long", 100.0, time.time(), _policy())
    assert t.peak_px == 100.0
    # A mid BELOW the peak does not move it.
    assert t.ratchet_peak(99.5, 99.9) is False
    assert t.peak_px == 100.0
    # A candle high above the peak ratchets it up (the GRASS wick case).
    assert t.ratchet_peak(101.5, 100.9) is True
    assert t.peak_px == 101.5
    # Lower highs never ratchet backwards.
    assert t.ratchet_peak(101.0, 100.5) is False
    assert t.peak_px == 101.5
    # A mid print above the peak also counts as a peak candidate.
    assert t.ratchet_peak(102.0, 101.0) is True
    assert t.peak_px == 102.0
    # For a short, highs are irrelevant — only lows ratchet. Sane candle
    # (hi>lo) whose low is still above the peak -> no move.
    s = DSLTracker("SPK2", "short", 100.0, time.time(), _policy())
    assert s.ratchet_peak(100.8, 100.1) is False
    assert s.peak_px == 100.0
    # A lower low ratchets it (and proves hi is ignored: hi=100.5 would be a
    # no-op source; the move comes from lo=99.7).
    assert s.ratchet_peak(100.5, 99.7) is True
    assert s.peak_px == 99.7


def test_ratchet_short_arms_from_candle_low():
    t = DSLTracker("SPK", "short", 100.0, time.time(), _policy())
    assert t.peak_px == 100.0
    # A candle low below the peak ratchets it down (better = lower for shorts).
    assert t.ratchet_peak(98.0, 98.2) is True
    assert t.peak_px == 98.2
    # Higher lows never ratchet backwards.
    assert t.ratchet_peak(99.0, 98.5) is False
    assert t.peak_px == 98.2
    # lo alone (hi=None) still ratchets.
    assert t.ratchet_peak(None, 97.8) is True
    assert t.peak_px == 97.8


def test_ratchet_no_op_returns_false():
    t = DSLTracker("NOP", "long", 100.0, time.time(), _policy())
    assert t.ratchet_peak(None, None) is False
    assert t.peak_px == 100.0


def test_ratchet_persists_state_on_change():
    # A registered (in-registry) tracker persists the ratcheted peak so a
    # daemon restart keeps the armed floor.
    dsl_exit.register_position(
        coin="PST", side="long", entry_px=100.0, policy=_policy(),
    )
    t = dsl_exit._active_positions["PST_long"]
    assert t.ratchet_peak(101.0, 100.5) is True
    dsl_exit.load_state(force=True)
    revived = dsl_exit._active_positions.get("PST_long")
    assert revived is not None
    assert revived.peak_px == 101.0


# ── fast_exit_pass: end-to-end with stubbed network ──────────────────────────

def _candle(h, l):
    return types.SimpleNamespace(h=h, l=l)


def test_fast_exit_pass_catches_unsampled_spike_and_closes(monkeypatch):
    """GRASS-shaped, two ticks.

    Tick 1 (the spike): fresh mid 101.2 is above protect (101.0); the 1m
    candles peaked at 101.5 — a wick the main-loop mid never saw. ratchet_peak
    lifts peak to 101.5 and check() computes the phase-2 floor
    entry + (101.5-100)*(1-0.25) = 101.125; mark 101.2 is just above it ->
    hold, but the never-decrease ratchet stores 101.125.

    Tick 2 (the fade): mid back to 98.0. Fresh-computed floor would relax to
    the phase-1 stop (95.0), but the ratchet keeps it at 101.125 ->
    mark < floor -> floor_breach exit. Without the candle ratchet the peak
    would have stayed at 101.2 (the best sampled mid) and the floor at
    100.94 — the close still fires, but a LOWER/looser floor; the ratchet is
    what arms it off the TRUE extreme.
    """
    dsl_exit.register_position(
        coin="GRASS", side="long", entry_px=100.0, leverage=3,
        entry_atr_pct=1.0, policy=_policy(),
    )
    monkeypatch.setattr(executor, "fetch_hl_candles", lambda *a, **k: [
        _candle(100.2, 99.8), _candle(101.5, 100.0), _candle(101.3, 100.9),
    ])

    # Tick 1: mid inside the spike, above protect.
    monkeypatch.setattr(executor, "get_hl_price", lambda coin: 101.2)
    exits = executor.fast_exit_pass()
    assert exits == []  # still above the phase-2 floor
    tracker = dsl_exit._active_positions.get("GRASS_long")
    assert tracker is not None
    # Peak ratcheted to the candle high, not the sampled mid.
    assert tracker.peak_px == 101.5

    # Tick 2: fade back below entry — the ratcheted floor bites.
    monkeypatch.setattr(executor, "get_hl_price", lambda coin: 98.0)
    exits = executor.fast_exit_pass()
    assert len(exits) == 1
    ex = exits[0]
    assert ex["coin"] == "GRASS"
    assert ex["side"] == "long"
    assert "floor_breach" in ex["reason"]
    assert tracker.peak_px == 101.5
    # Leveraged pct reflects 3x on the -2.0% spot move.
    assert ex["unrealized_pct"] == pytest.approx(-2.0)
    assert ex["leveraged_pct"] == pytest.approx(-6.0)


def test_fast_exit_pass_no_fire_below_protect(monkeypatch):
    """No spike: peak stays below entry+protect, only the phase-1 max-loss
    floor is live, and the mid is above it -> no exit. Proves the pass
    doesn't over-close."""
    dsl_exit.register_position(
        coin="CALM", side="long", entry_px=100.0, leverage=3,
        entry_atr_pct=1.0, policy=_policy(),
    )
    monkeypatch.setattr(executor, "get_hl_price", lambda coin: 99.0)
    # Candles never exceed the sampled mid — nothing to ratchet to.
    monkeypatch.setattr(executor, "fetch_hl_candles", lambda *a, **k: [
        _candle(99.4, 98.8), _candle(99.1, 98.7),
    ])
    exits = executor.fast_exit_pass()
    assert exits == []
    tracker = dsl_exit._active_positions.get("CALM_long")
    assert tracker is not None
    # Long peaks only ratchet UP from entry: candle highs (99.4) below the
    # 100.0 entry never move it, so peak stays at entry and no phase-2
    # floor exists; the -1.0% mid is above the max-loss floor (95.0) -> hold.
    assert tracker.peak_px == 100.0


def test_fast_exit_pass_no_positions_is_noop(monkeypatch):
    # Empty registry -> no fetches at all, empty result.
    monkeypatch.setattr(
        executor, "get_hl_price",
        lambda coin: (_ for _ in ()).throw(AssertionError("no network expected")),
    )
    assert executor.fast_exit_pass() == []


def test_fast_exit_pass_tolerates_fetch_failure(monkeypatch):
    """A transient network failure skips the coin this tick, never raises."""
    dsl_exit.register_position(
        coin="FLAKY", side="long", entry_px=100.0, leverage=1,
        entry_atr_pct=1.0, policy=_policy(),
    )
    monkeypatch.setattr(
        executor, "get_hl_price",
        lambda coin: (_ for _ in ()).throw(ConnectionError("network down")),
    )
    monkeypatch.setattr(
        executor, "fetch_hl_candles",
        lambda *a, **k: (_ for _ in ()).throw(ConnectionError("network down")),
    )
    # No mid, no candles -> pass bails out without firing or raising.
    assert executor.fast_exit_pass() == []


# ── daemon wiring (static: trading_loop.py is not importable) ────────────────

def test_trading_loop_daemon_wiring():
    """The hermes-fast-exit daemon must be defined, started as a daemon
    thread, gated on config, and close through the idempotent path."""
    src = (
        __import__("pathlib").Path(executor.__file__)
        .parents[2] / "scripts" / "trading_loop.py"
    ).read_text()
    assert re.search(r"def _fast_exit_daemon", src), "daemon function missing"
    assert re.search(
        r"threading\.Thread\(target=_fast_exit_daemon, name=\"hermes-fast-exit\", daemon=True\)\.start\(\)",
        src,
    ), "daemon thread not started"
    assert re.search(r"fast_exit_interval_sec", src), "config gate missing"
    assert re.search(r"close_position_market\(coin, ex\[.reason.\]\)", src), (
        "daemon must close via the idempotent close_position_market path"
    )
    assert re.search(r"active_position_coins\(\)", src), (
        "daemon must idle (in-memory check) when no position is open"
    )