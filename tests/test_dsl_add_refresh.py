"""P0 bug (ALPHA-QUEUE, found 2026-07-13, fixed 2026-07-17; port of upstream
f94156259ec5): DSL tracker entry_px went stale on position ADDS.

Evidence (SKHY, manual short + add): fills avg entry 158.73, close 158.90 =
-0.1% spot realized. The tracker kept the FIRST fill entry (159.83), read
+0.64% spot / +6.44% ROE, showed a green win, and ran its profit floor off
the wrong basis. Symmetric risk: an add above tracked entry delays the stop
beyond intended. Fix: rehydrate_from_exchange tracks last-seen size and, on
any material size increase (>0.5%), refreshes entry_px to the exchange's
average and clamps peak_px to the new basis.
"""

from __future__ import annotations

import time

import pytest

from hermes_trader.agents import dsl_exit
from hermes_trader.agents.dsl_exit import DSLTracker, ExitPolicy


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


def _pos(coin, szi, entry, lev=5):
    return {"position": {"coin": coin, "szi": str(szi), "entryPx": str(entry),
                         "leverage": {"value": lev}}}


def _track(coin, side, entry, size, peak=None, lev=5):
    t = DSLTracker(coin, side, entry, time.time(), ExitPolicy(), leverage=lev)
    t.size = size
    if peak is not None:
        t.peak_px = peak
    with dsl_exit._registry_lock:
        dsl_exit._active_positions[f"{coin}_{side}"] = t
    return t


def test_skhy_replay_short_add_refreshes_entry_basis():
    # SKHY 2026-07-13 replay: short first fill @159.83; a manual add moved
    # the true average to 158.73. Old code kept 159.83 and read a +0.64%
    # phantom spot profit at the 158.90 close; the truth was -0.1%.
    t = _track("xyz:SKHY", "short", 159.83, 100.0)
    dsl_exit.rehydrate_from_exchange([_pos("xyz:SKHY", -200, 158.73)])
    assert t.entry_px == 158.73 and t.size == 200.0
    assert t.peak_px == 158.73          # short peak clamped to the new basis
    assert t._unrealized_pct(158.90) < 0   # the phantom win is gone


def test_long_add_above_entry_keeps_true_peak():
    t = _track("BTC", "long", 100.0, 10.0, peak=105.0)
    dsl_exit.rehydrate_from_exchange([_pos("BTC", 20, 102.0)])
    assert t.entry_px == 102.0 and t.size == 20.0
    assert t.peak_px == 105.0           # real price extreme survives


def test_long_add_with_peak_below_new_entry_clamps_peak():
    t = _track("ETH", "long", 100.0, 10.0, peak=100.5)
    dsl_exit.rehydrate_from_exchange([_pos("ETH", 20, 103.0)])
    assert t.peak_px == 103.0           # never a negative profit range


def test_short_add_with_peak_above_new_entry_clamps_peak():
    # Short mirror of the long clamp: an add whose average sits ABOVE the
    # tracked (lowest) peak must clamp the peak down to the new basis.
    t = _track("ETH", "short", 100.0, 10.0, peak=99.5)
    dsl_exit.rehydrate_from_exchange([_pos("ETH", -20, 97.0)])
    assert t.entry_px == 97.0 and t.size == 20.0
    assert t.peak_px == 97.0            # never a negative profit range


def test_partial_close_updates_size_not_entry():
    t = _track("SOL", "long", 50.0, 10.0, peak=60.0)
    dsl_exit.rehydrate_from_exchange([_pos("SOL", 5, 50.0)])
    assert t.size == 5.0 and t.entry_px == 50.0 and t.peak_px == 60.0


def test_legacy_unknown_size_adopted_without_entry_refresh():
    t = _track("DOGE", "long", 0.1, 0.0)   # size 0 = pre-fix state file
    dsl_exit.rehydrate_from_exchange([_pos("DOGE", 1000, 0.099)])
    assert t.size == 1000.0
    assert t.entry_px == 0.1               # no add proven — basis untouched


def test_unchanged_size_is_a_noop():
    t = _track("XRP", "short", 2.0, 100.0, peak=1.9)
    dsl_exit.rehydrate_from_exchange([_pos("XRP", -100, 2.0)])
    assert t.entry_px == 2.0 and t.peak_px == 1.9 and t.size == 100.0


def test_synthesized_tracker_knows_its_size():
    dsl_exit.rehydrate_from_exchange([_pos("NEW", 33, 1.5)])
    t = dsl_exit._active_positions["NEW_long"]
    assert t.size == 33.0 and t.entry_px == 1.5


def test_size_survives_state_roundtrip():
    t = DSLTracker("BTC", "long", 100.0, 123.0, ExitPolicy(), leverage=3)
    t.size = 42.5
    assert dsl_exit._tracker_from_dict(dsl_exit._tracker_to_dict(t)).size == 42.5


def test_legacy_state_dict_defaults_size_zero():
    d = dsl_exit._tracker_to_dict(DSLTracker("ETH", "short", 10.0, 1.0, ExitPolicy()))
    d.pop("size")
    assert dsl_exit._tracker_from_dict(d).size == 0.0


def test_legacy_state_file_loads_and_adopts_live_size(tmp_path):
    # Full legacy path: a state file written before the size field existed
    # must load as size 0, then the first rehydrate adopts the live size
    # silently (no entry-basis refresh, no error). The autouse fixture points
    # DSL_STATE_FILE at tmp_path, so just write the legacy payload there.
    legacy = dsl_exit._tracker_to_dict(
        DSLTracker("SOL", "long", 50.0, time.time(), ExitPolicy(), leverage=3))
    legacy.pop("size")
    import json
    (tmp_path / "dsl_state.json").write_text(json.dumps(
        {"version": 1, "saved_at": int(time.time() * 1000),
         "positions": [legacy]}))
    dsl_exit.load_state()
    t = dsl_exit._active_positions.get("SOL_long")
    assert t is not None and t.size == 0.0
    dsl_exit.rehydrate_from_exchange([_pos("SOL", 10, 50.0)])
    assert t.size == 10.0                   # adopted
    assert t.entry_px == 50.0               # basis untouched (no add proven)


# --- Port of upstream c2335d11d64a (manual re-open reconciliation) ---------
#
# Operator closed xyz:BE and reopened it at 3x by hand. The tracker kept
# saying leverage 1 and entry 268.15 while the live position was 3x at
# 267.87. The tracker's leverage is the divisor for every ROE-based stop,
# so max_loss_roe_pct 15 held against a stale 1x fires at a 15% PRICE move
# — 45% of margin on a position actually running 3x. The exit was three
# times looser than the policy said, on a live position, silently.


def test_reopen_adopted_live_leverage():
    # Tracker born at 1x; operator re-opened the same position at 3x by
    # hand. The rehydrate must adopt the live leverage — every ROE stop
    # divides by it.
    t = _track("xyz:BE", "short", 268.15, 40.0, peak=268.15, lev=1)
    dsl_exit.rehydrate_from_exchange([_pos("xyz:BE", -40, 268.15, lev=3)])
    assert t.leverage == 3, "stale leverage — every ROE stop is off by 3x"


def test_reopen_same_size_entry_drift_refreshes_basis():
    # Close-and-reopen at the SAME size: size reconciliation sees no change,
    # but the entry basis moved (268.15 -> 267.87, 0.104% > the 0.1%
    # tolerance). refresh_entry_basis must refresh it, clamping the peak
    # side-correctly — for a short the peak clamps DOWN to the new basis.
    t = _track("xyz:BE", "short", 268.15, 40.0, peak=268.15, lev=5)
    dsl_exit.rehydrate_from_exchange([_pos("xyz:BE", -40, 267.87, lev=5)])
    assert t.entry_px == 267.87, "stale entry basis — floors are wrong"
    assert t.size == 40.0
    assert t.peak_px == 267.87              # short peak clamped to new basis


def test_unchanged_reopen_keeps_peak():
    # A position that did NOT change must not be disturbed: refreshing the
    # basis every tick would reset peak tracking and never let a trail arm.
    # Same lev, same size, entry within the 0.1% tolerance.
    t = _track("xyz:BE", "short", 268.15, 40.0, peak=250.0, lev=5)
    dsl_exit.rehydrate_from_exchange([_pos("xyz:BE", -40, 268.15, lev=5)])
    assert t.entry_px == 268.15 and t.size == 40.0
    assert t.peak_px == 250.0, "peak tracking must survive an unchanged tick"


def test_legacy_size_zero_reopen_does_not_refresh_basis():
    # The guard upstream's first version broke: a state file written before
    # size tracking carries size 0, so a basis difference there cannot be
    # told apart from a legacy record that never stored one — refreshing on
    # that guess would move a live stop on no evidence. Legacy adopts the
    # size silently and leaves the basis alone.
    t = _track("DOGE", "long", 0.1, 0.0, peak=0.1, lev=5)   # size 0 = legacy
    dsl_exit.rehydrate_from_exchange([_pos("DOGE", 1000, 0.099, lev=5)])
    assert t.size == 1000.0               # adopted silently
    assert t.entry_px == 0.1              # basis untouched (no size evidence)
    assert t.peak_px == 0.1


def test_entry_wobble_within_tolerance_is_noop():
    # 0.05% is float/rounding noise, not a re-entry: within the 0.1%
    # tolerance nothing may move.
    t = _track("ETH", "long", 100.0, 10.0, peak=101.5, lev=5)
    dsl_exit.rehydrate_from_exchange([_pos("ETH", 10, 100.05, lev=5)])
    assert t.entry_px == 100.0 and t.size == 10.0
    assert t.peak_px == 101.5
