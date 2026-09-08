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


def _pos(coin, szi, entry):
    return {"position": {"coin": coin, "szi": str(szi), "entryPx": str(entry),
                         "leverage": {"value": 5}}}


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
