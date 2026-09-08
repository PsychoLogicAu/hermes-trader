"""SOPH hair-trigger fix tests (2026-09-08).

Two fixes for the <60s floor_breach exit class (9 ledger trades, SOPH
09-07 18:08 held 12s / SOPH 09-08 03:51 held 24s, both closed at ~entry while
price then ran +3.7–11.5% spot):

  1. Peak-seed fix (root cause) in ``executor.fast_exit_pass`` — the
     first daemon tick after a fill ratchets the peak to the ENTRY
     CANDLE's high, which includes the pre-fill spike the order was chasing.
     The 2026-09-03 JUP/ARB guard only dropped candles that CLOSED before
     entry; the entry candle (opened a few seconds pre-fill, still forming)
     survived and contaminated the peak. The fix excludes ANY candle whose
     OPEN is pre-fill (``c.t < entry_ms``); the live mid is the clean seed.

  2. Floor-exit grace window in ``dsl_exit.check()`` — for the first
     ``floor_exit_grace_sec`` seconds after entry, a floor-based breach
     (breakeven ratchet / phase-1 / phase-2 retrace) is suppressed. The
     catastrophic ``max_loss`` stop (checked earlier) and the exchange-side
     1.5x-ATR backup SL are UNAFFECTED during the grace.

All network is monkeypatched; DSL_STATE_FILE is pointed at tmp so no live
state file is ever written.
"""

from __future__ import annotations

import time
import types

import pytest

from hermes_trader.agents import dsl_exit, executor
from hermes_trader.agents.dsl_exit import DSLTracker, ExitPolicy, RetraceTier


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


def _candle(h, l, t=None):
    # t defaults to "now" (the still-forming candle) so callers that don't
    # care about the pre-fill boundary keep working.
    if t is None:
        t = int(time.time() * 1000)
    return types.SimpleNamespace(t=t, h=h, l=l)


def _live_policy(grace_sec: float = 0.0) -> ExitPolicy:
    """Mirror the live dsl_exit config (breakeven lock + 0.25 retrace),
    with the floor-exit grace optional."""
    return ExitPolicy(
        max_loss_pct=5.0, max_loss_roe_pct=30.0, protect_pct=1.0,
        retrace_threshold=0.25, hard_timeout_minutes=1800.0,
        breakeven_trigger_pct=0.7, breakeven_lock_pct=0.05,
        phase2_tiers=[RetraceTier(0.0, 0.25)],
        floor_exit_grace_sec=grace_sec,
    )


# ── 1. Peak-seed fix: the entry candle's pre-fill spike must not arm ────────

def test_peak_seed_ignores_entry_candle_spike_long(monkeypatch):
    """SOPH repro (2026-09-07 18:08, held 12s).

    The fill lands MID-CANDLE: the still-forming 1m candle OPENED ~5s before
    the fill and its high (0.004890) is the pre-fill spike the order chased
    (the 18:05–18:06 1m candles ran 0.004715→0.005164 then faded 6%). The
    live mid at the first daemon tick (12s after the fill) is back at ~entry
    (0.004843).

    WITHOUT the fix the first pass ratchets peak to the entry candle's high
    → peak +0.97% arms the 0.7% breakeven lock → floor = entry×1.0005 →
    mark 0.004843 < floor → floor_breach at ~entry (fees only). WITH the fix
    the entry candle is filtered (its open is pre-fill), the peak seeds from
    the live mid only, the breakeven lock never arms, and the position holds.
    """
    entry_px = 0.004843
    now_ms = int(time.time() * 1000)
    entry_ms = now_ms - 12_000  # filled 12s ago
    dsl_exit.register_position(
        coin="SOPH", side="long", entry_px=entry_px, leverage=3,
        entry_time=entry_ms / 1000.0, policy=_live_policy(),
    )
    # Three 1m candles: the 08:05 candle CLOSED before the fill (spike high
    # 0.005164), the entry candle OPENED at 08:08:00 (5s pre-fill) with the
    # contaminated high 0.004890 and is still forming, plus one older closed
    # candle. None opened at/after the fill.
    monkeypatch.setattr(executor, "fetch_hl_candles", lambda *a, **k: [
        _candle(0.004708, 0.004665, t=entry_ms - 180_000),  # closed pre-fill
        _candle(0.005164, 0.004697, t=entry_ms - 120_000),  # 08:05 spike, closed
        _candle(0.004890, 0.004803, t=entry_ms - 5_000),    # entry candle, forming
    ])
    # Mid has pulled back to ~entry after the spike faded.
    monkeypatch.setattr(executor, "get_hl_price", lambda coin: entry_px)

    exits = executor.fast_exit_pass()
    assert exits == [], f"entry-candle pre-fill spike must not close a fresh position: {exits}"
    tr = dsl_exit._active_positions.get("SOPH_long")
    assert tr is not None
    # Peak seeded from the live mid (= entry), NOT the entry candle high 0.004890
    # (let alone the 0.005164 spike). Breakeven trigger (0.7%) is far away.
    assert tr.peak_px <= entry_px + 1e-9
    assert tr.peak_px < 0.004890, "peak ratcheted off the contaminated entry candle"
    # And no floor was ratcheted above entry (the poison state).
    assert tr._last_floor is None or tr._last_floor <= entry_px + 1e-9


def test_peak_seed_ignores_entry_candle_dip_short(monkeypatch):
    """Short-side mirror of the SOPH repro.

    A short chased a DIP: the entry candle's LOW (opened pre-fill) carries the
    pre-fill spike-down. Ratcheting the (short) peak off it would arm the
    breakeven floor and instantly close a fresh short that never moved.
    """
    entry_px = 0.004843
    now_ms = int(time.time() * 1000)
    entry_ms = now_ms - 12_000
    dsl_exit.register_position(
        coin="SOPH", side="short", entry_px=entry_px, leverage=3,
        entry_time=entry_ms / 1000.0, policy=_live_policy(),
    )
    monkeypatch.setattr(executor, "fetch_hl_candles", lambda *a, **k: [
        _candle(0.004900, 0.004708, t=entry_ms - 120_000),  # dip low, closed pre-fill
        _candle(0.004860, 0.004800, t=entry_ms - 5_000),    # entry candle low, forming
    ])
    monkeypatch.setattr(executor, "get_hl_price", lambda coin: entry_px)

    exits = executor.fast_exit_pass()
    assert exits == [], f"entry-candle pre-fill dip must not close a fresh short: {exits}"
    tr = dsl_exit._active_positions.get("SOPH_short")
    assert tr is not None
    # Short peak = best (lowest) price seen; must NOT have ratcheted down to the
    # pre-fill dip low 0.004800 — only the live mid (= entry) is clean.
    assert tr.peak_px >= entry_px - 1e-9
    assert tr.peak_px > 0.004810, "peak ratcheted off the contaminated entry candle low"


def test_peak_seed_still_ratchets_post_entry_candle(monkeypatch):
    """The fix must NOT neuter the pass's raison d'être: a genuine POST-ENTRY
    candle (opened after the fill) still ratchets the peak, so a real intrabar
    wick is still caught (GRASS shape)."""
    now_ms = int(time.time() * 1000)
    entry_ms = now_ms - 60_000  # filled 60s ago
    dsl_exit.register_position(
        coin="GRASS", side="long", entry_px=100.0, leverage=3,
        entry_time=entry_ms / 1000.0, policy=_live_policy(),
    )
    monkeypatch.setattr(executor, "fetch_hl_candles", lambda *a, **k: [
        _candle(100.2, 99.8, t=entry_ms - 30_000),  # closed pre-fill -> filtered
        _candle(101.5, 100.0, t=entry_ms + 10_000),  # opened AFTER fill -> ratchetable
        _candle(101.3, 100.9, t=now_ms),              # forming, post-entry
    ])
    monkeypatch.setattr(executor, "get_hl_price", lambda coin: 101.2)

    exits = executor.fast_exit_pass()
    tr = dsl_exit._active_positions.get("GRASS_long")
    assert tr is not None
    # Post-entry candle high (101.5) is the true peak — ratcheted.
    assert tr.peak_px == 101.5
    # Peak +1.5% > protect 1.0% armed the phase-2 floor:
    #   100 + (101.5-100)*(1-0.25) = 101.125. Mid 101.2 is just above it -> hold.
    assert exits == []
    assert tr._last_floor == pytest.approx(101.125)


def test_peak_seed_subsumes_old_pre_entry_filter(monkeypatch):
    """Regression guard for the 2026-09-03 JUP/ARB case: a candle that CLOSED
    before entry (open well pre-fill) is still excluded — the new filter is
    strictly stricter, not a replacement that forgets the old class."""
    entry_px = 0.22656
    now_ms = int(time.time() * 1000)
    entry_ms = now_ms - 15_000
    dsl_exit.register_position(
        coin="JUP", side="long", entry_px=entry_px, leverage=5,
        entry_time=entry_ms / 1000.0, policy=_live_policy(),
    )
    monkeypatch.setattr(executor, "fetch_hl_candles", lambda *a, **k: [
        _candle(0.23163, 0.22786, t=entry_ms - 120_000),  # 09:00, +2.2% pre-fill spike
        _candle(0.22849, 0.22564, t=entry_ms - 60_000),   # 09:01, closed pre-fill
        _candle(0.22662, 0.22541, t=entry_ms + 40_000),   # forming, flat, post-entry
    ])
    monkeypatch.setattr(executor, "get_hl_price", lambda coin: 0.2266)

    exits = executor.fast_exit_pass()
    assert exits == []
    tr = dsl_exit._active_positions.get("JUP_long")
    assert tr is not None
    assert tr.peak_px < 0.23000, "peak ratcheted off a pre-entry candle"


# ── 2. Floor-exit grace window ───────────────────────────────────────────────

def _manual(peak: float, grace_sec: float, entry_time: float) -> DSLTracker:
    """A long whose peak has armed the breakeven lock and whose mark has
    pulled back to entry — the SOPH contaminated-peak state. The next
    check() at mark=entry is a floor breach the instant the floor is above
    entry."""
    tr = DSLTracker("SOPH", "long", 100.0, entry_time, _live_policy(grace_sec),
                    leverage=3, entry_atr_pct=0.0)
    # Peak +1.5% armed the breakeven lock (trigger 0.7%): floor = 100×1.0005.
    tr.peak_px = peak
    tr.check(101.5)  # arms phase-2 + breakeven floor, persists _last_floor
    return tr


def test_grace_suppresses_breach_within_window():
    """Within the grace, the contaminated floor breach is held (state kept)."""
    now = time.time()
    tr = _manual(101.5, grace_sec=90.0, entry_time=now - 20.0)  # 20s old < 90s
    v = tr.check(100.0)  # mark pulled back to entry
    assert v.exit is False
    assert v.reason.startswith("floor_exit_grace")
    # State kept armed — the net is ready the moment the grace expires.
    assert tr.peak_px == 101.5
    assert tr._last_floor is not None and tr._last_floor > 100.0


def test_grace_fires_after_window():
    """Past the grace, the identical state breaches and exits."""
    now = time.time()
    tr = _manual(101.5, grace_sec=90.0, entry_time=now - 100.0)  # 100s old > 90s
    v = tr.check(100.0)
    assert v.exit is True
    assert "floor_breach" in v.reason


def test_grace_off_behaves_as_before():
    """grace_sec=0 (the default) = no suppression: the breach fires
    immediately even seconds after entry."""
    now = time.time()
    tr = _manual(101.5, grace_sec=0.0, entry_time=now - 5.0)
    v = tr.check(100.0)
    assert v.exit is True
    assert "floor_breach" in v.reason


def test_grace_does_not_suppress_max_loss():
    """The catastrophic max_loss stop is UNAFFECTED by the grace: a deep
    loss below entry exits even 5s after entry."""
    now = time.time()
    tr = _manual(101.5, grace_sec=90.0, entry_time=now - 5.0)  # 5s old
    # 5%+ adverse spot move = max_loss (5.0% spot cap, ROE 30/3=10% not binding).
    v = tr.check(94.0)
    assert v.exit is True
    assert "max_loss" in v.reason


def test_grace_short_side_suppressed_then_fires():
    """Short-side mirror: breach within the grace is held, then exits."""
    now = time.time()
    tr = DSLTracker("SOPH", "short", 100.0, now - 20.0,
                    _live_policy(90.0), leverage=3, entry_atr_pct=0.0)
    tr.peak_px = 98.5  # best (lowest) price; short breakeven floor = 100×0.9995
    tr.check(98.5)
    v = tr.check(100.0)  # mark pushed back up to entry within the grace
    assert v.exit is False
    assert v.reason.startswith("floor_exit_grace")


# ── 3. Config plumbing ───────────────────────────────────────────────────────

def test_policy_from_config_reads_grace(monkeypatch):
    """floor_exit_grace_sec flows from the live .agent-config.json dsl_exit
    block into a SYNTHESIZED tracker's policy (the post-restart path), and a
    missing key falls back to 0.0 (off) — never a crash, never an accidental
    grace."""
    monkeypatch.setattr(
        "hermes_trader.agents.config_store.read_agent_config",
        lambda: {"dsl_exit": {"floor_exit_grace_sec": 90.0}},
    )
    pol = dsl_exit._policy_from_config()
    assert pol.floor_exit_grace_sec == 90.0

    monkeypatch.setattr(
        "hermes_trader.agents.config_store.read_agent_config",
        lambda: {"dsl_exit": {}},
    )
    pol = dsl_exit._policy_from_config()
    assert pol.floor_exit_grace_sec == 0.0


def test_state_roundtrip_preserves_grace():
    """A tracker persisted with a grace rehydrates with the same grace, and a
    legacy state file (written before the field existed) falls back to 0.0."""
    dsl_exit.register_position(
        coin="SOPH", side="long", entry_px=100.0,
        policy=_live_policy(grace_sec=90.0),
    )
    dsl_exit.load_state(force=True)
    revived = dsl_exit._active_positions.get("SOPH_long")
    assert revived is not None
    assert revived.policy.floor_exit_grace_sec == 90.0

    # Legacy payload without the key -> 0.0.
    import json
    legacy = json.loads(json.dumps({
        "coin": "OLD", "side": "long", "leverage": 1, "entry_px": 100.0,
        "entry_time": time.time(), "entry_atr_pct": 0.0, "peak_px": 100.0,
        "consecutive_breaches": 0, "last_floor": None,
        "policy": {"max_loss_pct": 5.0, "max_loss_roe_pct": 30.0,
                    "protect_pct": 1.0, "retrace_threshold": 0.25,
                    "hard_timeout_minutes": 1800.0,
                    "phase2_tiers": [{"pct_above_entry": 0.0, "retrace_threshold": 0.25}]},
    }))
    tr = dsl_exit._tracker_from_dict(legacy)
    assert tr.policy.floor_exit_grace_sec == 0.0
