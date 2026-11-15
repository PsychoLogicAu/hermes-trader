"""Persistent candle store (client/candlestore.py) — core + confirmed-bar guard.

Pins the store's correctness-critical behaviors:
  * upsert dedupe (PK) and window semantics (t0 incl / t1 excl, ascending)
  * CONFIRMED-BARS-AT-WRITE: a bar with t + dur > now is NEVER stored
    (candleSnapshot's last row is the still-forming candle — a store that
    contains it poisons every replay that trusts it)
  * coverage()/coins()/count_bars() views
  * posture: enabled = env dir + config block enabled (hot, 5s cache);
    env alone does NOT enable; harvest is a pure no-op when off
"""

import pytest

from hermes_trader.client import candlestore as cs


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Isolated store: env dir -> tmp, config posture forced on."""
    cs.reset_store()
    monkeypatch.setenv("HERMES_CANDLESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(cs, "_cfg", lambda: {
        "enabled": True, "harvest": True,
        "intervals": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "prune_max_days": {"1m": 7}})
    yield cs.get_store()
    cs.reset_store()


T0 = 1_700_000_000_000  # arbitrary fixed base (2023-11-14 19:46:40 UTC)


def _bars(n, interval="5m", t0=T0):
    step = cs.MS_PER_INTERVAL[interval]
    return [{"t": t0 + i * step, "o": 1.0 + i, "h": 2.0 + i,
             "l": 0.5 + i, "c": 1.5 + i, "v": 10.0 + i} for i in range(n)]


# ── upsert + series ────────────────────────────────────────────────────

def test_upsert_dedupe(store):
    bars = _bars(10)
    assert store.upsert_bars("BTC", "5m", bars, now_ms=T0 + 10 ** 10) == 10
    assert store.upsert_bars("BTC", "5m", bars, now_ms=T0 + 10 ** 10) == 0
    assert len(store.series("BTC", "5m")) == 10


def test_upsert_never_overwrites_existing(store):
    b1 = _bars(3)
    b2 = [dict(b, c=999.0) for b in b1]  # same t, mutated values
    store.upsert_bars("BTC", "5m", b1, now_ms=T0 + 10 ** 10)
    store.upsert_bars("BTC", "5m", b2, now_ms=T0 + 10 ** 10)
    assert [x["c"] for x in store.series("BTC", "5m")] == [b["c"] for b in b1]


def test_series_window_semantics(store):
    now = T0 + 3 * 86_400_000
    store.upsert_bars("ETH", "5m", _bars(3 * 288), now_ms=now)
    all_bars = store.series("ETH", "5m")
    assert len(all_bars) == 3 * 288
    assert all_bars == sorted(all_bars, key=lambda b: b["t"])
    day2_lo = T0 + 86_400_000
    win = store.series("ETH", "5m", day2_lo, day2_lo + 86_400_000)
    assert len(win) == 288
    assert win[0]["t"] == day2_lo                    # t0 inclusive
    assert win[-1]["t"] < day2_lo + 86_400_000       # t1 exclusive
    # either end optional
    assert len(store.series("ETH", "5m", t0_ms=T0 + 2 * 86_400_000)) == 288
    assert len(store.series("ETH", "5m", t1_ms=T0 + 86_400_000)) == 288


def test_multi_coin_interval_isolation(store):
    now = T0 + 10 ** 10
    store.upsert_bars("BTC", "5m", _bars(5), now_ms=now)
    store.upsert_bars("BTC", "1h", _bars(5, "1h"), now_ms=now)
    store.upsert_bars("ETH", "5m", _bars(5), now_ms=now)
    assert len(store.series("BTC", "5m")) == 5
    assert len(store.series("BTC", "1h")) == 5
    assert store.coins() == ["BTC", "ETH"]
    assert store.count_bars() == 15
    assert store.count_bars("BTC", "5m") == 5
    cov = store.coverage("BTC", "5m")
    assert cov and sum(n for _, n in cov) == 5


# ── confirmed-bar guard (the feature's highest-value invariant) ────────

def test_forming_bar_never_stored(store):
    """Last bar of a snapshot whose interval has NOT closed yet is dropped."""
    bars = _bars(5)  # last bar t = T0+4*300000; closes at T0+5*300000
    now = T0 + 4 * 300_000 + 1  # forming: last bar's end > now
    assert store.upsert_bars("BTC", "5m", bars, now_ms=now) == 4
    got = store.series("BTC", "5m")
    assert [b["t"] for b in got] == [b["t"] for b in bars[:4]]


def test_boundary_confirmed_at_exact_close(store):
    """t + dur == now is CONFIRMED (the forming bar's end instant has passed)."""
    bars = _bars(5)
    now = T0 + 5 * 300_000  # exactly the last bar's close instant
    assert store.upsert_bars("BTC", "5m", bars, now_ms=now) == 5


def test_guard_applies_through_harvest_entry(store):
    """harvest() routes through the same write guard (Candle attr objects)."""
    from hermes_trader.models.types import Candle
    cands = [Candle(t=T0 + i * 300_000, o=1, h=2, l=0.5, c=1.5, v=10)
             for i in range(5)]
    cs.harvest("SOL", "5m", cands, now_ms=T0 + 4 * 300_000 + 1)
    assert len(store.series("SOL", "5m")) == 4  # Candle input, forming excluded


# ── posture ────────────────────────────────────────────────────────────

def test_posture_off_without_env(monkeypatch):
    monkeypatch.delenv("HERMES_CANDLESTORE_DIR", raising=False)
    cs.reset_store()
    assert cs.get_store() is None
    assert cs.enabled() is False
    cs.harvest("BTC", "5m", _bars(3))  # pure no-op, no raise, no file


def test_posture_config_master_switch(tmp_path, monkeypatch):
    """Env mount alone does NOT enable — the config block must say so."""
    monkeypatch.setenv("HERMES_CANDLESTORE_DIR", str(tmp_path))
    cs.reset_store()
    monkeypatch.setattr(cs, "_cfg", lambda: {"enabled": False})
    assert cs.enabled() is False
    cs.harvest("BTC", "5m", _bars(3), now_ms=T0 + 10 ** 10)  # master off
    cs.reset_store()
    import os
    assert not os.path.exists(str(tmp_path / "candles.db"))  # off => no file
    monkeypatch.setattr(cs, "_cfg", lambda: {"enabled": True, "harvest": True})
    assert cs.enabled() is True
    cs.reset_store()


def test_harvest_fail_open(store, monkeypatch):
    """A broken store never raises into a live fetch."""
    def boom(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(store, "upsert_bars", boom)
    cs.harvest("BTC", "5m", _bars(3), now_ms=T0 + 10 ** 10)  # must not raise