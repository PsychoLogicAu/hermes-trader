"""Concurrency regression (reviewer finding t_18e7517a 2026-11-15).

One shared sqlite connection (check_same_thread=False) serves harvest from
scan worker threads, the backfill planner/runner, and prune. Reviewer probe
reproduced InterfaceError('bad parameter') / fetchone()->None crashes 3/3
with UNLOCKED reads; this test fails on that shape and passes with the
RLock-around-every-_con-use fix. If this test ever flakes or fails, the
connection-sharing discipline broke — do not 'fix' by loosening asserts.
"""
import threading
import time

import pytest

from hermes_trader.client import candlestore as cs

T0 = 1_700_000_000_000
DAY = 86_400_000


@pytest.fixture()
def store(tmp_path, monkeypatch):
    cs.reset_store()
    monkeypatch.setenv("HERMES_CANDLESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(cs, "_cfg", lambda: {
        "enabled": True, "harvest": True, "backfill": True,
        "intervals": ["5m"], "max_store_mb": 1024,
        "prune_max_days": {"1m": 7},
        "backfill_max_weight_per_min": 200, "backfill_interval_min": 10,
        "retention_days": {}})
    yield cs.get_store()
    cs.reset_store()


def _bars(base, n=50):
    return [{"t": base + i * 300_000, "o": 1.0, "h": 2.0, "l": 0.5,
             "c": 1.5, "v": 10.0} for i in range(n)]


def test_concurrent_writers_planners_readers_pruneers(store):
    errors = []
    deadline = time.monotonic() + 2.0

    def guard(fn):
        def run():
            try:
                fn()
            except Exception as e:  # any exception IS the race
                errors.append((fn.__name__, repr(e)))
        return run

    def harvest_worker():
        i = 0
        while time.monotonic() < deadline:
            store.upsert_bars(f"C{i % 4}", "5m",
                              _bars(T0 + (i % 200) * 300_000), now_ms=T0 + 10 * DAY)
            i += 1

    def planner_worker():
        while time.monotonic() < deadline:
            store.backfill_plans(["C0", "C1", "C2"], ["5m"], now_ms=T0 + 10 * DAY)

    def reader_worker():
        while time.monotonic() < deadline:
            store.series("C0", "5m", T0, T0 + DAY)
            store.coverage("C1", "5m")
            store.count_bars("C2", "5m")
            store.coins()
            store.est_bytes()

    def pruner_worker():
        while time.monotonic() < deadline:
            store.prune_hard_caps(now_ms=T0 + 10 * DAY, prune_max_days={"1m": 7})
            store.prune_to_size(max_bytes=10 ** 12, batch_rows=200,
                                now_ms=T0 + 10 * DAY)

    ths = ([threading.Thread(target=guard(harvest_worker)) for _ in range(3)]
           + [threading.Thread(target=guard(planner_worker)) for _ in range(2)]
           + [threading.Thread(target=guard(reader_worker)) for _ in range(3)]
           + [threading.Thread(target=guard(pruner_worker)) for _ in range(2)])
    for t in ths:
        t.start()
    for t in ths:
        t.join(10)
    assert not any(t.is_alive() for t in ths)
    assert errors == [], f"connection race: {errors[:5]}"
    # store still coherent after the storm
    assert store.count_bars() > 0
    assert store.series("C0", "5m")[:1]
