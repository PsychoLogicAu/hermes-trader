"""Gap planner + paced backfill + size-cap pruning (t_18e7517a plan T4/T5).

Planner discipline (the retention traps):
  * NEVER plan a fetch whose start sits inside the dead window past the
    interval horizon — candleSnapshot returns [] there silently; the plan
    clamps to the horizon and reports the past-horizon span as unrecoverable.
  * every planned window is <= MAX_BARS_PER_FETCH (5000) — older end is
    silently truncated otherwise.
Runner discipline:
  * fetch None (rate budget exhausted / transient) = skip, retry next cycle;
    nothing written, no poisoning.
  * empty list = legitimate no-history; counts toward `empty`, stores nothing.
  * max_fetches bounds the cycle to the configured sustained weight/min.
Pruning discipline (history is unrecoverable, so prune is deliberate):
  * hard day cap (1m@7d) runs every cycle.
  * size-cap prune deletes in VALUE ORDER (1m -> 15m -> 5m -> ...),
    oldest-first, one tier before touching the next.
"""

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
        "intervals": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "max_store_mb": 1024, "prune_max_days": {"1m": 7},
        "backfill_max_weight_per_min": 200, "backfill_interval_min": 10,
        "retention_days": {}})
    yield cs.get_store()
    cs.reset_store()


def _bars(n, interval="5m", t0=T0):
    step = cs.MS_PER_INTERVAL[interval]
    return [{"t": t0 + i * step, "o": 1.0, "h": 2.0,
             "l": 0.5, "c": 1.5, "v": 10.0} for i in range(n)]


def _mk_bars(n, interval, t0):
    step = cs.MS_PER_INTERVAL[interval]
    return [{"t": t0 + i * step, "o": 1.0, "h": 2.0,
             "l": 0.5, "c": 1.5, "v": 10.0} for i in range(n)]


# ── gap planner ────────────────────────────────────────────────────────

def test_trailing_gap_planned(store):
    now = T0 + 2 * DAY
    store.upsert_bars("BTC", "5m", _bars(288), now_ms=now)  # day 1 fully stocked
    g = store.gap_plan("BTC", "5m", now_ms=now)
    assert g["gap_ms"] == DAY
    assert g["unrecoverable_ms"] == 0  # ~1d < 17d horizon
    assert g["backfillable_ms"] == g["gap_ms"]


def test_horizon_clamp(store):
    now = T0 + 30 * DAY  # store's single bar is 30d old; 5m horizon 17d
    store.upsert_bars("BTC", "5m", _bars(1), now_ms=now)
    g = store.gap_plan("BTC", "5m", now_ms=now)
    dur = cs.MS_PER_INTERVAL["5m"]
    # gap = 30d - dur; the part older than the 17d horizon is unrecoverable,
    # the fetchable remainder is exactly [now-17d, now).
    assert g["gap_ms"] == 30 * DAY - dur
    assert g["unrecoverable_ms"] == (30 - cs.RETENTION_DAYS["5m"]) * DAY - dur
    assert g["backfillable_ms"] == cs.RETENTION_DAYS["5m"] * DAY


def test_horizon_clamp_rounding(store):
    """backfillable = min(gap, horizon - age-of-last-bar) — no negatives."""
    now = T0 + 30 * DAY
    store.upsert_bars("BTC", "5m", _bars(1), now_ms=now)
    g = store.gap_plan("BTC", "5m", now_ms=now)
    assert g["backfillable_ms"] >= 0
    assert g["unrecoverable_ms"] + g["backfillable_ms"] == g["gap_ms"]


def test_fresh_store_no_plan(store):
    now = T0 + DAY
    # day 1 fully stocked; last bar closes exactly AT now (confirmed boundary)
    store.upsert_bars("BTC", "5m", _bars(288), now_ms=now)
    g = store.gap_plan("BTC", "5m", now_ms=now)
    assert g["backfillable_ms"] == 0
    assert store.backfill_plans(["BTC"], ["5m"], now_ms=now) == []


def test_empty_store_seeds_horizon(store):
    now = T0 + 30 * DAY
    g = store.gap_plan("NEWCOIN", "5m", now_ms=now)
    assert g["backfillable_ms"] == cs.RETENTION_DAYS["5m"] * DAY
    plans = store.backfill_plans(["NEWCOIN"], ["5m"], now_ms=now)
    step = cs.MS_PER_INTERVAL["5m"]
    assert sum((p["end_ms"] - p["start_ms"]) // step for p in plans) * step \
        == cs.RETENTION_DAYS["5m"] * DAY
    assert plans[0]["end_ms"] == now  # newest window first
    if len(plans) > 1:
        assert plans[0]["start_ms"] >= plans[-1]["end_ms"]


def test_plan_windows_respect_5000_cap(store):
    # 1m horizon 3.5d = 5040 bars > 5000 -> must split into <=5000-bar windows
    now = T0 + 30 * DAY
    plans = store.backfill_plans(["NEW"], ["1m"], now_ms=now)
    assert len(plans) == 2
    step = cs.MS_PER_INTERVAL["1m"]
    for p in plans:
        assert (p["end_ms"] - p["start_ms"]) // step <= cs.MAX_BARS_PER_FETCH


# ── backfill runner ────────────────────────────────────────────────────

def test_backfill_once_stores_fetched(store):
    now = T0 + 30 * DAY
    plans = store.backfill_plans(["BTC"], ["5m"], now_ms=now)

    def fetch(coin, interval, start_ms, end_ms):
        step = cs.MS_PER_INTERVAL[interval]
        n = (end_ms - start_ms) // step
        return _mk_bars(n, interval, start_ms)

    stats = cs.backfill_once(store, plans, fetch_fn=fetch, now_ms=now)
    assert stats["fetched"] == len(plans) and stats["failed"] == 0
    got = store.series("BTC", "5m")
    assert got, "seed stored bars"
    assert got[0]["t"] >= now - cs.RETENTION_DAYS["5m"] * DAY  # horizon respected
    assert got[-1]["t"] + cs.MS_PER_INTERVAL["5m"] <= now       # confirmed only


def test_backfill_empty_and_rate_skipped(store):
    now = T0 + 30 * DAY
    # 1m seed = 5040 bars -> two windows: first [] (no history), second None
    # (rate budget exhausted). Neither may write or poison anything.
    plans = store.backfill_plans(["BTC"], ["1m"], now_ms=now)

    seq = iter([[], None] + [[]] * 10)

    def fetch(coin, interval, start_ms, end_ms):
        return next(seq)

    stats = cs.backfill_once(store, plans, fetch_fn=fetch, now_ms=now)
    assert stats["empty"] >= 1
    assert stats["failed"] >= 1
    assert store.count_bars() == 0  # nothing written from []/None


def test_backfill_max_fetches_binds(store):
    now = T0 + 30 * DAY
    plans = store.backfill_plans(["BTC", "ETH"], ["5m", "1h"], now_ms=now)

    def fetch(coin, interval, start_ms, end_ms):
        return []

    stats = cs.backfill_once(store, plans, fetch_fn=fetch, now_ms=now,
                             max_fetches=2)
    assert stats["fetches_used"] == 2
    assert stats["deferred"] == len(plans) - 2


# ── cycle wiring (posture + injection) ─────────────────────────────────

def test_run_cycle_off_by_default(store, monkeypatch):
    monkeypatch.setattr(cs, "_cfg", lambda: {"enabled": False})
    stats = cs.run_backfill_cycle(coins=["BTC"])
    assert stats is None or stats.get("skipped"), "backfill obeys master switch"


def test_run_cycle_injected(store, monkeypatch):
    def fetch(coin, interval, start_ms, end_ms):
        step = cs.MS_PER_INTERVAL[interval]
        n = (end_ms - start_ms) // step
        return _mk_bars(n, interval, start_ms)

    stats = cs.run_backfill_cycle(fetch_fn=fetch, coins=["BTC", "ETH"])
    assert stats["fetched"] >= 2
    assert store.coins() == ["BTC", "ETH"]


# ── pruning ────────────────────────────────────────────────────────────

def test_prune_hard_day_cap_1m(store):
    now = T0 + 10 * DAY
    store.upsert_bars("BTC", "1m", _mk_bars(10 * 1440, "1m", T0), now_ms=now)
    n = store.prune_hard_caps(now_ms=now, prune_max_days={"1m": 7})
    assert n == 3 * 1440
    assert store.series("BTC", "1m")[0]["t"] >= now - 7 * DAY


def test_prune_size_value_order_oldest_first(store):
    now = T0 + 3 * DAY
    store.upsert_bars("BTC", "5m", _mk_bars(3 * 288, "5m", T0), now_ms=now)
    store.upsert_bars("BTC", "1m", _mk_bars(3 * 1440, "1m", T0), now_ms=now)
    removed = store.prune_to_size(max_bytes=1, batch_rows=500, now_ms=now)
    assert removed == 500
    remaining_1m = store.series("BTC", "1m")
    assert remaining_1m[0]["t"] == T0 + 500 * 60_000  # oldest 1m went first
    assert len(store.series("BTC", "5m")) == 3 * 288  # 5m untouched
    # exhausting the 1m tier spills into 15m/5m only then:
    removed = store.prune_to_size(max_bytes=1,
                                  batch_rows=remaining_1m and len(remaining_1m),
                                  now_ms=now)
    assert len(store.series("BTC", "1m")) == 0
    assert len(store.series("BTC", "5m")) == 3 * 288  # 15m empty; 5m still intact


def test_thread_not_started_when_disabled(monkeypatch):
    monkeypatch.delenv("HERMES_CANDLESTORE_DIR", raising=False)
    cs.reset_store()
    assert cs.start_backfill_thread() is None

# ── empty-history cooldown (reviewer t_18e7517a 2026-11-15) ─────────────

def test_empty_cooldown_suppresses_replan(store):
    """A (coin,iv) that returns [] (delisted/pre-listing) must not be
    re-fetched forever: backfill_once marks it; planner skips it until the
    cooldown passes — guaranteed-[] responses burn weight budget."""
    now = T0 + 30 * DAY
    plans = store.backfill_plans(["DEAD"], ["5m"], now_ms=now)
    assert plans
    stats = cs.backfill_once(store, plans,
                             fetch_fn=lambda c, i, s, e: [], now_ms=now)
    assert stats["empty"] >= 1 and store.count_bars() == 0
    # same instant: inside cooldown -> not planned again
    assert store.backfill_plans(["DEAD"], ["5m"], now_ms=now,
                                empty_cooldown_ms=14 * DAY) == []
    # cooldown expired -> planned again (it may relist / HL may differ)
    assert store.backfill_plans(["DEAD"], ["5m"], now_ms=now + 15 * DAY,
                                empty_cooldown_ms=14 * DAY)
