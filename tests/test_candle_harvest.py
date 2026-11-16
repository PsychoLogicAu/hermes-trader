"""Harvest hook inside hl_client.fetch_hl_candles (t_18e7517a plan T3).

Pins:
  * a successful fetch persists its CONFIRMED bars to the store — the last
    (still-forming) candle from the snapshot is excluded at write time
  * master-off byte-identity: with posture off the store module is never
    touched (get_store patched to raise would prove it) and the fetch return
    value is unchanged
  * harvest=False (store on, harvest off) writes nothing
  * transient failures (non-list after retries) harvest nothing
  * repeated cached reads harvest once (harvest rides FETCHES, not reads)
"""

import pytest

from hermes_trader.client import candlestore as cs
from hermes_trader.client import hl_client

STEP = hl_client._MS_PER_CANDLE["5m"]


def _raw_bars_ending_now(n: int, end_ms: int):
    """n bars whose LAST bar starts AT end_ms — i.e. the forming candle."""
    start = end_ms - (n - 1) * STEP
    return [{"t": start + i * STEP, "o": "1", "h": "2", "l": "0.5",
             "c": str(i), "v": "10"} for i in range(n)]


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated window store + fake HTTP + fixed clock (+ store posture on)."""
    cs.reset_store()
    monkeypatch.setattr(hl_client, "_CANDLE_WINDOWS", {})
    monkeypatch.setattr(hl_client, "_CANDLE_TTL_OVERRIDE_S", None)
    clock = {"now": 1_700_000_500.0}  # fixed wall clock for snapshots
    monkeypatch.setattr(hl_client.time, "time", lambda: clock["now"])

    def fake_post(path, payload, timeout=5):
        req = payload["req"]
        n = (req["endTime"] - req["startTime"]) // STEP
        return _raw_bars_ending_now(n, req["endTime"])

    monkeypatch.setattr(hl_client, "_http_post", fake_post)
    monkeypatch.setenv("HERMES_CANDLESTORE_DIR", str(tmp_path))
    state = {"enabled": True, "harvest": True}
    monkeypatch.setattr(cs, "_cfg", lambda: state)
    yield {"tmp": tmp_path, "clock": clock, "state": state,
           "store": cs.get_store()}
    cs.reset_store()


def test_harvest_on_fetch_excludes_forming_candle(env):
    out = hl_client.fetch_hl_candles("BTC", "5m", 100)
    assert len(out) == 100  # caller still gets its 100 bars
    stored = env["store"].series("BTC", "5m")
    assert len(stored) == 99  # forming candle at write-time excluded
    assert [b["c"] for b in stored] == [c.c for c in out[:-1]]


def test_master_off_never_touches_store(env, monkeypatch):
    """Posture off => the store machinery is never invoked (master-off pin)."""
    monkeypatch.delenv("HERMES_CANDLESTORE_DIR", raising=False)
    cs.reset_store()

    def boom(*a, **k):
        raise AssertionError("store touched while master-off")

    monkeypatch.setattr(cs, "get_store", boom)
    out = hl_client.fetch_hl_candles("BTC", "5m", 100)
    assert len(out) == 100  # fetch path byte-identical behavior


def test_harvest_off_writes_nothing(env):
    env["state"]["harvest"] = False
    hl_client.fetch_hl_candles("BTC", "5m", 100)
    assert env["store"].count_bars() == 0


def test_transient_failure_harvests_nothing(env, monkeypatch):
    monkeypatch.setattr(hl_client.time, "sleep", lambda s: None)

    def fail(path, payload, timeout=5):
        return None

    monkeypatch.setattr(hl_client, "_http_post", fail)
    assert hl_client.fetch_hl_candles("BTC", "5m", 100) == []
    assert env["store"].count_bars() == 0


def test_cached_reads_harvest_once(env):
    hl_client.fetch_hl_candles("BTC", "5m", 100)  # fetch -> harvest
    hl_client.fetch_hl_candles("BTC", "5m", 50)   # slice hit, no new fetch
    assert env["store"].count_bars() == 99        # harvested once