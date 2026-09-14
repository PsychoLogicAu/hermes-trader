"""Stale-tracker settlement: exchange-side closes booked at detection time.

Covers scripts→hermes_trader/stale_close_settle.py + the dsl_exit hook that
captures stale entry context (ZETA 2026-09-13 incident: exchange SL filled,
tracker dropped as stale, no CLOSE row and no loss cooldown until the NEXT
restart's reconcile — FIL-style re-chases went unblocked in between).

Offline: fills fetch and memory/config are injected; ledger writes land on the
conftest temp path (HERMES_LEDGER_FILE isolation).
"""
import json
import time

import pytest

from hermes_trader import ledger
from hermes_trader.agents import dsl_exit
from hermes_trader.stale_close_settle import settle_stale_closes


@pytest.fixture
def ledger_file(tmp_path):
    path = tmp_path / "trades.jsonl"
    old = ledger.LEDGER_FILE
    ledger.LEDGER_FILE = str(path)
    yield path
    ledger.LEDGER_FILE = old


class FakeMemory:
    def __init__(self):
        self.closes = []
        self.cooldowns = {}

    def record_close(self, c):
        self.closes.append(c)

    def set_loss_cooldown(self, coin, until_ms):
        self.cooldowns[coin] = int(until_ms)


def _read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _stale_rec(coin="ZETA", side="long", entry_px=0.0394, size=9241.7,
               leverage=3, entry_time=None):
    return {"coin": coin, "side": side, "entry_px": entry_px, "size": size,
            "leverage": leverage,
            "entry_time": entry_time or (time.time() - 1500)}


def _fill(coin="ZETA", dirn="Close Long", px=0.03811, sz=9241.7, ts=None,
          closed_pnl="-11.92"):
    f = {"coin": coin, "dir": dirn, "px": str(px), "sz": str(sz),
         "time": int(((ts or time.time())) * 1000)}
    if closed_pnl is not None:
        f["closedPnl"] = closed_pnl
    return f


# ── dsl_exit: stale drop returns entry context ──────────────────────────────

def _isolate_dsl_state(monkeypatch, tmp_path):
    state = str(tmp_path / ".dsl-state.json")
    monkeypatch.setattr(dsl_exit, "DSL_STATE_FILE", state)
    dsl_exit._active_positions.clear()
    return dsl_exit


def test_rehydrate_returns_stale_entry_context(monkeypatch, tmp_path):
    _isolate_dsl_state(monkeypatch, tmp_path)
    # A live tracker's size comes from exchange reconciliation (register itself
    # leaves 0 until a rehydrate syncs it); simulate one cycle of that here.
    t = dsl_exit.register_position("ZETA", "long", 0.0394, leverage=3)
    dsl_exit.rehydrate_from_exchange(
        [{"position": {"coin": "ZETA", "szi": "9241.7", "entryPx": "0.0394",
                       "leverage": {"value": 3}}}])

    recs = dsl_exit.rehydrate_from_exchange([])  # exchange says: flat

    assert len(recs) == 1
    r = recs[0]
    assert (r["coin"], r["side"]) == ("ZETA", "long")
    assert r["entry_px"] == pytest.approx(0.0394)
    assert r["size"] == pytest.approx(9241.7)
    assert r["leverage"] == 3
    assert r["entry_time"] == pytest.approx(t.entry_time)


def test_rehydrate_returns_empty_when_nothing_dropped(monkeypatch, tmp_path):
    _isolate_dsl_state(monkeypatch, tmp_path)
    dsl_exit.register_position("ETH", "long", 3000.0)
    recs = dsl_exit.rehydrate_from_exchange(
        [{"position": {"coin": "ETH", "szi": "0.5", "entryPx": "3000"}}])
    assert recs == []


def test_rehydrate_preserved_dex_not_reported(monkeypatch, tmp_path):
    """A tracker preserved because its dex query failed is NOT a stale record."""
    _isolate_dsl_state(monkeypatch, tmp_path)
    dsl_exit.register_position("xyz:MUA", "long", 1.0)
    recs = dsl_exit.rehydrate_from_exchange([], queried_dexes=set())
    assert recs == []
    assert "xyz:MUA_long" in dsl_exit._active_positions


# ── settle: booking + cooldown ──────────────────────────────────────────────

def test_single_fill_books_close_and_arms_cooldown(ledger_file):
    mem = FakeMemory()
    now = time.time()
    fills = [_fill(ts=now - 60)]
    settled = settle_stale_closes(
        [_stale_rec(entry_time=now - 1500)],
        fetch_fills=lambda: fills, memory=mem,
        read_agent_config=lambda: {"loss_cooldown_min": 180,
                                   "stop_loss_cooldown_min": 360},
        now_ms=int(now * 1000))

    assert len(settled) == 1
    rows = _read(ledger_file)
    assert len(rows) == 1 and rows[0]["event"] == "CLOSE"
    assert rows[0]["coin"] == "ZETA" and rows[0]["side"] == "long"
    assert rows[0]["exit_px"] == pytest.approx(0.03811)
    # Exchange closedPnl is authoritative when present.
    assert rows[0]["realized_pnl_usd"] == pytest.approx(-11.92)
    assert "exchange_close" in (rows[0]["exit_reason"] or "")
    assert rows[0]["exit_type"] == "exchange_close"
    # Outcome store saw it too.
    assert len(mem.closes) == 1 and mem.closes[0]["coin"] == "ZETA"
    # Stop-class cooldown (360 > 180) armed.
    until = mem.cooldowns["ZETA"]
    assert now * 1000 < until <= (now + 361 * 60) * 1000


def test_winner_books_close_but_no_cooldown(ledger_file):
    mem = FakeMemory()
    now = time.time()
    fills = [_fill(px=0.0410, closed_pnl="12.50", ts=now - 30)]
    settled = settle_stale_closes(
        [_stale_rec(entry_time=now - 900)],
        fetch_fills=lambda: fills, memory=mem,
        read_agent_config=lambda: {"loss_cooldown_min": 180},
        now_ms=int(now * 1000))
    assert len(settled) == 1
    assert len(_read(ledger_file)) == 1
    assert mem.cooldowns == {}


def test_no_fill_evidence_books_nothing(ledger_file):
    """A stale drop with no attributable fill (flaky read?) must not invent a
    CLOSE — startup reconcile stays the backstop."""
    mem = FakeMemory()
    settled = settle_stale_closes(
        [_stale_rec()], fetch_fills=lambda: [], memory=mem,
        read_agent_config=lambda: {"loss_cooldown_min": 180})
    assert settled == []
    assert not ledger_file.exists() or _read(ledger_file) == []
    assert mem.cooldowns == {}


def test_fill_before_entry_ignored(ledger_file):
    """Only fills AFTER the dropped position's entry can settle it (no_pyramid
    means no overlap, but a previous cycle's close must not be reused)."""
    mem = FakeMemory()
    now = time.time()
    fills = [_fill(ts=now - 7200)]  # old close from a prior trade
    settled = settle_stale_closes(
        [_stale_rec(entry_time=now - 600)],
        fetch_fills=lambda: fills, memory=mem,
        read_agent_config=lambda: {"loss_cooldown_min": 180},
        now_ms=int(now * 1000))
    assert settled == []
    assert not ledger_file.exists() or _read(ledger_file) == []


def test_opposite_direction_fill_ignored(ledger_file):
    mem = FakeMemory()
    now = time.time()
    fills = [_fill(dirn="Close Short", ts=now - 10)]
    settled = settle_stale_closes(
        [_stale_rec(entry_time=now - 600)],
        fetch_fills=lambda: fills, memory=mem,
        read_agent_config=lambda: {"loss_cooldown_min": 180},
        now_ms=int(now * 1000))
    assert settled == []


def test_ambiguous_split_fills_without_closedpnl_skipped(ledger_file):
    mem = FakeMemory()
    now = time.time()
    fills = [_fill(sz=5000, closed_pnl=None, ts=now - 20),
             _fill(sz=4241.7, closed_pnl=None, ts=now - 10)]
    settled = settle_stale_closes(
        [_stale_rec(entry_time=now - 900)],
        fetch_fills=lambda: fills, memory=mem,
        read_agent_config=lambda: {"loss_cooldown_min": 180},
        now_ms=int(now * 1000))
    assert settled == []
    assert not ledger_file.exists() or _read(ledger_file) == []


def test_split_fills_with_closedpnl_aggregate_and_arm(ledger_file):
    mem = FakeMemory()
    now = time.time()
    fills = [_fill(sz=5000, closed_pnl="-8.00", ts=now - 20),
             _fill(sz=4241.7, closed_pnl="-3.92", ts=now - 10)]
    settled = settle_stale_closes(
        [_stale_rec(entry_time=now - 900)],
        fetch_fills=lambda: fills, memory=mem,
        read_agent_config=lambda: {"loss_cooldown_min": 180,
                                   "stop_loss_cooldown_min": 360},
        now_ms=int(now * 1000))
    assert len(settled) == 1
    rows = _read(ledger_file)
    assert len(rows) == 1
    assert rows[0]["realized_pnl_usd"] == pytest.approx(-11.92)
    assert "ZETA" in mem.cooldowns


def test_newest_first_fill_order_handled(ledger_file):
    """Real HL userFills is newest-first: exit px must be the CHRONOLOGICALLY
    last fill, not the first element of the response."""
    mem = FakeMemory()
    now = time.time()
    fills = [_fill(px=0.03815, closed_pnl="-6.00", ts=now - 5),   # newest first
             _fill(px=0.03811, closed_pnl="-5.92", ts=now - 40)]
    settled = settle_stale_closes(
        [_stale_rec(entry_time=now - 900)],
        fetch_fills=lambda: fills, memory=mem,
        read_agent_config=lambda: {"loss_cooldown_min": 180},
        now_ms=int(now * 1000))
    assert len(settled) == 1
    rows = _read(ledger_file)
    assert rows[0]["exit_px"] == pytest.approx(0.03815)
    assert rows[0]["realized_pnl_usd"] == pytest.approx(-11.92)


def test_fetch_failure_is_silent_and_defers(ledger_file):
    def boom():
        raise RuntimeError("network down")
    mem = FakeMemory()
    settled = settle_stale_closes(
        [_stale_rec()], fetch_fills=boom, memory=mem,
        read_agent_config=lambda: {"loss_cooldown_min": 180})
    assert settled == []
    assert not ledger_file.exists() or _read(ledger_file) == []


def test_pnl_formula_matches_reconcile_convention():
    """No closedPnl on the fill → executor/reconcile formula: leveraged spot
    move minus 2×taker-fee estimate; net_usd = gross − fee."""
    from hermes_trader.stale_close_settle import _compute_record
    now_ms = int(time.time() * 1000)
    rec = {"coin": "ZETA", "side": "long", "entry_px": 0.0394, "size": 9241.7,
           "leverage": 3, "entry_time": now_ms / 1000 - 1500}
    fills = [{"px": "0.03811", "sz": "9241.7", "time": now_ms}]  # no closedPnl
    out = _compute_record(rec, fills)
    spot = (0.03811 - 0.0394) / 0.0394 * 100
    assert out["spot_pct"] == pytest.approx(round(spot, 4))
    # reconcile convention: fees_pct = 0.025 × 2 × lev = 0.15 at 3x
    assert out["realized_pnl_pct"] == pytest.approx(
        round(spot * 3 - 0.025 * 2 * 3, 4))
    notional = 9241.7 * 0.0394
    # fee_usd follows the reconcile convention: round(notional*(fees_pct/lev)/100, 4)
    fee = round(notional * (0.15 / 3) / 100.0, 4)
    assert out["realized_pnl_usd"] == pytest.approx(
        round(notional * spot / 100.0 - fee, 4), abs=0.01)
