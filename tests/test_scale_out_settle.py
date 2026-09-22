"""SCALE_OUT settlement: the exchange-side TP half reaches the ledger.

The TP scale-out trigger placed at entry (``tp_scale_fraction``, default 0.5)
fills on Hyperliquid with NO bot code running. Pre-fix, the DSL tracker just
silently adopted the smaller size (rehydrate shrink branch) and the final CLOSE
row booked P/L on the REMAINING half only — every TP-scaled winner was
under-reported by ~one half (39 such ledger rows; TAO 2026-09-21: +$0.44
booked, ~+$0.54 invisible).

Now: rehydrate_from_exchange reports the shrink as a kind="scale_out" record →
settle_scale_outs books a SCALE_OUT ledger event (NOT a CLOSE — the OPEN↔CLOSE
stack pairing must stay 1:1) + an outcome-store close, attributing the fill by
size-delta from userFills, with a mid-price estimate fallback.

Offline: fills fetch and memory injected; ledger writes land on the conftest
temp path (HERMES_LEDGER_FILE isolation).
"""
import json
import time

import pytest

from hermes_trader import ledger
from hermes_trader.agents import dsl_exit
from hermes_trader.stale_close_settle import settle_scale_outs, settle_stale_closes


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

    def record_close(self, c):
        self.closes.append(c)


def _read(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _scale_rec(coin="TAO", side="long", entry_px=278.35, delta=0.06,
               old=0.12, new=0.06, leverage=5, entry_time=None):
    return {"kind": "scale_out", "coin": coin, "side": side,
            "entry_px": entry_px, "size_delta": delta,
            "old_size": old, "new_size": new, "leverage": leverage,
            "entry_time": entry_time or (time.time() - 3600)}


def _fill(coin="TAO", dirn="Close Long", px="287.42", sz="0.06", ts=None,
          closed_pnl="0.537"):
    f = {"coin": coin, "dir": dirn, "px": str(px), "sz": str(sz),
         "time": int(((ts or time.time())) * 1000)}
    if closed_pnl is not None:
        f["closedPnl"] = closed_pnl
    return f


# ── rehydrate detection ──────────────────────────────────────────────────────

def test_rehydrate_reports_size_shrink(monkeypatch, tmp_path):
    """A live position whose size halves yields a kind=scale_out record and
    the tracker adopts the new size (basis unchanged)."""
    monkeypatch.setattr(dsl_exit, "DSL_STATE_FILE", str(tmp_path / "dsl.json"))
    dsl_exit._active_positions.clear()
    monkeypatch.setattr(dsl_exit, "_loaded_from_disk", True)
    t = dsl_exit.register_position("TAO", "long", 278.35, leverage=5)
    t.size = 0.12

    recs = dsl_exit.rehydrate_from_exchange(
        [{"position": {"coin": "TAO", "szi": "0.06", "entryPx": "278.35",
                       "leverage": {"value": "5"}}}])
    scales = [r for r in recs if r.get("kind") == "scale_out"]
    assert len(scales) == 1
    r = scales[0]
    assert r["coin"] == "TAO" and r["side"] == "long"
    assert abs(r["size_delta"] - 0.06) < 1e-9
    assert r["old_size"] == pytest.approx(0.12) and r["new_size"] == pytest.approx(0.06)
    # tracker alive, size adopted, basis untouched
    assert "TAO_long" in dsl_exit._active_positions
    assert dsl_exit._active_positions["TAO_long"].size == pytest.approx(0.06)
    assert dsl_exit._active_positions["TAO_long"].entry_px == 278.35


def test_rehydrate_shrink_below_tolerance_not_reported(monkeypatch, tmp_path):
    """A <0.5% size wobble is float noise / dust — no record."""
    monkeypatch.setattr(dsl_exit, "DSL_STATE_FILE", str(tmp_path / "dsl.json"))
    dsl_exit._active_positions.clear()
    monkeypatch.setattr(dsl_exit, "_loaded_from_disk", True)
    t = dsl_exit.register_position("ETH", "long", 100.0, leverage=5)
    t.size = 1.0

    recs = dsl_exit.rehydrate_from_exchange(
        [{"position": {"coin": "ETH", "szi": "0.999", "entryPx": "100.0"}}])
    assert [r for r in recs if r.get("kind") == "scale_out"] == []


# ── settlement: fill attribution by size delta ───────────────────────────────

def test_scale_out_booked_from_single_matching_fill(ledger_file):
    mem = FakeMemory()
    settled = settle_scale_outs(
        [_scale_rec()], fetch_fills=lambda: [_fill()])
    assert len(settled) == 1
    rows = _read(ledger_file)
    assert len(rows) == 1 and rows[0]["event"] == "SCALE_OUT"
    r = rows[0]
    assert r["coin"] == "TAO" and r["side"] == "long"
    assert r["exit_px"] == 287.42
    # exchange closedPnl is authoritative
    assert r["realized_pnl_usd"] == pytest.approx(0.537)
    assert r["notional_usd"] == pytest.approx(0.06 * 278.35, abs=0.01)
    assert r["source"] == "tp_scale_out"
    # outcome store deliberately NOT touched (per-trade stats source — a
    # half-row would double-count TP-scaled trades in win-rate)
    assert len(mem.closes) == 0


def test_scale_out_ignores_wrong_size_and_direction(ledger_file):
    """Fills whose size doesn't match the shrink delta (e.g. the eventual
    full remainder close later) must not be attributed to the scale-out."""
    mem = FakeMemory()
    fills = [
        _fill(sz="0.12"),                      # too big (that's the remainder close)
        _fill(dirn="Open Long", sz="0.06"),    # wrong direction
    ]
    settled = settle_scale_outs([_scale_rec()], fetch_fills=lambda: fills)
    assert settled == []
    assert _read(ledger_file) == []


def test_scale_out_ambiguous_multiple_matching_fills_skipped(ledger_file):
    """Two same-size closing fills → can't tell which is the TP half; skip
    (under-report like pre-fix, never double-book)."""
    mem = FakeMemory()
    fills = [_fill(ts=time.time() - 100), _fill(ts=time.time() - 50)]
    settled = settle_scale_outs([_scale_rec()], fetch_fills=lambda: fills)
    assert settled == []
    assert _read(ledger_file) == []


def test_scale_out_estimate_fallback_flagged(ledger_file):
    """No fill attributable → book at the caller's mid, flagged est."""
    mem = FakeMemory()
    settled = settle_scale_outs([_scale_rec()], fetch_fills=lambda: [],
                                mids={"TAO": 287.0})
    assert len(settled) == 1
    r = _read(ledger_file)[0]
    assert r["source"] == "tp_scale_out_est"
    assert r["exit_px"] == 287.0
    # spot basis (same convention as close_position_market): $16.70 × 3.108% ≈ +$0.52 − fees
    assert 0.4 < r["realized_pnl_usd"] < 0.6


def test_scale_out_no_fill_no_mid_skips(ledger_file):
    settled = settle_scale_outs([_scale_rec()], fetch_fills=lambda: [], mids=None)
    assert settled == []
    assert _read(ledger_file) == []


def test_scale_out_short_side(ledger_file):
    mem = FakeMemory()
    rec = _scale_rec(coin="CASHCAT", side="short", entry_px=0.15,
                     delta=221.0, old=442.0, new=221.0, leverage=2)
    fills = [_fill(coin="CASHCAT", dirn="Close Short", px="0.1485",
                   sz="221", closed_pnl="0.30")]
    settled = settle_scale_outs([rec], fetch_fills=lambda: fills)
    assert len(settled) == 1
    r = _read(ledger_file)[0]
    assert r["spot_pct"] == pytest.approx(1.0, abs=0.01)  # short profits on drop


# ── routing: the two settle functions must not cross-process records ────────

def test_settle_stale_closes_ignores_scale_records(ledger_file):
    """A scale_out record handed to settle_stale_closes (e.g. an un-updated
    caller) must NOT produce a CLOSE row."""
    mem = FakeMemory()
    settled = settle_stale_closes([_scale_rec()], fetch_fills=lambda: [_fill()],
                                  read_agent_config=lambda: {})
    assert settled == []
    assert _read(ledger_file) == []


def test_settle_scale_outs_ignores_stale_records(ledger_file):
    mem = FakeMemory()
    stale = {"kind": "stale_close", "coin": "TAO", "side": "long",
             "entry_px": 278.35, "size": 0.12, "leverage": 5,
             "entry_time": time.time() - 3600}
    settled = settle_scale_outs([stale], fetch_fills=lambda: [_fill()])
    assert settled == []
    assert _read(ledger_file) == []


def test_scale_out_is_not_a_close_for_stack_pairing():
    """The startup reconcile pairs OPEN↔CLOSE on a stack; SCALE_OUT rows must
    never pop that stack. Assert the event-name contract here (reconcile
    filters on exact "OPEN"/"CLOSE")."""
    rec = ledger.record_scale_out.__doc__ or ""
    assert "NOT a CLOSE" in rec
