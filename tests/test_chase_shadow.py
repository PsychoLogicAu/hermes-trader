"""CHASE-SHADOW instrument (agents/chase_shadow.py) — t_c95ea397.

Pins the properties that make a log-only accrual safe and comparable:
  * observation-only: returns None; emits exactly ONE line per SHORT entry
    evaluation; nothing for longs; nothing when config disables it
  * field math matches the 105d backtest harness (scratch/_ext_bt.py
    feats()/btc_mom4h + _ext_bt2.py btc_drift24) on CONFIRMED 1h bars
  * would_block fires only at dd24h <= bar (default -8.0)
  * fail-open: empty store / raising store NEVER propagate an exception;
    a data gap logs a nodata line (denominator preserved)
"""

import json
import logging
import time

import pytest

from hermes_trader.client import candlestore as cs
from hermes_trader.agents import chase_shadow as chs
from hermes_trader.agents.chase_shadow import (
    coin_features, btc_features, log_chase_shadow)

T0 = 1_700_000_000_000
H = 3_600_000


@pytest.fixture()
def store(tmp_path, monkeypatch):
    cs.reset_store()
    chs._ERR_STATE.update(count=0, last_msg=None)  # once-warned must not dedupe across tests
    monkeypatch.setenv("HERMES_CANDLESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(cs, "_cfg", lambda: {
        "enabled": True, "harvest": True,
        "intervals": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "prune_max_days": {"1m": 7}})
    yield cs.get_store()
    cs.reset_store()


def _coin_bars(closes, t0=T0):
    return [{"t": t0 + i * H, "o": c, "h": c, "l": c, "c": c, "v": 1.0}
            for i, c in enumerate(closes)]


# closes: 36 flat, 14 at 120 (24h high), then crash to 62 -> dd24h = -48.33
CRASH_CLOSES = [100.0] * 36 + [120.0] * 14 + [62.0] * 10
FLAT_CLOSES = [100.0] * 30 + [108.0] * 8 + [104.0] * 12   # dd24h = -3.70 -> no block


def _load(store, closes, coin="FOO", t0=T0):
    n = len(closes)
    store.upsert_bars(coin, "1h", _coin_bars(closes, t0), now_ms=t0 + (n + 2) * H)


def _btc(store, t0=T0, n=60):
    closes = [100.0 + i for i in range(n)]
    store.upsert_bars("BTC", "1h", _coin_bars(closes, t0), now_ms=t0 + (n + 2) * H)


def _lines(caplog):
    return [r.getMessage() for r in caplog.records
            if "CHASE-SHADOW" in r.getMessage()]


# ── feature math (mirror of the backtest harness) ──────────────────────

def test_features_match_backtest_math():
    bars = _coin_bars(CRASH_CLOSES)
    tdec = T0 + (len(CRASH_CLOSES) + 1) * H
    f = coin_features(bars, tdec)
    assert f is not None
    assert f["dd24h"] == pytest.approx((62.0 / 120.0 - 1) * 100, abs=1e-9)
    assert f["off24lo"] == pytest.approx(0.0, abs=1e-9)  # low IS the last bar
    assert f["mark"] == pytest.approx(62.0)
    # ema4p: recompute per harness (alpha .4 over all closes, seed first)
    e = 100.0
    for x in [100.0] * 36 + [120.0] * 14 + [62.0] * 10:
        e = 0.4 * x + 0.6 * e
    assert f["ema4p"] == pytest.approx((62.0 / e - 1) * 100, abs=1e-9)
    # tape age: first bar whose CLOSE is the 24h low is at index 50; tdec is
    # 11 bar-hours later
    assert f["tape_age_h"] == pytest.approx(11.0, abs=1e-9)


def test_features_need_30_bars():
    assert coin_features(_coin_bars([100.0] * 29), T0 + 40 * H) is None


def test_confirmed_only_filter():
    # a bar starting after tdec - 1h is still forming -> excluded
    bars = _coin_bars([100.0] * 30 + [50.0])
    tdec = T0 + 30 * H  # last bar (t=T0+30H) not confirmed yet
    f = coin_features(bars, tdec)
    assert f["dd24h"] == pytest.approx(0.0)


def test_btc_fields():
    bars = _coin_bars([100.0 + i for i in range(60)])
    f = btc_features(bars, T0 + 61 * H)
    assert f["btc_drift24"] == pytest.approx((159.0 / 135.0 - 1) * 100, abs=1e-9)
    assert f["btc_mom4h"] == pytest.approx((159.0 / 155.0 - 1) * 100, abs=1e-9)
    g = btc_features(_coin_bars([100.0] * 10), T0 + 11 * H)
    assert g["btc_drift24"] is None and g["btc_mom4h"] is not None


# ── line behavior ───────────────────────────────────────────────────────

def test_would_block_fires(store, caplog):
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    assert log_chase_shadow("FOO", "short", {}) is None
    lines = _lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert "would_block=Y" in line and "dd24h=-48.33" in line
    assert "ema4p=" in line and "off24lo=" in line and "tape_age_h=" in line
    assert "btc_drift24=" in line and "btc_mom4h=" in line and "mark=62" in line


def test_no_block_when_shallow(store, caplog):
    _load(store, FLAT_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short", {})
    lines = _lines(caplog)
    assert len(lines) == 1 and "would_block=N" in lines[0]


def test_configured_bar(store, caplog):
    _load(store, FLAT_CLOSES)  # dd24h = -3.85
    _btc(store)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short", {"chase_shadow": {"dd24h_bar_pct": -3.0}})
    assert "would_block=Y" in _lines(caplog)[0]


def test_long_side_silent(store, caplog):
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "long", {})
    assert _lines(caplog) == []


def test_disabled_silent(store, caplog):
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short", {"chase_shadow": {"enabled": False}})
    assert _lines(caplog) == []


def test_nodata_line_when_thin(store, caplog):
    _load(store, [100.0] * 10)  # <30 confirmed bars -> no opinion
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short", {})
    lines = _lines(caplog)
    assert len(lines) == 1 and "nodata" in lines[0] and "would_block=N" in lines[0]


def test_fail_open_on_broken_store(store, monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("store exploded")
    monkeypatch.setattr(cs, "series", boom)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short", {})  # must not raise
    assert any("fail-open" in m for m in _lines(caplog))


def test_no_store_env_is_nodata_not_crash(monkeypatch, caplog):
    cs.reset_store()
    monkeypatch.delenv("HERMES_CANDLESTORE_DIR", raising=False)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short", {})  # must not raise
    lines = _lines(caplog)
    assert len(lines) == 1 and "nodata" in lines[0]


# ── audit regression pins (independent review 2026-11-26, needs-fix major) ──
# Guards used to run OUTSIDE the try: {"chase_shadow": true} raised
# AttributeError INTO the executor (maybe_execute, unwrapped) and aborted a
# short-entry evaluation. Every guard is now inside; these pins falsify the
# escape for the whole malformed-config class.

def test_bool_shorthand_config_no_escape(store, caplog):
    """{"chase_shadow": true} — the exact shorthand C6 advertises — must be
    treated as defaults (enabled), NEVER raise into the caller."""
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    for cfg in ({"chase_shadow": True}, {"chase_shadow": []},
                {"chase_shadow": "on"}, {"chase_shadow": None}):
        log_chase_shadow("FOO", "short", cfg)  # must not raise
    assert all("would_block=Y" in m for m in _lines(caplog))  # defaults applied


def test_non_dict_config_no_escape(store, caplog):
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    for cfg in ([], ["x"], 42, "s", 0, None):
        log_chase_shadow("FOO", "short", cfg)  # must not raise
    assert len(_lines(caplog)) == 6


def test_enabled_string_false_silences(store, caplog):
    """The kill switch must work when JSON quoting turns false into "false"."""
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    for s in ("false", "False", " off ", "0", "no"):
        log_chase_shadow("FOO", "short", {"chase_shadow": {"enabled": s}})
    assert _lines(caplog) == []
    log_chase_shadow("FOO", "short", {"chase_shadow": {"enabled": "true"}})
    assert len(_lines(caplog)) == 1  # a truthy string keeps accrual on


def test_nan_bar_never_blocks(store, caplog):
    _load(store, CRASH_CLOSES)  # dd24h = -48.33
    _btc(store)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short",
                     {"chase_shadow": {"dd24h_bar_pct": float("nan")}})
    line = _lines(caplog)[0]
    assert "would_block=N" in line  # NaN comparisons are False: never fires


def test_garbage_bar_fail_opens_with_warn(store, caplog):
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short", {"chase_shadow": {"dd24h_bar_pct": "x"}})
    lines = _lines(caplog)
    assert len(lines) == 1 and "fail-open" in lines[0]


def test_trade_side_garbage_no_escape(store, caplog):
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", None, {})
    log_chase_shadow("FOO", 5, {})
    assert _lines(caplog) == []


# ---- re-entry lens (t_c95ea397 checkpoint, 11-21 plummet analysis) --------

def _write_ledger(tmp_path, monkeypatch, rows):
    from hermes_trader import ledger as L
    p = tmp_path / "trades.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(L, "LEDGER_FILE", str(p))


def _close_row(coin, ts, pnl, exit_type="floor_breach", fee=0.1):
    return {"event": "CLOSE", "coin": coin, "side": "short", "ts": ts,
            "ts_iso": "2026-11-26T00:00:00Z", "entry_px": 1.0, "exit_px": 1.0,
            "realized_pnl_usd": pnl, "fee_usd": fee, "funding_cost_usd": 0.0,
            "exit_type": exit_type}


def test_reentry_flag_y_after_recent_win(store, tmp_path, monkeypatch, caplog):
    now = int(time.time() * 1000)
    _write_ledger(tmp_path, monkeypatch,
                  [_close_row("FOO", now - 2 * 3_600_000, 1.5)])
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short", {})
    line = _lines(caplog)[0]
    assert "reentry_flag=Y" in line
    assert "prev_net=+1.40" in line      # 1.5 - 0.1 fee
    assert "prev_stop=N" in line


def test_no_flag_after_loss_or_wide_gap(store, tmp_path, monkeypatch, caplog):
    now = int(time.time() * 1000)
    _write_ledger(tmp_path, monkeypatch,
                  [_close_row("FOO", now - 2 * 3_600_000, -3.0, "max_loss")])
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short", {})
    line = _lines(caplog)[0]
    assert "reentry_flag=N" in line      # prior close lost -> not the cell
    assert "prev_stop=Y" in line
    # same win, but 6h old: outside the default 4h candidate gap
    _write_ledger(tmp_path, monkeypatch,
                  [_close_row("FOO", now - 6 * 3_600_000, 1.5)])
    caplog.clear()
    log_chase_shadow("FOO", "short", {})
    assert "reentry_flag=N" in _lines(caplog)[0]


def test_reentry_fields_configurable_and_na(store, tmp_path, monkeypatch, caplog):
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    # missing ledger file -> NA fields, line still emitted, no crash
    from hermes_trader import ledger as L
    monkeypatch.setattr(L, "LEDGER_FILE", str(tmp_path / "missing.jsonl"))
    log_chase_shadow("FOO", "short", {})
    assert "reentry_flag=N gap_h=NA prev_net=NA" in _lines(caplog)[0]
    # candidate gap configurable
    now = int(time.time() * 1000)
    _write_ledger(tmp_path, monkeypatch,
                  [_close_row("FOO", now - 5 * 3_600_000, 1.5)])
    caplog.clear()
    log_chase_shadow("FOO", "short", {"chase_shadow": {"reentry_gap_max_h": 6.0}})
    assert "reentry_flag=Y" in _lines(caplog)[0]


def test_garbage_ledger_rows_fail_open(store, tmp_path, monkeypatch, caplog):
    from hermes_trader import ledger as L
    p = tmp_path / "trades.jsonl"
    p.write_text('not json\n{"event": "CLOSE", "coin": "FOO", "ts": bad}\n')
    monkeypatch.setattr(L, "LEDGER_FILE", str(p))
    _load(store, CRASH_CLOSES)
    _btc(store)
    caplog.set_level(logging.WARNING)
    log_chase_shadow("FOO", "short", {})
    lines = _lines(caplog)
    assert len(lines) == 1
    # lens fields go NA, but the CHASE line still computes and emits
    assert "dd24h=-" in lines[0] and "reentry_flag=N gap_h=NA" in lines[0]
