"""Tests for the squeeze-breakout SHADOW signal (no network).

The rule under test is a faithful port of scratch/research/signals.py
`squeeze_breakout` (the OOS-verified candidate): 1h Donchian(48) close
breakout with a decisive body, fresh only within `fresh_min` of the 1h
bar CLOSE. Here we unit-test the port, the cache/fetch layer, the ledger
dedup, and the executor trade-result attach.

Note on config: tests/conftest.py points `HERMES_AGENT_CONFIG_FILE` at a
disposable temp file so no test can read or write the live config through
`read_agent_config()`. Module functions are monkeypatched on the
`squeeze_signal` module itself (where they bind the name), and the one
"live config" test reads the real repo file by path (read-only) because it
is verifying a shipped artifact, not exercising the hot path.
"""

import json
import os
import time

import pytest

import hermes_trader.agents.squeeze_signal as sq
from hermes_trader.models.types import Candle

H = 3_600_000

# Sentinel: distinguishes "caller didn't pass final" from an explicit value.
_UNSET = object()

# A valid long breakout bar: close 104 > channel high 101, decisive body
# (|104-102|=2 >= 0.5*range=1.75). Order is (open, high, low, close).
_DEFAULT_LONG = (102.0, 105.0, 101.5, 104.0)


def _candles(n=50, base=100.0, final=_DEFAULT_LONG, close_age_ms=300):
    """n 1h candles: first n-1 flat in [base-1, base+1]; last bar = final.

    `final` is (open, high, low, close) of the breakout bar. The last bar
    opened `now - H - close_age_ms` and is confirmed (close_age_ms < H).
    """
    now = int(time.time() * 1000)
    t0 = now - H * n
    out = [
        Candle(t=t0 + i * H, o=base, h=base + 1.0, l=base - 1.0, c=base, v=1000.0)
        for i in range(n - 1)
    ]
    o, h, l, c = final
    out.append(Candle(t=now - H - close_age_ms, o=o, h=h, l=l, c=c, v=2000.0))
    return out


def _cfg(enabled=True, lookback=48, fresh_min=15.0, ttl=300.0):
    return {"enabled": enabled, "debug": False, "lookback": lookback,
            "fresh_min": fresh_min, "cache_ttl_seconds": ttl}


def _eval(coin="TEST", cfg=None, candles=None, side="long",
          final=_UNSET, close_age_ms=300):
    if candles is None:
        if final is _UNSET:
            candles = _candles(close_age_ms=close_age_ms)
        else:
            candles = _candles(final=final, close_age_ms=close_age_ms)
    return sq._evaluate(coin, side, cfg or _cfg(), candles)


def _cfg_gate(enabled=True, extreme_pct=5.0):
    c = _cfg(enabled=enabled)
    c["extreme_pct"] = extreme_pct
    return c


# ── Rule port ─────────────────────────────────────────────────────────────────
def test_long_breakout_fires():
    # close 104 > channel high 101, decisive body, 300ms fresh.
    sig = _eval()
    assert sig.active and sig.side == "long"
    assert sig.chan_high == 101.0 and sig.chan_low == 99.0
    assert sig.close == 104.0
    assert sig.ext_pct is not None
    assert sig.ext_pct == pytest.approx((104.0 - 101.0) / 104.0 * 100.0)
    assert sig.score is not None
    assert 0.6 <= sig.score <= 1.0
    assert sig.fresh_age_min is not None
    assert 0 <= sig.fresh_age_min < 15.0
    assert sig.atr1h is not None and sig.atr1h > 0
    assert sig.atr1h_pct is not None
    assert sig.breakout_bar_t is not None


def test_short_breakout_fires():
    # close 92 < channel low 99, body 4 >= 0.5*range 2.
    sig = _eval(final=(96.0, 95.0, 91.0, 92.0))
    assert sig.active and sig.side == "short"
    assert sig.ext_pct is not None
    assert sig.ext_pct == pytest.approx((99.0 - 92.0) / 92.0 * 100.0)


def test_wick_only_pier_rejected():
    # closes above the channel but body is 0.2 of a 2.0 range (< 50%).
    sig = _eval(final=(101.0, 102.5, 100.5, 101.2))
    assert not sig.active
    assert sig.error and "wick-only" in sig.error


def test_inside_channel_rejected():
    # strictly inside the channel: close 100.5 with chan [99, 101].
    sig = _eval(final=(100.0, 101.0, 99.5, 100.5))
    assert not sig.active
    assert sig.error == "inside channel"


def test_stale_rejected():
    # 1h close 30m ago > 15m window -> stale.
    sig = _eval(close_age_ms=30 * 60_000)
    assert not sig.active
    assert sig.error and "stale" in sig.error


def test_forming_bar_dropped():
    # A still-forming bar on top of a confirmed non-breakout bar is ignored
    # -> the latest CONFIRMED bar is inside the channel.
    candles = _candles(final=(100.0, 101.0, 99.5, 100.5))
    last = candles[-1]  # the confirmed (non-breakout) bar
    candles.append(Candle(t=last.t + H, o=100.5, h=110.0, l=100.5, c=109.0, v=9999.0))
    sig = _eval(candles=candles)
    assert not sig.active
    assert sig.error == "inside channel"


def test_insufficient_history():
    sig = _eval(candles=_candles(n=10))
    assert not sig.active
    assert sig.error and "insufficient" in sig.error


# ── Cache / fetch layer ───────────────────────────────────────────────────────
def test_disabled_config_is_inactive_no_fetch(monkeypatch):
    calls = []
    monkeypatch.setattr(sq, "read_agent_config",
                        lambda: {"squeeze_signal": _cfg(enabled=False)})
    monkeypatch.setattr(sq, "fetch_hl_candles",
                        lambda *a, **k: (calls.append(a), _candles())[1])
    sq._cache.clear()
    sig = sq._fetch("TEST", "long")
    assert not sig.active and sig.error == "disabled"
    assert calls == []  # config short-circuits before any fetch


def test_fetch_caches_and_is_reentrant(monkeypatch):
    calls = []
    monkeypatch.setattr(sq, "read_agent_config",
                        lambda: {"squeeze_signal": _cfg()})

    def _fake(coin, interval, count):
        calls.append((coin, interval, count))
        return _candles()

    monkeypatch.setattr(sq, "fetch_hl_candles", _fake)
    sq._cache.clear()
    s1 = sq._fetch("TEST", "long")
    s2 = sq._fetch("TEST", "long")
    s3 = sq._fetch("TEST", "short")  # alignment recomputed per call
    assert s1 is s2 is s3 and s1.active
    assert len(calls) == 1  # 2nd and 3rd reads are cache hits
    assert s3.verdict_side == "short" and s3.side == "long"
    # request window = lookback + warmup
    assert calls[0] == ("TEST", "1h", 73)


# ── Composite entry-gate flag (chan_pos + extreme_no_breakout) ────────────────
# With the standard _candles fixture the 48-bar channel is [99, 101], so
# chan_pos = (close - 99) / 2 — the fixture close 100.5 reads 0.75.
def test_chan_pos_inside_channel():
    sig = _eval(final=(100.0, 101.0, 99.5, 100.5))
    assert sig.chan_pos == pytest.approx(0.75)


def test_chan_pos_breakout_above_one():
    sig = _eval()  # close 104 above channel high 101
    assert sig.chan_pos == pytest.approx(2.5)


def test_chasing_long_at_top_no_breakout_flagged():
    # close 100.95 inside the channel: 97.5% of the 48h range — a long here
    # is chasing the top with no fresh breakout confirmation.
    sig = _eval(final=(100.0, 101.0, 99.5, 100.95))
    assert not sig.active
    assert sig.chan_pos == pytest.approx(0.975)
    assert sig.extreme_no_breakout is True


def test_confirmed_breakout_suppresses_flag():
    # The same top-of-range zone WITH a fresh aligned breakout is the
    # confirmed case the gate is designed to let through.
    sig = _eval()  # active long breakout, chan_pos 2.5
    assert sig.active and sig.side == "long"
    assert sig.chan_pos == pytest.approx(2.5)
    assert sig.extreme_no_breakout is False


def test_chasing_short_at_bottom_flagged():
    # close 99.05 inside the channel, 2.5% of the 48h range — a short here
    # is chasing the bottom with no confirmation.
    sig = _eval(final=(100.0, 100.5, 99.0, 99.05), side="short")
    assert not sig.active
    assert sig.chan_pos == pytest.approx(0.025)
    assert sig.extreme_no_breakout is True


def test_short_breakout_suppresses_flag():
    sig = _eval(final=(96.0, 95.0, 91.0, 92.0), side="short")
    assert sig.active and sig.side == "short"
    assert sig.chan_pos == pytest.approx(-3.5)  # (92 - 99) / (101 - 99)
    assert sig.extreme_no_breakout is False


def test_opposite_extreme_not_flagged():
    # Top of the range is the WRONG extreme for a short candidate.
    sig = _eval(final=(100.0, 101.0, 99.5, 100.95), side="short")
    assert sig.chan_pos == pytest.approx(0.975)
    assert sig.extreme_no_breakout is False


def test_mid_range_not_flagged():
    sig = _eval(final=(100.0, 101.0, 99.5, 100.5), side="long")
    assert sig.chan_pos == pytest.approx(0.75)
    assert sig.extreme_no_breakout is False


def test_zero_extreme_pct_disables_flag():
    sig = _eval(final=(100.0, 101.0, 99.5, 100.95), cfg=_cfg_gate(extreme_pct=0.0))
    assert sig.chan_pos == pytest.approx(0.975)
    assert sig.extreme_no_breakout is False


def test_stale_breakout_is_unconfirmed(monkeypatch):
    # A breakout that aged out past fresh_min is NOT a confirmation: the
    # candidate sits above the channel high with no fresh signal -> flagged.
    # (The 15m freshness window is the anti-re-fire gate; chasing the top
    # 40 minutes after the 1h close is exactly the PURR pattern.)
    sig = _eval(close_age_ms=30 * 60_000)
    assert not sig.active and "stale" in (sig.error or "")
    assert sig.chan_pos == pytest.approx(2.5)
    assert sig.extreme_no_breakout is True


def test_cache_hit_recomputes_flag_for_new_side(monkeypatch):
    # The flag depends on the candidate side — a cache hit for a different
    # side must re-derive it from the stored chan_pos without a refetch.
    # (Only ACTIVE signals are cached; a default long breakout reads
    # chan_pos 2.5, so for a long candidate the flag is False and stays
    # False for a short candidate — the recompute is still exercised.)
    monkeypatch.setattr(sq, "read_agent_config",
                        lambda: {"squeeze_signal": _cfg_gate()})

    calls = []

    def _fake(coin, interval, count):
        calls.append(coin)
        return _candles()  # active long breakout

    monkeypatch.setattr(sq, "fetch_hl_candles", _fake)
    sq._cache.clear()
    long_sig = sq._fetch("TEST", "long")
    assert long_sig.active and long_sig.extreme_no_breakout is False
    short_sig = sq._fetch("TEST", "short")  # cache hit, no refetch
    assert short_sig is long_sig
    assert len(calls) == 1
    assert short_sig.verdict_side == "short"
    assert short_sig.extreme_no_breakout is False  # recomputed for new side

    # Direct: _set_gate IS side-dependent on the stored chan_pos.
    sig = sq._inactive("TEST", "long", "disabled", 48)
    sig.chan_pos = 0.975
    sq._set_gate(sig, _cfg_gate())
    assert sig.extreme_no_breakout is True     # long at the top, no breakout
    sig.verdict_side = "short"
    sq._set_gate(sig, _cfg_gate())
    assert sig.extreme_no_breakout is False    # wrong extreme for a short


# ── Executor trade-result attach ──────────────────────────────────────────────
def _attach(sig, result, coin, side, monkeypatch):
    monkeypatch.setattr(sq, "get_squeeze_signal_sync", lambda c, s: sig)
    monkeypatch.setattr(sq, "record_shadow", lambda c, s, x, **k: None)
    import hermes_trader.agents.executor as ex
    ex._attach_squeeze_to_result(result, coin, side)
    return result


def test_attach_result_active_sets_fields(monkeypatch):
    sig = _eval()  # active long
    result = {"executed": True, "mode": "LIVE", "analysis_id": "A1"}
    _attach(sig, result, "TEST", "long", monkeypatch)
    assert result["squeeze_side"] == "long"
    assert result["squeeze_score"] is not None
    assert result["squeeze_ext_pct"] is not None
    assert result["squeeze_aligned"] is True
    assert result["squeeze_error"] is None
    assert "squeeze_counter_signal" not in result
    assert result["squeeze_chan_pos"] == pytest.approx(2.5, abs=0.001)
    assert result["squeeze_extreme_no_breakout"] is False


def test_attach_result_chasing_flag_on_inactive(monkeypatch):
    # The chasing case lives in the INACTIVE path (inside channel, no
    # fresh breakout) — the new gate fields must be attached there too.
    sig = _eval(final=(100.0, 101.0, 99.5, 100.95))  # chan_pos 0.975
    assert sig.extreme_no_breakout is True
    result = {"executed": False, "mode": "SHADOW"}
    _attach(sig, result, "TEST", "long", monkeypatch)
    assert result["squeeze_side"] is None
    assert result["squeeze_error"] == "inside channel"
    assert result["squeeze_chan_pos"] == pytest.approx(0.975, abs=0.001)
    assert result["squeeze_extreme_no_breakout"] is True


def test_attach_result_counter_signal_flagged(monkeypatch):
    sig = _eval()  # active long; candidate trades short -> AGAINST
    result = {"executed": True, "mode": "LIVE"}
    _attach(sig, result, "TEST", "short", monkeypatch)
    assert result["squeeze_side"] == "long"
    assert result["squeeze_aligned"] is False
    assert result["squeeze_counter_signal"] is True


def test_attach_result_inactive_sets_error(monkeypatch):
    sig = _eval(close_age_ms=30 * 60_000)  # stale -> inactive
    result = {"executed": False, "mode": "SHADOW"}
    _attach(sig, result, "TEST", "long", monkeypatch)
    assert result["squeeze_side"] is None
    assert result["squeeze_aligned"] is None
    assert result["squeeze_error"] and "stale" in result["squeeze_error"]


# ── Ledger dedup (one row per breakout bar) ───────────────────────────────────
def test_record_shadow_dedup_per_breakout_bar(tmp_path, monkeypatch):
    # HERMES_LEDGER_FILE is pointed at a temp path by tests/conftest.py.
    import hermes_trader.ledger as ledger_mod

    path = ledger_mod.LEDGER_FILE
    assert path and os.path.isabs(path), "ledger file path not isolated"

    def _read_rows():
        try:
            with open(path) as f:
                return [json.loads(l) for l in f if l.strip()]
        except FileNotFoundError:
            return []

    before = len(_read_rows())
    sig = _eval()
    sq._logged_breakouts.clear()
    sq.record_shadow("TEST", "long", sig, analysis_id="A1", mode="LIVE")
    sq.record_shadow("TEST", "long", sig, analysis_id="A2", mode="LIVE")  # dup
    rows = _read_rows()[before:]
    assert len(rows) == 1
    r = rows[0]
    assert r["event"] == "SHADOW"
    assert r["signal"] == "squeeze_breakout"
    assert r["coin"] == "TEST" and r["side"] == "long"
    assert r["analysis_id"] == "A1"
    assert r["breakout_bar_t"] == sig.breakout_bar_t
    entry, atr1 = sig.close, sig.atr1h
    assert entry is not None and atr1 is not None
    assert r["stop_px"] == pytest.approx(entry - atr1 * 1.5)
    assert r["tp_px"] == pytest.approx(entry + atr1 * 1.0)
    assert sig.logged is True

    # A NEW breakout bar (different t) logs again.
    sig2 = _eval()
    assert sig2.breakout_bar_t is not None
    sig2.breakout_bar_t = sig2.breakout_bar_t + H
    sig2.logged = False
    sq.record_shadow("TEST", "long", sig2, analysis_id="A3", mode="LIVE")
    assert len(_read_rows()[before:]) == 2


# ── Shipped artifact ──────────────────────────────────────────────────────────
def test_live_config_block_present():
    # The live .agent-config.json (repo root) carries the researched default
    # params. Reads the real file by path (read-only) — NOT read_agent_config(),
    # which conftest redirects to a disposable temp file.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo_root, ".agent-config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    sqcfg = cfg.get("squeeze_signal") or {}
    assert sqcfg.get("enabled") is True
    assert sqcfg.get("lookback") == 48
    assert sqcfg.get("fresh_min") == 15
    assert sqcfg.get("cache_ttl_seconds") == 300
    assert sqcfg.get("extreme_pct") == 5.0