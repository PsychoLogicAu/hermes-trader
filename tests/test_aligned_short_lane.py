"""Tests for the lane shadow accrual (board t_2e23804b + SPEC ADDENDUM).

Two halves:
  1. perception._aligned_short_probe — the per-scan candidate probe, BOTH
     sides of the lane family:
       aligned_short: dn-momentum net (6h close-to-close <= -mom_pct,
         momentum-shadow semantics — NOT the Theil-Sen fired flag),
         composite >= comp_bar, ema8(4h) < ema21(4h) alignment.
       early_long (tag variant, gated by early_long_enabled, code default
         OFF): up arm evaluated + net >= +mom_pct + comp bar; promising
         cells TAGGED in variant (momhi_1dF / struct_1dT / aligned), rows
         matching none accrue as other_upfired — tagged, NOT filtered.
     Candidates ride a thread-safe side channel
     (take_aligned_short_lane_candidates drains it), never the
     surfacing/research path.
  2. risk_gates.aligned_short_lane_emit — the WOULD ADMIT line: live-state
     mirror (slots incl. this-cycle admits, no open position, per-coin
     re-entry cooldown), fields coin/ts/mid/comp/mom6h_net/ema4h/btc_tape/
     conf-if-asked + addendum fields lane/variant/ema4h_dist(+bucket)/
     ema1d/struct_flags/btc_tape_drift/anti/lane_priority(mode#rank).
     lane_priority orders the cycle walk (short_first default | long_first
     | composite) — tested explicitly, incl. the slot-contention case.
     Shadow-only: the function logs, it cannot admit.
  3. risk_gates.aligned_short_lane_live_admit + build_aligned_short_lane_analysis
     — the LIVE promotion experiment (owner call 2026-10-30, 5 lane slots):
     the named cell only, dedicated lane slots ON TOP of the shared pool,
     fail-closed on tape gap / unreadable occupancy, and the
     runner_entry_gate.cohort_admissions key that is the kill switch for the
     conf-0.0 mechanical entry at BOTH conf floors. Plus the ledger side:
     config_snapshot.lane on the OPEN row and count_open_with_lane().

All candle fetches and the BTC tape read are monkeypatched — no network.
"""
from __future__ import annotations

import json
import math
import sys
import time
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from hermes_trader.agents import perception, risk_gates  # noqa: E402
from hermes_trader.models.types import Candle  # noqa: E402

LINE_RE = (
    r"^\[aligned-short-lane\]\[SHADOW\] WOULD ADMIT (\S+) (SHORT|LONG) "
    r"ts=(\d+) mid=(\S+) comp=(\S+) mom6h_net=([+-]\d+\.\d)% "
    r"ema4h=(\d+/\d+) (\S+)([<>])(\S+) btc_tape=(\S+) "
    r"conf=(\S+) slots=(\d+)/(\d+) "
    r"lane=(\S+) variant=(\S+) ema4h_dist=([+-]\d+\.\d+)% "
    r"ema4h_dist_bucket=(deep|nearx) ema1d=(\S+) struct_flags=(\S+) "
    r"btc_tape_drift=(\S+) anti=(\S+) lane_priority=(\S+) "
)


def _closes_candles(closes, t0=0, step=300_000):
    return [Candle(t=t0 + i * step, o=c, h=c, l=c, c=c, v=1.0)
            for i, c in enumerate(closes)]


def _dn_hit(move=-5.0, reason=None):
    fired = move <= -3.0
    r = reason or (f"{move:.1f}% over 72 bars (downtrend)" if fired else "flat")
    return {"name": "downtrendMomentum", "score": 5.0 if fired else 0,
            "reason": r, "fired": fired}


def _up_hit(move=5.0, reason=None):
    fired = move >= 3.0
    r = reason or (f"{move:.1f}% over 72 bars (uptrend)" if fired else "flat")
    return {"name": "uptrendMomentum", "score": 5.0 if fired else 0,
            "reason": r, "fired": fired}


def _struct_hit(name, fired=True):
    return {"name": name, "score": 1.0 if fired else 0,
            "reason": "structure", "fired": fired}


def _scan_config(**over):
    cfg = {
        "scan": {"candleInterval": "5m", "candleCount": 100, "cacheTtlMs": 0,
                 "cacheTtlMs1h": 0},
        "thresholds": {"trendMomentumLookback": 72, "trendMomentumPct": 3.0,
                       "sigmaThreshold": 3.0, "breakoutLookback": 48,
                       "bbLength": 20, "bbStdDev": 2.0, "adxPeriod": 14,
                       "momentumLookback": 6, "momentumPct": 3.0},
        "runner_entry_gate": {"min_composite": 30.0},
        "weights": {},
    }
    cfg.update(over)
    return cfg


# ── probe: entry-candidate conditions (aligned_short side) ──────────────────

class TestProbe:
    LANE = {"enabled": True}

    def _c5(self, net_pct=-5.0, n=100):
        # flat 100s, last close moved by net_pct vs close 72 bars back
        closes = [100.0] * n
        closes[-1] = 100.0 * (1 + net_pct / 100)
        return _closes_candles(closes)

    def _patch4h(self, monkeypatch, closes4h):
        monkeypatch.setattr(
            perception, "_fetch_candles_sync",
            lambda coin, interval, count, ttl: _closes_candles(closes4h, step=14_400_000))

    def test_happy_path_fields(self, monkeypatch):
        self._patch4h(monkeypatch, [200.0 - i for i in range(40)])  # falling -> ema8<ema21
        probe = perception._aligned_short_probe(
            "TESTC", self._c5(-5.0), 97.5, 35.0, [_dn_hit(-5.0)], self.LANE, _scan_config())
        assert probe is not None
        assert probe["coin"] == "TESTC"
        assert probe["lane"] == "aligned_short"
        assert probe["side"] == "short"
        assert probe["variant"] == ["aligned_dn4h"]
        assert probe["ema4h_bucket"] == "deep"
        assert probe["ema4h_dist_pct"] < -0.7
        assert probe["mom6h_net"] == pytest.approx(-5.0, abs=0.01)
        assert probe["comp"] == 35.0
        assert probe["ema4h_pair"] == "8/21"
        assert probe["ema4h_fast"] < probe["ema4h_slow"]
        assert probe["ts"] > 0

    def test_net_too_shallow_rejected(self, monkeypatch):
        self._patch4h(monkeypatch, [200.0 - i for i in range(40)])
        assert perception._aligned_short_probe(
            "TESTC", self._c5(-2.0), 98.0, 35.0, [_dn_hit(-2.0)],
            self.LANE, _scan_config()) is None

    def test_composite_below_bar_rejected(self, monkeypatch):
        self._patch4h(monkeypatch, [200.0 - i for i in range(40)])
        assert perception._aligned_short_probe(
            "TESTC", self._c5(-5.0), 97.5, 29.9, [_dn_hit(-5.0)],
            self.LANE, _scan_config()) is None
        # lane comp_bar override accepts it
        p = perception._aligned_short_probe(
            "TESTC", self._c5(-5.0), 97.5, 29.9, [_dn_hit(-5.0)],
            {**self.LANE, "comp_bar": 20.0}, _scan_config())
        assert p is not None

    def test_ema_misalignment_rejected(self, monkeypatch):
        self._patch4h(monkeypatch, [100.0 + i for i in range(40)])  # rising -> ema8>ema21
        assert perception._aligned_short_probe(
            "TESTC", self._c5(-5.0), 97.5, 35.0, [_dn_hit(-5.0)],
            self.LANE, _scan_config()) is None

    def test_4h_data_gap_no_opinion(self, monkeypatch):
        monkeypatch.setattr(perception, "_fetch_candles_sync",
                            lambda *a, **k: None)
        assert perception._aligned_short_probe(
            "TESTC", self._c5(-5.0), 97.5, 35.0, [_dn_hit(-5.0)],
            self.LANE, _scan_config()) is None

    def test_dn_arm_insufficient_history_rejected(self, monkeypatch):
        self._patch4h(monkeypatch, [200.0 - i for i in range(40)])
        assert perception._aligned_short_probe(
            "TESTC", self._c5(-5.0), 97.5, 35.0,
            [_dn_hit(-5.0, reason="insufficient_history")],
            self.LANE, _scan_config()) is None

    def test_5m_history_too_short_rejected(self, monkeypatch):
        self._patch4h(monkeypatch, [200.0 - i for i in range(40)])
        assert perception._aligned_short_probe(
            "TESTC", self._c5(-5.0, n=40), 97.5, 35.0, [_dn_hit(-5.0)],
            self.LANE, _scan_config()) is None

    def test_nearx_bucket_is_attribute_not_block(self, monkeypatch):
        # nearx_band widened -> a still-aligned short lands in the nearx
        # bucket but is NOT rejected (anti-rule recorded, never blocks v1)
        self._patch4h(monkeypatch, [200.0 - i for i in range(40)])
        p = perception._aligned_short_probe(
            "TESTC", self._c5(-5.0), 97.5, 35.0, [_dn_hit(-5.0)],
            {**self.LANE, "nearx_band_pct": 50.0}, _scan_config())
        assert p is not None and p["ema4h_bucket"] == "nearx"


# ── probe: EARLY-LONG tag variant (SPEC ADDENDUM 2026-10-22) ───────────────

class TestEarlyLongProbe:
    LANE = {"enabled": True, "early_long_enabled": True}

    def _c5(self, net_pct=5.0, n=100):
        closes = [100.0] * n
        closes[-1] = 100.0 * (1 + net_pct / 100)
        return _closes_candles(closes)

    def _patch_tf(self, monkeypatch, closes4h, closes1d):
        def fake(coin, interval, count, ttl):
            if interval == "4h":
                return _closes_candles(closes4h, step=14_400_000)
            if interval == "1d":
                return _closes_candles(closes1d, step=86_400_000)
            return None
        monkeypatch.setattr(perception, "_fetch_candles_sync", fake)

    def test_momhi_1dF_tagged(self, monkeypatch):
        self._patch_tf(monkeypatch,
                       [100.0 - i for i in range(40)],   # 4h falling
                       [200.0 - i for i in range(40)])   # 1d falling (1dF)
        p = perception._aligned_short_probe(
            "TESTC", self._c5(+7.0), 107.0, 35.0, [_up_hit(7.0)],
            self.LANE, _scan_config())
        assert p is not None
        assert p["lane"] == "early_long" and p["side"] == "long"
        assert "momhi_1dF" in p["variant"]
        assert p["ema1d_fast"] < p["ema1d_slow"]
        assert p["ema4h_bucket"] == "deep"
        assert p["struct_flags"] == []

    def test_mom_lo_not_tagged_as_momhi(self, monkeypatch):
        # net +4% (< mom_hi 6) on 1dF: not the momhi cell — accrues tagged
        # other_upfired, still logged (tagged not filtered).
        self._patch_tf(monkeypatch,
                       [100.0 - i for i in range(40)],
                       [200.0 - i for i in range(40)])
        p = perception._aligned_short_probe(
            "TESTC", self._c5(+4.0), 104.0, 35.0, [_up_hit(4.0)],
            self.LANE, _scan_config())
        assert p is not None and p["variant"] == ["other_upfired"]

    def test_struct_1dT_tagged(self, monkeypatch):
        self._patch_tf(monkeypatch,
                       [100.0 - i for i in range(40)],   # 4h falling
                       [100.0 + i for i in range(40)])   # 1d rising (1dT)
        p = perception._aligned_short_probe(
            "TESTC", self._c5(+4.0), 104.0, 35.0,
            [_up_hit(4.0), _struct_hit("trendFlip1h"),
             _struct_hit("higherLows1h", fired=False)],
            self.LANE, _scan_config())
        assert p is not None and "struct_1dT" in p["variant"]
        assert p["struct_flags"] == ["trendFlip1h"]

    def test_aligned_tag_recorded_not_filtered(self, monkeypatch):
        # fully aligned long (4hT+1dT), NO struct: variant=[aligned]; the
        # tapeup_only anti-check happens at EMIT time — probe still logs it.
        self._patch_tf(monkeypatch,
                       [100.0 + i for i in range(40)],
                       [100.0 + i for i in range(40)])
        p = perception._aligned_short_probe(
            "TESTC", self._c5(+4.0), 104.0, 35.0, [_up_hit(4.0)],
            self.LANE, _scan_config())
        assert p is not None and p["variant"] == ["aligned"]
        assert p["struct_flags"] == []

    def test_struct_and_not_aligned_4h_tags_struct_only(self, monkeypatch):
        self._patch_tf(monkeypatch,
                       [100.0 - i for i in range(40)],
                       [100.0 + i for i in range(40)])
        p = perception._aligned_short_probe(
            "TESTC", self._c5(+7.0), 107.0, 35.0,
            [_up_hit(7.0), _struct_hit("volumeBuildup1h")],
            self.LANE, _scan_config())
        assert p is not None
        assert p["variant"] == ["struct_1dT"]  # 4h not aligned up -> no aligned tag

    def test_disabled_by_default_byte_identical(self, monkeypatch):
        # config without early_long_enabled: early-long shapes return None —
        # the code deploy lands byte-identical until the config flip.
        self._patch_tf(monkeypatch,
                       [200.0 - i for i in range(40)],
                       [200.0 - i for i in range(40)])
        assert perception._aligned_short_probe(
            "TESTC", self._c5(+7.0), 107.0, 35.0, [_up_hit(7.0)],
            {"enabled": True}, _scan_config()) is None

    def test_up_arm_insufficient_history_rejected(self, monkeypatch):
        self._patch_tf(monkeypatch,
                       [200.0 - i for i in range(40)],
                       [200.0 - i for i in range(40)])
        assert perception._aligned_short_probe(
            "TESTC", self._c5(+7.0), 107.0, 35.0,
            [_up_hit(7.0, reason="insufficient_history")],
            self.LANE, _scan_config()) is None

    def test_1d_data_gap_no_opinion(self, monkeypatch):
        monkeypatch.setattr(
            perception, "_fetch_candles_sync",
            lambda coin, interval, count, ttl:
            _closes_candles([100.0 + i for i in range(40)], step=14_400_000)
            if interval == "4h" else None)
        assert perception._aligned_short_probe(
            "TESTC", self._c5(+7.0), 107.0, 35.0, [_up_hit(7.0)],
            self.LANE, _scan_config()) is None

    def test_mom_hi_configurable(self, monkeypatch):
        self._patch_tf(monkeypatch,
                       [100.0 - i for i in range(40)],
                       [200.0 - i for i in range(40)])
        p = perception._aligned_short_probe(
            "TESTC", self._c5(+4.0), 104.0, 35.0, [_up_hit(4.0)],
            {**self.LANE, "mom_hi_pct": 3.5}, _scan_config())
        assert p is not None and "momhi_1dF" in p["variant"]

    def test_short_side_wins_when_net_negative(self, monkeypatch):
        # a dn candidate never takes the early_long path even when enabled
        self._patch_tf(monkeypatch,
                       [200.0 - i for i in range(40)],
                       [200.0 - i for i in range(40)])
        p = perception._aligned_short_probe(
            "TESTC", self._c5(-5.0), 97.5, 35.0, [_dn_hit(-5.0)],
            self.LANE, _scan_config())
        assert p is not None and p["lane"] == "aligned_short"


# ── side channel: scan records, loop drains ─────────────────────────────────

class TestSideChannel:
    def setup_method(self):
        perception._lane_candidates.clear()

    def test_take_drains(self):
        perception._note_lane_candidate({"coin": "A"})
        perception._note_lane_candidate({"coin": "B"})
        assert [c["coin"] for c in perception.take_aligned_short_lane_candidates()] == ["A", "B"]
        assert perception.take_aligned_short_lane_candidates() == []

    def test_scan_records_candidate_even_when_coin_dropped(self, monkeypatch):
        """A coin the composite gate drops must still accrue (the lane
        cohort is defined by the probe, not by surfacing)."""
        perception._lane_candidates.clear()
        # grind down -5% over the last 72 bars (no single-bar spike, so no
        # burst bypass; composite lands ~50 -> dropped by min_score 90)
        closes = [100.0] * 28 + [100.0 - 5.0 * (i + 1) / 72 for i in range(72)]

        def fake_fetch(coin, interval, count, ttl):
            if interval == "5m" or interval == "1h":
                return _closes_candles(closes,
                                       step=300_000 if interval == "5m" else 3_600_000)
            return _closes_candles([200.0 - i for i in range(40)], step=14_400_000)
        monkeypatch.setattr(perception, "_fetch_candles_sync", fake_fetch)

        market = {"coin": "TESTC", "type": "perp"}
        cfg = _scan_config(weights={"trendStrength": 1.0, "volumeSpike": 1.0,
                                    "downtrendMomentum": 0.0},
                           aligned_short_lane={"enabled": True})
        ok, result = perception._scan_single_market(
            market, 95.0, cfg, 90.0, None, False, False)  # trend bypass OFF -> dropped
        assert ok and result is None  # live path untouched: coin not surfaced
        cands = perception.take_aligned_short_lane_candidates()
        assert [c["coin"] for c in cands] == ["TESTC"]
        assert cands[0]["comp"] >= 30.0
        assert cands[0]["mom6h_net"] == pytest.approx(-5.0, abs=0.01)
        assert cands[0]["lane"] == "aligned_short"

    def test_lane_disabled_records_nothing(self, monkeypatch):
        perception._lane_candidates.clear()
        closes = [100.0] * 100
        closes[-1] = 95.0
        monkeypatch.setattr(perception, "_fetch_candles_sync",
                            lambda coin, interval, count, ttl: _closes_candles(closes))
        market = {"coin": "TESTC", "type": "perp"}
        ok, _ = perception._scan_single_market(market, 95.0, _scan_config(), 90.0)
        assert ok
        assert perception.take_aligned_short_lane_candidates() == []

    def test_single_call_site(self):
        src = Path(perception.__file__).read_text()
        assert src.count("_note_lane_candidate(") == 2  # def + one call site


# ── emitter: WOULD ADMIT line + live-state mirror ───────────────────────────

CAND = [{"coin": "APT", "ts": 1792600000000, "mid": 8.12, "comp": 34.0,
         "side": "SHORT", "lane": "aligned_short", "variant": ["aligned_dn4h"],
         "mom6h_net": -6.3, "ema4h_fast": 8.02, "ema4h_slow": 8.44,
         "ema4h_pair": "8/21", "ema4h_dist_pct": -4.976,
         "ema4h_bucket": "deep"}]

CAND_L = [{"coin": "LONGC", "ts": 1792600000001, "mid": 9.0, "comp": 60.0,
           "side": "LONG", "lane": "early_long", "variant": ["momhi_1dF"],
           "mom6h_net": 7.1, "ema4h_fast": 9.3, "ema4h_slow": 9.0,
           "ema4h_pair": "8/21", "ema4h_dist_pct": 3.333,
           "ema4h_bucket": "deep", "ema1d_fast": 7.8, "ema1d_slow": 8.2,
           "ema1d_pair": "8/21", "struct_flags": []}]


def _emit_lane(monkeypatch, caplog, cands=None, cfg=None,
               open_coins=frozenset(), slots_used=2, max_slots=8, conf=None,
               tape_drift=-4.2, tape_vol=3.1):
    from hermes_trader.agents import market_regime
    monkeypatch.setattr(market_regime, "btc_tape_activity",
                        lambda force=False: {"vol": tape_vol, "drift": tape_drift})
    caplog.set_level("WARNING", logger="hermes_trader.agents.risk_gates")
    return risk_gates.aligned_short_lane_emit(
        cands if cands is not None else list(CAND),
        cfg if cfg is not None else {"enabled": True},
        open_coins=set(open_coins), slots_used=slots_used,
        max_slots=max_slots, conf_by_coin=conf or {})


class TestEmit:
    def setup_method(self):
        risk_gates._as_lane_last_emit.clear()

    def _emit(self, monkeypatch, caplog, **kw):
        return _emit_lane(monkeypatch, caplog, **kw)

    def test_line_format_all_fields(self, monkeypatch, caplog):
        n = self._emit(monkeypatch, caplog, conf={"APT": 0.72})
        assert n == 1
        line = next(r.message for r in caplog.records
                    if "aligned-short-lane" in r.getMessage())
        m = re_match(line)
        assert m, line
        assert m.group(1) == "APT"
        assert m.group(2) == "SHORT"
        assert m.group(3) == "1792600000000"
        assert m.group(6) == "-6.3"
        assert m.group(8) == "8.02" and m.group(9) == "<" and m.group(10) == "8.44"
        assert m.group(11) == "vol+3.10%/drift-4.20%"
        assert m.group(12) == "0.72"
        assert m.group(15) == "aligned_short"
        assert m.group(16) == "aligned_dn4h"
        assert m.group(18) == "deep"
        assert m.group(19) == "n/a"
        assert m.group(20) == "-"
        assert m.group(22) == "-"
        assert m.group(23) == "short_first#1"
        assert "NOT admitting (shadow v1" in line

    def test_long_line_fields(self, monkeypatch, caplog):
        n = self._emit(monkeypatch, caplog, cands=list(CAND_L), conf={})
        assert n == 1
        line = next(r.message for r in caplog.records
                    if "aligned-short-lane" in r.getMessage())
        m = re_match(line)
        assert m, line
        assert m.group(1) == "LONGC" and m.group(2) == "LONG"
        assert m.group(9) == ">"          # 4h fast ABOVE slow (4hT long)
        assert m.group(15) == "early_long"
        assert m.group(16) == "momhi_1dF"
        assert m.group(19) == "8/21:7.8<8.2"  # 1d fast BELOW slow (1dF)
        assert m.group(20) == "-"
        assert m.group(21) == "-4.20%"
        assert "conf=n/a(not-asked)" in line

    def test_conf_if_asked_variants(self, monkeypatch, caplog):
        self._emit(monkeypatch, caplog, conf={})
        lines = [r.getMessage() for r in caplog.records if "WOULD ADMIT" in r.getMessage()]
        assert len(lines) == 1 and "conf=n/a(not-asked)" in lines[0]
        caplog.clear()
        risk_gates._as_lane_last_emit.clear()
        self._emit(monkeypatch, caplog, conf={"APT": None})
        lines = [r.getMessage() for r in caplog.records if "WOULD ADMIT" in r.getMessage()]
        assert "conf=n/a" in lines[0] and "not-asked" not in lines[0]

    def test_disabled_config_emits_nothing(self, monkeypatch, caplog):
        assert self._emit(monkeypatch, caplog, cfg={}) == 0
        assert not [r for r in caplog.records if "WOULD ADMIT" in r.getMessage()]

    def test_open_position_skipped(self, monkeypatch, caplog):
        assert self._emit(monkeypatch, caplog, open_coins={"APT"}) == 0

    def test_full_slots_skipped(self, monkeypatch, caplog):
        assert self._emit(monkeypatch, caplog, slots_used=8, max_slots=8) == 0

    def test_reentry_cooldown_dedups(self, monkeypatch, caplog):
        assert self._emit(monkeypatch, caplog) == 1
        caplog.clear()
        assert self._emit(monkeypatch, caplog) == 0  # same coin, inside cooldown
        # shorter cooldown configured -> re-emits
        caplog.clear()
        assert self._emit(monkeypatch, caplog,
                          cfg={"enabled": True, "reentry_cooldown_min": 0}) == 1

    def test_tape_data_gap_still_emits(self, monkeypatch, caplog):
        from hermes_trader.agents import market_regime
        monkeypatch.setattr(market_regime, "btc_tape_activity",
                            lambda force=False: None)
        caplog.set_level("WARNING", logger="hermes_trader.agents.risk_gates")
        n = risk_gates.aligned_short_lane_emit(
            list(CAND), {"enabled": True}, open_coins=set(), slots_used=1,
            max_slots=8, conf_by_coin={})
        assert n == 1
        line = [r.getMessage() for r in caplog.records
                if "WOULD ADMIT" in r.getMessage()][0]
        assert "btc_tape=n/a" in line and "btc_tape_drift=n/a" in line

    def test_btc_tape_never_blocks_shadow_v1(self, monkeypatch, caplog):
        # quiet tape (the old quiet-tape gate would veto): field logged, not a block
        assert self._emit(monkeypatch, caplog, tape_vol=0.2, tape_drift=0.1) == 1

    def test_anti_nearx_recorded_not_blocking(self, monkeypatch, caplog):
        c = dict(CAND[0], ema4h_bucket="nearx", ema4h_dist_pct=-0.31)
        assert self._emit(monkeypatch, caplog, cands=[c]) == 1
        line = [r.getMessage() for r in caplog.records
                if "WOULD ADMIT" in r.getMessage()][0]
        assert "ema4h_dist_bucket=nearx" in line and "anti=nearx" in line

    def test_anti_tapeup_only_recorded_not_blocking(self, monkeypatch, caplog):
        c = dict(CAND_L[0], variant=["aligned"])
        assert self._emit(monkeypatch, caplog, cands=[c], tape_drift=0.5) == 1
        line = [r.getMessage() for r in caplog.records
                if "WOULD ADMIT" in r.getMessage()][0]
        assert "anti=tapeup_only" in line
        caplog.clear()
        risk_gates._as_lane_last_emit.clear()
        c2 = dict(c, struct_flags=["trendFlip1h"])  # struct support -> not tapeup-only
        assert self._emit(monkeypatch, caplog, cands=[c2], tape_drift=0.5) == 1
        line = [r.getMessage() for r in caplog.records
                if "WOULD ADMIT" in r.getMessage()][0]
        assert "anti=-" in line and "struct_flags=trendFlip1h" in line


# ── emitter: LANE PRIORITY (slot order across the two lanes) ───────────────

class TestLanePriority:
    def setup_method(self):
        risk_gates._as_lane_last_emit.clear()

    def _both(self):
        return [dict(CAND[0]), dict(CAND_L[0])]  # LONGC has comp 60 > APT 34

    def _lines(self, caplog):
        return [r.getMessage() for r in caplog.records
                if "WOULD ADMIT" in r.getMessage()]

    def test_short_first_books_short_before_long_despite_comp(self, monkeypatch, caplog):
        n = _emit_lane(monkeypatch, caplog, cands=self._both(),
                       cfg={"enabled": True, "lane_priority": "short_first"})
        assert n == 2
        lines = self._lines(caplog)
        assert "APT SHORT" in lines[0]
        assert "lane_priority=short_first#1" in lines[0]
        assert "LONGC LONG" in lines[1] and "lane_priority=short_first#2" in lines[1]

    def test_long_first_reverses_order(self, monkeypatch, caplog):
        n = _emit_lane(monkeypatch, caplog, cands=self._both(),
                       cfg={"enabled": True, "lane_priority": "long_first"})
        assert n == 2
        lines = self._lines(caplog)
        assert "LONGC LONG" in lines[0] and "lane_priority=long_first#1" in lines[0]
        assert "APT SHORT" in lines[1] and "lane_priority=long_first#2" in lines[1]

    def test_composite_mode_orders_by_comp_across_lanes(self, monkeypatch, caplog):
        n = _emit_lane(monkeypatch, caplog, cands=self._both(),
                       cfg={"enabled": True, "lane_priority": "composite"})
        assert n == 2
        lines = self._lines(caplog)
        assert "LONGC LONG" in lines[0] and "lane_priority=composite#1" in lines[0]
        assert "APT SHORT" in lines[1]

    def test_slot_contention_slot_goes_to_top_rank_short(self, monkeypatch, caplog):
        # one free slot, both lanes candidate: short_first books the short
        n = _emit_lane(monkeypatch, caplog, cands=self._both(),
                       slots_used=7, max_slots=8)
        assert n == 1
        lines = self._lines(caplog)
        assert "APT SHORT" in lines[0] and "LONGC" not in lines[0]

    def test_slot_contention_long_first_books_long(self, monkeypatch, caplog):
        n = _emit_lane(monkeypatch, caplog, cands=self._both(),
                       cfg={"enabled": True, "lane_priority": "long_first"},
                       slots_used=7, max_slots=8)
        assert n == 1
        lines = self._lines(caplog)
        assert "LONGC LONG" in lines[0] and "APT" not in lines[0]

    def test_malformed_mode_falls_back_short_first(self, monkeypatch, caplog):
        n = _emit_lane(monkeypatch, caplog, cands=self._both(),
                       cfg={"enabled": True, "lane_priority": "sideways"})
        assert n == 2
        assert "lane_priority=short_first#1" in self._lines(caplog)[0]


def re_match(line):
    import re
    return re.match(LINE_RE, line)


# ── integration: RAW probe record -> emit line ──────────────────────────────
# In production the emitter's input is exactly what the probe returns
# (lowercase side, no ema1d fields on shorts). The fixture classes above
# hand-build records, so they cannot catch a producer/consumer shape drift
# (a lowercase side reaching the line would break the review script's
# (SHORT|LONG) parse and the ema1d n/a guard). Feed the REAL probe output
# through the REAL emitter and assert the accrued line parses.

class TestProbeToEmitIntegration:
    def setup_method(self):
        risk_gates._as_lane_last_emit.clear()

    def _lines(self, caplog):
        return [r.getMessage() for r in caplog.records
                if "WOULD ADMIT" in r.getMessage()]

    def test_probe_short_record_emits_parseable_line(self, monkeypatch, caplog):
        monkeypatch.setattr(
            perception, "_fetch_candles_sync",
            lambda coin, interval, count, ttl: _closes_candles(
                [200.0 - i for i in range(40)], step=14_400_000))
        closes = [100.0] * 100
        closes[-1] = 95.0
        rec = perception._aligned_short_probe(
            "TESTC", _closes_candles(closes), 97.5, 35.0, [_dn_hit(-5.0)],
            {"enabled": True}, _scan_config())
        assert rec is not None and rec["side"] == "short"  # lowercase producer
        assert "ema1d_pair" not in rec
        assert _emit_lane(monkeypatch, caplog, cands=[rec],
                          conf={"TESTC": 0.66}) == 1
        line = self._lines(caplog)[0]
        m = re_match(line)
        assert m, line
        assert m.group(2) == "SHORT"           # normalized on the line
        assert m.group(15) == "aligned_short"
        assert m.group(19) == "n/a"            # short: no 1d fetch, honest n/a

    def test_probe_long_record_emits_parseable_line(self, monkeypatch, caplog):
        def fake(coin, interval, count, ttl):
            if interval == "4h":
                return _closes_candles([100.0 - i for i in range(40)],
                                       step=14_400_000)
            if interval == "1d":
                return _closes_candles([200.0 - i for i in range(40)],
                                       step=86_400_000)
            return None
        monkeypatch.setattr(perception, "_fetch_candles_sync", fake)
        closes = [100.0] * 100
        closes[-1] = 107.0
        rec = perception._aligned_short_probe(
            "TESTC", _closes_candles(closes), 107.0, 35.0, [_up_hit(7.0)],
            {"enabled": True, "early_long_enabled": True}, _scan_config())
        assert rec is not None and rec["side"] == "long"
        assert _emit_lane(monkeypatch, caplog, cands=[rec], conf={}) == 1
        line = self._lines(caplog)[0]
        m = re_match(line)
        assert m, line
        assert m.group(2) == "LONG"
        assert m.group(15) == "early_long" and m.group(16) == "momhi_1dF"
        assert "<" in m.group(19)              # 1dF: fast below slow
        assert "btc_tape_drift=" in line and "lane_priority=short_first#1" in line


# ---------------------------------------------------------------------------
# LIVE admission (owner call 2026-10-30, board t_2e23804b follow-on):
# aligned_short_lane_live_admit + build_aligned_short_lane_analysis + the
# cohort_admissions conf-floor weld + the ledger occupancy read + the lane
# tag on the OPEN snapshot. This is the code that books real money, so the
# filters, both slot caps, every fail-closed path and the kill-switch key are
# pinned here.
# ---------------------------------------------------------------------------

LIVE_CFG = {
    "enabled": True,
    "reentry_cooldown_min": 180,
    "live": {"enabled": True, "max_lane_slots": 5, "mom_max_pct": -6.0,
             "require_bucket": "deep", "btc24_max_pct": -2.0,
             "vol_min_pct": 2.0, "net6h_max_pct": -2.0,
             "cohort_tag": "aligned_short_live"},
}

COHORT_KEY = "mechanical/aligned_short_live"


def _live_cfg(**over):
    cfg = {"enabled": True, "reentry_cooldown_min": 180,
           "live": dict(LIVE_CFG["live"])}
    cfg["live"].update(over.pop("live", {}))
    cfg.update(over)
    return cfg


def _live_admit(monkeypatch, caplog, cands=None, cfg=None, open_coins=frozenset(),
                slots_used=0, max_slots=10, lane_open=0, last_ts=None,
                tape_drift: float = -4.2, tape_vol: Optional[float] = 3.1,
                tape_net6h: Optional[float] = None,
                tape_gap=False, tape_missing=False):
    from hermes_trader.agents import market_regime

    def _tape(force=False):
        if tape_gap:
            raise RuntimeError("tape feed down")
        if tape_missing:
            return {"vol": tape_vol, "drift": None}
        out = {"vol": tape_vol, "drift": tape_drift}
        if tape_net6h is not None:
            out["net6h"] = tape_net6h   # absent = the fast clause cannot fire
        return out

    monkeypatch.setattr(market_regime, "btc_tape_activity", _tape)
    caplog.set_level("INFO", logger="hermes_trader.agents.risk_gates")
    return risk_gates.aligned_short_lane_live_admit(
        cands if cands is not None else list(CAND),
        cfg if cfg is not None else _live_cfg(),
        open_coins=set(open_coins), slots_used=slots_used, max_slots=max_slots,
        lane_open=lane_open, last_entry_ts_by_coin=last_ts or {})


def _msgs(caplog, needle="aligned-short-lane"):
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


def _cand(coin, comp=34.0, mom=-8.0, bucket="deep", lane="aligned_short",
          variant=("aligned_dn4h",), side="short"):
    return {"coin": coin, "ts": 1792600000000, "mid": 1.0, "comp": comp,
            "side": side, "lane": lane, "variant": list(variant),
            "mom6h_net": mom, "ema4h_fast": 0.9, "ema4h_slow": 1.0,
            "ema4h_pair": "8/21", "ema4h_dist_pct": -5.0,
            "ema4h_bucket": bucket}


class TestLiveAdmit:
    def test_inert_without_live_block(self, monkeypatch, caplog):
        assert _live_admit(monkeypatch, caplog, cfg={"enabled": True}) == []
        assert _msgs(caplog) == []          # absent key = zero work, zero noise

    def test_inert_when_live_disabled(self, monkeypatch, caplog):
        cfg = _live_cfg()
        cfg["live"]["enabled"] = False
        assert _live_admit(monkeypatch, caplog, cfg=cfg) == []
        assert _msgs(caplog) == []

    def test_admits_the_cell(self, monkeypatch, caplog):
        specs = _live_admit(monkeypatch, caplog)   # CAND: deep, -6.3%, dn4h
        assert len(specs) == 1
        s = specs[0]
        assert s["coin"] == "APT" and s["side"] == "short"
        assert s["cohort_tag"] == "aligned_short_live"
        assert s["lane"] == "aligned_short" and s["mom6h_net"] == -6.3
        line = _msgs(caplog)[0]
        assert "[aligned-short-lane][LIVE] ADMIT APT SHORT" in line
        assert "lane_slots=1/5" in line and "cohort=mechanical/aligned_short_live" in line
        assert "arm=confirmed" in line and "btc_tape_vol=+3.10%" in line

    def test_only_the_named_cell(self, monkeypatch, caplog):
        for bad in (_cand("A", lane="early_long"),
                    _cand("B", variant=("momhi_1dF",)),
                    _cand("C", side="long"),
                    _cand("D", bucket="nearx")):
            assert _live_admit(monkeypatch, caplog, cands=[bad]) == [], bad

    def test_momentum_depth_is_the_gate(self, monkeypatch, caplog):
        assert _live_admit(monkeypatch, caplog, cands=[_cand("A", mom=-4.0)]) == []
        assert len(_live_admit(monkeypatch, caplog, cands=[_cand("A", mom=-4.0)],
                               cfg=_live_cfg(live={"mom_max_pct": -3.0}))) == 1

    def test_idle_unless_the_tape_is_dumping(self, monkeypatch, caplog):
        assert _live_admit(monkeypatch, caplog, tape_drift=+0.5) == []
        assert _live_admit(monkeypatch, caplog, tape_drift=0.0) == []  # >= cap
        assert "idle" in _msgs(caplog)[0]

    def test_tape_gap_fails_closed(self, monkeypatch, caplog):
        assert _live_admit(monkeypatch, caplog, tape_gap=True) == []
        assert _live_admit(monkeypatch, caplog, tape_missing=True) == []
        assert any("fail closed" in m for m in _msgs(caplog))

    def test_unprovable_occupancy_fails_closed(self, monkeypatch, caplog):
        assert _live_admit(monkeypatch, caplog, lane_open=-1) == []
        assert any("fail closed" in m for m in _msgs(caplog))

    def test_lane_slot_cap_binds_first(self, monkeypatch, caplog):
        cands = [_cand(c, comp=float(i)) for i, c in enumerate("ABCDEF")]
        got = _live_admit(monkeypatch, caplog, cands=cands, lane_open=4)
        assert [s["coin"] for s in got] == ["F"]          # comp-desc (F=5.0)
        assert _live_admit(monkeypatch, caplog, cands=cands, lane_open=5) == []

    def test_shared_pool_cap_also_binds(self, monkeypatch, caplog):
        cands = [_cand(c, comp=float(i)) for i, c in enumerate("ABC")]
        got = _live_admit(monkeypatch, caplog, cands=cands,
                          slots_used=9, max_slots=10, lane_open=0)
        assert [s["coin"] for s in got] == ["C"]

    def test_open_position_and_cooldown_skip(self, monkeypatch, caplog):
        assert _live_admit(monkeypatch, caplog, open_coins=frozenset({"APT"})) == []
        now = int(time.time() * 1000)
        assert _live_admit(monkeypatch, caplog, last_ts={"APT": now - 60_000}) == []
        # Outside the 180min lane cooldown -> admitted again.
        assert len(_live_admit(monkeypatch, caplog,
                               last_ts={"APT": now - 200 * 60_000})) == 1

    def test_missing_moment_field_skips(self, monkeypatch, caplog):
        c = _cand("A")
        c.pop("mom6h_net")
        assert _live_admit(monkeypatch, caplog, cands=[c]) == []


class TestLaneArmClauses:
    """The tape gate that replaced 'operator notices the dump' (owner call
    2026-10-30). Confirmed clause = drift<=-2 AND vol>=2; fast clause =
    BTC 6h net <= -2. Measured: armed 7% of hours, +$134 at 5 slots, and the
    book it filters out is a net LOSER (-$23)."""

    def test_confirmed_clause_needs_vol_expansion(self, monkeypatch, caplog):
        assert _live_admit(monkeypatch, caplog, tape_drift=-4.2,
                           tape_vol=1.0) == []          # deep drift, flat vol
        assert "tape is not a dump" in _msgs(caplog)[0]

    def test_fast_arms_on_the_6h_onset_alone(self, monkeypatch, caplog):
        # The 10-22 shape: 24h drift still shallow, the dump just started.
        got = _live_admit(monkeypatch, caplog, tape_drift=-0.5, tape_vol=0.5,
                          tape_net6h=-2.5)
        assert len(got) == 1 and got[0]["arm"] == "fast"
        assert "arm=fast" in _msgs(caplog)[0]

    def test_both_clauses_labelled(self, monkeypatch, caplog):
        got = _live_admit(monkeypatch, caplog, tape_drift=-4.2, tape_vol=3.1,
                          tape_net6h=-6.0)
        assert got[0]["arm"] == "confirmed+fast"
        assert "arm=confirmed+fast" in _msgs(caplog)[0]

    def test_missing_scalar_disables_only_its_own_clause(self, monkeypatch, caplog):
        # vol n/a -> confirmed cannot fire, fast still can.
        assert len(_live_admit(monkeypatch, caplog, tape_vol=None,
                               tape_net6h=-3.0)) == 1
        # net6h n/a -> fast cannot fire; confirmed still can.
        assert len(_live_admit(monkeypatch, caplog, tape_drift=-4.2,
                               tape_vol=3.1)) == 1
        # both n/a -> nothing arms.
        assert _live_admit(monkeypatch, caplog, tape_vol=None) == []

    def test_arm_thresholds_are_config_knobs(self, monkeypatch, caplog):
        assert _live_admit(monkeypatch, caplog, tape_drift=-4.2, tape_vol=3.1,
                           cfg=_live_cfg(live={"vol_min_pct": 5.0})) == []
        # fast arm switched off -> the 6h onset alone no longer admits
        assert _live_admit(monkeypatch, caplog, tape_drift=-0.5, tape_vol=0.5,
                           tape_net6h=-2.5,
                           cfg=_live_cfg(live={"net6h_max_pct": -99.0})) == []
        # drift cap tightened -> deep-but-shallow-vol tape stays idle
        assert _live_admit(monkeypatch, caplog, tape_drift=-1.5, tape_vol=9.0) == []


class TestTapeNet6h:
    """`btc_tape_activity` must carry net6h — the lane's fast arm reads it,
    and it has to come from the SAME 290-bar window (no second price call) and
    describe ONLY the last 72 bars, or the fast arm is just the drift gate
    wearing a different hat."""

    def _tape(self, monkeypatch, closes):
        from hermes_trader.agents import market_regime as mr
        candles = [Candle(t=i * 300_000, o=c, h=c, l=c, c=c, v=1.0)
                   for i, c in enumerate(closes)]
        monkeypatch.setattr(mr, "fetch_hl_candles",
                            lambda coin, interval="5m", count=290: candles)
        monkeypatch.setattr(mr, "_tape_cache", (None, 0.0))
        res = mr.btc_tape_activity(force=True)
        assert res is not None
        return res

    def test_net6h_measures_the_last_72_bars_only(self, monkeypatch):
        closes = [100.0] * 290
        for i in range(290 - 72, 290):     # -3% step inside the 6h window
            closes[i] = 97.0
        tape = self._tape(monkeypatch, closes)
        # reference bar = closes[-73] (72 steps back = exactly 6h) is pre-step
        assert tape["net6h"] == pytest.approx(-3.0)
        assert tape["drift"] == pytest.approx(-3.0)

    def test_old_move_moves_drift_not_net6h(self, monkeypatch):
        closes = [100.0] * 290
        for i in range(100, 290):          # -5% step ~16h back, flat since
            closes[i] = 95.0
        tape = self._tape(monkeypatch, closes)
        assert tape["drift"] == pytest.approx(-5.0)
        assert tape["net6h"] == pytest.approx(0.0)

    def test_vol_and_drift_keys_unchanged(self, monkeypatch):
        tape = self._tape(monkeypatch, [100.0 + (i % 3) for i in range(290)])
        assert set(tape) == {"vol", "drift", "net6h"}
        # first close 100.0, last (i=289, 289%3==1) 101.0 -> +1.0% over 24h
        assert tape["vol"] > 0 and tape["drift"] == pytest.approx(1.0)
        assert tape["net6h"] == pytest.approx(0.0)   # 72 bars = 24 cycles, net 0


class TestLaneCohortWeld:
    """The kill switch: without the cohort_admissions key, a conf-0.0 lane
    entry blocks at BOTH conf floors; with it, it passes the floors and only
    the other gates stand between it and the market."""

    def _spec(self):
        return {"coin": "APT", "side": "short", "mid": 8.12, "comp": 34.0,
                "mom6h_net": -8.0, "ema4h_bucket": "deep", "ema4h_dist_pct": -5.0,
                "lane": "aligned_short", "variant": ["aligned_dn4h"],
                "btc_tape_drift": -4.2, "cohort_tag": "aligned_short_live",
                "ts": 1792600000000}

    def test_analysis_is_honest_about_having_no_seat(self):
        a = risk_gates.build_aligned_short_lane_analysis(self._spec())
        assert a["confidence"] == 0.0
        assert a["verdict"] == "SHORT" and a["side"] == "short"
        assert a["primary_model"] == "mechanical"
        assert a["cohort_tag"] == a["_lane_tag"] == "aligned_short_live"
        assert a["id"] and a["perception_id"] and a["created_at"]
        assert "NO seat call" in a["reasoning"]

    def test_runner_gate_admits_only_with_the_key(self):
        from hermes_trader.agents.executor import _runner_entry_block_reason
        a = risk_gates.build_aligned_short_lane_analysis(self._spec())
        cfg = {"runner_entry_gate": {"enabled": True, "allow_shorts": True,
                                     "cohort_admissions": {COHORT_KEY: 0.0}}}
        assert _runner_entry_block_reason(a, cfg) == ""
        cfg2 = {"runner_entry_gate": {"enabled": True, "allow_shorts": True,
                                      "cohort_admissions": {}}}
        assert "conf" in _runner_entry_block_reason(a, cfg2)

    def test_runner_gate_respects_the_shorts_kill_switch(self):
        from hermes_trader.agents.executor import _runner_entry_block_reason
        a = risk_gates.build_aligned_short_lane_analysis(self._spec())
        cfg = {"runner_entry_gate": {"enabled": True, "allow_shorts": False,
                                     "cohort_admissions": {COHORT_KEY: 0.0}}}
        assert _runner_entry_block_reason(a, cfg) != ""

    def test_classic_conf_gate_admits_only_with_the_key(self):
        ctx = risk_gates.GateContext(
            confidence=0.0, current_positions=[], trade_notional_usd=300.0,
            daily_pnl=0.0, market_volume_24h_usd=50_000_000.0, coin="APT",
            trade_side="short", has_binary_news_risk=False, equity=38000.0,
            total_open_notional=0.0, cohort_tag="aligned_short_live",
            primary_model="mechanical")
        blocked = {"pass": False, "reason": "confidence 0.00 < 0.65"}
        cfg = {"runner_entry_gate": {"cohort_admissions": {COHORT_KEY: 0.0}}}
        assert risk_gates._cohort_conf_admission(ctx, cfg, blocked, 0.65)["pass"]
        out = risk_gates._cohort_conf_admission(
            ctx, {"runner_entry_gate": {"cohort_admissions": {}}}, blocked, 0.65)
        assert out == blocked


class TestLedgerLaneTag:
    def _rows(self, tmp_path, monkeypatch):
        from hermes_trader import ledger
        path = tmp_path / "trades.jsonl"
        rows = [
            {"event": "OPEN", "coin": "APT", "side": "short", "entry_px": 8.12,
             "config_snapshot": {"lane": "aligned_short_live"}},
            {"event": "OPEN", "coin": "BTC", "side": "short", "entry_px": 100.0,
             "config_snapshot": {"regime": "down"}},
            {"event": "OPEN", "coin": "SOL", "side": "short", "entry_px": 99.0,
             "config_snapshot": {"lane": "aligned_short_live"}},
            "not json at all",
            {"event": "OPEN", "coin": None, "side": "short", "entry_px": 1.0},
            {"event": "CLOSE", "coin": "SOL", "side": "short", "entry_px": 99.0},
        ]
        path.write_text("\n".join(
            r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n",
            encoding="utf-8")
        monkeypatch.setattr(ledger, "LEDGER_FILE", str(path))
        return ledger

    def test_counts_only_tagged_unclosed(self, tmp_path, monkeypatch):
        ledger = self._rows(tmp_path, monkeypatch)
        assert ledger.count_open_with_lane("aligned_short_live") == 1
        assert ledger.count_open_with_lane("other_lane") == 0
        assert ledger.count_open_with_lane("") == 0

    def test_missing_ledger_is_zero_not_crash(self, tmp_path, monkeypatch):
        from hermes_trader import ledger
        monkeypatch.setattr(ledger, "LEDGER_FILE", str(tmp_path / "nope.jsonl"))
        assert ledger.count_open_with_lane("aligned_short_live") == 0

    def test_open_snapshot_unchanged_without_lane(self, monkeypatch):
        from hermes_trader.agents.executor import build_open_config_snapshot
        base = build_open_config_snapshot("down", "dsl", 1.5, 2.0)
        assert "lane" not in base
        tagged = build_open_config_snapshot("down", "dsl", 1.5, 2.0,
                                            lane="aligned_short_live")
        assert tagged["lane"] == "aligned_short_live"
        assert {k: v for k, v in tagged.items() if k != "lane"} == base


class TestTapeEval:
    """The SHARED tape evaluator (refactor 2026-10-30, owner idea tying the
    lane arm to state_admission's short band, t_860f95f8). ONE boundary
    convention for every tape-armed consumer: drift band
    [btc24_min_pct, btc24_max_pct) — lower INCLUSIVE, upper EXCLUSIVE, the
    shape state_admission already used — vol_min_pct inclusive, net6h_max_pct
    inclusive. Missing data only ever REMOVES: a clause can never arm on a
    gap, and an UNSET key means the clause is OFF, not unbounded."""

    CFG = {"btc24_max_pct": -2.0, "vol_min_pct": 2.0, "net6h_max_pct": -2.0}

    def _ev(self, sc, cfg=None):
        from hermes_trader.agents.market_regime import tape_eval
        return tape_eval(sc, self.CFG if cfg is None else cfg)

    def test_band_is_half_open(self):
        assert self._ev({"drift": -2.0})["band_hit"] is False    # max EXCLUSIVE
        assert self._ev({"drift": -2.01})["band_hit"] is True
        cfg = dict(self.CFG, btc24_min_pct=-3.0)
        assert self._ev({"drift": -3.0}, cfg)["band_hit"] is True   # min INCLUSIVE
        assert self._ev({"drift": -3.01}, cfg)["band_hit"] is False

    def test_unset_band_is_off_not_unbounded(self):
        got = self._ev({"drift": -40.0, "vol": 9.0}, {"vol_min_pct": 2.0})
        assert got["band_hit"] is False and got["confirmed_dump"] is False

    def test_missing_scalar_never_arms(self):
        assert self._ev({"drift": None, "vol": 9.0})["confirmed_dump"] is False
        assert self._ev({"drift": -4.0, "vol": None})["confirmed_dump"] is False
        assert self._ev({"drift": -4.0})["fast_arm"] is False
        assert sorted(self._ev({"drift": -4.0})["missing"]) == ["net6h", "vol"]

    def test_clauses_are_independent(self):
        # deep drift + flat vol -> confirmed off, but the 6h onset still arms
        got = self._ev({"drift": -4.0, "vol": 1.0, "net6h": -2.0})
        assert got["confirmed_dump"] is False and got["fast_arm"] is True
        # net6h_max_pct unset -> the fast clause is not configured at all
        assert self._ev({"drift": -4.0, "net6h": -40.0},
                        {"btc24_max_pct": -2.0,
                         "vol_min_pct": 2.0})["fast_arm"] is False

    def test_malformed_bound_disables_its_own_clause(self):
        # garbage reads as "not configured": cannot arm, cannot crash — the
        # fail direction that only ever REMOVES entries
        got = self._ev({"drift": -4.0, "vol": 3.0, "net6h": -9.0},
                       {"btc24_max_pct": "nonsense", "vol_min_pct": 2.0,
                        "net6h_max_pct": None})
        assert got["band_hit"] is False and got["confirmed_dump"] is False
        assert got["fast_arm"] is False

    def test_same_scalars_opposite_readings(self):
        # THE reason the thresholds stay per-consumer: the lane wants the deep
        # dump, the state_admission short band wants MILD drift and refuses
        # exactly this tape.
        sc = {"drift": -4.2, "vol": 3.1}
        assert self._ev(sc)["band_hit"] is True
        assert self._ev(sc, {"btc24_min_pct": -2.0,
                             "btc24_max_pct": 0.0})["band_hit"] is False


class TestLaneTapeStateStamp:
    """`live.tape_state` opts the lane into the t_860f95f8 phase label so lane
    and clause lines cross-read. LOG-ONLY (a candle gap never changes a
    verdict) and OFF by default (no candle read when the block is unset — the
    clause's own cost rule)."""

    def _hist(self):
        base = 1_700_000_000_000
        closes = ([100.0] * 30 + [97.0, 95.0, 92.0, 90.0, 88.0]
                  + [89.0, 92.0, 95.0, 97.0, 99.0] * 3)
        return [(base + i * 3_600_000, c) for i, c in enumerate(closes)]

    def test_off_by_default_costs_no_candle_read(self, monkeypatch, caplog):
        from hermes_trader.agents import market_regime

        def _boom(count=216, force=False):
            raise AssertionError("history fetched with tape_state unset")

        monkeypatch.setattr(market_regime, "btc_tape_history", _boom)
        got = _live_admit(monkeypatch, caplog, tape_drift=-4.2, tape_vol=3.1)
        assert len(got) == 1 and "tape_state" not in got[0]
        assert "tape_state=" not in _msgs(caplog)[0]

    def test_stamp_renders_on_spec_and_admit_line(self, monkeypatch, caplog):
        from hermes_trader.agents import market_regime
        monkeypatch.setattr(market_regime, "btc_tape_history",
                            lambda count=216, force=False: self._hist())
        got = _live_admit(monkeypatch, caplog, tape_drift=-4.2, tape_vol=3.1,
                          cfg=_live_cfg(live={"tape_state": {}}))
        stamp = got[0]["tape_state"]
        assert stamp.startswith("deep=") and "shallow=" in stamp
        assert stamp not in ("no-history", "malformed-history")
        assert f"tape_state={stamp}" in _msgs(caplog)[0]
        reason = risk_gates.build_aligned_short_lane_analysis(got[0])["reasoning"]
        assert f"tape_state={stamp}" in reason

    def test_history_gap_labels_and_still_admits(self, monkeypatch, caplog):
        from hermes_trader.agents import market_regime

        def _down(count=216, force=False):
            raise RuntimeError("candle feed down")

        monkeypatch.setattr(market_regime, "btc_tape_history", _down)
        got = _live_admit(monkeypatch, caplog, tape_drift=-4.2, tape_vol=3.1,
                          cfg=_live_cfg(live={"tape_state": {}}))
        assert len(got) == 1 and got[0]["tape_state"] == "no-history"
