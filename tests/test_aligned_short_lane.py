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

All candle fetches and the BTC tape read are monkeypatched — no network.
"""
from __future__ import annotations

import math
import sys
import time
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
        assert m.group(19) == "8/21:7.8>8.2"
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
