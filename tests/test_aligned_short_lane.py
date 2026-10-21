"""Tests for the ALIGNED-SHORT lane shadow accrual (board t_2e23804b).

Two halves:
  1. perception._aligned_short_probe — the per-scan candidate probe:
     dn-momentum net (6h close-to-close <= -mom_pct, momentum-shadow
     semantics — NOT the Theil-Sen fired flag), composite >= comp_bar, and
     ema8(4h) < ema21(4h) alignment. Candidates ride a thread-safe side
     channel (take_aligned_short_lane_candidates drains it), never the
     surfacing/research path.
  2. risk_gates.aligned_short_lane_emit — the WOULD ADMIT line: live-state
     mirror (slots, no open position, per-coin re-entry cooldown), fields
     coin/ts/mid/comp/mom6h_net/ema4h/btc_tape/conf-if-asked. Shadow-only:
     the function logs, it cannot admit.

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
    r"^\[aligned-short-lane\]\[SHADOW\] WOULD ADMIT (\S+) SHORT "
    r"ts=(\d+) mid=(\S+) comp=(\S+) mom6h_net=([+-]\d+\.\d)% "
    r"ema4h=(\d+/\d+) (\S+)<(\S+) btc_tape=(\S+) "
    r"conf=(\S+) slots=(\d+)/(\d+) "
)


def _closes_candles(closes, t0=0, step=300_000):
    return [Candle(t=t0 + i * step, o=c, h=c, l=c, c=c, v=1.0)
            for i, c in enumerate(closes)]


def _dn_hit(move=-5.0, reason=None):
    fired = move <= -3.0
    r = reason or (f"{move:.1f}% over 72 bars (downtrend)" if fired else "flat")
    return {"name": "downtrendMomentum", "score": 5.0 if fired else 0,
            "reason": r, "fired": fired}


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


# ── probe: entry-candidate conditions ───────────────────────────────────────

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
         "mom6h_net": -6.3, "ema4h_fast": 8.02, "ema4h_slow": 8.44,
         "ema4h_pair": "8/21"}]


class TestEmit:
    def setup_method(self):
        risk_gates._as_lane_last_emit.clear()

    def _tape(self, monkeypatch):
        from hermes_trader.agents import market_regime
        monkeypatch.setattr(market_regime, "btc_tape_activity",
                            lambda force=False: {"vol": 3.1, "drift": -4.2})

    def _emit(self, monkeypatch, caplog, cands=None, cfg=None,
              open_coins=frozenset(), slots_used=2, max_slots=8, conf=None):
        self._tape(monkeypatch)
        caplog.set_level("WARNING", logger="hermes_trader.agents.risk_gates")
        return risk_gates.aligned_short_lane_emit(
            cands if cands is not None else list(CAND),
            cfg if cfg is not None else {"enabled": True},
            open_coins=set(open_coins), slots_used=slots_used,
            max_slots=max_slots, conf_by_coin=conf or {})

    def test_line_format_all_fields(self, monkeypatch, caplog):
        n = self._emit(monkeypatch, caplog, conf={"APT": 0.72})
        assert n == 1
        line = next(r.message for r in caplog.records
                    if "aligned-short-lane" in r.getMessage())
        m = re_match(line)
        assert m, line
        assert m.group(1) == "APT"
        assert m.group(2) == "1792600000000"
        assert m.group(5) == "-6.3"
        assert m.group(7) == "8.02" and m.group(8) == "8.44"
        assert m.group(9) == "vol+3.10%/drift-4.20%"
        assert m.group(10) == "0.72"
        assert "NOT admitting (shadow v1" in line

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
        assert "btc_tape=n/a" in [r.getMessage() for r in caplog.records
                                  if "WOULD ADMIT" in r.getMessage()][0]

    def test_btc_tape_never_blocks_shadow_v1(self, monkeypatch, caplog):
        from hermes_trader.agents import market_regime
        monkeypatch.setattr(market_regime, "btc_tape_activity",
                            lambda force=False: {"vol": 0.2, "drift": 0.1})  # very quiet tape
        assert self._emit(monkeypatch, caplog) == 1  # field logged, not a block


def re_match(line):
    import re
    return re.match(LINE_RE, line)
