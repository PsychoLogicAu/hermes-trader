"""Breadth-onset shadow tags (burstcnt / flip_age_m / retrace_pp) — t_d8b728bb.

Record-only attributes on the aligned-short-lane WOULD ADMIT line so the
NEXT dump's shadow book prices the breadth-onset variant natively
(offline sample was n=9, med MFE12 -3.59% vs lane ~-1.5: suggestion-grade,
accrual is the validator). These tests pin the EMIT side: default-on tags,
hot-rollback knob (aligned_short_lane.breadth_tags=false -> byte-clean
pre-tag line), degradation to '-' on missing input, and the perception
flip-tracker transition semantics. Nothing here can affect firing.
"""
import time

from hermes_trader.agents import perception, risk_gates
from hermes_trader.models.types import TriggerHit


def _hit(name: str, fired: bool) -> TriggerHit:
    return {"name": name, "score": 0.5, "reason": "test", "fired": fired}

CAND = [{"coin": "APT", "ts": 1792600000000, "mid": 8.12, "comp": 34.0,
         "side": "SHORT", "lane": "aligned_short", "variant": ["aligned_dn4h"],
         "mom6h_net": -6.3, "ema4h_fast": 8.02, "ema4h_slow": 8.44,
         "ema4h_pair": "8/21", "ema4h_dist_pct": -4.976,
         "ema4h_bucket": "deep"}]


def _emit(monkeypatch, caplog, tape_extra=None, lane_cfg=None, pre=None):
    from hermes_trader.agents import market_regime
    _t = {"vol": 3.1, "drift": -4.2}
    _t.update(tape_extra or {})
    monkeypatch.setattr(market_regime, "btc_tape_activity",
                        lambda force=False: _t)
    caplog.set_level("WARNING", logger="hermes_trader.agents.risk_gates")
    risk_gates._as_lane_last_emit.clear()
    perception._dn_arm_prev.clear()
    perception._dn_flip_at.clear()
    perception._dn_flip_events.clear()
    if pre:
        pre()
    n = risk_gates.aligned_short_lane_emit(
        list(CAND), lane_cfg if lane_cfg is not None else {"enabled": True},
        open_coins=set(), slots_used=2, max_slots=8,
        conf_by_coin={"APT": 0.72})
    line = next(r.message for r in caplog.records
                if "aligned-short-lane" in r.getMessage())
    return n, line


class TestEmitTags:
    def test_defaults_degrade_to_dash(self, monkeypatch, caplog):
        n, line = _emit(monkeypatch, caplog)
        assert n == 1
        assert "burstcnt=0 flip_age_m=- retrace_pp=-" in line

    def test_retrace_from_tape(self, monkeypatch, caplog):
        _n, line = _emit(monkeypatch, caplog,
                         tape_extra={"drift_min12": -4.35})
        assert "retrace_pp=+0.15" in line and "anti=-" in line

    def test_burst_and_age_rendered(self, monkeypatch, caplog):
        def pre():
            now = time.time()
            perception.note_dn_arm_flips(
                [_hit("downtrendMomentum", True)], "APT", now=now)
            perception.note_dn_arm_flips(
                [_hit("downtrendMomentum", True)], "BTC",
                now=now + 5)
        _n, line = _emit(monkeypatch, caplog,
                         tape_extra={"drift_min12": -4.5}, pre=pre)
        assert "burstcnt=2 flip_age_m=0" in line

    def test_knob_off_is_byte_clean(self, monkeypatch, caplog):
        _n, line = _emit(monkeypatch, caplog, tape_extra={"drift_min12": -4.15},
                         lane_cfg={"enabled": True, "breadth_tags": False})
        assert "burstcnt" not in line and "retrace_pp" not in line
        assert "lane_priority=short_first#1 — NOT admitting" in line


class TestFlipTracker:
    def test_transition_semantics(self):
        perception._dn_arm_prev.clear()
        perception._dn_flip_at.clear()
        perception._dn_flip_events.clear()
        t0 = 1_800_000_000.0
        fired = [_hit("downtrendMomentum", True)]
        flat = [_hit("downtrendMomentum", False)]
        perception.note_dn_arm_flips(fired, "X", now=t0)
        perception.note_dn_arm_flips(fired, "X", now=t0 + 60)   # no double
        assert perception.dn_burst_count(3600.0, now=t0 + 60) == 1
        perception.note_dn_arm_flips(flat, "X", now=t0 + 120)
        perception.note_dn_arm_flips(fired, "X", now=t0 + 180)  # re-arm flips
        assert perception.dn_burst_count(3600.0, now=t0 + 180) == 2
        assert perception.dn_flip_age_s("X", now=t0 + 240) == 60.0
        assert perception.dn_flip_age_s("NOPE", now=t0 + 240) is None
        # window edge: both flips (t0, t0+180) are still INSIDE a 30-min
        # window at t0+240; they fall out only past t0+1800
        assert perception.dn_burst_count(1800.0, now=t0 + 240) == 2

    def test_other_arms_ignored(self):
        perception._dn_arm_prev.clear()
        perception._dn_flip_events.clear()
        perception.note_dn_arm_flips(
            [_hit("uptrendMomentum", True)], "Y", now=time.time())
        assert perception.dn_burst_count() == 0
