"""t_62c58463 — Theil-Sen momentum estimator (t_38944f65 NXPC post-mortem).

The live close-to-close net reads a pump bar's MEMORY, not a trend: on
2026-10-15 it fired uptrendMomentum +13.3% UP through six red hours into the
NXPC long. These tests pin the robust Theil-Sen arm's behaviour on the
incident shapes and the three config modes.
"""

import math

import pytest

from hermes_trader.indicators.triggers import (
    _theilsen_move_pct,
    downtrend_momentum,
    uptrend_momentum,
)
from hermes_trader.models.types import Candle


def _closes_to_candles(closes):
    return [Candle(t=i * 300_000, o=c, h=c, l=c, c=c, v=1.0) for i, c in enumerate(closes)]


def _path(moves, start=1.0):
    """Build a close path from per-bar % moves."""
    p, out = start, [start]
    for m in moves:
        p *= 1 + m / 100
        out.append(p)
    return out


# ---- scenario paths (72-bar windows, threshold 5.0 like live config) ----
LB = 72
TH = 5.0

GRIND_UP = _path([0.12] * LB)                       # steady uptrend
PUMP_CRASH = _path([2.4] * 12 + [-0.25] * 60)       # NXPC shape: pump then six red hours
BLEED_PARABOLA = _path([-0.2] * 40 + [1.5] * 32)    # owner's regime-change case
FLAT = _path([0.0] * LB)
PUMP_FLAT = _path([2.4] * 12 + [0.0] * 60)


class TestTheilsenMovePct:
    def test_insufficient_history_returns_none(self):
        assert _theilsen_move_pct(_closes_to_candles([1.0] * 10), 72) is None

    def test_pump_crash_reads_down(self):
        """The incident: net says +13% UP; Theil-Sen must say DOWN."""
        candles = _closes_to_candles(PUMP_CRASH)
        net = (PUMP_CRASH[-1] / PUMP_CRASH[-LB - 1] - 1) * 100
        ts = _theilsen_move_pct(candles, LB)
        assert net > TH                      # the bug: net fires UP
        assert ts is not None and ts < -TH   # the fix: robust slope reads DN

    def test_bleed_parabola_reads_up(self):
        """Owner's regime-change case: median-of-bars majority-vote says DN;
        Theil-Sen must read UP."""
        ts = _theilsen_move_pct(_closes_to_candles(BLEED_PARABOLA), LB)
        assert ts is not None and ts > TH

    def test_grind_up_reads_up(self):
        ts = _theilsen_move_pct(_closes_to_candles(GRIND_UP), LB)
        assert ts is not None and ts > TH

    def test_flat_reads_flat(self):
        ts = _theilsen_move_pct(_closes_to_candles(FLAT), LB)
        assert ts is not None and abs(ts) < TH

    def test_pump_flat_not_a_trend(self):
        """Pump then dead tape: net says UP (pump memory); Theil-Sen must not."""
        ts = _theilsen_move_pct(_closes_to_candles(PUMP_FLAT), LB)
        assert ts is not None and abs(ts) < TH

    def test_nonpositive_closes_returns_none(self):
        bad = GRIND_UP[:]
        bad[10] = 0.0
        assert _theilsen_move_pct(_closes_to_candles(bad), LB) is None


class TestUptrendMomentumModes:
    def test_off_mode_is_legacy_net(self):
        hit = uptrend_momentum(_closes_to_candles(PUMP_CRASH), LB, TH)
        assert hit["fired"] is True          # legacy behaviour preserved
        assert "shadow_ts_move" not in hit
        assert "live_net_move" not in hit

    def test_shadow_mode_keeps_net_fired_and_attaches_arm(self):
        hit = uptrend_momentum(_closes_to_candles(PUMP_CRASH), LB, TH, theilsen_mode="shadow")
        assert hit["fired"] is True          # net still decides
        assert hit.get("shadow_ts_fired") is False
        assert hit.get("shadow_ts_move") < -TH   # arm recorded the truth

    def test_live_mode_theilsen_decides(self):
        hit = uptrend_momentum(_closes_to_candles(PUMP_CRASH), LB, TH, theilsen_mode="live")
        assert hit["fired"] is False         # the incident would NOT surface UP
        assert hit.get("live_net_move") > TH  # flight recorder keeps the net

    def test_live_mode_falls_back_to_net_when_ts_none(self):
        candles = _closes_to_candles(GRIND_UP)
        hit = uptrend_momentum(candles, LB, TH, theilsen_mode="live")
        assert hit["fired"] is True          # genuine grind still fires

    def test_off_mode_unchanged_signature_on_grind(self):
        off = uptrend_momentum(_closes_to_candles(GRIND_UP), LB, TH)
        shadow = uptrend_momentum(_closes_to_candles(GRIND_UP), LB, TH, theilsen_mode="shadow")
        live = uptrend_momentum(_closes_to_candles(GRIND_UP), LB, TH, theilsen_mode="live")
        assert off["fired"] == shadow["fired"] == live["fired"] is True
        assert off["score"] == shadow["score"] == pytest.approx(live["score"], rel=0.5)


class TestDowntrendMomentumMirror:
    def test_live_mode_surfaces_the_incident_as_downtrend(self):
        hit = downtrend_momentum(_closes_to_candles(PUMP_CRASH), LB, TH, theilsen_mode="live")
        assert hit["fired"] is True          # the decline gets surfaced for SHORT
        assert "downtrend" in hit["reason"]

    def test_off_mode_misses_the_incident(self):
        hit = downtrend_momentum(_closes_to_candles(PUMP_CRASH), LB, TH)
        assert hit["fired"] is False         # net +13% hides the decline entirely

    def test_shadow_mode_mirror_fields(self):
        hit = downtrend_momentum(_closes_to_candles(PUMP_CRASH), LB, TH, theilsen_mode="shadow")
        assert hit["fired"] is False
        assert hit.get("shadow_ts_fired") is True

    def test_bleed_parabola_not_surfaced_as_downtrend_in_live(self):
        hit = downtrend_momentum(_closes_to_candles(BLEED_PARABOLA), LB, TH, theilsen_mode="live")
        assert hit["fired"] is False


# ---- [momentum-shadow] accrual line (t_62c58463 follow-up, 2026-10-19) ----
# The per-arm pair collapsed to ONE line per coin-scan: both arms read the
# same signed scalar (net and Theil-Sen are compared against +/-thr), so the
# mirror arm's line was entailed duplication. _momentum_shadow_line is the
# single format owner; these tests pin the shape AND fire-decision equivalence
# with the old per-arm union (so the offline accrual join sees exactly the
# same coin-scans, just one line instead of up to two).

import re

from hermes_trader.agents.perception import _momentum_shadow_line

LINE_RE = re.compile(
    r"^\[momentum-shadow\] (\S+) mode=(live|shadow) "
    r"(?:net=([-+]?\d+\.\d)%/(UP|DN|flat) )?"
    r"(live=(UP|DN|flat) )?ts=(([+-]?\d+\.\d)%/)?(UP|DN|flat)$")


def _hit(name, fired, **extras):
    h = {"name": name, "score": 0.0, "reason": "flat", "fired": fired}
    h.update(extras)
    return h


def _old_live_lines(up, dn, thr):
    """Verbatim replica of the pre-collapse per-arm live branch."""
    out = []
    for h, is_up in ((up, True), (dn, False)):
        net = h.get("live_net_move")
        ts = ("UP" if is_up else "DN") if h["fired"] else "flat"
        if net is not None and (net != 0.0 or h["fired"]):
            ns = "UP" if net >= thr else ("DN" if net <= -thr else "flat")
            if ns != ts or h["fired"]:
                out.append((net, ns, ts))
    return out


def _old_shadow_lines(up, dn):
    """Verbatim replica of the pre-collapse per-arm shadow branch."""
    out = []
    for h, is_up in ((up, True), (dn, False)):
        if "shadow_ts_move" in h:
            live = ("UP" if is_up else "DN") if h["fired"] else "flat"
            ts = ("UP" if is_up else "DN") if h.get("shadow_ts_fired") else "flat"
            if live != ts or h["fired"] or h.get("shadow_ts_fired"):
                out.append((live, h["shadow_ts_move"], ts))
    return out


class TestMomentumShadowLine:
    def test_live_pair_collapses_to_single_golden_line(self):
        """The 2026-10-19 RUNE tape: old printed uptrendMomentum ts=UP AND a
        downtrendMomentum ts=flat mirror; the collapse keeps one line."""
        up = _hit("uptrendMomentum", True, live_net_move=6.3)
        dn = _hit("downtrendMomentum", False, live_net_move=6.3)
        assert _momentum_shadow_line("RUNE", up, dn, "live", TH) == (
            "[momentum-shadow] RUNE mode=live net=+6.3%/UP ts=UP")

    def test_live_downtrend_fired_golden(self):
        """NXPC shape: net DN, Theil-Sen DN — one line, ts=DN."""
        up = _hit("uptrendMomentum", False, live_net_move=-9.1)
        dn = _hit("downtrendMomentum", True, live_net_move=-9.1)
        assert _momentum_shadow_line("NXPC", up, dn, "live", TH) == (
            "[momentum-shadow] NXPC mode=live net=-9.1%/DN ts=DN")

    def test_live_disagreement_golden(self):
        """USELESS shape: ts fired UP but net below thr -> net side flat."""
        up = _hit("uptrendMomentum", True, live_net_move=4.8)
        dn = _hit("downtrendMomentum", False, live_net_move=4.8)
        assert _momentum_shadow_line("USELESS", up, dn, "live", TH) == (
            "[momentum-shadow] USELESS mode=live net=+4.8%/flat ts=UP")

    def test_live_fire_decision_and_scalars_match_old_union(self):
        nets = [-9.1, -5.0, 0.0, 4.8, 5.0, 6.3]
        for fu in (False, True):
            for fd in (False, True):
                if fu and fd:
                    continue  # impossible vs +/-thr (thr > 0)
                for net in nets:
                    up = _hit("uptrendMomentum", fu, live_net_move=net)
                    dn = _hit("downtrendMomentum", fd, live_net_move=net)
                    old = _old_live_lines(up, dn, TH)
                    line = _momentum_shadow_line("COIN", up, dn, "live", TH)
                    assert (line is None) == (not old)
                    if old and line:
                        m = LINE_RE.match(line)
                        assert m, line
                        net_val, net_side = m.group(3), m.group(4)
                        ts_side = m.group(9)
                        assert float(net_val) == pytest.approx(old[0][0], abs=0.05)
                        assert net_side == old[0][1]
                        # scalar ts verdict: old arms' non-flat side
                        old_ts = next((t for _, _, t in old if t != "flat"), "flat")
                        assert ts_side == old_ts

    def test_shadow_fire_decision_and_scalars_match_old_union(self):
        for fu in (False, True):
            for fd in (False, True):
                if fu and fd:
                    continue
                for tfu in (False, True):
                    for tfd in (False, True):
                        if tfu and tfd:
                            continue
                        tsv = -7.0 if (tfd or not tfu) else 6.0
                        up = _hit("uptrendMomentum", fu,
                                  shadow_ts_move=tsv, shadow_ts_fired=tfu)
                        dn = _hit("downtrendMomentum", fd,
                                  shadow_ts_move=tsv, shadow_ts_fired=tfd)
                        old = _old_shadow_lines(up, dn)
                        line = _momentum_shadow_line("COIN", up, dn, "shadow", TH)
                        assert (line is None) == (not old)
                        if old and line:
                            m = LINE_RE.match(line)
                            assert m, line
                            assert m.group(6) == (
                                "UP" if fu else "DN" if fd else "flat")
                            assert m.group(9) == (
                                "UP" if tfu else "DN" if tfd else "flat")
                            assert float(m.group(8)) == pytest.approx(tsv, abs=0.05)

    def test_both_flat_is_silent(self):
        up = _hit("uptrendMomentum", False, live_net_move=2.0)
        dn = _hit("downtrendMomentum", False, live_net_move=2.0)
        assert _momentum_shadow_line("COIN", up, dn, "live", TH) is None
        up = _hit("uptrendMomentum", False)
        dn = _hit("downtrendMomentum", False)
        assert _momentum_shadow_line("COIN", up, dn, "shadow", TH) is None

    def test_no_extras_is_silent(self):
        """insufficient_history hits carry no arm extras -> nothing to record."""
        up = _hit("uptrendMomentum", False)
        dn = _hit("downtrendMomentum", False)
        assert _momentum_shadow_line("COIN", up, dn, "live", TH) is None

    def test_perception_wires_the_helper_once(self):
        """Call-site pin: the emit point goes through the single format owner
        and the old per-arm f-strings are gone."""
        import inspect

        src = inspect.getsource(
            __import__("hermes_trader.agents.perception", fromlist=["x"]))
        assert src.count("_momentum_shadow_line(") == 2  # def + one call site
        assert '{h["name"]} mode=live' not in src
        assert '{h["name"]} mode=shadow' not in src
