"""Up-gate native tag on the early_long WOULD ADMIT line — t_982bccdd.

Record-only attribute so the proposed up-tape-gated early-long door accrues
natively (grep `upgate=allow`): the 2026-11-23 counterfactual
(scratch/_early_long_lane_cf.py, n=193) prices the door REGIME-GATED —
up-tape +$0.83/tr vs down/chop -$0.42..-0.80 with ~zero alpha. Pins:
allow/deny semantics (drift > bar, strict), hot-tunable bar
(upgate_drift_min), hot-rollback knob (upgate_tag=false -> byte-clean),
LONG-only placement (SHORT lines byte-identical), and '-' degradation on
missing/non-finite inputs — never an opinion, never a raise.
"""
from hermes_trader.agents import market_regime, risk_gates

LONG_CAND = {"coin": "TAO", "ts": 1794973294664, "mid": 300.0, "comp": 37.8,
             "side": "long", "lane": "early_long", "variant": ["momhi_1dF"],
             "mom6h_net": 6.8, "ema4h_fast": 301.0, "ema4h_slow": 299.0,
             "ema4h_pair": "8/21", "ema4h_dist_pct": 0.667,
             "ema4h_bucket": "deep", "ema1d_fast": 290.0, "ema1d_slow": 280.0,
             "ema1d_pair": "8/21"}
SHORT_CAND = {"coin": "APT", "ts": 1794973294664, "mid": 8.12, "comp": 34.0,
              "side": "short", "lane": "aligned_short",
              "variant": ["aligned_dn4h"], "mom6h_net": -6.3,
              "ema4h_fast": 8.02, "ema4h_slow": 8.44, "ema4h_pair": "8/21",
              "ema4h_dist_pct": -4.976, "ema4h_bucket": "deep"}

_DEFAULT = object()   # lets a test pass tape=None on purpose


def _emit(monkeypatch, caplog, tape=_DEFAULT, lane_cfg=None, cands=None):
    _t = {"vol": 3.1, "drift": 1.77} if tape is _DEFAULT else tape
    monkeypatch.setattr(market_regime, "btc_tape_activity",
                        lambda force=False: _t)
    caplog.clear()
    caplog.set_level("WARNING", logger="hermes_trader.agents.risk_gates")
    risk_gates._as_lane_last_emit.clear()
    n = risk_gates.aligned_short_lane_emit(
        list(cands if cands is not None else [LONG_CAND, SHORT_CAND]),
        lane_cfg if lane_cfg is not None else {"enabled": True},
        open_coins=set(), slots_used=0, max_slots=8, conf_by_coin={})
    lines = [r.message for r in caplog.records
             if "WOULD ADMIT" in r.getMessage()]
    long_line = next((l for l in lines if " LONG " in l), None)
    short_line = next((l for l in lines if " SHORT " in l), None)
    assert long_line is not None and short_line is not None, lines
    return n, long_line, short_line


class TestUpGateTag:
    def test_allow_on_positive_drift_long_only(self, monkeypatch, caplog):
        n, long_line, short_line = _emit(monkeypatch, caplog)
        assert n == 2
        assert "upgate=allow(drift+1.77%)" in long_line
        assert "upgate" not in short_line

    def test_deny_on_nonpositive_drift(self, monkeypatch, caplog):
        _n, line, _s = _emit(monkeypatch, caplog, tape={"vol": 3.0,
                                                        "drift": -0.81})
        assert "upgate=deny(drift-0.81%)" in line
        _n, line, _s = _emit(monkeypatch, caplog, tape={"vol": 3.0,
                                                        "drift": 0.0})
        assert "upgate=deny(drift+0.00%)" in line  # positive is STRICT

    def test_bar_is_hot_tunable(self, monkeypatch, caplog):
        cfg = {"enabled": True, "upgate_drift_min": 1.0}
        _n, line, _s = _emit(monkeypatch, caplog, tape={"vol": 3.0,
                                                        "drift": 0.5},
                             lane_cfg=cfg)
        assert "upgate=deny(drift+0.50%)" in line
        _n, line, _s = _emit(monkeypatch, caplog, tape={"vol": 3.0,
                                                        "drift": 1.2},
                             lane_cfg=cfg)
        assert "upgate=allow(drift+1.20%)" in line

    def test_missing_tape_degrades_to_dash(self, monkeypatch, caplog):
        # Contract shape for a data gap: btc_tape_activity() returns the
        # WHOLE dict as None (vol/drift are numeric whenever it returns).
        _n, line, _s = _emit(monkeypatch, caplog, tape=None)
        assert "upgate=-" in line

    def test_finite_helper_rejects_nonparseable_and_nonfinite(self):
        f = risk_gates._as_finite_float
        assert f(None) is None and f("abc") is None
        assert f("Infinity") is None and f("1e999") is None
        assert f(float("nan")) is None and f(float("inf")) is None
        assert f(-0.81) == -0.81 and f("+1.5") == 1.5 and f(0) == 0.0

    def test_nonfinite_bar_falls_back_to_zero(self, monkeypatch, caplog):
        cfg = {"enabled": True, "upgate_drift_min": "Infinity"}
        _n, line, _s = _emit(monkeypatch, caplog,
                             tape={"vol": 3.0, "drift": 0.5}, lane_cfg=cfg)
        assert "upgate=allow(drift+0.50%)" in line

    def test_knob_off_is_byte_clean(self, monkeypatch, caplog):
        cfg = {"enabled": True, "upgate_tag": False}
        _n, long_line, short_line = _emit(monkeypatch, caplog, lane_cfg=cfg)
        assert "upgate" not in long_line and "upgate" not in short_line
        assert long_line.rstrip().endswith("retrace_pp=- — NOT admitting "
                                           "(shadow v1; exit mirror max_loss "
                                           "5 / protect 1 / retrace .25 / "
                                           "breach 2 / stale 720m priced "
                                           "off-line)")

    def test_short_line_byte_identical_knob_on_vs_off(self, monkeypatch,
                                                      caplog):
        _n, _l_on, short_on = _emit(monkeypatch, caplog)
        caplog.clear()
        _n, _l_off, short_off = _emit(
            monkeypatch, caplog, lane_cfg={"enabled": True,
                                           "upgate_tag": False})
        assert short_on == short_off
