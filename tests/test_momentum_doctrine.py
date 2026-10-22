"""Tests for the F3 momentum doctrine (board t_757ed56a, round-4 gauntlet).

The doctrine rewrites the ONE side's direction criterion (clause + conviction
band) ONLY on qual rows: momentum arm fired + 4h EMA8 >0.7% from the EMA21
cross + decisive live 6h net (up >= +6.0 / dn <= -5.0). Everything else must
stay byte-identical — the gauntlet lesson is that emission SCOPE is the
safety story (F1/F2 churned 184 verdicts by emitting on all fired rows; F3's
canary on 1,328 unfired rows was 0.000 drift / 0 flips).

Covers: the qual predicate boundaries, the exact replacement text (must
equal the measured arm's transforms), build_questions doctrine=None
byte-identity, and the systemone_primary_verdict integration (knob off =
byte-identical request; knob on + qual = trend_context_6h + text applied;
knob on + non-qual = byte-identical). No network: call_systemone is
monkeypatched (same pattern as tests/test_systemone_primary.py).
"""
import copy
import json

import pytest

from hermes_trader.agents import decision_voice as dv


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    import hermes_trader.agents.config_store as cs
    from pathlib import Path
    Path(cs.CONFIG_PATH).unlink(missing_ok=True)
    monkeypatch.setenv("HERMES_DV_FILE", str(tmp_path / "dv.jsonl"))


def _perc(up=None, dn=None, **over):
    """perception dict with momentum hits; up/dn = (fired, live_net_move)."""
    trig = []
    if up is not None:
        trig.append({"name": "uptrendMomentum", "fired": up[0], "live_net_move": up[1]})
    if dn is not None:
        trig.append({"name": "downtrendMomentum", "fired": dn[0], "live_net_move": dn[1]})
    p = {"id": "pid-md-1", "coin": "APT", "mid": 0.84,
         "composite_score": 20.6, "daily_move_pct": 11.1, "triggers": trig}
    p.update(over)
    return p


def _tf4(e8, e21):
    return {"ema8": e8, "ema21": e21}


# ── qual predicate ────────────────────────────────────────────────────────

def test_qual_long_breakaway():
    d = dv.momentum_doctrine_eval(_perc(up=(True, 7.3)), _tf4(0.987, 1.0))
    assert d and d["side"] == "long"
    assert d["mom6h_pct"] == 7.3 and d["ema4h_to_cross_pct"] == -1.3

def test_qual_short_established():
    d = dv.momentum_doctrine_eval(_perc(dn=(True, -6.1)), _tf4(0.97, 1.0))
    assert d and d["side"] == "short"

def test_near_cross_excluded():
    # dist -0.65% > -0.7 bar: the near-cross population measured
    # both-halves-NEGATIVE (-$17); the doctrine must never fire there.
    assert dv.momentum_doctrine_eval(_perc(up=(True, 9.9)), _tf4(0.9935, 1.0)) is None

def test_weak_momentum_excluded():
    assert dv.momentum_doctrine_eval(_perc(up=(True, 5.9)), _tf4(0.987, 1.0)) is None
    assert dv.momentum_doctrine_eval(_perc(dn=(True, -4.9)), _tf4(0.97, 1.0)) is None

def test_boundaries_inclusive():
    assert dv.momentum_doctrine_eval(_perc(up=(True, 6.0)), _tf4(0.9929, 1.0))["side"] == "long"
    assert dv.momentum_doctrine_eval(_perc(dn=(True, -5.0)), _tf4(0.9929, 1.0))["side"] == "short"

def test_unfired_and_missing_excluded():
    assert dv.momentum_doctrine_eval(_perc(up=(False, 9.0)), _tf4(0.987, 1.0)) is None
    assert dv.momentum_doctrine_eval(_perc(up=(True, None)), _tf4(0.987, 1.0)) is None
    assert dv.momentum_doctrine_eval(_perc(), _tf4(0.987, 1.0)) is None
    assert dv.momentum_doctrine_eval(_perc(up=(True, 9.0)), {}) is None
    assert dv.momentum_doctrine_eval(_perc(up=(True, 9.0)), {"ema8": None, "ema21": None}) is None


# ── text exactness (must equal the measured F3 arm's strings) ────────────

_EXPECT_LONG = ("Clean bullish 4h/1d EMA alignment, or a fired uptrendMomentum "
                "trigger (a fresh 6h trend momentum up) as trend authority"
                " — this is a BREAKAWAY: the 6h momentum trigger fired decisively "
                "while the 4h EMA8 is still far (>0.7%) from the EMA21 cross, i.e. "
                "the move is far from confirmation rather than chasing it; if entry "
                "structure is clean, conviction in the 0.66-0.78 band is warranted"
                ", or whale-accumulation counter-trend flag; entry structure present.")
_EXPECT_SHORT = ("Clean bearish 4h/1d EMA alignment, or a fired downtrendMomentum "
                 "trigger (a fresh 6h trend momentum down) as trend authority"
                 " — this confirms an ESTABLISHED downtrend: the 6h momentum trigger "
                 "is firing down decisively while the 4h EMA8 sits far (>0.7%) below "
                 "the EMA21 cross; if entry structure is clean, conviction in the "
                 "0.66-0.78 band is warranted"
                 "; a 1h bounce in a downtrend is a short, not a dip-buy.")

def test_replacement_text_exact():
    ql = dv.build_questions(False, doctrine={"side": "long"})
    assert ql["direction"]["criteria"]["long"] == _EXPECT_LONG
    assert ql["direction"]["criteria"]["short"] == dv.DIRECTION_QUESTION["criteria"]["short"]
    qs = dv.build_questions(False, doctrine={"side": "short"})
    assert qs["direction"]["criteria"]["short"] == _EXPECT_SHORT
    assert qs["direction"]["criteria"]["long"] == dv.DIRECTION_QUESTION["criteria"]["long"]
    # instructions and every other question untouched
    assert ql["direction"]["instructions"] == dv.DIRECTION_QUESTION["instructions"]
    assert ql["conviction"] == dv.CONVICTION_QUESTION and ql["trap"] == dv.TRAP_QUESTION

def test_build_questions_default_byte_identical():
    assert json.dumps(dv.build_questions(False)) == json.dumps(
        {"direction": dv.DIRECTION_QUESTION, "conviction": dv.CONVICTION_QUESTION,
         "trap": dv.TRAP_QUESTION})
    assert json.dumps(dv.build_questions(True, news_live=True, doctrine=None)) == \
        json.dumps(dv.build_questions(True, news_live=True))


# ── primary-seat integration ──────────────────────────────────────────────

def _cfg(**over):
    c = {"engine": "systemone", "model": "d3-27b", "api_key": "",
         "base_url": "http://x", "timeout_s": 5, "max_state_chars": 12000,
         "fade_fields": False, "composite_score": True, "vwap_fields": True,
         "breakout_side_label": True, "crowded_mover_label": True,
         "setup_shadow": True, "conf_multiplier": 1.0}
    c.update(over)
    return c


def _capture_call(monkeypatch):
    box = {}
    def fake(base_url, api_key, model, state, questions, timeout_s=None):
        box["state"] = copy.deepcopy(state)
        box["questions"] = copy.deepcopy(questions)
        return None, 10, 5   # force early-None return; capture already done
    monkeypatch.setattr(dv, "call_systemone", fake)
    return box


def _run(monkeypatch, cfg, perc, tf4h=None):
    box = _capture_call(monkeypatch)
    out = dv.systemone_primary_verdict(
        "APT", perc, {}, tf4h or _tf4(0.987, 1.0), {}, "0.01%/hr", "no news", [],
        0.5, 10, cfg)
    assert out is None          # early-return path, request already captured
    return box


def test_knob_off_request_byte_identical(monkeypatch):
    box = _run(monkeypatch, _cfg(), _perc(up=(True, 7.3)))
    assert "trend_context_6h" not in box["state"]
    assert box["questions"]["direction"]["criteria"]["long"] == \
        dv.DIRECTION_QUESTION["criteria"]["long"]

def test_knob_on_qual_emits(monkeypatch):
    box = _run(monkeypatch, _cfg(momentum_doctrine=True), _perc(up=(True, 7.3)))
    assert box["state"]["trend_context_6h"] == {
        "side": "long", "mom6h_pct": 7.3, "ema4h_to_cross_pct": -1.3}
    assert box["questions"]["direction"]["criteria"]["long"] == _EXPECT_LONG

def test_knob_on_nonqual_byte_identical(monkeypatch):
    # fired but near-cross (the poison cell) and fired-but-weak: NO emission.
    near = _tf4(0.9935, 1.0)   # dist -0.65% > -0.7 bar
    for perc, tf in ((_perc(up=(True, 9.9)), near), (_perc(up=(True, 5.9)), _tf4(0.987, 1.0))):
        box = _run(monkeypatch, _cfg(momentum_doctrine=True), perc, tf)
        assert "trend_context_6h" not in box["state"]
        assert box["questions"]["direction"]["criteria"]["long"] == \
            dv.DIRECTION_QUESTION["criteria"]["long"]

def test_observer_never_sees_doctrine(monkeypatch):
    # dv_observe path builds questions without a doctrine argument — pin the
    # call-signature contract: build_questions(held, news_live=...) default.
    q = dv.build_questions(held=True, news_live=True)
    assert q["direction"]["criteria"]["long"] == dv.DIRECTION_QUESTION["criteria"]["long"]
