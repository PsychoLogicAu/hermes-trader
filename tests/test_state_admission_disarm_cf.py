"""LOG-ONLY disarm counterfactual probe (board t_0c635008, 2026-11-25).

Why this exists: the 11-07..25 short-clause shadow accrual shows all 3,196
ARM-SKIP decisions carry stamp flag 'd' (disarmed) on BOTH legs and zero carry
'x' (past clock), so widening deep_arm_clock_h / shallow_arm_clock_h admits
exactly zero — the disarm rule is the only arm door. Two cohort reads of that
door disagree (~+$80 alpha on the Nov accrual vs -$272 on the 10-29/31 read),
so it gets PRICED by accrual rather than flipped.

What these tests pin:
  1. the probe changes NOTHING — verdict, why-line and live arms byte-identical;
  2. a probe that would arm emits exactly one INFO accrual line;
  3. a probe that still refuses emits nothing (the refusal IS the answer);
  4. malformed / absent / no-history probe blocks can never raise and never
     change the ARM-SKIP fail-closed behavior;
  5. fan-out is bounded (DISARM_CF_MAX);
  6. one poison probe value (JSON Infinity -> OverflowError on int()) loses only
     its own accrual row and logs a WARNING the LIVE logger actually emits
     (audit F2, 2026-11-25);
  7. a non-finite value in the LIVE disarm_confirm_bars knob fails closed to the
     stricter conf=1 rule instead of raising past the guards and out of the
     decision path (audit F4 — pre-existing flaw class the new knob invites).
"""
import datetime as dt
import inspect
import logging

import pytest

from hermes_trader.agents import risk_gates
from hermes_trader.agents.risk_gates import (
    DISARM_CF_MAX,
    disarm_cf_log_lines,
    parse_disarm_cf,
    state_admission,
    tape_state_arms,
)

H = 3_600_000
T0 = int(dt.datetime(2026, 11, 1, tzinfo=dt.UTC).timestamp() * 1000)
LOGGER = "hermes_trader.agents.risk_gates"

# The LIVE posture of the armed clause (config-models/mammoth-1.json).
LIVE_TAPE = {"deep_arm_pct": -3.0, "deep_arm_clock_h": 48,
             "shallow_arm_pct": -2.0, "shallow_arm_clock_h": 12,
             "require_arm": True, "disarm_confirm_bars": 2,
             "disarm_hard_above_pct": 0.5}


def drift_hist(drift):
    """Turn a drift% series into hourly closes with the LIVE index math
    (close[t] vs close[t-24]), so the gate sees exactly the drift we name."""
    closes = [100.0] * len(drift)
    for i, d in enumerate(drift):
        if i >= 24:
            closes[i] = closes[i - 24] * (1.0 + d / 100.0)
    return [(T0 + i * H, closes[i]) for i in range(len(drift))]


def ramp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


# Crash through the floor, recover back into the band (both arms arm), then ONE
# 3-bar venture to +0.8% (>= hard-kill 0.5), then slide back into the band from
# above and end at -1.2% — inside [-2.0, 0.0) so the tape gate passes and the
# refusal we are studying is the ARM gate. This reproduces the live finding:
# both legs read 'd' (disarmed), not 'x'.
DRIFT = ([0.0] * 40 + ramp(0.0, -6.0, 20) + ramp(-6.0, -1.0, 20)
         + [0.8, 0.9, 0.7] + ramp(0.7, -1.2, 6))
HIST = drift_hist(DRIFT)


def _analysis(hist=HIST, side="short", conf=0.30, coin="TESTX"):
    return {"side": side, "confidence": conf, "coin": coin,
            "btc_drift_24h": DRIFT[-1], "btc_history_1h": hist}


def _clause(tape_extra=None, shadow=True):
    tape = dict(LIVE_TAPE)
    if tape_extra:
        tape.update(tape_extra)
    return {"shadow": shadow, "short": {
        "btc24_min_pct": -2.0, "btc24_max_pct": 0.0, "min_conf": 0.1,
        "tape_state": tape}}


def _lines(caplog):
    return [r.getMessage() for r in caplog.records
            if "SHADOW-DISARM" in r.getMessage()]


# ---- the refusal under study: live posture = ARM-SKIP, both legs 'd' --------

def test_live_posture_refuses_with_disarmed_flags():
    arms = tape_state_arms(HIST, LIVE_TAPE, band_max=0.0)
    assert not arms["deep"]["active"] and not arms["shallow"]["active"]
    assert arms["deep"]["disarmed"] and arms["shallow"]["disarmed"]
    adm, why, _ = state_admission(_analysis(), _clause(), True)
    assert not adm and "ARM-SKIP" in why and "d<=" in why


# ---- 1. the probe changes nothing ------------------------------------------

def test_probe_block_is_verdict_and_why_byte_identical():
    a_plain = state_admission(_analysis(), _clause(), True)
    a_cf = state_admission(_analysis(), _clause(
        {"disarm_cf": [{"hard_above_pct": None, "confirm_bars": 6},
                       {"above_pct": None, "hard_above_pct": None}]}), True)
    assert a_cf == a_plain, "a log-only probe may not move the verdict or why-line"


def test_probe_absent_arms_are_byte_identical():
    assert tape_state_arms(HIST, LIVE_TAPE, band_max=0.0) == tape_state_arms(
        HIST, LIVE_TAPE, band_max=0.0, disarm_probe=None)


def test_probe_seam_actually_relaxes_the_disarm_rule():
    live = tape_state_arms(HIST, LIVE_TAPE, band_max=0.0)
    rel = tape_state_arms(HIST, LIVE_TAPE, band_max=0.0,
                          disarm_probe={"hard_above_pct": None, "confirm_bars": 6})
    assert live["deep"]["disarmed"] and not rel["deep"]["disarmed"]
    assert rel["deep"]["active"] and rel["shallow"]["active"]


# ---- 2/3. accrual lines -----------------------------------------------------

def test_relaxed_probe_arming_emits_one_info_line(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        adm, why, _ = state_admission(_analysis(), _clause(
            {"disarm_cf": {"tag": "nohard", "hard_above_pct": None,
                           "confirm_bars": 6}}), True)
    assert not adm and "ARM-SKIP" in why          # refusal stands
    got = _lines(caplog)
    assert len(got) == 1, got
    line = got[0]
    assert "WOULD ADMIT TESTX SHORT" in line and "PROBE ONLY, NOT admitting" in line
    assert "[nohard]" in line and "live arm [deep=" in line
    assert "d<=" in line.split("but relaxed")[0]        # live side reads 'd'
    assert "would arm [deep=" in line and "+<=" in line.split("but relaxed")[1]


def test_probe_that_still_refuses_emits_nothing(caplog):
    # hard kill raised to +2.0 but the 3-bar venture still persists past the
    # live 2-bar confirm rule -> relaxed probe agrees with the live refusal.
    with caplog.at_level(logging.INFO, logger=LOGGER):
        adm, why, _ = state_admission(_analysis(), _clause(
            {"disarm_cf": {"hard_above_pct": 2.0}}), True)
    assert not adm and "ARM-SKIP" in why
    assert _lines(caplog) == []


def test_probe_off_by_default_emits_nothing(caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        state_admission(_analysis(), _clause(), True)
    assert _lines(caplog) == []


# ---- 4. malformed / missing can never raise, never gate ---------------------

def test_malformed_probe_blocks_are_inert(caplog):
    for bad in ("yes", 42, None, [{}], [{"nothing": 1}],
                [{"hard_above_pct": "banana"}], [{"confirm_bars": "x"}],
                {"disarm_cf": "all"}):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=LOGGER):
            adm, why, _ = state_admission(_analysis(), _clause(
                {"disarm_cf": bad} if not isinstance(bad, dict) or "disarm_cf" not in bad
                else bad), True)
        assert not adm and "ARM-SKIP" in why
        assert _lines(caplog) == [], f"{bad!r} must accrue nothing"


def test_no_history_fail_closed_and_quiet(caplog):
    for hist in (None, [], [("x", "y")] * 40):
        caplog.clear()
        a = _analysis(hist=hist)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            adm, why, _ = state_admission(a, _clause(
                {"disarm_cf": {"hard_above_pct": None, "confirm_bars": 6}}), True)
        assert not adm and "ARM-SKIP" in why      # fail-CLOSED is unchanged
        assert _lines(caplog) == []


def test_probe_helper_never_raises():
    # A fault inside the probe must degrade to no lines, not propagate into the
    # decision (eval_all_gates / the runner gate have no per-gate try).
    assert disarm_cf_log_lines({"coin": "X", "side": "short"},
                               {"disarm_cf": [{"hard_above_pct": object()}]},
                               0.0, "deep=n shallow=n") == []
    assert disarm_cf_log_lines({}, {"disarm_cf": {"confirm_bars": 6}}, 0.0, "") == []


# ---- audit F2 (2026-11-25): one poison probe may not zero the whole list -----

@pytest.mark.parametrize("poison", [float("inf"), float("-inf"), float("nan")])
def test_poison_probe_value_cannot_zero_the_list(caplog, poison):
    # F4's guard turns a non-finite confirm_bars into the stricter conf=1, so the
    # poison probe stays disarmed and simply accrues nothing; F2's per-probe guard
    # is the backstop. Either way the two GOOD probes must still accrue and the
    # decision must still be the live ARM-SKIP refusal.
    with caplog.at_level(logging.INFO, logger=LOGGER):
        adm, why, _ = state_admission(_analysis(), _clause({"disarm_cf": [
            {"tag": "good1", "hard_above_pct": None, "confirm_bars": 6},
            {"tag": "poison", "confirm_bars": poison},
            {"tag": "good2", "above_pct": None, "hard_above_pct": None},
        ]}), True)
    assert not adm and "ARM-SKIP" in why
    got = _lines(caplog)
    assert len(got) == 2, f"good probes must still accrue: {got}"
    assert "[good1]" in got[0] and "[good2]" in got[1]
    assert "[poison]" not in "".join(got)


def test_per_probe_fault_guard_is_isolated_and_audible(monkeypatch, caplog):
    # The per-probe try/except is defense-in-depth: prove a fault raised for ONE
    # probe (simulated here) drops that probe only, and is logged at WARNING —
    # the live logger does not emit DEBUG, so a DEBUG-only fault note would make
    # lost accrual invisible (audit F2).
    real = risk_gates.tape_state_arms

    def flaky(hist, cfg, band_max=None, disarm_probe=None):
        if disarm_probe and disarm_probe.get("confirm_bars") == 7:
            raise RuntimeError("simulated probe-internal fault")
        return real(hist, cfg, band_max=band_max, disarm_probe=disarm_probe)

    monkeypatch.setattr(risk_gates, "tape_state_arms", flaky)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        adm, why, _ = state_admission(_analysis(), _clause({"disarm_cf": [
            {"tag": "good1", "hard_above_pct": None, "confirm_bars": 6},
            {"tag": "flaky", "confirm_bars": 7},
            {"tag": "good2", "above_pct": None, "hard_above_pct": None},
        ]}), True)
    assert not adm and "ARM-SKIP" in why
    got = _lines(caplog)
    assert len(got) == 2 and "[good1]" in got[0] and "[good2]" in got[1], got
    faults = [r.getMessage() for r in caplog.records
              if "DISARM-CF-FAULT" in r.getMessage() and r.levelno >= logging.WARNING]
    assert len(faults) == 1 and "[flaky]" in faults[0], faults
    assert "SHADOW-DISARM" not in "".join(f for f in faults)  # fault marker distinct


# ---- audit F4: non-finite in the LIVE disarm knob must not crash the decision --

@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_live_disarm_conf_fails_closed_not_upwards(bad):
    # Pre-existing flaw class this knob invites: JSON Infinity parses, then
    # int(inf) raised OverflowError PAST both guards and out of state_admission
    # (no per-gate try at either call site). Now: treated as a malformed knob,
    # i.e. conf=1 = the STRICTER single-close disarm. Never an exception, never
    # a looser arm.
    cfg = dict(LIVE_TAPE)
    cfg["disarm_confirm_bars"] = bad
    arms = tape_state_arms(HIST, cfg, band_max=0.0)
    assert isinstance(arms, dict), "must not degrade to 'malformed' refusal path"
    assert arms["deep"]["disarmed"] and arms["shallow"]["disarmed"]
    assert not arms["deep"]["active"] and not arms["shallow"]["active"]
    adm, why, _ = state_admission(_analysis(),
                                  _clause({"disarm_confirm_bars": bad}), True)
    assert not adm and "ARM-SKIP" in why


# ---- 5. parser + fan-out bound ---------------------------------------------

def test_parser_accepts_dict_or_list_and_labels_probes():
    one = parse_disarm_cf({"hard_above_pct": None})
    assert one == [{"tag": "hard_above_pct=off", "probe": {"hard_above_pct": None}}]
    many = parse_disarm_cf([{"tag": "A", "confirm_bars": 6},
                            {"above_pct": None, "hard_above_pct": None},
                            "junk", {}])
    assert [p["tag"] for p in many] == ["A", "above_pct=off,hard_above_pct=off"]


def test_fanout_bounded(caplog):
    probes = [{"tag": f"p{i}", "hard_above_pct": None, "confirm_bars": 6 + i}
              for i in range(DISARM_CF_MAX + 3)]
    with caplog.at_level(logging.INFO, logger=LOGGER):
        state_admission(_analysis(), _clause({"disarm_cf": probes}), True)
    assert len(_lines(caplog)) == DISARM_CF_MAX


# ---- wiring pins -----------------------------------------------------------

def test_wiring_single_site_and_marker_unique():
    src = inspect.getsource(risk_gates)
    assert src.count("[gate][STATE-ADMIT][SHADOW-DISARM]") == 1
    assert src.count("disarm_cf_log_lines(") == 2      # def + the one ARM-SKIP site
    assert src.count('disarm_probe=p["probe"]') == 1    # the one probe call site
    assert src.count("def tape_state_arms(") == 1       # seam lives in one function
