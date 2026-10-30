"""Per-side shadow override tests (owner capability: SHORT live while the
LONG arm keeps accruing in shadow). Sub-block `shadow` beats clause-level
shadow for that side only; absent = byte-identical prior behavior."""
from typing import Any

from hermes_trader.agents.risk_gates import _as_shadow, state_admission

_LONG: dict[str, Any] = {"btc24_max_pct": -3.0, "min_conf": 0.0}
_SHORT: dict[str, Any] = {"btc24_min_pct": -2.0, "btc24_max_pct": 0.0, "min_conf": 0.0}


def _clause(clause_shadow: Any, long_shadow: Any = "__absent__", short_shadow: Any = "__absent__"):
    lg = dict(_LONG)
    sh = dict(_SHORT)
    if long_shadow != "__absent__":
        lg["shadow"] = long_shadow
    if short_shadow != "__absent__":
        sh["shadow"] = short_shadow
    c = {"long": lg, "short": sh}
    if clause_shadow is not None:
        c["shadow"] = clause_shadow
    return c


def _a(side, drift):
    return {"side": side, "confidence": 0.30, "btc_drift_24h": drift}


def test_owner_target_short_live_long_accrues():
    c = _clause(False, long_shadow=True)  # clause false, long.shadow true
    adm_l, why_l, sh_l = state_admission(_a("long", -4.0), c, True)
    adm_s, why_s, sh_s = state_admission(_a("short", -1.0), c, True)
    assert adm_l and sh_l is True          # long: admitted but log-only
    assert adm_s and sh_s is False         # short: LIVE
    assert "rescue tape" in why_l and "kept tape" in why_s


def test_clause_kill_switch_still_wins_by_default():
    c = _clause(True)  # clause shadow true, no per-side keys
    _, _, sh_l = state_admission(_a("long", -4.0), c, True)
    _, _, sh_s = state_admission(_a("short", -1.0), c, True)
    assert sh_l and sh_s  # byte-identical prior behavior


def test_side_override_can_arm_the_other_way():
    c = _clause(True, short_shadow=False)  # clause true, short live
    _, _, sh_l = state_admission(_a("long", -4.0), c, True)
    _, _, sh_s = state_admission(_a("short", -1.0), c, True)
    assert sh_l is True and sh_s is False


def test_side_can_shadow_while_clause_live():
    c = _clause(False, short_shadow=True)  # clause false, short accrues
    _, _, sh_l = state_admission(_a("long", -4.0), c, True)
    _, _, sh_s = state_admission(_a("short", -1.0), c, True)
    assert sh_l is False and sh_s is True


def test_string_forms_accepted():
    c = _clause("false", short_shadow="true")
    _, _, sh_s = state_admission(_a("short", -1.0), c, True)
    assert sh_s is True
    assert _as_shadow("False") is False and _as_shadow(" true") is True


def test_inert_side_returns_clause_shadow():
    # No short sub-block: side-level shadow cannot exist; clause value used.
    c = {"long": dict(_LONG), "shadow": False}
    _, why, sh = state_admission(_a("short", -1.0), c, True)
    assert sh is False and "no 'short' block" in why


def test_admit_verdicts_unchanged_by_shadow_split():
    # Shadow flag changes ONLY the live/log routing, never the admit verdict.
    c_live = _clause(False)
    c_shadow = _clause(True)
    for side, drift in (("long", -4.0), ("short", -1.0)):
        a1, w1, _ = state_admission(_a(side, drift), c_live, True)
        a2, w2, _ = state_admission(_a(side, drift), c_shadow, True)
        assert a1 == a2 and w1 == w2
