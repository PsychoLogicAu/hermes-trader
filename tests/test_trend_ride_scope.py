"""B.27 scope knob: `dsl_exit.regime_aware.scope` + the late-chase entry tag.

Two coupled pieces (branch feature/late-chase-volume-shadow):

1. select_exit_params(..., entry_class=...) honours `regime_aware.scope`:
     absent / "" / "all"  -> every regime-up trade gets trend_ride (original)
     "late_chase"         -> ONLY entries tagged late-chase-bypass get it;
                             everything else stays scalp
     unrecognised value   -> FAILS SAFE to scalp (+ warning)
   `enabled` stays the master switch: off => scalp for everyone.

2. _runner_entry_block_reason tags `analysis["_entry_class"] = "late_chase"`
   exactly when the late-chase bypass is LIVE (bypassed True). Shadow accruals
   and blocks never tag; fresh-impulse admits never tag. The open path passes
   the tag to select_exit_params, so a per-position ExitPolicy is chosen at
   OPEN and the DSL tracker carries it for its whole life (state round-trips
   policy via asdict — mid-life config flips never touch open positions).

Rationale (.hermes/WATCHLIST.md §B.27 SCOPE A/B): late-chase-admitted entries
in an up regime are a good trend_ride population (+$7.6 net delta, both halves
positive); regime-up-WIDE trend_ride is tail-carried (top-5 = 172% of net).
"""
import logging
import types

from hermes_trader.agents import executor

TR = {"protect_pct": 3.0, "retrace_threshold": 0.55}


def _dsl(scope=None, enabled=True):
    ra = {"enabled": enabled, "trend_ride": dict(TR)}
    if scope is not None:
        ra["scope"] = scope
    return {"protect_pct": 1.0, "retrace_threshold": 0.25, "regime_aware": ra}


# ── select_exit_params: scope semantics ──────────────────────────────────────

def test_scope_absent_is_book_wide_trend_ride():
    pp, rt, tiers, label = executor.select_exit_params(_dsl(), "up")
    assert (pp, rt, label) == (3.0, 0.55, "trend_ride(up-regime)")
    # tagged entry gets it too — scope absent ignores entry_class entirely
    pp2, *_ = executor.select_exit_params(_dsl(), "up", entry_class="late_chase")
    assert pp2 == 3.0


def test_scope_all_same_as_absent():
    for scope in ("all", "", None):
        pp, rt, tiers, label = executor.select_exit_params(_dsl(scope), "up",
                                                           entry_class=None)
        assert (pp, rt, label) == (3.0, 0.55, "trend_ride(up-regime)")


def test_scope_late_chase_releases_only_tagged():
    pp, rt, tiers, label = executor.select_exit_params(
        _dsl("late_chase"), "up", entry_class="late_chase")
    assert (pp, rt) == (3.0, 0.55)
    assert "late_chase" in label


def test_scope_late_chase_keeps_untagged_on_scalp():
    pp, rt, tiers, label = executor.select_exit_params(
        _dsl("late_chase"), "up", entry_class=None)
    assert (pp, rt, label) == (1.0, 0.25, "scalp")


def test_scope_late_chase_ignores_other_entry_classes():
    pp, rt, tiers, label = executor.select_exit_params(
        _dsl("late_chase"), "up", entry_class="fresh_impulse")
    assert (pp, rt, label) == (1.0, 0.25, "scalp")


def test_scope_late_chase_down_regime_stays_scalp():
    # scope never LOOSENS outside up-regime — regime stays the master condition
    pp, rt, tiers, label = executor.select_exit_params(
        _dsl("late_chase"), "down", entry_class="late_chase")
    assert (pp, rt, label) == (1.0, 0.25, "scalp")


def test_scope_unknown_fails_safe_to_scalp(caplog):
    with caplog.at_level(logging.WARNING):
        pp, rt, tiers, label = executor.select_exit_params(
            _dsl("late-chase"), "up", entry_class="late_chase")  # typo'd value
    assert (pp, rt, label) == (1.0, 0.25, "scalp")
    assert "unrecognised" in caplog.text


def test_master_switch_off_beats_scope():
    pp, rt, tiers, label = executor.select_exit_params(
        _dsl("late_chase", enabled=False), "up", entry_class="late_chase")
    assert (pp, rt, label) == (1.0, 0.25, "scalp")


def test_scope_case_and_whitespace_tolerant():
    pp, *_ = executor.select_exit_params(_dsl(" Late_Chase "), "up",
                                         entry_class="late_chase")
    assert pp == 3.0


# ── the tag: set ONLY on a live late-chase bypass ────────────────────────────

def _gate(**over):
    g = {
        "enabled": True,
        "min_confidence": 0.70,
        "min_composite": 30.0,
        "min_hip3_composite": 50.0,
        "bypass_late_trend_chase": True,
        "bypass_late_trend_chase_min_conf": 0.90,
        "late_chase_dynamic_per_signal_drop": 0.10,
        "late_chase_timesfm_vote": False,
        "late_chase_timesfm_drop": 0.0,
    }
    g.update(over)
    return {"runner_entry_gate": g}


def _analysis(conf=0.78, **over):
    a = {"coin": "JUP", "side": "long", "confidence": conf,
         "composite_score": 40.0, "uptrend_momentum_fired": True}
    a.update(over)
    return a


def _chronos(monkeypatch, aligned):
    sig = types.SimpleNamespace(median_pct=0.2 if aligned else -0.2,
                                spread_pct=0.5, error=None)
    monkeypatch.setattr(executor, "get_chronos_signal_sync", lambda c, s: sig)


def test_live_bypass_tags_entry_class(monkeypatch):
    _chronos(monkeypatch, aligned=True)  # 1 signal -> bar 0.90-0.10=0.80... conf .82 passes
    a = _analysis(conf=0.82)
    assert executor._runner_entry_block_reason(a, _gate()) == ""
    assert a.get("_entry_class") == "late_chase"


def test_fresh_impulse_admit_does_not_tag(monkeypatch):
    # fresh breakout+volume admits before the late-chase branch is ever reached
    a = _analysis(conf=0.78, momentum_burst_fired=True,
                  volume_spike_fired=True)
    assert executor._runner_entry_block_reason(a, _gate()) == ""
    assert "_entry_class" not in a


def test_blocked_does_not_tag(monkeypatch):
    _chronos(monkeypatch, aligned=False)  # no corroboration -> bar stays 0.90
    a = _analysis(conf=0.78)
    reason = executor._runner_entry_block_reason(a, _gate())
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "_entry_class" not in a


def test_shadow_bypass_does_not_tag(monkeypatch):
    # conf clears the bar but bypass is OFF with shadow_mode on: accrual logs,
    # trade still blocks, and NO tag may leak to a later admit.
    _chronos(monkeypatch, aligned=True)
    g = _gate(bypass_late_trend_chase=False,
              bypass_late_trend_chase_shadow_mode=True)
    a = _analysis(conf=0.82)
    reason = executor._runner_entry_block_reason(a, g)
    assert reason.startswith("runner_gate_blocked")
    assert "_entry_class" not in a
