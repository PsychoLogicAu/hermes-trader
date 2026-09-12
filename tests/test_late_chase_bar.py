"""Dynamic late-chase confidence bar, lowered by independent-signal corroboration.

The runner gate's late-trend-chase block fires on "uptrend with no fresh
breakout/burst" — the 2026-08-29 TRUMP/SKR/NIL regime (154 blocks in ~80 min,
all TA-CONFIRMED, conf pinned 0.75-0.78). `bypass_late_trend_chase` existed at
0.90 but the LLM tops out at ~0.78, so it was dead code.

New behavior: when the block reason WOULD be late-trend-chase, the confidence
bar for the existing bypass is a function of how many INDEPENDENT signals
(corroboration, not the LLM's own confidence) agree with the side:

    n=0 → fixed bar (bypass_late_trend_chase_min_conf, 0.90)
    n=1 → fixed bar - late_chase_dynamic_per_signal_drop (live 0.10 → 0.80)
    n=2 → fixed bar - 2*drop (0.70), clamped to >= min_confidence

Corroboration signals at gate time (independent pipelines, not the LLM):
  chronos_aligned — get_chronos_signal_sync median sign vs side (300s cache)
  timesfm_aligned — get_timesfm_signal_sync median sign vs side (300s cache);
      counted toward the bar only when `late_chase_timesfm_vote` is true —
      while off, an aligned read yields a [COUNTERFACTUAL] rescue log instead
      (sample accrual; live bar byte-identical either way). When counted, the
      tf vote lowers the bar by its OWN `late_chase_timesfm_drop` (separate
      knob, default 0.0 = inert; live 0.03), NOT the shared
      `late_chase_dynamic_per_signal_drop` (chronos stays at 0.10).
      The weak separate lever is what makes the AND shape binding: tf alone
      (0.90-0.03=0.87) can never release (LLM tops out ~0.82), while
      chronos+tf reaches 0.90-0.10-0.03=0.77 and releases the 0.78 rung.

(The squeeze_aligned Donchian breakout vote was removed with the
squeeze_breakout cull 2026-09-04 — forward P/L sim −$102.55/210 entries,
see .hermes/WATCHLIST.md §D.7. The bar now maxes at 2 signals.)

Both must be exactly True; missing / None / error count as 0. The dynamic bar
is clamped to >= min_confidence (never undercuts the hard floor). Feature
switch: late_chase_dynamic_per_signal_drop <= 0 (or key absent) → the fixed
bar, byte-identical to the pre-feature gate, and NO signal fetches.

Long side only: the late-trend-chase block only fires for side == "long".
"""
import logging
import types

from hermes_trader.agents import executor


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
    a = {
        "coin": "SKR",
        "side": "long",
        "confidence": conf,
        "composite_score": 0.0,
        "uptrend_momentum_fired": True,
    }
    a.update(over)
    return a


def _chronos(monkeypatch, aligned, error=None):
    sig = types.SimpleNamespace(median_pct=0.2 if aligned else -0.2, error=error)
    monkeypatch.setattr(executor, "get_chronos_signal_sync", lambda c, s: sig)
    return sig


def _patch_chronos_fail(monkeypatch):
    monkeypatch.setattr(
        executor, "get_chronos_signal_sync",
        lambda c, s: (_ for _ in ()).throw(RuntimeError("boom")))


# ── Corroboration helper ──────────────────────────────────────────────────────

def test_corroboration_counts_one_when_chronos_only(monkeypatch):
    # squeeze_aligned vote removed (squeeze_breakout cull 2026-09-04):
    # chronos aligned alone is the max-count case.
    _chronos(monkeypatch, aligned=True)
    n, names, shadow = executor._late_chase_corroboration(_analysis(), _gate(), "long")
    assert n == 1
    assert names == ("chronos_aligned",)
    assert shadow == ()


def test_corroboration_zero_when_chronos_error(monkeypatch):
    _chronos(monkeypatch, aligned=True, error="fetch failed")
    n, names, shadow = executor._late_chase_corroboration(_analysis(), _gate(), "long")
    assert n == 0  # chronos error → not a vote; no other signals remain
    assert names == ()
    assert shadow == ()


def test_corroboration_zero_when_both_down(monkeypatch):
    _patch_chronos_fail(monkeypatch)
    n, names, shadow = executor._late_chase_corroboration(_analysis(), _gate(), "long")
    assert n == 0
    assert names == ()
    assert shadow == ()


def test_corroboration_none_when_feature_disabled(monkeypatch):
    _chronos(monkeypatch, aligned=True)
    n, names, shadow = executor._late_chase_corroboration(
        _analysis(), _gate(late_chase_dynamic_per_signal_drop=0.0), "long")
    assert n is None  # feature off → gate must behave exactly as before
    assert names == ()
    assert shadow == ()


def test_corroboration_none_when_key_absent(monkeypatch):
    # Live-config compat: a gate dict without the new keys must mean "off",
    # with no signal fetch (default drop 0.0).
    calls = []
    monkeypatch.setattr(
        executor, "get_chronos_signal_sync",
        lambda c, s: calls.append(1) or types.SimpleNamespace(
            median_pct=0.2, error=None))
    g = _gate()
    del g["runner_entry_gate"]["late_chase_dynamic_per_signal_drop"]
    n, names, shadow = executor._late_chase_corroboration(_analysis(), g, "long")
    assert n is None
    assert calls == []


# ── TimesFM additive vote (shadow by default) ─────────────────────────────────

def _timesfm(monkeypatch, aligned, error=None, enabled=True):
    sig = types.SimpleNamespace(median_pct=0.3 if aligned else -0.3, error=error)
    from hermes_trader.agents import timesfm_signal
    monkeypatch.setattr(timesfm_signal, "get_timesfm_signal_sync",
                        lambda c, s: sig)
    return sig


def _gate_tf(**over):
    g = _gate(**over)
    g["timesfm_signal"] = {"enabled": True}
    return g


def test_timesfm_shadow_vote_off_does_not_count(monkeypatch):
    # timesfm enabled, flag OFF: aligned read lands in shadow_names only;
    # the counted n is unchanged (live bar byte-identical).
    _chronos(monkeypatch, aligned=True)
    _timesfm(monkeypatch, aligned=True)
    n, names, shadow = executor._late_chase_corroboration(
        _analysis(), _gate_tf(), "long")
    assert n == 1
    assert names == ("chronos_aligned",)
    assert shadow == ("timesfm_aligned",)


def test_timesfm_vote_on_counts_third_signal(monkeypatch):
    _chronos(monkeypatch, aligned=True)
    _timesfm(monkeypatch, aligned=True)
    g = _gate_tf(late_chase_timesfm_vote=True)
    n, names, shadow = executor._late_chase_corroboration(
        _analysis(), g, "long")
    assert n == 2
    assert names == ("chronos_aligned", "timesfm_aligned")
    assert shadow == ()


def test_timesfm_vote_skipped_when_signal_disabled(monkeypatch):
    # No timesfm_signal key at all → no fetch, no vote, no shadow name.
    # (squeeze vote removed with the 2026-09-04 cull → chronos is the only
    # shared-drop signal, so n == 1 here.)
    calls = []
    from hermes_trader.agents import timesfm_signal
    monkeypatch.setattr(
        timesfm_signal, "get_timesfm_signal_sync",
        lambda c, s: calls.append(1) or types.SimpleNamespace(
            median_pct=0.3, error=None))
    _chronos(monkeypatch, aligned=True)
    n, names, shadow = executor._late_chase_corroboration(_analysis(), _gate(), "long")
    assert n == 1
    assert names == ("chronos_aligned",)
    assert calls == []
    assert shadow == ()


def test_timesfm_error_counts_zero(monkeypatch):
    _chronos(monkeypatch, aligned=True)
    _timesfm(monkeypatch, aligned=True, error="model down")
    g = _gate_tf(late_chase_timesfm_vote=True)
    n, names, shadow = executor._late_chase_corroboration(
        _analysis(), g, "long")
    assert n == 1  # error → no vote even with the flag on; chronos alone
    assert names == ("chronos_aligned",)
    assert "timesfm_aligned" not in names
    assert shadow == ()


def test_timesfm_vote_flag_off_bar_unchanged(monkeypatch):
    # Gate-level: with the vote flag OFF, conf 0.78 with chronos+timesfm
    # aligned (squeeze not) stays BLOCKED at bar 0.80 — the live bar.
    _chronos(monkeypatch, aligned=True)
    _timesfm(monkeypatch, aligned=True)
    reason = executor._runner_entry_block_reason(
        _analysis(conf=0.78), _gate_tf())
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.80" in reason
    # release-side forensics: the block line carries the gate's OWN conf so a
    # floor-raise replay never has to join conf from another log line (2026-09-06).
    assert "conf 0.78" in reason


def test_timesfm_vote_on_releases(monkeypatch):
    # Vote ON with its OWN (weaker) tf drop 0.03: chronos (0.10) + timesfm
    # (0.03) → bar 0.90-0.13=0.77, so conf 0.78 (the live LLM ceiling)
    # releases. This is the intended AND-rung, distinct from the shared-0.10
    # over-release the separate knob exists to avoid.
    _chronos(monkeypatch, aligned=True)
    _timesfm(monkeypatch, aligned=True)
    g = _gate_tf(late_chase_timesfm_vote=True, late_chase_timesfm_drop=0.03)
    assert executor._runner_entry_block_reason(
        _analysis(conf=0.78), g) == ""


def test_timesfm_vote_on_tf_drop_defaults_zero_keeps_chronos_bar(monkeypatch):
    # Vote ON but tf_drop at its 0.0 default (inert): the tf vote contributes
    # nothing, so the bar is just chronos's shared 0.10 → 0.80. Conf 0.78 <
    # 0.80 stays BLOCKED. Pins that the knob is separate and defaults off —
    # flipping the vote alone never silently re-prices the tf vote at 0.10.
    _chronos(monkeypatch, aligned=True)
    _timesfm(monkeypatch, aligned=True)
    reason = executor._runner_entry_block_reason(
        _analysis(conf=0.78), _gate_tf(late_chase_timesfm_vote=True))
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.80" in reason


def test_timesfm_alone_can_never_release(monkeypatch):
    # Only timesfm aligned (chronos + squeeze both not). Vote ON, tf_drop
    # 0.03 → bar 0.90-0.03=0.87. Even the LLM's top confidence (0.82) is
    # below 0.87, so a timesfm-only vote can never unlock a trade: the value
    # is the AND shape (tf corroborating chronos), not tf per se.
    _chronos(monkeypatch, aligned=False)
    _timesfm(monkeypatch, aligned=True)
    g = _gate_tf(late_chase_timesfm_vote=True, late_chase_timesfm_drop=0.03)
    reason = executor._runner_entry_block_reason(_analysis(conf=0.82), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.87" in reason


def test_timesfm_vote_off_tf_drop_is_ignored(monkeypatch, caplog):
    # Vote OFF: even with a positive tf_drop present, the tf vote is not
    # counted and its drop is not applied — chronos alone → bar 0.80, and the
    # [COUNTERFACTUAL] rescue line uses tf_drop, not the shared drop.
    _chronos(monkeypatch, aligned=True)
    _timesfm(monkeypatch, aligned=True)
    g = _gate_tf(late_chase_timesfm_vote=False, late_chase_timesfm_drop=0.03)
    # conf 0.78 < 0.80 (tf vote not counted) → still blocked at the chronos bar,
    # and the rescue candidate bar is the tf-drop (0.77), not shared-drop (0.70)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(_analysis(conf=0.78), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.80" in reason
    rescued = [r for r in caplog.records
               if "RESCUED by timesfm additive vote" in r.getMessage()]
    assert len(rescued) == 1
    assert "candidate bar 0.77" in rescued[0].getMessage()
    assert "candidate bar 0.70" not in rescued[0].getMessage()


# ── Gate behavior ─────────────────────────────────────────────────────────────

def test_no_corroboration_keeps_fixed_bar(monkeypatch):
    _chronos(monkeypatch, aligned=False)
    reason = executor._runner_entry_block_reason(
        _analysis(conf=0.86), _gate())
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.90" in reason
    # Feature is live, counted 0 signals — the note says so (operability),
    # but the bar itself is unchanged.
    assert "0 signals aligned: none" in reason


def test_one_signal_lowers_bar_to_080(monkeypatch):
    _chronos(monkeypatch, aligned=True)
    # 0.82 >= 0.80 (1-signal bar) but < 0.90 (old fixed bar) → allowed
    assert executor._runner_entry_block_reason(
        _analysis(conf=0.82), _gate()) == ""
    # 0.79 < 0.80 → still blocked, and the message names the lowered bar
    reason = executor._runner_entry_block_reason(
        _analysis(conf=0.79), _gate())
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.80" in reason


def test_two_shared_signal_rung_gone_after_squeeze_cull(monkeypatch):
    # The squeeze_breakout cull (2026-09-04) removed the squeeze_aligned
    # vote, so the n=2 shared-drop rung (0.90 - 2*0.10 = 0.70) no longer
    # exists: chronos is the only shared-drop signal, so the best dynamic
    # bar is 0.80. Conf 0.78 (the live LLM ceiling) that the 0.70 rung used
    # to release is now BLOCKED — the cull is strictly conservative. (The
    # AND rung via the timesfm vote still reaches 0.77 —
    # test_timesfm_vote_on_releases.)
    _chronos(monkeypatch, aligned=True)
    reason = executor._runner_entry_block_reason(_analysis(conf=0.78), _gate())
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.80" in reason
    # The floor still binds below min_confidence (different message).
    reason = executor._runner_entry_block_reason(
        _analysis(conf=0.69), _gate())
    assert "confidence 0.69 < 0.70" in reason


def test_dynamic_bar_never_undercuts_min_confidence(monkeypatch):
    _chronos(monkeypatch, aligned=True)
    # min_confidence 0.82 sits above (0.90 - 2*0.10 = 0.70): the bar clamps
    # to 0.82, so conf == min_confidence enters (no crash, no false block at
    # the late-chase check), and below-min_conf is the min-conf check's job.
    g = _gate(min_confidence=0.82)
    assert executor._runner_entry_block_reason(_analysis(conf=0.82), g) == ""
    reason = executor._runner_entry_block_reason(_analysis(conf=0.80), g)
    assert "confidence 0.80 < 0.82" in reason


def test_feature_off_matches_old_behavior(monkeypatch):
    _chronos(monkeypatch, aligned=True)
    g = _gate(late_chase_dynamic_per_signal_drop=0.0)
    reason = executor._runner_entry_block_reason(_analysis(conf=0.78), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.90" in reason  # fixed bar, no dynamic adjustment at all
    assert "dynamic" not in reason


def test_feature_off_makes_no_signal_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        executor, "get_chronos_signal_sync",
        lambda c, s: calls.append(1) or types.SimpleNamespace(
            median_pct=0.2, error=None))
    g = _gate(late_chase_dynamic_per_signal_drop=0.0)
    assert executor._runner_entry_block_reason(_analysis(conf=0.78), g).startswith(
        "runner_gate_blocked (late trend-only chase")
    assert calls == []  # no signal fetch when the feature is off


def test_bypass_flag_still_required_for_dynamic_bar(monkeypatch):
    _chronos(monkeypatch, aligned=True)
    g = _gate(bypass_late_trend_chase=False)
    reason = executor._runner_entry_block_reason(_analysis(conf=0.78), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.80" in reason  # bar computed, but the flag gates the allow


def test_short_path_untouched_by_corroboration(monkeypatch):
    # Shorts hit the structured_short block, not late-chase; the helper must
    # not be consulted and the message must not carry a dynamic bar.
    _chronos(monkeypatch, aligned=True)
    reason = executor._runner_entry_block_reason(_analysis(
        conf=0.80, side="short", downtrend_momentum_fired=False,
        composite_score=10, slow_burn_count=0),
        _gate(allow_shorts=True, min_short_confidence=0.72,
              min_short_composite=25.0))
    assert "short needs downtrend momentum" in reason
    assert "bar" not in reason


def test_fresh_impulse_entry_never_takes_the_dynamic_path(monkeypatch):
    # A genuinely fresh impulse (volume+breakout + structure) passes before
    # the late-chase check; corroboration signals must not be consulted.
    calls = []
    monkeypatch.setattr(
        executor, "get_chronos_signal_sync",
        lambda c, s: calls.append(1) or types.SimpleNamespace(
            median_pct=None, error=None))
    a = _analysis(conf=0.75, composite_score=35, slow_burn_count=0,
                  volume_spike_fired=True, breakout_fired=True)
    assert executor._runner_entry_block_reason(a, _gate()) == ""
    assert calls == []


# ── Tri-state bypass (2026-09-06): off / on / shadow ─────────────────────────
# `bypass_late_trend_chase` true = live bypass (existing behavior); false +
# `bypass_late_trend_chase_shadow_mode` true = SHADOW: the trade is still
# blocked with the BYTE-IDENTICAL reason (external log parsers keep working)
# and a [gate][SHADOW] line accrues the would-bypass decision; both absent
# = off. The bool is authoritative — shadow only matters when the bool is
# false (fail-safe default: absent keys → off).

def test_shadow_mode_blocks_but_accrues(monkeypatch, caplog):
    # conf 0.82 >= 1-signal bar 0.80: in shadow mode the entry is NOT
    # bypassed (block reason, byte-identical prefix + bar to off-mode) and a
    # loud WOULD HAVE BYPASSED line accrues.
    _chronos(monkeypatch, aligned=True)
    g = _gate(bypass_late_trend_chase=False,
              bypass_late_trend_chase_shadow_mode=True)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(_analysis(conf=0.82), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.80" in reason  # identical to off-mode reason
    accruals = [r for r in caplog.records
                if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert len(accruals) == 1
    assert "SKR LONG" in accruals[0].getMessage()
    assert "shadow mode" in accruals[0].getMessage()


def test_shadow_mode_conf_below_bar_logs_nothing(monkeypatch, caplog):
    # conf 0.78 < bar 0.80 (chronos aligned): even in shadow mode there is no
    # would-bypass to accrue — the block reason is byte-identical and no
    # [gate][SHADOW] late_chase_bypass line is emitted.
    _chronos(monkeypatch, aligned=True)
    g = _gate(bypass_late_trend_chase=False,
              bypass_late_trend_chase_shadow_mode=True)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(_analysis(conf=0.78), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.80" in reason
    accruals = [r for r in caplog.records
                if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert accruals == []


def test_shadow_flag_ignored_when_bypass_bool_true(monkeypatch, caplog):
    # Precedence: bool True is authoritative — the entry bypasses (live) and
    # the shadow accrual line does NOT double-log on top of the bypass.
    _chronos(monkeypatch, aligned=True)
    g = _gate(bypass_late_trend_chase=True,
              bypass_late_trend_chase_shadow_mode=True)
    with caplog.at_level(logging.INFO,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(_analysis(conf=0.82), g)
    assert reason == ""
    shadow = [r for r in caplog.records
              if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert shadow == []
    bypassed = [r for r in caplog.records
                if "late-trend chase bypassed on SKR" in r.getMessage()]
    assert len(bypassed) == 1


def test_off_mode_default_byte_identical(monkeypatch, caplog):
    # Absent shadow key (fail-safe default) and bool false: blocked with the
    # exact off-mode reason and NO shadow accrual — byte-compat pin for the
    # external 06:00 UTC cron parsers that match this reason text.
    _chronos(monkeypatch, aligned=True)
    g = _gate(bypass_late_trend_chase=False)  # no shadow key at all
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(_analysis(conf=0.82), g)
    assert reason == ("runner_gate_blocked (late trend-only chase; no fresh "
                      "breakout/burst, conf 0.82, bar 0.80) "
                      "(dynamic bar 0.80 from 0.90, 1 signal aligned: "
                      "chronos_aligned)")
    accruals = [r for r in caplog.records
                if "WOULD HAVE BYPASSED" in r.getMessage()]
    assert accruals == []


def test_shadow_mode_shadow_block_counterfactual_coexists(monkeypatch, caplog):
    # Shadow mode + timesfm rescue counterfactual: both accrue on the same
    # blocked entry (shadow says "the live bypass would have let this through
    # at the live bar"; the tf-rescue line is about the tf vote) — the shadow
    # branch must not swallow the existing rescue-side counterfactual.
    _chronos(monkeypatch, aligned=True)
    _timesfm(monkeypatch, aligned=True)
    g = _gate_tf(late_chase_timesfm_vote=False, late_chase_timesfm_drop=0.03,
                 bypass_late_trend_chase=False,
                 bypass_late_trend_chase_shadow_mode=True)
    with caplog.at_level(logging.WARNING,
                         logger="hermes_trader.agents.executor"):
        reason = executor._runner_entry_block_reason(_analysis(conf=0.78), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    accruals = [r for r in caplog.records
                if "WOULD HAVE BYPASSED" in r.getMessage()]
    rescued = [r for r in caplog.records
               if "RESCUED by timesfm additive vote" in r.getMessage()]
    # conf 0.78 >= candidate bar 0.77 (0.80 - 0.03) → rescue accrues; and 0.78
    # >= live bar 0.80? No — 0.78 < 0.80 → shadow would-bypass does NOT accrue.
    # (Shadow accrues only when conf >= the LIVE bar.)
    assert len(rescued) == 1
    assert accruals == []


# ── Mover exemption removed (2026-09-12, VVV −$18.81 max_loss) ───────────────
# `structured_daily_mover` used to satisfy the `uptrend and not (...)` guard so
# 24h-mover admits skipped the conf-vs-bar late-chase test entirely. The 28-trade
# mover cohort 09-06→11 ran −$77.97 (−$2.78/trade vs the book's −$0.35). Movers
# now hit the SAME dynamic bar; the mover admission still counts toward the final
# structure requirement (a mover that passes the conf-vs-bar test is admitted via
# structure, not forced through).

def _mover_analysis(conf=0.75, **over):
    a = _analysis(conf=conf, daily_mover_fired=True, slow_burn_count=2,
                  composite_score=33.5)
    a.update(over)
    return a


def test_mover_uptrend_only_now_hits_late_chase_test(monkeypatch):
    # VVV 09-11 14:53 shape: daily mover, uptrend momentum, NO fresh impulse,
    # conf 0.75, chronos aligned (1 signal → bar 0.80). Old code: admitted
    # via structured_daily_mover, zero late-chase coverage. New: blocked.
    _chronos(monkeypatch, aligned=True)
    g = _gate(bypass_late_trend_chase=False,
              bypass_late_trend_chase_shadow_mode=True,
              mover_min_confidence=0.72, mover_min_composite=30.0)
    reason = executor._runner_entry_block_reason(_mover_analysis(conf=0.75), g)
    assert reason.startswith("runner_gate_blocked (late trend-only chase")
    assert "bar 0.80" in reason
    assert "conf 0.75" in reason


def test_mover_conf_at_or_above_bar_still_admitted_via_structure(monkeypatch):
    # The exemption removal must not kill the branch: a mover with conf 0.82
    # >= the 1-signal bar 0.80 (bypass ON) enters through the structure rung.
    _chronos(monkeypatch, aligned=True)
    g = _gate(bypass_late_trend_chase=True,
              mover_min_confidence=0.72, mover_min_composite=30.0)
    assert executor._runner_entry_block_reason(
        _mover_analysis(conf=0.82), g) == ""


def test_mover_without_uptrend_momentum_falls_through_unchanged(monkeypatch):
    # Mover admit with uptrend_momentum NOT fired never took the late-chase
    # path (uptrend gate) before or after the change: structure admission
    # alone is still the admission rung. Regression guard — the fix must not
    # have accidentally blocked the non-uptrend mover shape.
    _chronos(monkeypatch, aligned=True)
    g = _gate(mover_min_confidence=0.72, mover_min_composite=30.0)
    a = _mover_analysis(conf=0.75, uptrend_momentum_fired=False)
    assert executor._runner_entry_block_reason(a, g) == ""


def test_mover_fresh_impulse_never_takes_late_chase_path(monkeypatch):
    # A mover WITH fresh impulse keeps the fast path (byte-identical to the
    # pre-change fresh-impulse behavior) and never consults corroboration.
    calls = []
    monkeypatch.setattr(
        executor, "get_chronos_signal_sync",
        lambda c, s: calls.append(1) or types.SimpleNamespace(
            median_pct=None, error=None))
    a = _mover_analysis(conf=0.75, composite_score=35,
                        volume_spike_fired=True, breakout_fired=True)
    assert executor._runner_entry_block_reason(a, _gate()) == ""
    assert calls == []
