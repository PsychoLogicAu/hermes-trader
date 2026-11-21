"""Tests for the GENERIC dedicated clause slot budget (t_38fa6d3a).

First consumer: the ANSWERING seat's profile
`state_admission.short.{max_clause_slots, max_clause_slots_shadow}` — a
dedicated n-slot budget on top of the shared `max_concurrent` pool, so a
sub-floor clause can no longer own the whole book. Shadow-first: absent key =
inert + ZERO ledger reads (byte-identical); `max_clause_slots_shadow` absent/
true = log-only WOULD SKIP accrual, only an explicit false binds. A configured
cap under the contention floor is clamped UP (below it the slot-freeing
supervision switches off for exactly this population — sizing rule on
t_38fa6d3a). Occupancy comes from `ledger.count_open_with_lane(<tag>)` over
OPEN rows tagged `config_snapshot.lane`.

Covers: pure helper contract, the ledger-reading wrapper (zero-reads on
absent key, fail-closed on unreadable ledger), both gate call sites (runner +
classic), the clamp, the shadow accrual, and the ledger tag that makes the
clause cohort countable.
"""

import logging

from hermes_trader import ledger
from hermes_trader.agents import executor, market_regime, risk_gates


def _gate(**over):
    g = {"enabled": True, "min_confidence": 0.65, "min_composite": 30.0,
         "allow_shorts": True}
    g.update(over)
    # Contention floor pinned explicitly (5) so budget tests exercise the cap,
    # not the 60%-of-cap ratio default: every test measures the same floor the
    # production config pins (dsl_exit.stale_flat_min_positions).
    return {"runner_entry_gate": g, "max_concurrent": 10,
            "dsl_exit": {"stale_flat_min_positions": 5}}


def _cfg(**over):
    c = {"runner_entry_gate": {"allow_shorts": True},
         "min_ai_confidence": 0.65, "max_concurrent": 10,
         "dsl_exit": {"stale_flat_min_positions": 5}}
    c.update(over)
    return c


def _analysis(**over):
    a = {
        "coin": "TEST",
        "side": "short",
        "confidence": 0.47,
        "composite_score": 0.0,
        "primary_model": "test-model",
        "daily_move_pct": 0.0,
        "daily_mover_fired": False,
        "btc_drift_24h": -1.5,  # in the [-2.0, 0.0) band
    }
    a.update(over)
    return a


def _clause(**short_over):
    sh = {"btc24_min_pct": -2.0, "btc24_max_pct": 0.0}
    sh.update(short_over)
    return {
        "shadow": False,
        "long": {"btc24_max_pct": -3.0},
        "short": sh,
    }


def _stub_profile(monkeypatch, clause):
    monkeypatch.setattr(
        executor, "read_model_profile_for_model",
        lambda name: {"state_admission": clause} if clause is not None else {})


def _stub_rg_profile(monkeypatch, clause):
    monkeypatch.setattr(
        risk_gates, "read_model_profile_for_model",
        lambda name: {"state_admission": clause} if clause is not None else {})


def _stub_tape(monkeypatch, drift=-1.5):
    monkeypatch.setattr(market_regime, "btc_tape_activity",
                        lambda *a, **k: {"vol": 1.2, "drift": drift})


def _boom_read(tag="state_admission"):
    raise AssertionError(f"ledger read must not happen (tag={tag})")


# ── pure helper contract ─────────────────────────────────────────────────

def test_helper_absent_key_inert():
    assert risk_gates.clause_slot_budget({"btc24_max_pct": 0.0}, 3) is None


def test_helper_non_dict_inert():
    assert risk_gates.clause_slot_budget(None, 3) is None
    assert risk_gates.clause_slot_budget("max_clause_slots", 3) is None


def test_helper_malformed_never_inert_when_key_present():
    # A typo in an ARMED budget must never degrade to inert (inert = looser
    # than the owner wrote = over-admit). Floor present -> armed at floor;
    # floor absent -> fail-CLOSED at 0, still armed + malformed marker.
    b = risk_gates.clause_slot_budget({"max_clause_slots": "lots"}, 0, floor=5)
    assert b is not None and b["limit"] == 5 and b["malformed"] and b["allow"]
    b = risk_gates.clause_slot_budget({"max_clause_slots": "lots"}, 0)
    assert b is not None and b["limit"] == 0 and b["malformed"]
    # Log-only default never blocks, but the binding (shadow:false) posture
    # fails CLOSED at 0 rather than going inert.
    assert b["allow"] and b["shadow_would_skip"]
    b = risk_gates.clause_slot_budget(
        {"max_clause_slots": "lots", "max_clause_slots_shadow": False}, 0)
    assert not b["allow"] and not b["shadow_would_skip"]
    assert "malformed slots value" in b["reason"]


def test_helper_non_finite_slots_cannot_raise_and_never_loosens():
    # `float("inf")`/`"Infinity"`/`"1e999"` parse fine and int() on them raises
    # OverflowError — the exact Infinity-bar class this repo fixed twice
    # (3c15e35d, 258284fc). Here they land in the malformed path, armed.
    for v in (float("inf"), float("-inf"), "Infinity", "-Infinity", "inf",
              1e999, "1e999", float("nan")):
        b = risk_gates.clause_slot_budget(
            {"max_clause_slots": v, "max_clause_slots_shadow": False}, 0,
            floor=5)
        assert b is not None and b["malformed"] and b["limit"] == 5, repr(v)
        assert b["allow"], repr(v)          # armed at the floor, not unlimited
        b = risk_gates.clause_slot_budget(
            {"max_clause_slots": v, "max_clause_slots_shadow": False}, 0)
        assert b is not None and b["limit"] == 0 and not b["allow"], repr(v)


def test_helper_allow_and_block_boundaries():
    cfg = {"max_clause_slots": 5, "max_clause_slots_shadow": False}
    b = risk_gates.clause_slot_budget(cfg, 4)
    assert b["allow"] and not b["shadow_would_skip"] and b["reason"] == ""
    b = risk_gates.clause_slot_budget(cfg, 5)
    assert not b["allow"] and not b["shadow_would_skip"]
    assert "5/5" in b["reason"]
    # used BELOW the limit is not "full", exactly at it is.
    b = risk_gates.clause_slot_budget(cfg, 6)
    assert not b["allow"]


def test_helper_shadow_default_is_log_only_and_only_explicit_false_binds():
    for val, binds in ((None, False), (True, False), ("true", False),
                       ("banana", False), (False, True), ("false", True),
                       ("False", True)):
        cfg = {"max_clause_slots": 2}
        if val is not None:
            cfg["max_clause_slots_shadow"] = val
        b = risk_gates.clause_slot_budget(cfg, 99)
        assert b["shadow"] is (not binds), repr(val)
        assert b["allow"] is (not binds), repr(val)
        assert b["shadow_would_skip"] is (not binds), repr(val)


def test_helper_zero_budget_fails_closed():
    cfg = {"max_clause_slots": 0, "max_clause_slots_shadow": False}
    b = risk_gates.clause_slot_budget(cfg, 0)
    assert not b["allow"] and not b["shadow_would_skip"]
    # Log-only default still accrues the would-refusal (never a looser budget).
    b = risk_gates.clause_slot_budget({"max_clause_slots": -1}, 0)
    assert b["shadow"] and b["allow"] and b["shadow_would_skip"]


def test_helper_unprovable_occupancy_is_full():
    cfg = {"max_clause_slots": 5, "max_clause_slots_shadow": False}
    b = risk_gates.clause_slot_budget(cfg, -1)  # ledger -1 contract
    assert not b["allow"]


def test_helper_clamps_cap_up_to_contention_floor():
    # Configured 3 with floor 5: cap is corrected UP (supervision blindness
    # below the floor), not honored.
    cfg = {"max_clause_slots": 3, "max_clause_slots_shadow": False}
    b = risk_gates.clause_slot_budget(cfg, 4, floor=5)
    assert b["limit"] == 5 and b["allow"]  # would have blocked at 3
    b = risk_gates.clause_slot_budget(cfg, 5, floor=5)
    assert not b["allow"]


def test_helper_zero_or_negative_is_never_clamped_up_to_the_floor():
    # Reviewer LE2 (2026-11-21): the original code clamped BEFORE the zero
    # check, so an owner writing 0 ("refuse all clause admits") got a live
    # 5-slot book. Fail-closed must win over the sizing clamp.
    for bad in (0, -3):
        cfg = {"max_clause_slots": bad, "max_clause_slots_shadow": False}
        b = risk_gates.clause_slot_budget(cfg, 0, floor=5)
        assert b["limit"] == bad and not b["allow"], bad
        assert not b["shadow_would_skip"] and b["malformed"] is False
        assert "clause admits refused" in b["reason"]
        # Log-only default still never blocks, but accrues the refusal.
        b = risk_gates.clause_slot_budget({"max_clause_slots": bad}, 0, floor=5)
        assert b["allow"] and b["shadow_would_skip"]


def test_contention_floor_non_finite_or_zero_pin():
    # A non-finite pin falls back to the ratio (never 0, never a raise); an
    # explicit 0 is a legitimate "no floor" and must survive verbatim.
    assert risk_gates.contention_floor(
        {"max_concurrent": 10,
         "dsl_exit": {"stale_flat_min_positions": float("inf")}}) == 6
    assert risk_gates.contention_floor(
        {"max_concurrent": 10,
         "dsl_exit": {"stale_flat_min_positions": "banana"}}) == 6
    assert risk_gates.contention_floor(
        {"max_concurrent": 10,
         "dsl_exit": {"stale_flat_min_positions": 0}}) == 0
    assert risk_gates.contention_floor(
        {"max_concurrent": 10,
         "dsl_exit": {"stale_flat_min_positions": "5"}}) == 5
    assert risk_gates.contention_floor(
        {"max_concurrent": float("nan")}) == 3   # ratio default of cap 5


def test_wrapper_unreadable_occupancy_is_full_not_inert(monkeypatch, caplog):
    # A raising ledger read (or import failure) must NOT turn a restrictive
    # knob off — it reads as FULL.
    caplog.set_level(logging.WARNING, logger="hermes_trader.agents.risk_gates")

    def _raise(tag):
        raise RuntimeError("ledger unreadable")

    monkeypatch.setattr(ledger, "count_open_with_lane", _raise)
    b = risk_gates.clause_slot_budget_check(
        _clause(max_clause_slots=5, max_clause_slots_shadow=False), "short",
        _cfg())
    assert b is not None and not b["allow"] and b["used"] == b["limit"] == 5
    assert "occupancy UNREADABLE" in caplog.text
    # And in shadow mode the same unreadable occupancy accrues, never blocks.
    b = risk_gates.clause_slot_budget_check(_clause(max_clause_slots=5),
                                            "short", _cfg())
    assert b["allow"] and b["shadow_would_skip"]


def test_helper_key_names_are_configurable_for_future_lanes():
    # Generic contract: a future lane carries its own key names/tag and calls
    # the same helper (no new gate plumbing).
    cfg = {"max_lane_slots": 2, "lane_shadow": False}
    b = risk_gates.clause_slot_budget(cfg, 2, slots_key="max_lane_slots",
                                      shadow_key="lane_shadow")
    assert not b["allow"] and b["limit"] == 2
    b = risk_gates.clause_slot_budget(cfg, 1, slots_key="max_lane_slots",
                                      shadow_key="lane_shadow")
    assert b["allow"]


# ── ledger-reading wrapper ───────────────────────────────────────────────

def test_wrapper_absent_key_does_no_ledger_read(monkeypatch):
    monkeypatch.setattr(ledger, "count_open_with_lane", _boom_read)
    assert risk_gates.clause_slot_budget_check(
        _clause(), "short", _cfg()) is None
    assert risk_gates.clause_slot_budget_check(
        None, "short", _cfg()) is None
    assert risk_gates.clause_slot_budget_check(
        {"short": {"btc24_min_pct": -2.0}}, "short", _cfg()) is None


def test_wrapper_counts_only_on_armed_key(monkeypatch):
    calls = []
    monkeypatch.setattr(ledger, "count_open_with_lane",
                        lambda tag: calls.append(tag) or 4)
    b = risk_gates.clause_slot_budget_check(
        _clause(max_clause_slots=5, max_clause_slots_shadow=False),
        "short", _cfg())
    assert b["used"] == 4 and b["allow"] and calls == ["state_admission"]
    # Side without the key is still zero-read and inert.
    b = risk_gates.clause_slot_budget_check(
        _clause(max_clause_slots=5), "long", _cfg())
    assert b is None and len(calls) == 1


# ── runner gate call site (executor) ─────────────────────────────────────

def _run(monkeypatch, clause, count, **analysis_over):
    monkeypatch.setattr(ledger, "count_open_with_lane",
                        lambda tag: count)
    _stub_profile(monkeypatch, clause)
    _stub_tape(monkeypatch)
    a = _analysis(**analysis_over)
    return a, executor._runner_entry_block_reason(a, _gate())


def test_gate_live_budget_full_blocks_clause_admit(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="hermes_trader.agents.executor")
    clause = _clause(max_clause_slots=5, max_clause_slots_shadow=False)
    a, reason = _run(monkeypatch, clause, 5)
    assert reason == "runner_gate_blocked (clause slot budget 5/5)"
    assert "[gate][CLAUSE-SLOTS] SKIP TEST SHORT" in caplog.text
    assert not a.get("_lane_tag")  # nothing executes -> no ledger tag


def test_gate_live_budget_under_cap_admits_and_tags_ledger(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="hermes_trader.agents.executor")
    clause = _clause(max_clause_slots=5, max_clause_slots_shadow=False)
    a, reason = _run(monkeypatch, clause, 4)
    assert reason == ""
    assert "[gate][STATE-ADMIT] ADMIT" in caplog.text
    assert a.get("_lane_tag") == "state_admission"  # countable occupancy


def test_gate_budget_shadow_default_logs_would_skip_and_still_admits(
        monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger="hermes_trader.agents.executor")
    clause = _clause(max_clause_slots=5)  # shadow key absent = log-only
    a, reason = _run(monkeypatch, clause, 99)
    assert reason == ""
    assert "[gate][CLAUSE-SLOTS][SHADOW] WOULD SKIP TEST SHORT" in caplog.text
    assert a.get("_lane_tag") == "state_admission"


def test_gate_absent_budget_key_is_byte_identical_and_zero_reads(monkeypatch):
    monkeypatch.setattr(ledger, "count_open_with_lane", _boom_read)
    _stub_profile(monkeypatch, _clause())  # no budget keys
    _stub_tape(monkeypatch)
    a = _analysis()
    assert executor._runner_entry_block_reason(a, _gate()) == ""
    assert not a.get("_lane_tag")


def test_gate_configured_cap_below_floor_clamped_up(monkeypatch):
    # max_concurrent 8 -> contention floor 5 (round(8*0.6)); configured 3
    # must NOT bind at 4 in use (clamp UP), must bind at 5.
    clause = _clause(max_clause_slots=3, max_clause_slots_shadow=False)
    monkeypatch.setattr(ledger, "count_open_with_lane", lambda tag: 4)
    _stub_profile(monkeypatch, clause)
    _stub_tape(monkeypatch)
    assert executor._runner_entry_block_reason(_analysis(), _gate()) == ""
    monkeypatch.setattr(ledger, "count_open_with_lane", lambda tag: 5)
    assert executor._runner_entry_block_reason(
        _analysis(), _gate()) == "runner_gate_blocked (clause slot budget 5/5)"


def test_gate_never_clobbers_another_lanes_ledger_tag(monkeypatch):
    # Reviewer (2026-11-21): a clause admit inside an aligned-lane analysis
    # must NOT retag that row — the aligned lane's own max_lane_slots
    # occupancy is a LIVE gate and would under-count (that lane over-admits).
    clause = _clause(max_clause_slots=5, max_clause_slots_shadow=False)
    a, reason = _run(monkeypatch, clause, 0, _lane_tag="aligned_short_lane")
    assert reason == ""                              # clause still admits
    assert a["_lane_tag"] == "aligned_short_lane"    # tag NOT stolen


def test_gate_redecides_its_own_tag_on_every_attempt(monkeypatch):
    # Rotation retry / manual re-run reuses the analysis dict: a stale clause
    # tag from a previous attempt must not survive a decision that declines.
    clause = _clause(max_clause_slots=5, max_clause_slots_shadow=False)
    a, reason = _run(monkeypatch, clause, 0, btc_drift_24h=1.5,
                     _lane_tag="state_admission")   # tape out of band
    assert reason != ""
    assert "_lane_tag" not in a


# ── classic gate call site (risk_gates) ──────────────────────────────────

def _ctx(**over):
    kw = dict(confidence=0.47, current_positions=[], trade_notional_usd=100.0,
              daily_pnl=0.0, market_volume_24h_usd=50_000_000.0, coin="TEST",
              trade_side="short", has_binary_news_risk=False, equity=1000.0,
              total_open_notional=0.0, primary_model="test-model",
              daily_move_pct=0.0, daily_mover_fired=False)
    kw.update(over)
    return risk_gates.GateContext(**kw)


def _classic(monkeypatch, clause, count, caplog=None):
    if caplog is not None:
        caplog.set_level(logging.WARNING, logger="hermes_trader.agents.risk_gates")
    monkeypatch.setattr(ledger, "count_open_with_lane", lambda tag: count)
    _stub_rg_profile(monkeypatch, clause)
    _stub_tape(monkeypatch)
    ctx = _ctx()
    return risk_gates._state_conf_admission(
        ctx, _cfg(), risk_gates.confidence_gate(ctx, 0.65), 0.65)


def test_classic_live_budget_full_blocks(monkeypatch, caplog):
    res = _classic(monkeypatch,
                   _clause(max_clause_slots=5, max_clause_slots_shadow=False),
                   5, caplog)
    assert not res.get("pass")
    assert res["reason"] == "clause slot budget 5/5"
    assert "[gate][CLAUSE-SLOTS] SKIP (classic conf gate)" in caplog.text


def test_classic_budget_shadow_logs_and_admits(monkeypatch, caplog):
    res = _classic(monkeypatch, _clause(max_clause_slots=5), 99, caplog)
    assert res.get("pass") and res.get("via") == "state_admission"
    assert res.get("clause_lane_tag") == "state_admission"
    assert "[gate][CLAUSE-SLOTS][SHADOW] WOULD SKIP (classic conf gate)" in caplog.text


def test_classic_absent_budget_key_is_byte_identical_and_zero_reads(monkeypatch):
    monkeypatch.setattr(ledger, "count_open_with_lane", _boom_read)
    _stub_rg_profile(monkeypatch, _clause())
    _stub_tape(monkeypatch)
    ctx = _ctx()
    res = risk_gates._state_conf_admission(
        ctx, _cfg(), risk_gates.confidence_gate(ctx, 0.65), 0.65)
    assert res.get("pass") and res.get("via") == "state_admission"
    # Strictly no new key at all (the tag key only rides an ARMED budget, so
    # the serialized 'Trade result:' payload and execute response stay as-is).
    assert "clause_lane_tag" not in res
