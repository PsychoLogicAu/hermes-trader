"""Tests for the clef decision-voice GATES (clef-gate scope Changes 1-3).

Two enforcement points, both model-scoped in the OBSERVER's model profile
(config-models/<profile>.json -> trap_veto/close_veto blocks {mode,
threshold, ...}; owner call 2026-10-18 — the bars travel with the model
that produces the noul), shadow-first per the house pattern
(.hermes/plans/2026-10-07-clef-gate-close-voice-scope.md):

  1. entry trap-veto (executor._dv_trap_veto_reason) — mode shadow logs
     WOULD-BE-BLOCKED and allows; live hard-blocks with the exact reason
     string through the runner_gate forensics path; off / absent block =
     inert; NO dv row is fail-open (a failed observer is 'no observation',
     never a block).
  2. close_veto accrual (research._dv_close_exit_accrual) — held +
     close_now >= threshold + primary != CLOSE -> WOULD-HAVE-CLOSED line
     (shadow) or fire-spec (live); grace_minutes suppresses young
     positions.

decision_voice_gate.enabled remains the master kill above both.

Plus the whitelist regression (the 2026-08 pitfall): trap/close_now survive
analysis -> entry-context -> close-row join.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import executor, research  # noqa: E402


# ── helpers ────────────────────────────────────────────────────────────────

def _profile_file(trap_mode="shadow", trap_thr=0.45, close_mode="shadow",
                  close_thr=0.50, grace=90, blocks=True):
    """Write config-models/dv-test.json — the gate blocks are per-model
    (hot-read by dv_gate_block through MODEL_CONFIG_DIR)."""
    import hermes_trader.agents.config_store as cs
    prof = {"model": "vllm-test"}
    if blocks:
        prof["trap_veto"] = {"mode": trap_mode, "threshold": trap_thr}
        prof["close_veto"] = {"mode": close_mode, "threshold": close_thr,
                              "grace_minutes": grace}
    pf = Path(cs.MODEL_CONFIG_DIR) / "dv-test.json"
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(json.dumps(prof))


def _cfg(trap_mode="shadow", trap_thr=0.45, close_mode="shadow",
         close_thr=0.50, grace=90, enabled=True, blocks=True):
    """Agent-config dict for the executor gate fns (master kill + dv slot
    pointing at the dv-test profile); also writes the profile file."""
    _profile_file(trap_mode=trap_mode, trap_thr=trap_thr,
                  close_mode=close_mode, close_thr=close_thr, grace=grace,
                  blocks=blocks)
    return {
        "decision_voice_gate": {"enabled": enabled},
        "llm": {"decision_voice": {"enabled": True,
                                   "model_profile": "dv-test"}},
    }


def _analysis(trap=0.52, close_now=None, **over):
    a = {
        "id": "a1", "coin": "TEST", "verdict": "LONG", "side": "long",
        "confidence": 0.80, "composite_score": 55, "entry_px": 100.0,
        "perception_id": "pid-dv-1",
        "decision_voice_at_entry": {
            "model": "clef-flash", "verdict": "LONG", "confidence": 0.7,
            "side": "long", "trap": trap, "close_now": close_now,
        },
    }
    a.update(over)
    return a


# ── 1. trap-veto: hard block, exact reason string ─────────────────────────

def test_trap_veto_hard_block_reason_exact():
    r = executor._dv_trap_veto_reason(_analysis(trap=0.52),
                                      _cfg(trap_mode="live"))
    assert r == "dv_trap_veto (trap 0.52 >= 0.45)"


def test_trap_veto_below_threshold_passes():
    assert executor._dv_trap_veto_reason(_analysis(trap=0.40), _cfg()) == ""


# ── 2. trap-veto: shadow -> allowed + accrual line ────────────────────────

def test_trap_veto_shadow_logs_and_allows(caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.executor"):
        r = executor._dv_trap_veto_reason(_analysis(trap=0.52), _cfg())
    assert r == ""  # trade allowed
    lines = [rec.getMessage() for rec in caplog.records
             if "dv_trap_veto WOULD BE BLOCKED" in rec.getMessage()]
    assert len(lines) == 1
    msg = lines[0]
    assert "TEST LONG" in msg
    assert "trap 0.52 >= 0.45" in msg
    assert "conf 0.80" in msg
    assert "composite 55" in msg
    assert "pid-dv-1" in msg


# ── 3. fail-open: no dv row / missing scalar / gate off ───────────────────

def test_trap_veto_fail_open_no_dv_row():
    a = _analysis()
    a["decision_voice_at_entry"] = None
    assert executor._dv_trap_veto_reason(
        a, _cfg(trap_mode="live")) == ""


def test_trap_veto_fail_open_missing_trap_scalar():
    a = _analysis()
    a["decision_voice_at_entry"] = {"model": "clef-flash", "verdict": "LONG"}
    assert executor._dv_trap_veto_reason(
        a, _cfg(trap_mode="live")) == ""


def test_trap_veto_gate_disabled_no_opinion():
    assert executor._dv_trap_veto_reason(
        _analysis(trap=0.99), _cfg(enabled=False, trap_mode="live")) == ""


def test_trap_veto_profile_block_absent_gate_inert(caplog):
    """No gate blocks in the dv profile = no opinion at all (clean-break
    shape: the agent-config gate block no longer carries bars)."""
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.executor"):
        r = executor._dv_trap_veto_reason(_analysis(trap=0.99),
                                          _cfg(blocks=False))
    assert r == ""
    assert not [x for x in caplog.records
                if "dv_trap_veto" in x.getMessage()]


def test_trap_veto_mode_off_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.executor"):
        r = executor._dv_trap_veto_reason(_analysis(trap=0.99),
                                          _cfg(trap_mode="off"))
    assert r == ""
    assert not [x for x in caplog.records
                if "dv_trap_veto" in x.getMessage()]


def test_trap_veto_malformed_mode_fails_safe_to_shadow(caplog):
    """Garbage mode never hard-blocks: loud line + shadow behaviour."""
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.executor"):
        r = executor._dv_trap_veto_reason(_analysis(trap=0.52),
                                          _cfg(trap_mode="enable"))
    assert r == ""
    assert [x for x in caplog.records
            if "trap_veto mode 'enable'" in x.getMessage()
            and "fail-safe to shadow" in x.getMessage()]
    assert "WOULD BE BLOCKED" in caplog.text  # shadow accrual stands


# ── 4. close_veto accrual: held + >= thr + primary != CLOSE ───────────────

class _FakeTracker:
    def __init__(self, entry_px, entry_time, peak_px, mark_px):
        self.entry_px = entry_px
        self.entry_time = entry_time
        self.peak_px = peak_px
        self.last_mark_px = mark_px


def _dv_row(close_now=0.72, pid="pid-dv-9"):
    return {"coin": "TEST", "perception_id": pid, "dv_close_now": close_now}


def _held(coin="TEST", side="long"):
    return [{"coin": coin, "side": side, "size_usd": 33.0}]


@pytest.fixture
def _gate_cfg(monkeypatch):
    """Write the agent config (master kill + dv slot) AND the
    config-models/dv-test.json profile carrying the nested
    trap_veto/close_veto blocks (B.50+ model-scoped shape, hot read).
    profile=False removes the profile file entirely (block-absent
    defaults path)."""
    import hermes_trader.agents.config_store as cs
    from pathlib import Path

    def _write(close_mode="shadow", close_thr=0.50, grace=90, enabled=True,
               trap_mode="shadow", trap_thr=0.45, profile=True):
        cfg_path = Path(cs.CONFIG_PATH)
        cfg_path.unlink(missing_ok=True)
        cfg = {"mode": "LIVE",
               "decision_voice_gate": {"enabled": enabled},
               "llm": {"decision_voice": {"enabled": True,
                                          "model_profile": "dv-test"}}}
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        pf = Path(cs.MODEL_CONFIG_DIR) / "dv-test.json"
        pf.unlink(missing_ok=True)
        if profile:
            pf.parent.mkdir(parents=True, exist_ok=True)
            pf.write_text(json.dumps({
                "model": "vllm-test",
                "trap_veto": {"mode": trap_mode, "threshold": trap_thr},
                "close_veto": {"mode": close_mode, "threshold": close_thr,
                               "grace_minutes": grace}}))
    yield _write
    Path(cs.CONFIG_PATH).unlink(missing_ok=True)
    (Path(cs.MODEL_CONFIG_DIR) / "dv-test.json").unlink(missing_ok=True)


def test_close_veto_fires_held_split(caplog, _gate_cfg):
    _gate_cfg()
    trk = _FakeTracker(100.0, time.time() - 120 * 60, 101.0, 97.0)
    from hermes_trader.agents import dsl_exit as _dslx
    _dslx._active_positions["TEST_long"] = trk
    try:
        with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
            research._dv_close_exit_accrual("TEST", _dv_row(0.72), _held())
        lines = [r.getMessage() for r in caplog.records
                 if "dv_close_veto WOULD HAVE CLOSED" in r.getMessage()]
        assert len(lines) == 1
        msg = lines[0]
        assert "TEST long" in msg
        assert "close_now 0.72 >= 0.50" in msg
        assert "120min" in msg
        assert "-3.00%" in msg      # spot vs entry (mark 97 vs entry 100)
        assert "pid-dv-9" in msg
    finally:
        _dslx._active_positions.pop("TEST_long", None)


def test_close_veto_silent_below_threshold(caplog, _gate_cfg):
    _gate_cfg()
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
        research._dv_close_exit_accrual("TEST", _dv_row(0.40), _held())
    assert not [r for r in caplog.records
                if "dv_close_veto" in r.getMessage()]


def test_close_veto_silent_when_unheld(caplog, _gate_cfg):
    _gate_cfg()
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
        research._dv_close_exit_accrual("TEST", _dv_row(0.90), [])
    assert not [r for r in caplog.records
                if "dv_close_veto" in r.getMessage()]


def test_close_veto_silent_when_gate_disabled(caplog, _gate_cfg):
    _gate_cfg(enabled=False)
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
        research._dv_close_exit_accrual("TEST", _dv_row(0.90), _held())
    assert not [r for r in caplog.records
                if "dv_close_veto" in r.getMessage()]


# ── 5. grace_minutes suppresses young positions ───────────────────────────

def test_close_veto_grace_suppresses_young_position(caplog, _gate_cfg):
    _gate_cfg(grace=90)
    trk = _FakeTracker(100.0, time.time() - 30 * 60, 101.0, 97.0)  # 30min old
    from hermes_trader.agents import dsl_exit as _dslx
    _dslx._active_positions["TEST_long"] = trk
    try:
        with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
            research._dv_close_exit_accrual("TEST", _dv_row(0.90), _held())
        assert not [r for r in caplog.records
                    if "dv_close_veto" in r.getMessage()]
    finally:
        _dslx._active_positions.pop("TEST_long", None)


def test_close_veto_live_fires_spec(caplog, _gate_cfg):
    """B.50 wiring: LIVE mode returns the fire-spec (route_verdict acts on
    it) and logs the FIRED line. Tracker present so grace is verified."""
    _gate_cfg(close_mode="live")
    trk = _FakeTracker(100.0, time.time() - 120 * 60, 101.0, 97.0)
    from hermes_trader.agents import dsl_exit as _dslx
    _dslx._active_positions["TEST_long"] = trk
    try:
        with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
            spec = research._dv_close_exit_accrual("TEST", _dv_row(0.72), _held())
        assert spec is not None
        assert spec["coin"] == "TEST" and spec["side"] == "long"
        assert spec["close_now"] == 0.72 and spec["threshold"] == 0.50
        assert spec["spot"] == "-3.00%"
        lines = [r.getMessage() for r in caplog.records
                 if "dv_close_veto FIRED -> dv_close queued" in r.getMessage()]
        assert len(lines) == 1
        assert not [r for r in caplog.records
                    if "WOULD HAVE CLOSED" in r.getMessage()]
    finally:
        _dslx._active_positions.pop("TEST_long", None)


# ── 6. whitelist regression: analysis -> entry-context -> close row ───────

def test_trap_close_scalars_survive_entry_context_join(monkeypatch):
    """The 2026-08 pitfall test: a field not in the analysis whitelist
    silently never reaches the executor. Pin the full ride:
    analysis.decision_voice_at_entry -> memory entry-context 'decision_voice'
    -> close-row 'decision_voice_at_entry'."""
    from hermes_trader.agents.memory import AgentMemory

    a = _analysis(trap=0.61, close_now=0.55)

    # executor snapshots analysis -> entry context (the dict literal at the
    # record_entry_context call site): pin the mapping directly.
    m = AgentMemory()
    m._initialized = True
    ctx = {
        "decision_voice": a.get("decision_voice_at_entry"),
        "perception_id": a.get("perception_id"),
    }
    m.record_entry_context("TEST", "long", ctx)
    ec = m.pop_entry_context("TEST", "long")
    assert ec["decision_voice"]["trap"] == 0.61
    assert ec["decision_voice"]["close_now"] == 0.55

    # And the research() whitelist itself carries the scalars (source-level
    # pin — the dict literal must name both keys).
    src = Path(research.__file__).read_text()
    wl = src.split('"decision_voice_at_entry"', 1)[1].split("if dv_row", 1)[0]
    assert '"trap": dv_row["dv_trap"]' in wl
    assert '"close_now": dv_row["dv_close_now"]' in wl


# ── 7. B.50+ model-profile close_veto block (tri-state mode, welded) ──────

def _live_tracker(mins=120):
    return _FakeTracker(100.0, time.time() - mins * 60, 101.0, 97.0)


def _with_tracker(fn):
    from hermes_trader.agents import dsl_exit as _dslx
    _dslx._active_positions["TEST_long"] = _live_tracker()
    try:
        return fn()
    finally:
        _dslx._active_positions.pop("TEST_long", None)


def test_profile_absent_defaults_to_shadow(caplog, _gate_cfg):
    """No profile file at all -> fail-safe shadow at the 0.50 code
    default: accrues, never fires."""
    _gate_cfg(profile=False)
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
        spec = _with_tracker(lambda: research._dv_close_exit_accrual(
            "TEST", _dv_row(0.72), _held()))
    assert spec is None
    assert [r for r in caplog.records
            if "WOULD HAVE CLOSED" in r.getMessage()
            and "close_now 0.72 >= 0.50" in r.getMessage()]


def test_profile_off_silent(caplog, _gate_cfg):
    """mode 'off' silences the voice entirely (live bar irrelevant)."""
    _gate_cfg(close_mode="off", close_thr=0.20)
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
        spec = _with_tracker(lambda: research._dv_close_exit_accrual(
            "TEST", _dv_row(0.90), _held()))
    assert spec is None
    assert not [r for r in caplog.records if "dv_close_veto" in r.getMessage()]


def test_threshold_from_profile_block(caplog, _gate_cfg):
    """The bar travels with the model: block threshold 0.80 > close_now
    0.72 -> silent in live mode (no fire, no line)."""
    _gate_cfg(close_mode="live", close_thr=0.80)
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.research"):
        spec = _with_tracker(lambda: research._dv_close_exit_accrual(
            "TEST", _dv_row(0.72), _held()))
    assert spec is None
    assert not [r for r in caplog.records if "dv_close_veto" in r.getMessage()]


def test_profile_invalid_value_fails_safe_to_shadow(caplog, _gate_cfg):
    """A malformed close_veto mode never fires by accident: loud log,
    shadow behaviour."""
    _gate_cfg(close_mode="enable")  # garbage mode at a live-priced bar
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
        spec = _with_tracker(lambda: research._dv_close_exit_accrual(
            "TEST", _dv_row(0.72), _held()))
    assert spec is None
    assert [r for r in caplog.records
            if "close_veto mode 'enable'" in r.getMessage()
            and "fail-safe to shadow" in r.getMessage()]
    assert [r for r in caplog.records
            if "WOULD HAVE CLOSED" in r.getMessage()]


def test_profile_live_fail_closed_without_tracker(caplog, _gate_cfg):
    """Live mode never fires without dsl tracker data: age/grace would be
    unverifiable. Loud WOULD-NOT-FIRE line instead."""
    _gate_cfg(close_mode="live")
    from hermes_trader.agents import dsl_exit as _dslx
    _dslx._active_positions.pop("TEST_long", None)
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.research"):
        spec = research._dv_close_exit_accrual("TEST", _dv_row(0.90), _held())
    assert spec is None
    assert [r for r in caplog.records
            if "dv_close_veto WOULD NOT FIRE" in r.getMessage()]


def test_profile_live_grace_still_suppresses(caplog, _gate_cfg):
    """grace_minutes applies in live mode too (young position = no fire)."""
    _gate_cfg(close_mode="live", grace=90)
    from hermes_trader.agents import dsl_exit as _dslx
    _dslx._active_positions["TEST_long"] = _live_tracker(mins=30)
    try:
        with caplog.at_level(logging.INFO, logger="hermes_trader.agents.research"):
            spec = research._dv_close_exit_accrual("TEST", _dv_row(0.90), _held())
        assert spec is None
        assert not [r for r in caplog.records
                    if "dv_close_veto" in r.getMessage()]
    finally:
        _dslx._active_positions.pop("TEST_long", None)


def test_gate_enabled_false_kills_live_voice(_gate_cfg):
    """decision_voice_gate.enabled remains the master kill above the
    profile mode."""
    _gate_cfg(close_mode="live", enabled=False)
    assert _with_tracker(lambda: research._dv_close_exit_accrual(
        "TEST", _dv_row(0.99), _held())) is None


# ── 8. route_verdict injection: synthetic dv_close ────────────────────────

def _spec(**over):
    s = {"coin": "TEST", "side": "long", "close_now": 0.72, "threshold": 0.50,
         "age": "120min", "spot": "-3.00%", "peak": "+1.00%",
         "entry": "100", "mark": "97", "model": "vllm-test",
         "perception_id": "pid-x"}
    s.update(over)
    return s


def _rec_close(coin, reason):
    return {"coin": coin, "reason": reason}


def test_route_verdict_dv_close_fires_tagged():
    a = _analysis(verdict="PASS", dv_close_fire=_spec())
    r = executor.route_verdict(a, execute_fn=lambda x: None, close_fn=_rec_close)
    assert r["action"] == "close" and r["verdict"] == "CLOSE"
    assert r["close_source"] == "dv"
    assert r["result"]["coin"] == "TEST"
    assert r["result"]["reason"].startswith("dv_close: close_now 0.72 >= 0.50")
    # _exit_type maps the ledger class for the reason string:
    assert executor._exit_type(r["result"]["reason"]) == "dv_close"


def test_route_verdict_dv_close_overrides_long_on_held():
    a = _analysis(verdict="LONG", dv_close_fire=_spec())
    called = []
    r = executor.route_verdict(
        a, execute_fn=lambda x: called.append(x), close_fn=_rec_close)
    assert r["action"] == "close" and not called  # entry attempt skipped


def test_route_verdict_dv_close_respects_slot_defer_enforced(monkeypatch):
    """Owner call 2026-10-18: the slot-pressure guard applies to dv closes
    too — an enforced defer returns action none, deterministic exits own it."""
    defer = {"shadow": False, "used": 1, "cap": 8, "floor": 5,
             "peak_pct": 0.5, "protect_pct": 1.5,
             "reason": "ai_close_slot_defer: slots 1/8 < floor 5, never armed"}
    monkeypatch.setattr(executor, "_ai_close_slot_defer_check", lambda c: defer)
    a = _analysis(verdict="PASS", dv_close_fire=_spec())
    closed = []
    r = executor.route_verdict(a, execute_fn=lambda x: None,
                               close_fn=lambda c, reason: closed.append(c))
    assert r["action"] == "none" and r["close_source"] == "dv"
    assert "deferred_reason" in r and not closed
    # re-check happens every scan: the voice fires at its first PERMITTED veto


def test_route_verdict_dv_close_shadow_defer_closes(caplog, monkeypatch):
    defer = {"shadow": True, "used": 1, "cap": 8, "floor": 5,
             "peak_pct": 0.5, "protect_pct": 1.5, "reason": "would defer"}
    monkeypatch.setattr(executor, "_ai_close_slot_defer_check", lambda c: defer)
    a = _analysis(verdict="VETO", dv_close_fire=_spec())
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.executor"):
        r = executor.route_verdict(a, execute_fn=lambda x: None,
                                   close_fn=_rec_close)
    assert r["action"] == "close" and r["close_source"] == "dv"
    assert [x for x in caplog.records if "dv_close WOULD-HAVE-DEFERRED" in x.getMessage()]


def test_route_verdict_dv_close_contended_book_closes(monkeypatch):
    """Guard returns None (contended / armed / trend-released) -> fire."""
    monkeypatch.setattr(executor, "_ai_close_slot_defer_check", lambda c: None)
    a = _analysis(verdict="VETO", dv_close_fire=_spec())
    r = executor.route_verdict(a, execute_fn=lambda x: None, close_fn=_rec_close)
    assert r["action"] == "close" and r["close_source"] == "dv"


def test_route_verdict_primary_close_not_re_tagged(monkeypatch):
    """A primary CLOSE keeps its ai_close identity — the dv injection never
    double-routes (in the live flow a primary CLOSE skips the accrual, so a
    stale spec can only arrive via a replay)."""
    monkeypatch.setattr(executor, "_ai_close_slot_defer_check", lambda c: None)
    a = _analysis(verdict="CLOSE", dv_close_fire=_spec())
    r = executor.route_verdict(a, execute_fn=lambda x: None, close_fn=_rec_close)
    assert r["action"] == "close"
    assert "close_source" not in r
    assert r["result"]["reason"].startswith("ai_close")


def test_route_verdict_no_spec_routes_normally():
    a = _analysis()  # verdict LONG, no dv_close_fire key at all
    executed = []
    r = executor.route_verdict(
        a, execute_fn=lambda x: executed.append(x), close_fn=_rec_close)
    assert r["action"] == "execute" and executed


def test_analysis_whitelist_carries_dv_close_fire():
    """WHITELIST REGRESSION POINT pin: research's analysis dict must name
    the key or the injection can never route."""
    src = Path(research.__file__).read_text()
    assert '"dv_close_fire": _dv_close_fire,' in src
