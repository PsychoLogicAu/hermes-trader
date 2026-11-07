"""Every duelist CALL leaves a duel-log record — answered OR failed (t_ea2c5fea).

The card: "make every duelist call observable for later analysis" with the
required fields model / verdict / confidence / side / BRACKET / latency / scan
identifier / success-or-failure, on both the normal and the error path.

The answered path already carried all of them. What was missing was the ERROR
path: `call_duelist_with_fallback` swallows every fault and returns empty text,
and `_duelist_verdict` then wrote NOTHING. Measured 2026-11-07 on the live log
(/app/duel-logs/hermes-trader-duel.jsonl, 126,331 rows): the last row was
2026-11-06T19:04Z and the named seat had been ConnectError since — ~18 h of
scans, zero rows. In that corpus "the voice was asked and could not answer" is
literally indistinguishable from "no call happened", so the per-seat ceiling
fitted on the accrued rows has an invisible denominator.

These tests pin the two halves of the fix:
  * a failed call writes a compact row with the attribution + reason, and still
    returns None so NO decision surface ever sees it as an observation
    (duelist_at_entry, the session-log duel event, the B.10 close accrual,
    duel_store.resolve, the veto gate — all unchanged);
  * the report reads answered rows and failed rows as different populations.
No network: every LLM touch point is monkeypatched, as in test_duel_store.py.
"""

import json
import time

import pytest

from hermes_trader.agents import duel_store as ds
from hermes_trader.agents import research
from hermes_trader.models.types import Candle


# ── helpers (mirroring tests/test_duel_store.py's fixtures) ────────────────

def _candles(n: int = 40) -> list:
    base = 100.0
    return [
        Candle(t=1_700_000_000_000 + i * 3_600_000, o=base, h=base + 1,
               l=base - 1, c=base + (i % 5) * 0.2, v=1000)
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _isolated_duel_file(tmp_path, monkeypatch):
    p = tmp_path / "duel.jsonl"
    monkeypatch.setenv("HERMES_DUEL_FILE", str(p))
    yield p


@pytest.fixture
def _duelist_on(monkeypatch):
    """Named seat on a fake endpoint, no seat-backup configured."""
    monkeypatch.setenv("LLM_DUEL_MODEL", "duel-model")
    monkeypatch.setenv("LLM_DUEL_BASE_URL", "http://duel.test/v1")
    monkeypatch.setenv("LLM_DUEL_API_KEY", "dk")
    monkeypatch.delenv("LLM_DUEL_FALLBACK_MODEL", raising=False)


@pytest.fixture
def _session_log(tmp_path, monkeypatch):
    import hermes_trader.session_log as sl
    p = tmp_path / "session.jsonl"
    monkeypatch.setattr(sl, "SESSION_LOG_FILE", str(p))
    return p


def _read_events(p):
    if not p.exists():
        return []
    return [json.loads(ln) for ln in open(p).read().splitlines() if ln.strip()]


def _fail_meta(**over):
    """seat_meta as call_duelist_with_fallback returns it when NOTHING answered."""
    meta = {"seat": None, "answered_by": None, "primary_model": "duel-model",
            "fallback_model": None, "primary_fail": "exception:ConnectError",
            "fallback_fail": None, "breaker_open": False}
    meta.update(over)
    return meta


def _patch_primary(monkeypatch, primary_text: str = 'PASS\n{"verdict":"PASS","confidence":0.9}'):
    """Everything research() touches except the duelist leg."""
    monkeypatch.setattr(research, "fetch_hl_candles", lambda coin, tf, n: _candles())
    monkeypatch.setattr(research, "resolve_user_address", lambda: None)
    monkeypatch.setattr(research, "_fetch_funding_rate", lambda coin: "N/A")
    monkeypatch.setattr(research, "_fetch_news", lambda coin: "no news")
    monkeypatch.setattr(research, "_build_user_message", lambda *a, **k: "USER-PROMPT")
    monkeypatch.setattr(research, "build_system_prompt", lambda *a, **k: "SYS")
    monkeypatch.setattr(research, "_call_ai", lambda sp, um, **kw: (primary_text, None))


def _perception(pid: str = "pid-fail", coin: str = "SOL") -> dict:
    return {"id": pid, "coin": coin, "type": "perp", "mid": 200.0,
            "composite_score": 35.0, "triggers": []}


# ── the answered path: every required field on one row ─────────────────────

def test_answered_row_carries_every_required_field(
        monkeypatch, _duelist_on, _session_log, caplog):
    """model / verdict / confidence / side / bracket / latency / scan id / ok.

    The bracket is the one the L2 brief called the real gap: without the
    voice's own stop/tp the accrued corpus can only be priced on a fixed
    horizon, and the bracket first-touch ruler is what flipped the
    sole-voice verdict (t_4432a31a). Read snake_case — parse_verdict
    normalises the model's camelCase, and a runner that asked for `entryPx`
    silently lost the ruler once.
    """
    import logging
    caplog.set_level(logging.DEBUG)
    _patch_primary(monkeypatch)

    def answered(primary_cfg, fallback_cfg, sp, um, timeout_s=120.0):
        time.sleep(0.02)  # measurable wall time
        return ('LONG\n{"verdict":"LONG","confidence":0.72,"side":"long",'
                '"entryPx":201.5,"stopPx":195.0,"tpPx":212.0}', 40,
                {"seat": "primary", "answered_by": "duel-model",
                 "primary_model": "duel-model", "fallback_model": None,
                 "primary_fail": None, "fallback_fail": None,
                 "breaker_open": False})

    monkeypatch.setattr(research, "call_duelist_with_fallback", answered)
    analysis = research.research("SOL", _perception("pid-ok"))

    rows = ds.load_duels()
    assert len(rows) == 1
    r = rows[0]
    assert r["duelist_ok"] is True
    assert r["duelist_model"] == "duel-model"
    assert r["duelist_verdict"] == "LONG"
    assert r["duelist_confidence"] == 0.72
    assert r["duelist_side"] == "long"
    assert r["duelist_entry_px"] == 201.5
    assert r["duelist_stop_px"] == 195.0
    assert r["duelist_tp_px"] == 212.0
    assert r["duelist_ms"] >= 15 and r["duelist_server_ms"] == 40
    # the scan identifier — the documented join key to the ledger's rows
    assert r["perception_id"] == "pid-ok"
    assert r["coin"] == "SOL"
    assert analysis["perception_id"] == "pid-ok"
    # and the operator line shows the ruler without needing a join
    assert any("[duel] SOL:" in m and "br 201.5/195.0/212.0" in m
               for m in [rec.message for rec in caplog.records])


def test_no_bracket_omits_it_from_the_line(monkeypatch, _duelist_on, _session_log, caplog):
    """~44k spark rows carry no bracket: printing `br --/--/--` on all of them
    is noise. The fields are still present in the row, as None."""
    import logging
    caplog.set_level(logging.DEBUG)
    _patch_primary(monkeypatch)

    def abstain(primary_cfg, fallback_cfg, sp, um, timeout_s=120.0):
        return ('PASS\n{"verdict":"PASS","confidence":0.45}', None,
                {"seat": "primary", "answered_by": "duel-model",
                 "primary_model": "duel-model", "fallback_model": None,
                 "primary_fail": None, "fallback_fail": None,
                 "breaker_open": False})

    monkeypatch.setattr(research, "call_duelist_with_fallback", abstain)
    research.research("SOL", _perception("pid-nobrk"))

    rows = ds.load_duels()
    assert rows[0]["duelist_stop_px"] is None and rows[0]["duelist_tp_px"] is None
    lines = [m for m in (rec.message for rec in caplog.records) if "[duel] SOL:" in m]
    assert lines and all("br " not in m for m in lines)


# ── the error path: the row the corpus was missing ─────────────────────────

def test_failed_call_still_writes_a_row(monkeypatch, _duelist_on, _session_log, caplog):
    """Both seats down = a CALL happened and produced no observation.

    Before this, that scan left zero rows: the outage cohort and the
    never-called cohort were the same (invisible) set. Assert also the half
    that matters more than the row — nothing downstream may read it as an
    opinion.
    """
    import logging
    caplog.set_level(logging.DEBUG)
    _patch_primary(monkeypatch)

    def dead(primary_cfg, fallback_cfg, sp, um, timeout_s=120.0):
        time.sleep(0.02)  # a refused connection still has a wall time
        return "", None, _fail_meta(fallback_model="spark-model",
                                    fallback_fail="http-500")

    monkeypatch.setattr(research, "call_duelist_with_fallback", dead)
    analysis = research.research("SOL", _perception("pid-fail"))

    rows = ds.load_duels()
    assert len(rows) == 1
    r = rows[0]
    # the required fields, with no fabricated verdict
    assert r["duelist_ok"] is False
    assert r["perception_id"] == "pid-fail" and r["coin"] == "SOL"
    assert r["duelist_model"] is None          # nobody ANSWERED
    assert r["duelist_asked_model"] == "duel-model"
    assert r["duelist_asked_fallback_model"] == "spark-model"
    assert r["duelist_fail"] == "http-500"     # the single group-by field
    assert r["duelist_primary_fail"] == "exception:ConnectError"
    assert r["duelist_fallback_fail"] == "http-500"
    assert r["duelist_verdict"] is None
    assert r["duelist_confidence"] is None
    assert r["duelist_side"] is None
    assert r["duelist_entry_px"] is None and r["duelist_stop_px"] is None
    assert r["duelist_tp_px"] is None
    # latency of the FAILED attempt is still measured (timeout vs refused is
    # the difference between a runaway and a dead box)
    assert r["duelist_ms"] >= 15
    assert r["duelist_server_ms"] is None
    # the primary's side of the scan is intact, so the row joins to the scan
    assert r["primary_verdict"] == "PASS" and r["primary_confidence"] == 0.9

    # ── decision surfaces see NOTHING (log-only, behaviour unchanged) ──
    assert analysis["verdict"] == "PASS"
    assert analysis["duelist_at_entry"] is None
    assert not [e for e in _read_events(_session_log) if e.get("event") == "duel"]
    assert ds.resolve("SOL", "pid-fail") is None
    assert any("returned no text" in rec.message and "recorded as a failed call" in rec.message
               for rec in caplog.records)


def test_breaker_skip_is_attributable(monkeypatch, _duelist_on, _session_log):
    """A seat skipped by the failure breaker was not tried — a different
    operational story than an endpoint that was, and the row says which."""
    _patch_primary(monkeypatch)

    def cooling(primary_cfg, fallback_cfg, sp, um, timeout_s=120.0):
        return "", None, _fail_meta(primary_fail="breaker-open", breaker_open=True)

    monkeypatch.setattr(research, "call_duelist_with_fallback", cooling)
    research.research("SOL", _perception("pid-cool"))

    r = ds.load_duels()[0]
    assert r["duelist_ok"] is False
    assert r["duelist_fail"] == "breaker-open"
    assert r["duelist_breaker_open"] is True


def test_fault_inside_the_hook_after_the_call_is_recorded(monkeypatch, _duelist_on,
                                                          _session_log):
    """A crash between "the seat answered" and "the row was written" is still a
    call. Recorded with a `hook-exception:` reason, never as a verdict."""
    _patch_primary(monkeypatch)

    def answered(primary_cfg, fallback_cfg, sp, um, timeout_s=120.0):
        return 'LONG\n{"verdict":"LONG","confidence":0.7,"side":"long"}', 40, {
            "seat": "primary", "answered_by": "duel-model",
            "primary_model": "duel-model", "fallback_model": None,
            "primary_fail": None, "fallback_fail": None, "breaker_open": False}

    monkeypatch.setattr(research, "call_duelist_with_fallback", answered)
    real_parse = research.parse_verdict
    calls = {"n": 0}

    def boom_parse(*a, **k):
        calls["n"] += 1
        if calls["n"] > 1:            # primary parses fine; the duelist leg faults
            raise RuntimeError("parse blew up")
        return real_parse(*a, **k)

    monkeypatch.setattr(research, "parse_verdict", boom_parse)
    analysis = research.research("SOL", _perception("pid-hook"))

    assert analysis["verdict"] == "PASS"          # primary unaffected
    assert analysis["duelist_at_entry"] is None   # and no observation reached it
    rows = ds.load_duels()
    assert len(rows) == 1
    assert rows[0]["duelist_ok"] is False
    assert rows[0]["duelist_fail"] == "hook-exception:RuntimeError"
    assert rows[0]["duelist_verdict"] is None


def test_fault_before_the_seat_is_resolved_writes_no_row(monkeypatch, _duelist_on,
                                                         _session_log):
    """The boundary of "every CALL is recorded": a fault while resolving the
    seat means no seat was ever asked, and a fabricated call row would name a
    model that was never contacted. Silent is correct here."""
    _patch_primary(monkeypatch)
    monkeypatch.setattr(research, "duelist_config",
                        lambda: (_ for _ in ()).throw(RuntimeError("config read failed")))
    calls = []
    monkeypatch.setattr(research, "call_duelist_with_fallback",
                        lambda *a, **k: calls.append(a))

    analysis = research.research("SOL", _perception("pid-pre"))

    assert analysis["verdict"] == "PASS"
    assert not calls
    assert ds.load_duels() == []


def test_disabled_duelist_writes_no_row(monkeypatch, _session_log):
    """No seat configured = no call = nothing to log (the corpus must not fill
    up with records of a voice that was never deployed)."""
    monkeypatch.delenv("LLM_DUEL_MODEL", raising=False)
    monkeypatch.delenv("LLM_DUEL_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_DUEL_API_KEY", raising=False)
    _patch_primary(monkeypatch)
    calls = []
    monkeypatch.setattr(research, "call_duelist_with_fallback",
                        lambda *a, **k: calls.append(a))

    analysis = research.research("SOL", _perception("pid-off"))

    assert analysis["duelist_at_entry"] is None
    assert not calls
    assert ds.load_duels() == []


# ── the corpus readers must not confuse the two populations ────────────────

def test_resolve_ignores_failed_rows_but_not_legacy_ones(_isolated_duel_file):
    """`resolve` promises "the duelist row for a perception", i.e. its verdict.
    A failed-call row is not one, so entry-context/backfill callers keep the
    exact semantics they had before failure rows existed. Rows written before
    this change carry no `duelist_ok` key and still resolve."""
    ds.record_duel({"coin": "BTC", "perception_id": "pid-a", "duelist_model": "m",
                    "duelist_verdict": "SHORT", "duelist_confidence": 0.6})
    ds.record_duel({"coin": "BTC", "perception_id": "pid-a", "duelist_ok": False,
                    "duelist_fail": "timeout", "duelist_verdict": None})
    found = ds.resolve("BTC", "pid-a")
    assert found is not None and found["duelist_verdict"] == "SHORT"

    ds.record_duel({"coin": "ETH", "perception_id": "pid-b", "duelist_ok": False,
                    "duelist_fail": "timeout", "duelist_verdict": None})
    assert ds.resolve("ETH", "pid-b") is None
    assert ds.answered({"coin": "BTC"}) is True   # legacy default = answered
    assert ds.answered({"duelist_ok": False}) is False


def test_aggregate_scores_answers_and_counts_failures_separately(monkeypatch,
                                                                 _isolated_duel_file):
    """The A/B report's agreement rate, P&L join and latency stats are about
    ANSWERED calls. A failed row must not: (a) backfill a close's verdict,
    (b) count as a split, or (c) drag the latency table with a 120 s timeout.
    It does show up as a counted failure with its reason."""
    monkeypatch.setenv("LLM_DUEL_MODEL", "duel-model")
    ds.record_duel({"coin": "BTC", "perception_id": "pid-1", "mode": "LIVE",
                    "primary_model": "A", "duelist_model": "B",
                    "primary_verdict": "LONG", "primary_confidence": 0.8,
                    "duelist_verdict": "LONG", "duelist_confidence": 0.6,
                    "duelist_side": "long", "primary_ms": 100, "duelist_ms": 200})
    ds.record_duel({"coin": "ETH", "perception_id": "pid-2", "mode": "LIVE",
                    "primary_model": "A", "duelist_model": "B",
                    "primary_verdict": "LONG", "primary_confidence": 0.8,
                    "duelist_verdict": "SHORT", "duelist_confidence": 0.6,
                    "duelist_side": "short", "primary_ms": 200, "duelist_ms": 300})
    ds.record_duel({"coin": "SOL", "perception_id": "pid-3", "mode": "LIVE",
                    "primary_model": "A", "duelist_model": None,
                    "duelist_ok": False, "duelist_fail": "timeout",
                    "duelist_verdict": None, "duelist_confidence": None,
                    "duelist_side": None, "primary_ms": 150, "duelist_ms": 120_000})

    # a close whose perception matches ONLY the failed row: the backfill must
    # find nothing, so the close stays "no duelist verdict" instead of being
    # scored on a call that never answered. The outcome store is STUBBED rather
    # than written to — the conftest state dir is process-wide, so a
    # record_close() here would leak into the next module's aggregate test.
    from hermes_trader.agents.memory import memory
    monkeypatch.setattr(memory, "load", lambda *a, **k: None)
    monkeypatch.setattr(memory, "get_closes", lambda *a, **k: [{
        "coin": "SOL", "side": "long", "notional_usd": 100.0,
        "realized_pnl_pct": -3.0, "realized_pnl_usd": -3.0,
        "duelist_at_entry": None, "perception_id": "pid-3"}])

    r = ds.aggregate()
    assert r["duel_calls"] == 2 and r["answered_calls"] == 2
    assert r["failed_calls"] == 1
    assert r["call_success_rate"] == round(2 / 3, 3)
    assert r["duel_failure_mix"] == {"timeout": 1}
    assert r["agreement"] == {"agree": 1, "split": 1, "rate": 0.5}
    assert r["duelist_verdicts"] == {"LONG": 1, "SHORT": 1}
    # the 120 s timeout is NOT in the latency stats (n=2, max 300)
    assert r["latency"]["duelist"]["n"] == 2
    assert r["latency"]["duelist"]["max_ms"] == 300.0
    assert r["closes_with_duelist"] == 0
