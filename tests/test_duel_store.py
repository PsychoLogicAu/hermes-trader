"""Tests for the model-duel store (A/B LLM evaluation).

Covers: env gating (dormant unless a duelist model is named), the JSONL
store, the verdict-attribution P&L math, the aggregate report join, and the
research() integration (paired call, duel row, analysis field, session event).
No network: the LLM endpoints are monkeypatched like the shadow-signal tests.
"""

import json
import time

import pytest

from hermes_trader.agents import duel_store as ds
from hermes_trader.agents import research
from hermes_trader.agents.memory import memory
from hermes_trader.models.types import Candle
from hermes_trader.session_log import append as log_event


# ── helpers ────────────────────────────────────────────────────────────────

def _candles(n: int = 40) -> list:
    base = 100.0
    return [
        Candle(t=1_700_000_000_000 + i * 3_600_000, o=base, h=base + 1,
               l=base - 1, c=base + (i % 5) * 0.2, v=1000)
        for i in range(n)
    ]


def _duel_row(**over):
    row = {
        "ts": 0, "coin": "BTC", "perception_id": "pid-1", "mode": "SHADOW",
        "primary_model": "modelA", "duelist_model": "modelB",
        "primary_verdict": "LONG", "primary_confidence": 0.8,
        "duelist_verdict": "SHORT", "duelist_confidence": 0.6,
        "duelist_side": "short", "duelist_reasoning": "r",
        "primary_ms": 100, "duelist_ms": 200,
    }
    row.update(over)
    return row


@pytest.fixture(autouse=True)
def _isolated_duel_file(tmp_path, monkeypatch):
    """Every test appends to its own duel file (conftest already isolates the
    path process-wide, but this keeps assertions hermetic per test)."""
    p = tmp_path / "duel.jsonl"
    monkeypatch.setenv("HERMES_DUEL_FILE", str(p))
    yield p


@pytest.fixture
def _duelist_on(monkeypatch):
    """Enable the duelist against a fake endpoint."""
    monkeypatch.setenv("LLM_DUEL_MODEL", "duel-model")
    monkeypatch.setenv("LLM_DUEL_BASE_URL", "http://duel.test/v1")
    monkeypatch.setenv("LLM_DUEL_API_KEY", "dk")


@pytest.fixture
def _session_log(tmp_path, monkeypatch):
    """Redirect session_log.SESSION_LOG_FILE to a per-test file.

    session_log freezes the path at import (SESSION_LOG_PATH env), so the
    module attribute is monkeypatched — append() reads the global at call
    time, so this intercepts every write. Returns the Path; the test reads
    events from it. Without this, tests would append to the LIVE
    ~/.hermes-trader-session-log.jsonl.
    """
    import hermes_trader.session_log as sl
    p = tmp_path / "session.jsonl"
    monkeypatch.setattr(sl, "SESSION_LOG_FILE", str(p))
    return p


def _read_events(p):
    if not p.exists():
        return []
    return [json.loads(ln) for ln in open(p).read().splitlines() if ln.strip()]


# ── env gating ─────────────────────────────────────────────────────────────

def test_dormant_when_no_duelist_model(monkeypatch):
    monkeypatch.delenv("LLM_DUEL_MODEL", raising=False)
    monkeypatch.delenv("LLM_DUKE_MODEL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "primary-model")
    assert not ds.duelist_enabled()
    cfg = ds.duelist_config()
    # model does NOT fall back to LLM_MODEL (no silent self-duel)
    assert cfg["model"] == ""
    # base_url/api_key DO fall back to the primary endpoint (whatever that is —
    # a dev shell may export LLM_BASE_URL, so pin it here)
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    assert ds.duelist_config()["base_url"] == "https://openrouter.ai/api/v1"


def test_duelist_model_only_inherits_primary_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_DUEL_MODEL", "duel-model")
    monkeypatch.delenv("LLM_DUEL_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_DUEL_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://primary.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "pk")
    assert ds.duelist_enabled()
    cfg = ds.duelist_config()
    assert cfg["model"] == "duel-model"
    assert cfg["base_url"] == "http://primary.test/v1"
    assert cfg["api_key"] == "pk"


def test_duke_alias(monkeypatch):
    monkeypatch.delenv("LLM_DUEL_MODEL", raising=False)
    monkeypatch.setenv("LLM_DUKE_MODEL", "duke-model")
    assert ds.duelist_enabled()
    assert ds.duelist_config()["model"] == "duke-model"


# ── store ──────────────────────────────────────────────────────────────────

def test_record_and_load_roundtrip(_isolated_duel_file):
    ds.record_duel(_duel_row())
    ds.record_duel(_duel_row(coin="ETH", perception_id="pid-2"))
    rows = ds.load_duels()
    assert len(rows) == 2
    assert rows[0]["perception_id"] == "pid-1"
    assert rows[0]["ts"] >= 0  # auto-stamped


def test_load_skips_malformed_lines(_isolated_duel_file):
    with open(_isolated_duel_file, "w") as f:
        f.write("not json\n")
        f.write(json.dumps(_duel_row()) + "\n")
    assert len(ds.load_duels()) == 1


def test_resolve_finds_matching_perception(_isolated_duel_file):
    ds.record_duel(_duel_row(perception_id="pid-old"))
    ds.record_duel(_duel_row(perception_id="pid-1"))
    row = ds.resolve("BTC", "pid-1")
    assert row is not None and row["coin"] == "BTC"
    assert ds.resolve("BTC", "pid-missing") is None
    assert ds.resolve("", "pid-1") is None
    assert ds.resolve("BTC", "unknown") is None


# ── duelist call (no network) ──────────────────────────────────────────────

def test_call_duelist_returns_empty_without_key(monkeypatch):
    monkeypatch.setattr(research, "_async_do_call", None)  # must not be reached
    assert ds.call_duelist("", "http://x", "m", "s", "u") == ""


def test_call_duelist_posts_same_prompt(monkeypatch):
    """The duelist call must POST the SAME messages the primary gets —
    that byte-identity is the whole point of the A/B."""
    captured = {}

    async def fake_post(url, json, headers):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        class R:
            status_code = 200
            is_success = True
            text = ""
            def json(self):
                return {"choices": [{"message": {"content": "x"}}]}
        return R()

    captured_client = {}

    class FakeClient:
        def __init__(self, *a, **k):
            captured_client["init_kwargs"] = k
        async def __aenter__(self):
            self.post = fake_post
            return self
        async def __aexit__(self, *a):
            return False

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    out = ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    assert out == "x"
    assert captured["url"] == "http://duel.test/v1/chat/completions"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert captured["json"]["model"] == "m"
    assert captured["headers"]["Authorization"] == "Bearer dk"


def test_call_duelist_swallows_failures(monkeypatch):
    import httpx

    class BoomClient:
        def __init__(self, *a, **k):
            raise RuntimeError("boom")
        async def __aenter__(self):
            raise AssertionError("should not enter")

    monkeypatch.setattr(httpx, "AsyncClient", BoomClient)
    assert ds.call_duelist("dk", "http://duel.test/v1", "m", "s", "u") == ""


# ── P&L attribution math ───────────────────────────────────────────────────

def test_duelist_pnl_directional():
    live_long_win = {"side": "long", "realized_pnl_usd": 10.0, "realized_pnl_pct": 5.0}
    # concurs with live -> identical P&L
    assert ds._duelist_pnl(live_long_win, {"side": "long"}) == 10.0
    # opposes -> mirrored
    assert ds._duelist_pnl(live_long_win, {"side": "short"}) == -10.0
    # PASS or CLOSE -> flat (0.0): counts as a win only when live lost
    assert ds._duelist_pnl(live_long_win, {"side": None}) == 0.0
    assert ds._duelist_pnl(live_long_win, {"side": None, "verdict": "CLOSE"}) == 0.0
    # opposing verdict on a LOSING live trade -> the duelist wins
    live_long_loss = {"side": "long", "realized_pnl_usd": -8.0, "realized_pnl_pct": -4.0}
    assert ds._duelist_pnl(live_long_loss, {"side": "short"}) == 8.0


def test_model_stats_empty():
    s = ds._model_stats([])
    assert s["closes"] == 0 and s["realized_pnl_usd"] is None


def test_model_stats_basic():
    s = ds._model_stats([10.0, -5.0, 0.0])
    assert s["closes"] == 3
    assert s["wins"] == 1 and s["losses"] == 1
    assert s["realized_pnl_usd"] == 5.0
    assert s["avg_pnl_usd"] == round(5.0 / 3, 2)
    assert s["avg_win_usd"] == 10.0 and s["avg_loss_usd"] == 5.0


def test_latency_stats():
    # empty
    e = ds._latency_stats([])
    assert e["n"] == 0 and e["avg_ms"] is None
    # None values (rows written before latency tracking shipped) are excluded,
    # not treated as 0
    s = ds._latency_stats([None, 100.0, 300.0, None, 200.0])
    assert s["n"] == 3
    assert s["avg_ms"] == 200.0
    assert s["median_ms"] == 200.0
    assert s["max_ms"] == 300.0
    # even count -> median is the mean of the two middle values
    s2 = ds._latency_stats([100.0, 300.0])
    assert s2["median_ms"] == 200.0


# ── aggregate report ───────────────────────────────────────────────────────

def test_aggregate_join_and_scoring(monkeypatch, _isolated_duel_file):
    monkeypatch.setenv("LLM_DUEL_MODEL", "duel-model")
    # paired calls: one agree, one split — with distinct latencies so the
    # report's latency stats are verifiable
    ds.record_duel(_duel_row(perception_id="pid-1", coin="BTC",
                             primary_verdict="LONG", duelist_verdict="LONG",
                             duelist_side="long",
                             primary_ms=100, duelist_ms=200))
    ds.record_duel(_duel_row(perception_id="pid-2", coin="ETH",
                             primary_verdict="LONG", duelist_verdict="SHORT",
                             duelist_side="short",
                             primary_ms=200, duelist_ms=300))
    # realized closes: pid-1 won +10 (duelist concurs → +10); pid-2 lost -8
    # (duelist opposed → +8); a third close predates the duelist (no row).
    memory.record_close({
        "coin": "BTC", "side": "long", "notional_usd": 200.0,
        "realized_pnl_pct": 5.0, "realized_pnl_usd": 10.0,
        "duelist_at_entry": {"model": "duel-model", "verdict": "LONG",
                             "confidence": 0.8, "side": "long"},
        "perception_id": "pid-1",
    })
    memory.record_close({
        "coin": "ETH", "side": "long", "notional_usd": 200.0,
        "realized_pnl_pct": -4.0, "realized_pnl_usd": -8.0,
        "duelist_at_entry": {"model": "duel-model", "verdict": "SHORT",
                             "confidence": 0.6, "side": "short"},
        "perception_id": "pid-2",
    })
    memory.record_close({
        "coin": "SOL", "side": "long", "notional_usd": 100.0,
        "realized_pnl_pct": 2.0, "realized_pnl_usd": 3.0,
        "duelist_at_entry": None, "perception_id": "pid-legacy",
    })

    r = ds.aggregate()
    assert r["duel_calls"] == 2
    assert r["realized_closes"] == 3
    assert r["closes_with_duelist"] == 2
    assert r["agreement"] == {"agree": 1, "split": 1, "rate": 0.5}
    # primary: +10 -8 +3
    assert r["primary"]["realized_pnl_usd"] == 5.0
    assert r["primary"]["closes"] == 3
    # duelist-if-live: +10 (concurred) +8 (opposed the loser) — the legacy
    # close has no duelist verdict and doesn't score
    assert r["duelist_if_live"]["realized_pnl_usd"] == 18.0
    assert r["duelist_if_live"]["closes"] == 2
    assert r["duelist_if_live"]["wins"] == 2
    # latency is aggregated from the rows' *_ms fields
    assert r["latency"]["primary"] == {"n": 2, "avg_ms": 150.0,
                                       "median_ms": 150.0, "max_ms": 200.0}
    assert r["latency"]["duelist"]["n"] == 2 and r["latency"]["duelist"]["avg_ms"] == 250.0


def test_aggregate_latency_tolerates_legacy_rows(monkeypatch, _isolated_duel_file):
    """Rows written before the latency fields shipped (no *_ms) must not skew
    the report with zeros — they're excluded and reflected in n."""
    monkeypatch.setenv("LLM_DUEL_MODEL", "duel-model")
    # a row in the pre-latency shape: the *_ms keys are simply absent
    legacy = _duel_row(perception_id="pid-legacy-row")
    legacy.pop("primary_ms")
    legacy.pop("duelist_ms")
    ds.record_duel(legacy)
    ds.record_duel(_duel_row(perception_id="pid-new",
                             primary_ms=120, duelist_ms=340))
    r = ds.aggregate()
    # the absent fields read back as None and are excluded, not zeroed
    assert r["latency"]["primary"]["n"] == 1
    assert r["latency"]["primary"]["avg_ms"] == 120.0
    assert r["latency"]["duelist"]["n"] == 1
    assert r["latency"]["duelist"]["avg_ms"] == 340.0


# ── research() integration ─────────────────────────────────────────────────

def _patch_research_network(monkeypatch, primary_text, duelist_text):
    """Patch every external touch point in research(): candle fetch, account
    state, funding/news, and user-message assembly (the latter hides the
    _signals_block/_chronos_block network calls inside it)."""
    monkeypatch.setattr(research, "fetch_hl_candles", lambda coin, tf, n: _candles())
    monkeypatch.setattr(research, "resolve_user_address", lambda: None)
    monkeypatch.setattr(research, "_fetch_funding_rate", lambda coin: "N/A")
    monkeypatch.setattr(research, "_fetch_news", lambda coin: "no news")
    monkeypatch.setattr(research, "_build_user_message",
                        lambda *a, **k: "USER-PROMPT")
    monkeypatch.setattr(research, "build_system_prompt", lambda *a, **k: "SYS")

    def fake_call_ai(system_prompt, user_message, **kw):
        assert system_prompt == "SYS"
        assert user_message == "USER-PROMPT"
        time.sleep(0.02)  # measurable wall time for the primary_ms assert
        return primary_text

    monkeypatch.setattr(research, "_call_ai", fake_call_ai)

    def fake_duelist(key, base_url, model, system_prompt, user_message, timeout_s=120.0):
        assert key == "dk" and base_url == "http://duel.test/v1" and model == "duel-model"
        assert system_prompt == "SYS"  # same prompt, both models
        assert user_message == "USER-PROMPT"
        time.sleep(0.05)  # slower than the primary, so duelist_ms > primary_ms
        return duelist_text

    # research.py binds call_duelist at import (from duel_store import ...),
    # so patch the NAME IN THE research module — patching duel_store.call_duelist
    # would not intercept the call.
    monkeypatch.setattr(research, "call_duelist", fake_duelist)


def test_research_records_duel_row_when_enabled(
        monkeypatch, _isolated_duel_file, _duelist_on, _session_log, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    _patch_research_network(monkeypatch,
                            'PASS\n{"verdict":"PASS","confidence":0.5}',
                            'SHORT setup\n{"verdict":"SHORT","confidence":0.7,"side":"short"}')

    perception = {"id": "pid-int", "coin": "BTC", "type": "perp", "mid": 100.0,
                  "composite_score": 40.0, "triggers": []}
    analysis = research.research("BTC", perception)

    assert analysis["verdict"] == "PASS"  # primary verdict drives everything
    assert analysis["duelist_at_entry"] == {
        "model": "duel-model", "verdict": "SHORT", "confidence": 0.7, "side": "short",
    }
    # the duel row landed in the JSONL
    rows = ds.load_duels()
    assert len(rows) == 1
    assert rows[0]["perception_id"] == "pid-int"
    assert rows[0]["primary_verdict"] == "PASS"
    assert rows[0]["duelist_verdict"] == "SHORT"
    # and the session log got a duel event
    duel_events = [e for e in _read_events(_session_log) if e.get("event") == "duel"]
    assert len(duel_events) == 1
    assert duel_events[0]["coin"] == "BTC"
    assert duel_events[0]["agree"] is False
    # both calls' wall time was measured and carried through row → event
    assert rows[0]["primary_ms"] > 0 and rows[0]["duelist_ms"] > 0
    assert rows[0]["duelist_ms"] > rows[0]["primary_ms"]  # faked slower
    assert duel_events[0]["primary_ms"] == rows[0]["primary_ms"]
    assert duel_events[0]["duelist_ms"] == rows[0]["duelist_ms"]
    assert any("[duel] BTC:" in r.message and "SPLIT" in r.message
               for r in caplog.records)


def test_research_dormant_without_duelist(monkeypatch, _isolated_duel_file,
                                          _session_log, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    monkeypatch.delenv("LLM_DUEL_MODEL", raising=False)
    calls = []

    def spy_duelist(*a, **k):
        calls.append(a)

    # Patch the name in the research module (import-time binding), not duel_store's.
    monkeypatch.setattr(research, "call_duelist", spy_duelist)
    _patch_research_network(monkeypatch,
                            'PASS\n{"verdict":"PASS","confidence":0.5}', "unused")

    perception = {"id": "pid-off", "coin": "ETH", "type": "perp", "mid": 50.0,
                  "composite_score": 30.0, "triggers": []}
    analysis = research.research("ETH", perception)

    assert analysis["duelist_at_entry"] is None
    assert not calls  # no second LLM call at all
    assert ds.load_duels() == []
    assert not [e for e in _read_events(_session_log) if e.get("event") == "duel"]


def test_research_survives_duelist_outage(monkeypatch, _isolated_duel_file,
                                          _duelist_on, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(research, "fetch_hl_candles", lambda coin, tf, n: _candles())
    monkeypatch.setattr(research, "resolve_user_address", lambda: None)
    monkeypatch.setattr(research, "_fetch_funding_rate", lambda coin: "N/A")
    monkeypatch.setattr(research, "_fetch_news", lambda coin: "no news")
    monkeypatch.setattr(research, "_build_user_message", lambda *a, **k: "USER-PROMPT")
    monkeypatch.setattr(research, "build_system_prompt", lambda *a, **k: "SYS")
    monkeypatch.setattr(research, "_call_ai", lambda sp, um, **kw: 'PASS\n{"verdict":"PASS","confidence":0.9}')

    def boom(*a, **k):
        raise RuntimeError("duelist down")

    monkeypatch.setattr(research, "call_duelist", boom)

    perception = {"id": "pid-err", "coin": "SOL", "type": "perp", "mid": 200.0,
                  "composite_score": 35.0, "triggers": []}
    analysis = research.research("SOL", perception)

    assert analysis["verdict"] == "PASS"  # primary unaffected
    assert analysis["duelist_at_entry"] is None
    assert any("[duel] duelist hook failed" in r.message for r in caplog.records)


def test_parse_verdict_still_handles_duelist_shapes(monkeypatch):
    """The duelist reuses parse_verdict: verify a short-form SHORT verdict
    (the most common duelist disagreement) parses cleanly."""
    p = research.parse_verdict(
        '{"verdict":"SHORT","confidence":0.7,"side":"short"}',
        "BTC", {"mid": 100.0})
    assert p["verdict"] == "SHORT" and p["side"] == "short" and p["ai_down"] is False