"""Tests for the model-duel store (A/B LLM evaluation).

Covers: env gating (dormant unless a duelist model is named), the JSONL
store, the verdict-attribution P&L math, the aggregate report join, and the
research() integration (paired call, duel row, analysis field, session event).
No network: the LLM endpoints are monkeypatched like the shadow-signal tests.
"""

import json
import os
import time
import types

import pytest

import hermes_trader.agents.config_store as cs
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
    # The duelist URL/key must also be cleared: test_cleanup imports
    # hermes_trader.server, whose import-time _load_env_local_early() does
    # os.environ.setdefault() from .env.local — which now carries
    # LLM_DUEL_BASE_URL. conftest pops the vars before collection, but that
    # import re-leaks them mid-session, and duelist_config() reads env at CALL
    # time, so test order would otherwise decide the outcome.
    monkeypatch.delenv("LLM_DUEL_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_DUEL_API_KEY", raising=False)
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


# ── output budget (max_tokens) ────────────────────────────────────────────

def test_max_tokens_defaults_to_8192(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LLM_DUEL_MAX_TOKENS", raising=False)
    assert ds.resolve_max_tokens("LLM_MAX_TOKENS") == 8192
    assert ds.duelist_config()["max_tokens"] == 8192


def test_max_tokens_duelist_inherits_primary(monkeypatch):
    monkeypatch.delenv("LLM_DUEL_MAX_TOKENS", raising=False)
    monkeypatch.setenv("LLM_MAX_TOKENS", "1234")
    assert ds.duelist_config()["max_tokens"] == 1234
    monkeypatch.setenv("LLM_DUEL_MAX_TOKENS", "777")
    assert ds.duelist_config()["max_tokens"] == 777


def test_max_tokens_invalid_falls_back(monkeypatch):
    for bad in ("not-a-number", "-5", "0", "  "):
        monkeypatch.setenv("LLM_MAX_TOKENS", bad)
        assert ds.resolve_max_tokens("LLM_MAX_TOKENS") == 8192


# ── max_tokens: config `llm` block (P11 follow-on) ────────────────────────

@pytest.fixture
def agent_cfg(monkeypatch):
    """Own the agent-config FILE for the duration of one test (the
    CONFIG_PATH-direct isolation pattern — see test_sampling_profiles.py).
    conftest redirects HERMES_AGENT_CONFIG_FILE to a throwaway path BEFORE
    config_store freezes CONFIG_PATH, so writing to that file here never
    touches the live config; the resolvers read it at CALL time (hot)."""
    import hermes_trader.agents.config_store as cs
    cfg_path = cs.CONFIG_PATH
    had_cfg = os.path.exists(cfg_path)
    backup = ""
    if had_cfg:
        with open(cfg_path) as f:
            backup = f.read()

    def write_cfg(cfg):
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

    yield types.SimpleNamespace(write_cfg=write_cfg)

    if had_cfg:
        with open(cfg_path, "w") as f:
            f.write(backup)
    elif os.path.exists(cfg_path):
        os.remove(cfg_path)


def test_max_tokens_env_only_when_block_absent(monkeypatch, agent_cfg):
    """No-op guarantee: config block ABSENT → the env-only resolution
    (LLM_DUEL_MAX_TOKENS → LLM_MAX_TOKENS → 8192), byte-identical to the
    pre-config behavior."""
    monkeypatch.delenv("LLM_DUEL_MAX_TOKENS", raising=False)
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    agent_cfg.write_cfg({"mode": "SHADOW"})  # config present, no `llm` key
    assert ds.effective_primary_max_tokens() == 2048
    assert ds.duelist_config()["max_tokens"] == 2048


def test_max_tokens_config_wins_over_env(monkeypatch, agent_cfg):
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    monkeypatch.setenv("LLM_DUEL_MAX_TOKENS", "1024")
    agent_cfg.write_cfg({"llm": {"max_tokens": 4096, "duelist_max_tokens": 128}})
    assert ds.effective_primary_max_tokens() == 4096
    assert ds.duelist_config()["max_tokens"] == 128


def test_duelist_max_tokens_inherits_primary_resolved(monkeypatch, agent_cfg):
    """The duelist inherits the PRIMARY's FULLY-RESOLVED budget when its own
    config key is unset — so a primary budget set via config (not env) still
    propagates to the duelist."""
    monkeypatch.delenv("LLM_DUEL_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    agent_cfg.write_cfg({"llm": {"max_tokens": 4096}})
    assert ds.effective_primary_max_tokens() == 4096
    assert ds.effective_duelist_max_tokens() == 4096
    # A duelist config key beats the inherited value.
    agent_cfg.write_cfg({"llm": {"max_tokens": 4096, "duelist_max_tokens": 512}})
    assert ds.effective_duelist_max_tokens() == 512


def test_max_tokens_hot_read_follows_config_between_calls(monkeypatch, agent_cfg):
    """No module-level cache: flipping the file between calls is visible
    immediately (a max-tokens retune is a same-inode config flip)."""
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    agent_cfg.write_cfg({"llm": {"max_tokens": 2000}})
    assert ds.effective_primary_max_tokens() == 2000
    agent_cfg.write_cfg({"llm": {"max_tokens": 3000}})
    assert ds.effective_primary_max_tokens() == 3000
    agent_cfg.write_cfg({"mode": "SHADOW"})  # block removed → env/default
    assert ds.effective_primary_max_tokens() == 8192


@pytest.mark.parametrize("bad", ["not-a-number", "-5", 0, "", " ", True, None])
def test_max_tokens_invalid_config_falls_back(monkeypatch, agent_cfg, bad):
    """Invalid/non-positive config values fall through to the env fallback
    (never raise, never return 0)."""
    monkeypatch.setenv("LLM_MAX_TOKENS", "5555")
    agent_cfg.write_cfg({"llm": {"max_tokens": bad}})
    assert ds.effective_primary_max_tokens() == 5555
    assert ds._parse_max_tokens(ds.slot_get("primary", "max_tokens", "max_tokens")) is None


def test_fail_open_corrupt_config_max_tokens(monkeypatch, agent_cfg):
    """A corrupt config must never break the LLM path: the budget degrades to
    the env fallback (fail-open, same contract as the model/sampling
    resolvers)."""
    monkeypatch.setenv("LLM_MAX_TOKENS", "6666")
    monkeypatch.setenv("LLM_DUEL_MAX_TOKENS", "4444")
    agent_cfg.write_cfg({"llm": {"max_tokens": 9999}})
    with open(cs.CONFIG_PATH, "w") as f:
        f.write("{this is not valid json")
    assert ds.effective_primary_max_tokens() == 6666
    assert ds.effective_duelist_max_tokens() == 4444
    # And the duelist config path still resolves end to end.
    assert ds.duelist_config()["max_tokens"] == 4444


# ── max_tokens: nested llm.primary/llm.duelist slots (2026-09-10) ─────────

def test_nested_slots_win_over_flat_and_env(monkeypatch, agent_cfg):
    """The NESTED llm.<slot> shape wins over both the legacy flat keys and
    the env vars (priority: nested > flat > env > default)."""
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_DUEL_MODEL", "duel-env")
    monkeypatch.setenv("LLM_MAX_TOKENS", "100")
    monkeypatch.setenv("LLM_DUEL_MAX_TOKENS", "200")
    agent_cfg.write_cfg({"llm": {
        "model": "flat-primary",              # legacy flat (must lose)
        "duelist_model": "flat-duelist",      # legacy flat (must lose)
        "max_tokens": 300,                    # legacy flat (must lose)
        "duelist_max_tokens": 400,            # legacy flat (must lose)
        "primary": {"model": "nested-primary", "max_tokens": 1234},
        "duelist": {"model": "nested-duelist", "max_tokens": 567},
    }})
    assert ds.effective_primary_model() == "nested-primary"
    assert ds.effective_duelist_model() == "nested-duelist"
    assert ds.effective_primary_max_tokens() == 1234
    assert ds.effective_duelist_max_tokens() == 567
    assert ds.duelist_config()["max_tokens"] == 567


def test_partial_nested_slot_falls_back_to_flat(monkeypatch, agent_cfg):
    """A partial nested slot names only what it changes — every key it
    doesn't name falls through to the flat legacy key (per-key merge, NOT
    replace-whole-slot)."""
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    agent_cfg.write_cfg({"llm": {
        "model": "flat-primary",          # no nested primary.model -> flat
        "max_tokens": 999,                # no nested primary.max_tokens -> flat
        "primary": {"model": "nested-primary"},  # only model named
    }})
    assert ds.effective_primary_model() == "nested-primary"
    assert ds.effective_primary_max_tokens() == 999


def test_nested_sampling_wins_over_flat(monkeypatch, agent_cfg):
    """Nested llm.<slot>.sampling wins over the flat llm.sampling /
    llm.duelist_sampling; a partial nested sampling keeps code defaults
    (per-key, as before)."""
    agent_cfg.write_cfg({"llm": {
        "sampling": {"temperature": 0.3},                 # flat (must lose)
        "duelist_sampling": {"temperature": 0.4},         # flat (must lose)
        "primary": {"sampling": {"temperature": 0.9, "top_p": 0.5}},
        "duelist": {"sampling": {"top_k": 7}},
    }})
    p = research.effective_llm_sampling()
    assert p["temperature"] == 0.9 and p["top_p"] == 0.5
    d = ds.effective_duelist_sampling()
    # nested duelist top_k wins; the code-default keys it doesn't name survive
    assert d["top_k"] == 7 and d["temperature"] == 1.0


def test_nested_hot_read_follows_config_between_calls(monkeypatch, agent_cfg):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    agent_cfg.write_cfg({"llm": {"primary": {"max_tokens": 2000}}})
    assert ds.effective_primary_max_tokens() == 2000
    agent_cfg.write_cfg({"llm": {"primary": {"max_tokens": 3000}}})
    assert ds.effective_primary_max_tokens() == 3000
    agent_cfg.write_cfg({"mode": "SHADOW"})
    assert ds.effective_primary_max_tokens() == 8192


def test_nested_slot_absent_keeps_flat_noop(monkeypatch, agent_cfg):
    """No-op guarantee: a config with NO nested slots resolves exactly as the
    flat legacy keys did pre-nest (the pre-nest live config keeps working
    unchanged)."""
    monkeypatch.setenv("LLM_MODEL", "flat-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    agent_cfg.write_cfg({"llm": {
        "model": "flat-model",
        "max_tokens": 2048,
        "sampling": {"temperature": 0.2},
    }})
    assert ds.effective_primary_model() == "flat-model"
    assert ds.effective_primary_max_tokens() == 2048
    assert research.effective_llm_sampling()["temperature"] == 0.2


def test_fail_open_corrupt_config_nested(monkeypatch, agent_cfg):
    """A corrupt config degrades to the env fallback for EVERY resolver
    (models, max_tokens, sampling) — fail-open, nested or flat."""
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "5555")
    agent_cfg.write_cfg({"llm": {"primary": {"model": "cfg", "max_tokens": 1}}})
    with open(cs.CONFIG_PATH, "w") as f:
        f.write("{corrupt")
    assert ds.effective_primary_model() == "env-model"
    assert ds.effective_primary_max_tokens() == 5555
    assert research.effective_llm_sampling()["temperature"] == 0.1  # pure default


# ── chat_template_kwargs (per-request thinking toggle, 2026-09-10) ────────

def test_chat_template_kwargs_absent_body_unchanged(monkeypatch, agent_cfg):
    """NO-OP GUARANTEE: with the key absent, `chat_template_kwargs` is NOT
    in the POST body (primary AND duelist) — byte-identical wire to the
    pre-feature default."""
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    agent_cfg.write_cfg({"llm": {"primary": {"model": "m"},
                                 "duelist": {"model": "d"}}})
    loop = __import__("asyncio").new_event_loop()
    try:
        loop.run_until_complete(research._async_do_call("k", "http://x/v1", "m", "S", "U"))
        assert "chat_template_kwargs" not in captured["json"]
    finally:
        loop.close()
    ds.call_duelist("dk", "http://duel.test/v1", "d", "SYS", "USER")
    assert "chat_template_kwargs" not in captured["json"]


def test_chat_template_kwargs_primary_only(monkeypatch, agent_cfg):
    """Nested llm.primary.chat_template_kwargs reaches the PRIMARY body
    verbatim and does NOT leak into the duelist body."""
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    agent_cfg.write_cfg({"llm": {
        "primary": {"model": "m", "chat_template_kwargs": {"enable_thinking": False}},
        "duelist": {"model": "d"},
    }})
    loop = __import__("asyncio").new_event_loop()
    try:
        loop.run_until_complete(research._async_do_call("k", "http://x/v1", "m", "S", "U"))
        assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        loop.close()
    ds.call_duelist("dk", "http://duel.test/v1", "d", "SYS", "USER")
    assert "chat_template_kwargs" not in captured["json"]


def test_chat_template_kwargs_duelist_only(monkeypatch, agent_cfg):
    """The reverse: the duelist key reaches the DUELIST body only."""
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    agent_cfg.write_cfg({"llm": {
        "primary": {"model": "m"},
        "duelist": {"model": "d", "chat_template_kwargs": {"enable_thinking": True}},
    }})
    loop = __import__("asyncio").new_event_loop()
    try:
        loop.run_until_complete(research._async_do_call("k", "http://x/v1", "m", "S", "U"))
        assert "chat_template_kwargs" not in captured["json"]
    finally:
        loop.close()
    ds.call_duelist("dk", "http://duel.test/v1", "d", "SYS", "USER")
    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": True}


def test_chat_template_kwargs_invalid_value_is_noop(monkeypatch, agent_cfg):
    """A non-dict value (fail-open) is never sent and never raises."""
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    for bad in ("not-a-dict", ["list"], None, 5):
        agent_cfg.write_cfg({"llm": {"primary": {"model": "m",
                                                 "chat_template_kwargs": bad}}})
        loop = __import__("asyncio").new_event_loop()
        try:
            loop.run_until_complete(
                research._async_do_call("k", "http://x/v1", "m", "S", "U"))
        finally:
            loop.close()
        assert "chat_template_kwargs" not in captured["json"]
        assert ds.effective_chat_template_kwargs("primary") == {}


def test_chat_template_kwargs_hot_read(monkeypatch, agent_cfg):
    """No module-level cache: a flip between calls is visible immediately."""
    agent_cfg.write_cfg({"llm": {"primary": {"model": "m",
                                             "chat_template_kwargs": {"enable_thinking": True}}}})
    assert ds.effective_chat_template_kwargs("primary") == {"enable_thinking": True}
    agent_cfg.write_cfg({"llm": {"primary": {"model": "m",
                                             "chat_template_kwargs": {"enable_thinking": False}}}})
    assert ds.effective_chat_template_kwargs("primary") == {"enable_thinking": False}
    agent_cfg.write_cfg({"llm": {"primary": {"model": "m"}}})
    assert ds.effective_chat_template_kwargs("primary") == {}


def test_chat_template_kwargs_returns_copy(monkeypatch, agent_cfg):
    """The resolver returns a COPY — mutating the result must never touch
    the config dict (which the next hot read would re-serve)."""
    agent_cfg.write_cfg({"llm": {"primary": {"chat_template_kwargs": {"a": 1}}}})
    out = ds.effective_chat_template_kwargs("primary")
    out["injected"] = True
    assert ds.effective_chat_template_kwargs("primary") == {"a": 1}
    assert agent_cfg and ds.llm_slot("primary")["chat_template_kwargs"] == {"a": 1}


def _fake_httpx(monkeypatch, captured, content="x"):
    """httpx.AsyncClient stub capturing the POST body (same shape as the
    duelist prompt-identity test below)."""
    async def fake_post(url, json, headers):
        captured["url"] = url
        captured["json"] = json

        class R:
            status_code = 200
            is_success = True
            text = ""
            def json(self):
                return {"choices": [{"message": {"content": content}}]}
        return R()

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            self.post = fake_post
            return self
        async def __aexit__(self, *a):
            return False

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)


def test_duel_post_uses_env_max_tokens(monkeypatch):
    """The on-the-wire max_tokens must follow LLM_DUEL_MAX_TOKENS /
    LLM_MAX_TOKENS at call time — the 2026-08-25 incident assumed the
    hardcoded 8192 was a binding cap; it was only an output budget."""
    captured = {}
    _fake_httpx(monkeypatch, captured)
    monkeypatch.delenv("LLM_DUEL_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    assert captured["json"]["max_tokens"] == 8192
    monkeypatch.setenv("LLM_MAX_TOKENS", "2048")
    ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    assert captured["json"]["max_tokens"] == 2048
    monkeypatch.setenv("LLM_DUEL_MAX_TOKENS", "1024")
    ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    assert captured["json"]["max_tokens"] == 1024


def test_research_post_uses_llm_max_tokens(monkeypatch):
    import asyncio
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    loop = asyncio.new_event_loop()
    try:
        out = loop.run_until_complete(
            research._async_do_call("k", "http://x/v1", "m", "S", "U"))
        assert out == ("ok", None)
        assert captured["json"]["max_tokens"] == 8192
        monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
        loop.run_until_complete(
            research._async_do_call("k", "http://x/v1", "m", "S", "U"))
        assert captured["json"]["max_tokens"] == 4096
    finally:
        loop.close()


def test_research_post_uses_config_max_tokens(monkeypatch, agent_cfg):
    """End-to-end: the `llm.max_tokens` config value reaches the primary's
    POST body (the strongest no-op/presence guarantee — the on-the-wire
    max_tokens follows the config at call time)."""
    import asyncio
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    loop = asyncio.new_event_loop()
    try:
        agent_cfg.write_cfg({"llm": {"max_tokens": 5000}})
        loop.run_until_complete(
            research._async_do_call("k", "http://x/v1", "m", "S", "U"))
        assert captured["json"]["max_tokens"] == 5000
    finally:
        loop.close()


def test_duel_post_uses_config_max_tokens(monkeypatch, agent_cfg):
    """End-to-end: the `llm.duelist_max_tokens` config value reaches the
    duelist's POST body (config wins over the env fallback on the wire)."""
    captured = {}
    _fake_httpx(monkeypatch, captured)
    monkeypatch.setenv("LLM_DUEL_MAX_TOKENS", "1024")  # env must lose
    agent_cfg.write_cfg({"llm": {"duelist_max_tokens": 640}})
    ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    assert captured["json"]["max_tokens"] == 640


# ── sampling profile + timeout retry (2026-08-27) ─────────────────────────

def test_duel_post_sends_qwen_sampling_profile(monkeypatch):
    """The duelist payload must carry the model card's instruct-reasoning
    profile (temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
    presence_penalty=1.5, repetition_penalty=1.0) — the old hardcoded
    temperature=0.1 was a leftover from the OpenRouter era."""
    captured = {}
    _fake_httpx(monkeypatch, captured)
    monkeypatch.delenv("LLM_DUEL_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    assert captured["json"]["temperature"] == 1.0
    assert captured["json"]["top_p"] == 0.95
    assert captured["json"]["top_k"] == 20
    assert captured["json"]["min_p"] == 0.0
    assert captured["json"]["presence_penalty"] == 1.5
    assert captured["json"]["repetition_penalty"] == 1.0


def test_duel_retries_once_on_timeout(monkeypatch):
    """A ReadTimeout must trigger exactly one identical retry, then a
    successful second call wins."""
    import httpx
    calls = {"n": 0}
    body = {"choices": [{"message": {"content": "recovered"}}]}

    class R:
        status_code = 200
        is_success = True
        text = ""
        def json(self):
            return body

    async def fake_post(url, json, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("simulated server still generating")
        return R()

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            self.post = fake_post
            return self
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    out = ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER", timeout_s=0.5)
    # No `timings` in the fake response -> server_ms is None (fail-open).
    assert out == ("recovered", None)
    assert calls["n"] == 2  # first attempt timed out, exactly one retry


def test_duel_no_retry_on_second_timeout_and_never_raises(monkeypatch):
    """Two consecutive timeouts = give up after exactly 2 POSTs (no loop),
    and call_duelist never raises — it returns "" and logs LOUD (the old
    DEBUG line was how the doom-loop failures stayed invisible)."""
    import httpx

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            async def fake_post(url, json, headers):
                raise httpx.ReadTimeout("simulated")
            self.post = fake_post
            return self
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    out = ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER", timeout_s=0.5)
    assert out == ("", None)


def test_duel_non_timeout_failure_loud_and_no_retry(monkeypatch, caplog):
    """A non-timeout fault (e.g. dead endpoint) is not retried and IS
    logged at WARNING — visible in trader.log, not buried at DEBUG."""
    import httpx
    calls = {"n": 0}

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            async def fake_post(url, json, headers):
                calls["n"] += 1
                raise httpx.ConnectError("simulated dead endpoint")
            self.post = fake_post
            return self
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    with caplog.at_level("WARNING", logger="hermes_trader.agents.duel_store"):
        out = ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    assert out == ("", None)
    assert calls["n"] == 1  # no retry for non-timeout faults
    assert any("duelist call failed" in r.message for r in caplog.records)


def test_duel_timeout_loud_log(monkeypatch, caplog):
    """The timeout path's terminal line is WARNING with the runaway hint —
    the operator-visible half of the 2026-08-27 'make it visible' fix."""
    import httpx

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            async def fake_post(url, json, headers):
                raise httpx.ReadTimeout("simulated")
            self.post = fake_post
            return self
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    with caplog.at_level("WARNING", logger="hermes_trader.agents.duel_store"):
        ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER", timeout_s=0.5)
    assert any("TIMED OUT" in r.message and "runaway" in r.message
               for r in caplog.records)


def test_primary_call_ai_retries_once_on_timeout(monkeypatch):
    """_call_ai must survive a timeout, retry ONCE, and return the recovered
    (text, server_ms) tuple — and still return ("", None) (never raise) when
    both attempts time out. This is the primary path's half of the
    2026-08-27 single-retry fix, now extended with the server-reported
    processing time (the response's `timings` block)."""
    import httpx
    calls = {"n": 0}
    body = {
        "choices": [{"message": {"content": "primary-ok"}}],
        # llama.cpp-style timings: server_processing_ms = prompt + predicted.
        "timings": {"prompt_ms": 12.0, "predicted_ms": 2500.0},
    }

    class R:
        status_code = 200
        is_success = True
        text = ""
        def json(self):
            return body

    async def fake_post(url, json, headers):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("simulated server still generating")
        return R()

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            self.post = fake_post
            return self
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    out = research._call_ai("SYS", "USER", api_key="k",
                            base_url="http://x/v1", model="m")
    # (recovered text, server-reported processing time from `timings`)
    assert out == ("primary-ok", 2512)
    assert calls["n"] == 2


def test_primary_call_ai_never_raises_on_timeout(monkeypatch, caplog):
    """Both attempts time out -> _call_ai returns ("", None) (NOT an
    exception) and logs the terminal 'TIMED OUT on both attempts' WARNING.
    The caller (research_coin) relies on _call_ai never raising so a
    slow/dead LLM degrades to PASS-ai_down rather than crashing the worker."""
    import httpx

    class FakeClient:
        def __init__(self, *a, **k):
            pass
        async def __aenter__(self):
            async def fake_post(url, json, headers):
                raise httpx.ReadTimeout("simulated")
            self.post = fake_post
            return self
        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    with caplog.at_level("WARNING", logger="hermes_trader.agents.research"):
        out = research._call_ai("SYS", "USER", api_key="k",
                                base_url="http://x/v1", model="m")
    assert out == ("", None)
    assert any("both attempts" in r.message for r in caplog.records)


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
    assert ds.call_duelist("", "http://x", "m", "s", "u") == ("", None)


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
    # No `timings` in the fake response -> server_ms is None (fail-open).
    assert out == ("x", None)
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
    assert ds.call_duelist("dk", "http://duel.test/v1", "m", "s", "u") == ("", None)


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


# ── server_processing_ms (the queue-free latency source) ──────────────────

def test_server_processing_ms_from_timings():
    """The queue-free processing time is prompt_ms + predicted_ms (the
    llama.cpp `timings` block), as an int ms."""
    data = {"timings": {"prompt_ms": 33.0, "predicted_ms": 2131.1,
                        "predicted_n": 87}}
    assert ds.server_processing_ms(data) == 2164  # round(2164.1)


def test_server_processing_ms_fail_open():
    """Any absent / malformed shape degrades to None (never raises) — the
    OpenRouter / streaming case where `timings` isn't sent."""
    assert ds.server_processing_ms({}) is None                       # no timings
    assert ds.server_processing_ms({"choices": []}) is None          # no timings key
    assert ds.server_processing_ms(None) is None                     # not a dict
    assert ds.server_processing_ms("text") is None
    assert ds.server_processing_ms({"timings": "not-a-dict"}) is None
    assert ds.server_processing_ms({"timings": {}}) is None          # zero total
    assert ds.server_processing_ms({"timings": {"prompt_ms": None,
                                                "predicted_ms": None}}) is None
    # only one half present still yields that half's value
    assert ds.server_processing_ms({"timings": {"predicted_ms": 500.0}}) == 500


# ── aggregate report ───────────────────────────────────────────────────────

def test_aggregate_join_and_scoring(monkeypatch, _isolated_duel_file):
    monkeypatch.setenv("LLM_DUEL_MODEL", "duel-model")
    # paired calls: one agree, one split — with distinct latencies so the
    # report's latency stats are verifiable
    ds.record_duel(_duel_row(perception_id="pid-1", coin="BTC",
                             primary_verdict="LONG", duelist_verdict="LONG",
                             duelist_side="long",
                             primary_ms=100, duelist_ms=200,
                             primary_server_ms=80, duelist_server_ms=180))
    ds.record_duel(_duel_row(perception_id="pid-2", coin="ETH",
                             primary_verdict="LONG", duelist_verdict="SHORT",
                             duelist_side="short",
                             primary_ms=200, duelist_ms=300,
                             primary_server_ms=150, duelist_server_ms=260))
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
    # latency is aggregated from the rows' *_ms fields (wall clock)
    assert r["latency"]["primary"] == {"n": 2, "avg_ms": 150.0,
                                       "median_ms": 150.0, "max_ms": 200.0}
    assert r["latency"]["duelist"]["n"] == 2 and r["latency"]["duelist"]["avg_ms"] == 250.0
    # ...and from the *_server_ms fields (queue-free server processing)
    assert r["latency"]["primary_server"] == {"n": 2, "avg_ms": 115.0,
                                              "median_ms": 115.0, "max_ms": 150}
    assert r["latency"]["duelist_server"]["n"] == 2
    assert r["latency"]["duelist_server"]["avg_ms"] == 220.0


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
    # the *_server_ms fields are absent on BOTH rows here -> excluded (n=0),
    # mirroring the pre-server-shipping / no-`timings` endpoints
    assert r["latency"]["primary_server"]["n"] == 0
    assert r["latency"]["duelist_server"]["n"] == 0


# ── research() integration ─────────────────────────────────────────────────

def _patch_research_network(monkeypatch, primary_text, duelist_text,
                            primary_server_ms=None, duelist_server_ms=None):
    """Patch every external touch point in research(): candle fetch, account
    state, funding/news, and user-message assembly (the latter hides the
    _signals_block/_chronos_block network calls inside it). The two LLM fakes
    return the production (text, server_ms) tuples."""
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
        return primary_text, primary_server_ms

    monkeypatch.setattr(research, "_call_ai", fake_call_ai)

    def fake_duelist(key, base_url, model, system_prompt, user_message, timeout_s=120.0, max_tokens=None):
        assert key == "dk" and base_url == "http://duel.test/v1" and model == "duel-model"
        assert system_prompt == "SYS"  # same prompt, both models
        assert user_message == "USER-PROMPT"
        time.sleep(0.05)  # slower than the primary, so duelist_ms > primary_ms
        return duelist_text, duelist_server_ms

    # research.py binds call_duelist at import (from duel_store import ...),
    # so patch the NAME IN THE research module — patching duel_store.call_duelist
    # would not intercept the call.
    monkeypatch.setattr(research, "call_duelist", fake_duelist)


def test_research_records_duel_row_when_enabled(
        monkeypatch, _isolated_duel_file, _duelist_on, _session_log, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    # faked server-reported processing times: duelist's is smaller than its
    # wall time (the queue-wait gap), primary's is smaller than its wall too.
    _patch_research_network(monkeypatch,
                            'PASS\n{"verdict":"PASS","confidence":0.5}',
                            'SHORT setup\n{"verdict":"SHORT","confidence":0.7,"side":"short"}',
                            primary_server_ms=30, duelist_server_ms=40)

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
    # and the server-reported (queue-free) processing time rides along too
    assert rows[0]["primary_server_ms"] == 30
    assert rows[0]["duelist_server_ms"] == 40
    assert duel_events[0]["primary_server_ms"] == 30
    assert duel_events[0]["duelist_server_ms"] == 40
    assert any("[duel] BTC:" in r.message and "SPLIT" in r.message
               for r in caplog.records)
    # the log line carries the server time so the queue gap is visible
    assert any("ms srv" in r.message for r in caplog.records
               if "[duel] BTC:" in r.message)


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
    monkeypatch.setattr(research, "_call_ai",
                        lambda sp, um, **kw: ('PASS\n{"verdict":"PASS","confidence":0.9}', None))

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