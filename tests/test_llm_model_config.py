"""Model identity in the agent config's `llm` block (P8 follow-on).

The primary/duelist MODEL names used to be env-only (``LLM_MODEL`` /
``LLM_DUEL_MODEL``), so a model swap forced a container recreate. They now
resolve agent config ``llm.model`` / ``llm.duelist_model`` FIRST (hot, read
at call time — a swap is a same-inode config flip), then the env fallback,
then the code default; endpoint/key/max-tokens stay in ``.env.local``.

Pins the no-op guarantee (config block ABSENT = byte-identical to the old
env-only resolution, for every read site), the config-wins rule, the
duelist-disable rule (model absent everywhere = dormant; the duelist
deliberately does NOT fall back to the primary's model), hot-read with no
module-level cache, fail-open on a corrupt config, and that the resolved
model reaches the POST body + the analysis dict (what the ``Verdict:`` /
``Trade result`` lines log).

Config isolation mirrors tests/test_sampling_profiles.py: own the
``config_store.CONFIG_PATH`` file directly (backup/write/restore). No
network, no models — httpx.AsyncClient is stubbed with a FakeClient whose
fake_post captures the POST body (the proven capture pattern).
"""

import asyncio
import json
import os
import types

import httpx
import pytest

import hermes_trader.agents.config_store as cs
from hermes_trader.agents import duel_store as ds
from hermes_trader.agents import research

PRIMARY_ENV = "LLM_MODEL"
DUEL_ENV = "LLM_DUEL_MODEL"


# ── httpx stub (mirrors tests/test_sampling_profiles.py::_fake_httpx) ────

def _fake_httpx(monkeypatch, captured, content='{"verdict":"PASS","confidence":0.0}'):
    """httpx.AsyncClient stub capturing the POST body."""

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

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)


def _call_primary(captured, monkeypatch):
    """Drive research._call_ai (the real production entry point — it
    resolves model=None via the config→env chain) hermetically; return the
    captured POST body."""
    out = research._call_ai("S", "U", api_key="k", base_url="http://x/v1")
    assert captured.get("json") is not None
    return captured["json"]


# ── config isolation (CONFIG_PATH direct — see module docstring) ─────────

@pytest.fixture
def agent_cfg(monkeypatch):
    """Own the agent-config FILE for the duration of one test."""
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


# ── primary model resolution ─────────────────────────────────────────────

def test_primary_model_env_when_block_absent(monkeypatch, agent_cfg):
    """No-op guarantee: with the `llm` block ABSENT the helper returns the
    LLM_MODEL env value exactly as the old inline resolution did."""
    monkeypatch.setenv(PRIMARY_ENV, "env-model")
    agent_cfg.write_cfg({"mode": "SHADOW"})  # config present, no `llm` key
    assert ds.effective_primary_model() == "env-model"


def test_primary_model_env_fallback_chain(monkeypatch, agent_cfg):
    """LLM_MODEL unset → OPENROUTER_MODEL; both unset → the code default."""
    monkeypatch.delenv(PRIMARY_ENV, raising=False)
    agent_cfg.write_cfg({})
    monkeypatch.setenv("OPENROUTER_MODEL", "or-model")
    assert ds.effective_primary_model() == "or-model"
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert ds.effective_primary_model() == "x-ai/grok-4.3"


def test_primary_model_config_wins_over_env(monkeypatch, agent_cfg):
    monkeypatch.setenv(PRIMARY_ENV, "env-model")
    agent_cfg.write_cfg({"llm": {"model": "cfg-model"}})
    assert ds.effective_primary_model() == "cfg-model"


def test_primary_model_partial_llm_block_falls_back(monkeypatch, agent_cfg):
    """A `llm` block with sampling but no `model` key does NOT disable the
    env fallback — a partial block names only what it changes."""
    monkeypatch.setenv(PRIMARY_ENV, "env-model")
    agent_cfg.write_cfg({"llm": {"sampling": {"temperature": 0.3}}})
    assert ds.effective_primary_model() == "env-model"


def test_primary_model_hot_read_follows_config_between_calls(monkeypatch, agent_cfg):
    """No module-level cache: flipping the file between calls is visible
    immediately (the model-swap hot flip)."""
    agent_cfg.write_cfg({"llm": {"model": "m-one"}})
    assert ds.effective_primary_model() == "m-one"
    agent_cfg.write_cfg({"llm": {"model": "m-two"}})
    assert ds.effective_primary_model() == "m-two"
    agent_cfg.write_cfg({"mode": "SHADOW"})  # block removed → env
    monkeypatch.setenv(PRIMARY_ENV, "env-model")
    assert ds.effective_primary_model() == "env-model"


# ── duelist model resolution ─────────────────────────────────────────────

def test_duelist_model_env_when_block_absent(monkeypatch, agent_cfg):
    """No-op guarantee: block ABSENT → the LLM_DUEL_MODEL env value (the old
    env-only resolution)."""
    monkeypatch.setenv(DUEL_ENV, "duel-env")
    agent_cfg.write_cfg({"mode": "SHADOW"})
    assert ds.effective_duelist_model() == "duel-env"
    assert ds.duelist_enabled() is True


def test_duelist_model_config_wins_over_env(monkeypatch, agent_cfg):
    monkeypatch.setenv(DUEL_ENV, "duel-env")
    agent_cfg.write_cfg({"llm": {"duelist_model": "duel-cfg"}})
    assert ds.effective_duelist_model() == "duel-cfg"
    assert ds.duelist_config()["model"] == "duel-cfg"


def test_duelist_disabled_when_absent_everywhere(monkeypatch, agent_cfg):
    """Model absent in config AND env → the duelist is fully dormant."""
    monkeypatch.delenv(DUEL_ENV, raising=False)
    monkeypatch.setenv(PRIMARY_ENV, "primary-model")
    agent_cfg.write_cfg({"mode": "SHADOW"})
    assert ds.effective_duelist_model() is None
    assert ds.duelist_enabled() is False
    assert ds.duelist_config()["model"] == ""


def test_duelist_does_not_fall_back_to_primary(monkeypatch, agent_cfg):
    """A named primary NEVER implies a duelist — a silent "duel the primary
    against itself" would double LLM load with no A/B value."""
    monkeypatch.delenv(DUEL_ENV, raising=False)
    monkeypatch.setenv(PRIMARY_ENV, "primary-model")
    agent_cfg.write_cfg({"llm": {"model": "primary-model"}})
    assert ds.duelist_enabled() is False


def test_duelist_hot_read_follows_config_between_calls(monkeypatch, agent_cfg):
    agent_cfg.write_cfg({"llm": {"duelist_model": "d-one"}})
    assert ds.duelist_enabled() is True
    agent_cfg.write_cfg({"mode": "SHADOW"})  # duelist removed → dormant
    assert ds.duelist_enabled() is False


def test_duelist_config_endpoint_fallback_unchanged(monkeypatch, agent_cfg):
    """base_url/api_key keep their env fallbacks (config is model + sampling
    only) — a duelist on the primary's server needs only the model name."""
    monkeypatch.setenv(DUEL_ENV, "duel-env")
    # test_cleanup's import-time _load_env_local_early() re-leaks the LIVE
    # .env.local vars (LLM_DUEL_BASE_URL) mid-session; pin the primary
    # endpoint explicitly and clear the duelist-specific ones (same reason
    # as tests/test_duel_store.py::test_dormant_when_no_duelist_model).
    monkeypatch.delenv("LLM_DUEL_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_DUEL_API_KEY", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://primary.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "primary-key")
    agent_cfg.write_cfg({})
    cfg = ds.duelist_config()
    assert cfg["base_url"] == "http://primary.test/v1"
    assert cfg["api_key"] == "primary-key"
    assert cfg["model"] == "duel-env"


# ── the resolved model reaches the POST body + analysis dict ────────────

def test_post_body_carries_config_model(monkeypatch, agent_cfg):
    """The `model` field of the primary POST body is the config-resolved
    name (not a later env read), so a 404 on a mistyped id is loud, not a
    silent all-PASS mode."""
    monkeypatch.setenv(PRIMARY_ENV, "env-model")
    agent_cfg.write_cfg({"llm": {"model": "cfg-model"}})
    captured = {}
    _fake_httpx(monkeypatch, captured)
    body = _call_primary(captured, monkeypatch)
    assert body["model"] == "cfg-model"


def test_post_body_falls_back_to_env_model(monkeypatch, agent_cfg):
    monkeypatch.setenv(PRIMARY_ENV, "env-model")
    agent_cfg.write_cfg({"mode": "SHADOW"})
    captured = {}
    _fake_httpx(monkeypatch, captured)
    body = _call_primary(captured, monkeypatch)
    assert body["model"] == "env-model"


def test_research_analysis_carries_model_that_answered(monkeypatch, agent_cfg):
    """research.research stamps the RESOLVED model onto the analysis dict —
    the value the `Verdict:` log line and the `Trade result` row print — so
    the name always shows the model that actually spoke.

    Hermetic drive: every fetch/LLM/memory/news entry is stubbed; only the
    model resolver under test is left real (reading the written config).
    """
    from hermes_trader.agents import memory as mem_mod

    monkeypatch.setenv(PRIMARY_ENV, "env-model")
    agent_cfg.write_cfg({"llm": {"model": "cfg-model"}})

    perception = {
        "id": "test-perc", "coin": "TESTCOIN", "mid": 1.0,
        "composite_score": 50, "triggers": [],
    }

    import types as _types

    def _candles(n):
        return [_types.SimpleNamespace(t=i * 3600_000, o=1.0, h=1.1, l=0.9,
                                       c=1.0, v=100.0) for i in range(n)]

    monkeypatch.setattr(research, "fetch_hl_candles",
                        lambda c, i, n: _candles(max(n, 40)))
    monkeypatch.setattr(research, "_fetch_funding_rate", lambda c: "0.01")
    monkeypatch.setattr(research, "_fetch_news", lambda c: "no news")
    monkeypatch.setattr(research, "resolve_user_address", lambda: None)
    monkeypatch.setattr(research, "fetch_account_state", lambda *a, **k: {})
    monkeypatch.setattr(research, "_call_ai",
                        lambda *a, **k: ('{"verdict":"PASS","confidence":0.0}', None))
    monkeypatch.setattr(research.memory, "record_analysis", lambda a: None)
    monkeypatch.setattr(research.memory, "update_equity", lambda e: None)
    monkeypatch.setattr(research.memory, "get_win_rate",
                        lambda: {"rate": 0.5, "total": 10})

    out = research.research("TESTCOIN", perception)
    assert out["primary_model"] == "cfg-model"
    assert out["verdict"] == "PASS"


# ── fail-open ────────────────────────────────────────────────────────────

def test_fail_open_corrupt_config_model_falls_back(monkeypatch, agent_cfg):
    """A corrupt config must never break the LLM path: both model resolvers
    degrade to the env fallback (fail-open, same contract as the sampling
    helpers)."""
    monkeypatch.setenv(PRIMARY_ENV, "env-model")
    monkeypatch.setenv(DUEL_ENV, "duel-env")
    agent_cfg.write_cfg({"llm": {"model": "cfg-model"}})
    with open(cs.CONFIG_PATH, "w") as f:
        f.write("{this is not valid json")
    assert ds.effective_primary_model() == "env-model"
    assert ds.effective_duelist_model() == "duel-env"
    # And the full POST path still works end to end.
    captured = {}
    _fake_httpx(monkeypatch, captured)
    body = _call_primary(captured, monkeypatch)
    assert body["model"] == "env-model"
