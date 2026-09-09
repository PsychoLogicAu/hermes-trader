"""P11 — per-slot, hot-configurable LLM sampling profiles (block shape).

The primary POST body used to hardcode ``"temperature": 0.1`` and the
duelist body spread the module constant ``DUELIST_SAMPLING_PROFILE``;
neither could follow a model swap. Both slots now merge a per-key
override from the agent config's ``llm.sampling`` /
``llm.duelist_sampling`` blocks (inside the same ``llm`` block as the
model names — settings travel with the model), read at CALL time (hot —
no module-level cache). Code defaults are today's exact values, so with
the block ABSENT both bodies are byte-identical to the pre-change ones
(the no-op guarantee, pinned here).

Config isolation: we target ``config_store.CONFIG_PATH`` DIRECTLY
(backup / write / restore the file contents) rather than the
HERMES_AGENT_CONFIG_FILE env var — a few test modules re-point the env
var at import time (their own isolation dirs), which is too late for
config_store's module-level path freeze, so the env var can lag the
file read_agent_config actually opens mid-suite (same approach as
tests/test_cleanup.py::test_timesfm_block_in_prompt_sync_render).
No network, no models: httpx.AsyncClient is stubbed with a FakeClient
whose fake_post captures the POST body (the proven capture pattern from
tests/test_duel_store.py).
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


# The sampling fields either slot may carry — used to assert "no extra
# sampling keys leaked into the body".
SAMPLING_KEYS = (
    "temperature", "top_p", "top_k", "min_p",
    "presence_penalty", "repetition_penalty", "seed",
)

# The duelist's pre-change on-the-wire sampling (the 6-key constant).
DUELIST_DEFAULTS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}


# ── httpx stub (mirrors tests/test_duel_store.py::_fake_httpx) ──────────

def _fake_httpx(monkeypatch, captured, content="ok"):
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
    """Drive research._async_do_call (the enclosing LLM-call function)
    hermetically and return the captured POST body."""
    loop = asyncio.new_event_loop()
    try:
        out = loop.run_until_complete(
            research._async_do_call("k", "http://x/v1", "m", "S", "U"))
    finally:
        loop.close()
    assert out == "ok"
    return captured["json"]


# ── config isolation (CONFIG_PATH direct — see module docstring) ────────

@pytest.fixture
def agent_cfg(monkeypatch):
    """Own the agent-config FILE for the duration of one test: write a
    fresh dict via ``agent_cfg.write_cfg(cfg)``; the previous contents
    (or absence) are always restored afterwards."""
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


def _sampling_fields(body):
    return {k: body[k] for k in SAMPLING_KEYS if k in body}


# ── primary slot ────────────────────────────────────────────────────────

def test_primary_body_equals_old_when_config_absent(monkeypatch, agent_cfg):
    """No-op guarantee (primary): with the `llm` block ABSENT the body's
    sampling fields are EXACTLY {temperature: 0.1} — the pre-change
    hardcoded literal, no extra sampling keys."""
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    agent_cfg.write_cfg({"mode": "LIVE"})  # config present, key ABSENT
    body = _call_primary(captured, monkeypatch)
    assert _sampling_fields(body) == {"temperature": 0.1}


def test_primary_body_reflects_custom_llm_sampling(monkeypatch, agent_cfg):
    """A custom `llm.sampling` block lands in the primary POST body."""
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    agent_cfg.write_cfg({"llm": {"sampling": {"temperature": 0.3, "top_p": 0.9}}})
    body = _call_primary(captured, monkeypatch)
    assert body["temperature"] == 0.3
    assert body["top_p"] == 0.9


def test_primary_partial_override_keeps_defaults(monkeypatch, agent_cfg):
    """Merge rule is per-key, not replace-whole-dict: a block naming only
    `top_p` keeps the default temperature 0.1."""
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    agent_cfg.write_cfg({"llm": {"sampling": {"top_p": 0.9}}})
    body = _call_primary(captured, monkeypatch)
    assert body["temperature"] == 0.1
    assert body["top_p"] == 0.9


def test_primary_hot_read_follows_config_between_calls(monkeypatch, agent_cfg):
    """No module-level cache: flipping the value in the SAME config file
    between two calls must be visible in the second call's body."""
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    agent_cfg.write_cfg({"llm": {"sampling": {"temperature": 0.2}}})
    body1 = _call_primary(captured, monkeypatch)
    assert body1["temperature"] == 0.2
    agent_cfg.write_cfg({"llm": {"sampling": {"temperature": 0.4, "top_k": 40}}})
    body2 = _call_primary(captured, monkeypatch)
    assert body2["temperature"] == 0.4
    assert body2["top_k"] == 40


# ── duelist slot ────────────────────────────────────────────────────────

def test_duelist_body_equals_old_when_config_absent(monkeypatch, agent_cfg):
    """No-op guarantee (duelist): with the `llm` block ABSENT the body
    carries the pre-change 6-key constant, exactly."""
    captured = {}
    _fake_httpx(monkeypatch, captured)
    agent_cfg.write_cfg({"mode": "LIVE"})  # config present, key ABSENT
    ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    body = captured["json"]
    assert _sampling_fields(body) == DUELIST_DEFAULTS


def test_duelist_partial_override_keeps_other_defaults(monkeypatch, agent_cfg):
    """`llm.duelist_sampling: {temperature: 0.8}` overrides ONLY temperature;
    the other 5 default keys are still present (per-key merge, not
    replace-whole-dict)."""
    captured = {}
    _fake_httpx(monkeypatch, captured)
    agent_cfg.write_cfg({"llm": {"duelist_sampling": {"temperature": 0.8}}})
    ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    body = captured["json"]
    assert body["temperature"] == 0.8
    for key in ("top_p", "top_k", "min_p", "presence_penalty",
                "repetition_penalty"):
        assert body[key] == DUELIST_DEFAULTS[key]


def test_duelist_hot_read_follows_config_between_calls(monkeypatch, agent_cfg):
    """No module-level cache: the second call sees the NEW value written
    to the same file between calls."""
    captured = {}
    _fake_httpx(monkeypatch, captured)
    agent_cfg.write_cfg({"llm": {"duelist_sampling": {"temperature": 0.5}}})
    ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    assert captured["json"]["temperature"] == 0.5
    agent_cfg.write_cfg({"llm": {"duelist_sampling": {"temperature": 0.7}}})
    ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    assert captured["json"]["temperature"] == 0.7


# ── fail-open ───────────────────────────────────────────────────────────

def test_fail_open_corrupt_config_uses_pure_defaults(monkeypatch, agent_cfg):
    """A corrupt config file must never break the LLM call path: both
    slots succeed and fall back to their pure code defaults."""
    captured_p, captured_d = {}, {}
    _fake_httpx(monkeypatch, captured_p, content="ok")
    agent_cfg.write_cfg({"llm": {"sampling": {"temperature": 0.9}}})  # not used
    with open(cs.CONFIG_PATH, "w") as f:
        f.write("{this is not valid json")
    # Primary: the corrupt file degrades to the pure default.
    body = _call_primary(captured_p, monkeypatch)
    assert _sampling_fields(body) == {"temperature": 0.1}
    # Duelist: same file, same guarantee (the previous write_cfg's valid
    # JSON is already gone — we rewrote the file corrupt above).
    _fake_httpx(monkeypatch, captured_d)
    ds.call_duelist("dk", "http://duel.test/v1", "m", "SYS", "USER")
    assert captured_d["json"] is not None
    assert _sampling_fields(captured_d["json"]) == DUELIST_DEFAULTS


def test_fail_open_config_read_raises(monkeypatch, agent_cfg):
    """Even if read_agent_config itself blows up (not just a missing or
    corrupt file — e.g. an I/O fault), the helpers fail open to the pure
    defaults and never raise into the LLM call path."""
    agent_cfg.write_cfg({"mode": "LIVE"})
    monkeypatch.setattr(cs, "read_agent_config",
                        lambda: (_ for _ in ()).throw(OSError("io fault")))
    assert research.effective_llm_sampling() == {"temperature": 0.1}
    assert ds.effective_duelist_sampling() == DUELIST_DEFAULTS
    # And the full POST path still works end to end.
    captured = {}
    _fake_httpx(monkeypatch, captured, content="ok")
    body = _call_primary(captured, monkeypatch)
    assert _sampling_fields(body) == {"temperature": 0.1}
