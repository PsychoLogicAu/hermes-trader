"""Tests for the zstd-chunked prompt/response archive (prompt_log.py).

Pins:
  * disabled default -> record_call writes NOTHING (the shipping state);
  * enabled -> records land in calls.jsonl, rotation compresses to one chunk,
    iter_records reads rotated + active transparently;
  * size-triggered and day-roll-triggered rotation; retention by age and by
    total-size cap (oldest pruned first);
  * index.jsonl manifests rotations/deletions;
  * record_call never raises (write faults degrade to at most one warning);
  * research._archive_record shape: duelist rows reference (no prompt text),
    truncation guards, parsed-verdict sub-dict.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from hermes_trader.agents import prompt_log  # noqa: E402


class _Cfg:
    def __init__(self, d): self._d = d
    def get(self, k, default=None): return self._d.get(k, default)


def _enable_cfg(monkeypatch, **block):
    """Enable via the real config path (module-level read_agent_config), so
    prompt_log._cfg() merges defaults normally."""
    cfg = {"enabled": True}
    cfg.update(block)
    monkeypatch.setattr(prompt_log, "read_agent_config",
                        lambda: _Cfg({"prompt_log": cfg}))


@pytest.fixture()
def archive_dir(tmp_path, monkeypatch):
    """Fresh archive dir + enabled config + reset module state per test."""
    d = tmp_path / "prompt-log"
    monkeypatch.setenv("HERMES_PROMPT_LOG_DIR", str(d))
    # Reset the module-level rotation state (backend cache is fine to keep).
    prompt_log._state.pop("dir", None)
    prompt_log._state.pop("bytes", None)
    prompt_log._state.pop("day", None)
    prompt_log._state.pop("warned", None)
    return d


def _enable(monkeypatch, **cfg):
    base = {"enabled": True}
    base.update(cfg)
    monkeypatch.setattr(prompt_log, "read_agent_config",
                        lambda: _Cfg({"prompt_log": base}), raising=False)


def _rec(i=0, **kw):
    r = {"ts": int(time.time() * 1000) + i, "coin": "BTC", "role": "primary",
         "model": "m", "perception_id": f"p{i}", "system_prompt": "sys",
         "user_message": f"user {i}", "raw_response": f"resp {i}",
         "parsed": {"verdict": "PASS"}, "wall_ms": 10, "server_ms": None}
    r.update(kw)
    return r


def _patch_read_cfg(monkeypatch, cfg=None):
    # iter_records/stats call _cfg() (for the dir default); point it at a
    # harmless config so they don't read whatever file conftest redirected.
    monkeypatch.setattr(prompt_log, "read_agent_config",
                        lambda: _Cfg(cfg or {}), raising=False)


def test_disabled_writes_nothing(tmp_path, monkeypatch):
    d = tmp_path / "pl"
    monkeypatch.setenv("HERMES_PROMPT_LOG_DIR", str(d))
    _patch_read_cfg(monkeypatch, {})  # absent block -> disabled
    prompt_log.record_call(_rec())
    assert not d.exists() or not list(d.iterdir())


def test_enabled_appends_active_chunk(tmp_path, monkeypatch):
    d = tmp_path / "pl"
    monkeypatch.setenv("HERMES_PROMPT_LOG_DIR", str(d))
    _enable(monkeypatch)
    prompt_log.record_call(_rec(1))
    prompt_log.record_call(_rec(2))
    lines = (d / "calls.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["perception_id"] == "p1"


def test_size_rotation_compresses_and_truncates(tmp_path, monkeypatch):
    d = tmp_path / "pl"
    monkeypatch.setenv("HERMES_PROMPT_LOG_DIR", str(d))
    # Rotate when the active chunk exceeds 0.0001 MB (~105 bytes) — every
    # append after the first trips it.
    _enable(monkeypatch, max_chunk_raw_mb=0.0001)
    prompt_log.record_call(_rec(1))
    prompt_log.record_call(_rec(2))  # rotation happens BEFORE this append
    chunks = [p for p in d.iterdir() if p.name.startswith("calls-")]
    assert len(chunks) == 1
    assert chunks[0].name.endswith((".jsonl.zst", ".jsonl.gz"))
    blob = chunks[0].read_bytes()
    raw = prompt_log.decompress_blob(blob)
    assert b"p1" in raw                      # record 1 is inside the chunk
    active = (d / "calls.jsonl").read_text().splitlines()
    assert len(active) == 1 and "p2" in active[0]
    # index manifest recorded the rotation
    idx = [json.loads(l) for l in (d / "index.jsonl").read_text().splitlines()]
    rot = [e for e in idx if e["event"] == "rotated"]
    assert rot and rot[0]["chunk"] == chunks[0].name


def test_iter_records_reads_chunks_then_active(tmp_path, monkeypatch):
    d = tmp_path / "pl"
    monkeypatch.setenv("HERMES_PROMPT_LOG_DIR", str(d))
    _enable(monkeypatch)
    for i in range(1, 4):
        prompt_log.record_call(_rec(i))
    # Force one rotation manually.
    prompt_log._rotate(str(d), prompt_log._cfg())
    prompt_log.record_call(_rec(9))
    ids = [r["perception_id"] for r in prompt_log.iter_records(str(d))]
    assert ids == ["p1", "p2", "p3", "p9"]


def test_retention_days_prunes_oldest(tmp_path, monkeypatch):
    d = tmp_path / "pl"
    monkeypatch.setenv("HERMES_PROMPT_LOG_DIR", str(d))
    _enable(monkeypatch)
    prompt_log.record_call(_rec(1))
    prompt_log._rotate(str(d), prompt_log._cfg())
    chunks = [p for p in d.iterdir() if p.name.startswith("calls-")]
    assert len(chunks) == 1
    old = time.time() - 40 * 86400
    os.utime(chunks[0], (old, old))
    prompt_log._enforce_retention(str(d), {"retention_days": 30,
                                           "max_total_stored_mb": 0})
    assert not chunks[0].exists()
    idx = [json.loads(l) for l in (d / "index.jsonl").read_text().splitlines()]
    assert any(e["event"] == "deleted" and e["why"] == "retention_days" for e in idx)


def test_size_cap_prunes_oldest_first(tmp_path, monkeypatch):
    d = tmp_path / "pl"
    monkeypatch.setenv("HERMES_PROMPT_LOG_DIR", str(d))
    _enable(monkeypatch)
    prompt_log.record_call(_rec(1))
    prompt_log._rotate(str(d), prompt_log._cfg())
    prompt_log.record_call(_rec(2))
    prompt_log._rotate(str(d), prompt_log._cfg())
    chunks = sorted(p for p in d.iterdir() if p.name.startswith("calls-"))
    assert len(chunks) == 2
    real_bytes = sum(p.stat().st_size for p in chunks)
    newest = chunks[-1].stat().st_size
    # Cap between ONE chunk and the total -> only the newest survives.
    cfg = {"retention_days": 0,
           "max_total_stored_mb": (newest + 50) / (1024 * 1024)}
    assert real_bytes > newest + 50  # sanity: cap really excludes both chunks
    prompt_log._enforce_retention(str(d), cfg)
    remaining = sorted(p.name for p in d.iterdir() if p.name.startswith("calls-"))
    assert chunks[0].name not in remaining
    assert chunks[-1].name in remaining


def test_record_call_never_raises(tmp_path, monkeypatch):
    # Point the dir at an UNCREATABLE path (file exists where a dir must be).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setenv("HERMES_PROMPT_LOG_DIR", str(blocker / "sub"))
    _enable(monkeypatch)
    prompt_log.record_call(_rec())  # must not raise


def test_stats(tmp_path, monkeypatch):
    d = tmp_path / "pl"
    monkeypatch.setenv("HERMES_PROMPT_LOG_DIR", str(d))
    _enable(monkeypatch)
    _patch_read_cfg(monkeypatch, {"prompt_log": {"enabled": True}})
    prompt_log.record_call(_rec(1))
    s = prompt_log.stats(str(d))
    assert s["active_records"] == 1
    assert s["enabled"] is True


# ── research._archive_record shape ─────────────────────────────────────────

def test_archive_record_shapes(monkeypatch):
    from hermes_trader.agents import research as R

    class _C:
        def get(self, k, default=None):
            return {"max_prompt_chars": 50, "max_response_chars": 30}.get(k, default)
    monkeypatch.setattr(prompt_log, "_cfg", lambda: _C())
    # research reads read_agent_config for mode — give it a benign one.
    monkeypatch.setattr(R, "read_agent_config", lambda: {"mode": "LIVE"})

    parsed = {"verdict": "LONG", "confidence": 0.8, "side": "long",
              "entry_px": 1.0, "stop_px": 0.9, "tp_px": 1.2,
              "reasoning": "r" * 3000, "ai_down": False,
              "close_guard_downgraded": False}
    rec = R._archive_record(coin="BTC", perception={"id": "p1", "mid": 1.0,
                                                    "composite_score": 42},
                            role="primary", model="m", system_prompt="s" * 100,
                            user_message="u", raw_response="x" * 100,
                            parsed=parsed, wall_ms=5, server_ms=3,
                            prior_block_injected=True)
    assert rec["prior_block_injected"] is True
    assert rec["system_prompt"].endswith("[TRUNCATED]")
    assert len(rec["system_prompt"]) < 100
    assert rec["raw_response"].endswith("[TRUNCATED]")
    assert rec["parsed"]["verdict"] == "LONG"
    assert len(rec["parsed"]["reasoning"]) <= 2012  # 2000 + marker

    dl = R._archive_record(coin="BTC", perception={"id": "p1"}, role="duelist",
                           model="d", system_prompt=None, user_message=None,
                           raw_response="resp", parsed=parsed)
    assert dl["system_prompt"] is None and dl["user_message"] is None
    assert dl["raw_response"] == "resp"
