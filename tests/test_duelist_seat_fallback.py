"""Seat-backup for the duelist (t_848eb1c4): `llm.duelist.fallback`.

Motivation is measured, not hypothetical: the live second voice's box (the
Spark seat on the lemonade A/B host) was unreachable from 2026-11-06 19:04 UTC
onward — `ConnectError: All connection attempts failed` on EVERY scan, ~3,000
identical WARNING lines, zero duel rows, and every scan still paid the dead
call's latency. A second voice that is only sometimes up is the normal case, so
the named seat is tried first and the backup second, with a failure breaker so
a known-dead seat stops being poked while a backup exists.

Pinned here (against local stub OpenAI servers, never a real box):
  * a healthy named seat answers and the backup is never called;
  * a dead/erroring seat falls through to the backup, with the failure reason
    in the meta the duel row carries;
  * EACH seat calls with ITS OWN max_tokens / sampling / chat_template_kwargs —
    the cross-seat contamination that made the 2026-11-07 flash flip silently
    lose every duel row (thinking-capable model + the other seat's template
    kwargs = 200 with empty content = no row);
  * `empty-content` is a distinct failure class from an outage;
  * the breaker trips after N consecutive failures and re-probes after the
    cooldown; a success resets it;
  * with NO fallback configured the breaker is inert (a single seat must never
    be skipped — that would drop data rather than move it);
  * both seats down = no text, no duel row, and the attribution says why.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import duel_store as ds  # noqa: E402

# An address nothing listens on: httpx fails fast (ConnectError) rather than
# burning a test's timeout, which is the same reason vocabulary the live seat
# produced.
DEAD_URL = "http://127.0.0.1:1/v1"

ANSWER = json.dumps({"verdict": "PASS", "confidence": 0.42})


class _Stub:
    """Minimal OpenAI-compatible stub; records every request body it serves.

    `replies[i]` serves request i ({"code": int, "json": {}}); the last entry
    repeats, so a stub can be "always 500" with one element.
    """

    def __init__(self, replies):
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("content-length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except json.JSONDecodeError:
                    body = {}
                outer.bodies.append(body)
                reply = outer.replies[min(len(outer.bodies), len(outer.replies)) - 1]
                raw = json.dumps(reply.get("json") or {}).encode()
                self.send_response(reply.get("code", 200))
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format, *args):  # silence default stderr spam
                pass

        self.bodies: list[dict] = []
        self.replies = replies
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _completion(text: str) -> dict:
    return {"json": {"choices": [{"message": {"content": text}}]}}


def _seat(base_url: str, model: str, **over) -> dict:
    """A call config shaped exactly like duelist_config()'s output."""
    cfg = {"seat": "primary", "model": model, "base_url": base_url,
           "api_key": "sk-test", "max_tokens": 4096,
           "sampling": {"temperature": 0.7}, "chat_template_kwargs": {}}
    cfg.update(over)
    return cfg


@pytest.fixture(autouse=True)
def _reset_breaker():
    """The breaker is module state — reset so test order can't decide trips."""
    ds._seat_failures.clear()
    ds._seat_cooling_until.clear()
    yield
    ds._seat_failures.clear()
    ds._seat_cooling_until.clear()


@pytest.fixture
def stubs():
    made: list[_Stub] = []

    def make(replies) -> _Stub:
        s = _Stub(replies)
        made.append(s)
        return s

    yield make
    for s in made:
        s.stop()


# ── Routing ────────────────────────────────────────────────────────────────

def test_healthy_named_seat_never_touches_the_backup(stubs):
    p, fb = stubs([_completion(ANSWER)]), stubs([_completion(ANSWER)])
    text, _ms, meta = ds.call_duelist_with_fallback(
        _seat(p.base_url, "voice-a"), _seat(fb.base_url, "voice-b", seat="fallback"),
        "sys", "user")
    assert json.loads(text)["confidence"] == 0.42
    assert meta["seat"] == "primary" and meta["answered_by"] == "voice-a"
    assert meta["primary_fail"] is None and meta["breaker_open"] is False
    assert fb.bodies == []  # the backup is a backup, not a second opinion


def test_dead_named_seat_falls_back_and_says_why(stubs):
    fb = stubs([_completion(ANSWER)])
    text, _ms, meta = ds.call_duelist_with_fallback(
        _seat(DEAD_URL, "voice-a"), _seat(fb.base_url, "voice-b", seat="fallback"),
        "sys", "user")
    assert text.strip() and meta["seat"] == "fallback"
    assert meta["answered_by"] == "voice-b"
    assert meta["primary_fail"].startswith("exception:")
    assert fb.bodies[0]["model"] == "voice-b"


def test_both_seats_down_produces_no_text_and_full_attribution(stubs):
    _text, _ms, meta = ds.call_duelist_with_fallback(
        _seat(DEAD_URL, "voice-a"), _seat(DEAD_URL, "voice-b", seat="fallback"),
        "sys", "user")
    assert _text == ""
    assert meta["seat"] is None and meta["answered_by"] is None
    assert meta["primary_fail"].startswith("exception:")
    assert meta["fallback_fail"].startswith("exception:")


def test_http_error_on_the_named_seat_falls_back(stubs):
    p, fb = stubs([{"code": 500, "json": {"error": "boom"}}]), stubs([_completion(ANSWER)])
    _t, _ms, meta = ds.call_duelist_with_fallback(
        _seat(p.base_url, "voice-a"), _seat(fb.base_url, "voice-b", seat="fallback"),
        "sys", "user")
    assert meta["primary_fail"] == "http-500" and meta["seat"] == "fallback"
    assert len(p.bodies) == 1  # one attempt, then the backup — no hammering


def test_empty_content_is_a_config_fault_not_an_outage(stubs):
    # 200 + choices + no content: the thinking-mode signature (answer lands in
    # reasoning_content). Its own reason class, because the fix is
    # chat_template_kwargs, not a restart — and it still falls back.
    p, fb = (stubs([{"json": {"choices": [{"message": {"content": ""}}]}}]),
             stubs([_completion(ANSWER)]))
    _t, _ms, meta = ds.call_duelist_with_fallback(
        _seat(p.base_url, "voice-a"), _seat(fb.base_url, "voice-b", seat="fallback"),
        "sys", "user")
    assert meta["primary_fail"] == "empty-content" and meta["seat"] == "fallback"


# ── Per-seat request bodies (the contamination this feature must never
#    reintroduce) ───────────────────────────────────────────────────────────

def test_each_seat_posts_its_own_params_never_the_other_seats(stubs):
    p, fb = stubs([{"code": 500}]), stubs([_completion(ANSWER)])
    ds.call_duelist_with_fallback(
        _seat(p.base_url, "voice-a", max_tokens=4096,
              sampling={"temperature": 0.7, "top_p": 0.8},
              chat_template_kwargs={"enable_thinking": False}),
        _seat(fb.base_url, "voice-b", seat="fallback", max_tokens=8192,
              sampling={"temperature": 0.9}, chat_template_kwargs={}),
        "sys", "user")
    pb, fbb = p.bodies[0], fb.bodies[0]
    assert pb["chat_template_kwargs"] == {"enable_thinking": False}
    assert pb["max_tokens"] == 4096 and pb["temperature"] == 0.7
    # The backup must NOT inherit the named seat's template overrides: sending
    # enable_thinking to a model that has no such template arg, or omitting it
    # from one that needs it, both cost the duel row.
    assert "chat_template_kwargs" not in fbb
    assert fbb["max_tokens"] == 8192 and fbb["temperature"] == 0.9
    assert "top_p" not in fbb  # the named seat's sampling keys stay its own


# ── Failure breaker ────────────────────────────────────────────────────────

def test_breaker_stops_poking_a_seat_that_keeps_failing(stubs, monkeypatch):
    # Breaker read from a stub instead of a config file: threshold 2, cooldown
    # long enough to outlast the test.
    monkeypatch.setattr(ds, "fallback_breaker_cfg",
                        lambda: {"threshold": 2, "cooldown_s": 3600.0})
    p, fb = stubs([{"code": 500}]), stubs([_completion(ANSWER)])
    for _ in range(2):
        meta = ds.call_duelist_with_fallback(
            _seat(p.base_url, "voice-a"), _seat(fb.base_url, "voice-b", seat="fallback"),
            "sys", "user")[2]
        assert meta["primary_fail"] == "http-500"
    assert len(p.bodies) == 2
    third = ds.call_duelist_with_fallback(
        _seat(p.base_url, "voice-a"), _seat(fb.base_url, "voice-b", seat="fallback"),
        "sys", "user")[2]
    assert third["breaker_open"] is True and third["primary_fail"] == "breaker-open"
    assert third["seat"] == "fallback"
    assert len(p.bodies) == 2  # the dead seat was skipped, not retried


def test_a_success_resets_the_breaker(stubs, monkeypatch):
    monkeypatch.setattr(ds, "fallback_breaker_cfg",
                        lambda: {"threshold": 2, "cooldown_s": 3600.0})
    p, fb = (stubs([{"code": 500}, _completion(ANSWER)]),
             stubs([_completion(ANSWER)]))
    ds.call_duelist_with_fallback(
        _seat(p.base_url, "voice-a"), _seat(fb.base_url, "voice-b", seat="fallback"),
        "sys", "user")
    assert p.bodies and fb.bodies  # one failure, backup used
    meta = ds.call_duelist_with_fallback(
        _seat(p.base_url, "voice-a"), _seat(fb.base_url, "voice-b", seat="fallback"),
        "sys", "user")[2]
    assert meta["seat"] == "primary"  # the seat recovered, breaker reset
    assert not ds._seat_failures and not ds._seat_cooling_until


def test_no_backup_means_the_breaker_never_skips_the_only_seat(stubs, monkeypatch):
    # Without a backup there is nothing to fall back to: skipping the seat
    # would DROP data instead of moving it, so the breaker is inert.
    monkeypatch.setattr(ds, "fallback_breaker_cfg",
                        lambda: {"threshold": 1, "cooldown_s": 3600.0})
    p = stubs([{"code": 500}])
    for _ in range(3):
        _t, _ms, meta = ds.call_duelist_with_fallback(
            _seat(p.base_url, "voice-a"), None, "sys", "user")
        assert meta["primary_fail"] == "http-500" and meta["breaker_open"] is False
    assert len(p.bodies) == 3
    assert not ds._seat_failures
