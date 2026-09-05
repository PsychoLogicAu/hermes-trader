"""Tests for the deterministic CLOSE guard (2026-09-04, xyz:HOOD).

The research prompt lists the WHOLE open book ("Open positions: ADA long,
PURR long"), so a small model can reason about OTHER held positions and emit
a CLOSE verdict for the coin it's actually being researched — the Qwen2.5-
Coder-7B duelist did exactly this on xyz:HOOD (book = ADA + PURR, HOOD not
held): its Reasoning 5 was about ADA/PURR and it returned
{"verdict":"CLOSE","reasoning":"Structure flipped against current positions"}.

A CLOSE for a coin the account does NOT hold is by definition a misread of
another position's book state; the only thing the plumbing could do with it
is close_fn(coin) → `noop: already_flat`. `parse_verdict` now coerces such a
CLOSE to PASS when a `held_coins` book is supplied, tagging the result
`close_guard_downgraded=True`. `held_coins=None` disables the guard so
backtest / legacy callers keep the raw token (no replay-semantics shift).

Covers:
1. Guard arms: CLOSE on an unheld coin → PASS + downgraded flag.
2. Guard does NOT touch a legitimate CLOSE on a HELD coin.
3. Guard is off by default (held_coins=None) → raw CLOSE preserved.
4. The regex-fallback CLOSE path is also caught by the guard.
5. Non-CLOSE verdicts pass through untouched.
6. The flag rides into the returned dict and is False otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents.research import parse_verdict  # noqa: E402


CLOSE_TEXT = (
    '{"verdict":"CLOSE","confidence":0.9,"side":"null",'
    '"entryPx":null,"stopPx":null,"tpPx":null,'
    '"reasoning":"Structure flipped against current positions"}'
)
PERCEPTION = {"mid": 110.0}


# ── Guard arms: CLOSE on an unheld coin → PASS + flag ──────────────────────

def test_close_on_unheld_coin_downgraded_to_pass():
    # The literal 2026-09-04 HOOD case: book has ADA + PURR, coin is HOOD.
    p = parse_verdict(CLOSE_TEXT, "xyz:HOOD", PERCEPTION, held_coins={"ADA", "PURR"})
    assert p["verdict"] == "PASS"
    assert p["close_guard_downgraded"] is True
    # Confidence/reasoning survive — this is a downgraded CLOSE, not a
    # failure-PASS (ai_down only fires on empty ai_text).
    assert p["ai_down"] is False
    assert p["confidence"] == 0.9
    assert "Structure flipped" in p["reasoning"]


def test_close_on_unheld_coin_empty_book():
    # Empty book (no positions) — any CLOSE is a misread.
    p = parse_verdict(CLOSE_TEXT, "BTC", PERCEPTION, held_coins=set())
    assert p["verdict"] == "PASS"
    assert p["close_guard_downgraded"] is True


# ── Guard does NOT touch a legitimate CLOSE on a held coin ─────────────────

def test_close_on_held_coin_preserved():
    p = parse_verdict(CLOSE_TEXT, "ADA", PERCEPTION, held_coins={"ADA", "PURR"})
    assert p["verdict"] == "CLOSE"
    assert p["close_guard_downgraded"] is False
    assert p["confidence"] == 0.9


def test_close_on_held_hip3_coin_preserved():
    # HIP-3 coins carry the <dex>:<coin> prefix in BOTH the book and the
    # research coin (fetch_account_state normalizes to that form), so the
    # exact-set check must preserve a CLOSE on a held HIP-3 coin.
    p = parse_verdict(CLOSE_TEXT, "xyz:HOOD", PERCEPTION, held_coins={"xyz:HOOD", "ADA"})
    assert p["verdict"] == "CLOSE"
    assert p["close_guard_downgraded"] is False


# ── Guard is off by default (legacy / backtest semantics unchanged) ────────

def test_close_preserved_when_guard_disabled():
    # No held_coins → raw token preserved, flag False.
    p = parse_verdict(CLOSE_TEXT, "xyz:HOOD", PERCEPTION)
    assert p["verdict"] == "CLOSE"
    assert p["close_guard_downgraded"] is False


def test_close_preserved_when_held_coins_none_explicit():
    p = parse_verdict(CLOSE_TEXT, "xyz:HOOD", PERCEPTION, held_coins=None)
    assert p["verdict"] == "CLOSE"
    assert p["close_guard_downgraded"] is False


# ── Regex-fallback CLOSE path is also caught ────────────────────────────────

def test_close_regex_fallback_downgraded():
    # A malformed object that still matches the "verdict" scan ({"verdict":
    # CLOSE} — CLOSE unquoted, so json.loads fails) → the JSONDecodeError
    # branch derives CLOSE from the first line. The guard runs AFTER both
    # parse paths, so it catches this too.
    p = parse_verdict(
        "CLOSE\n{\"verdict\": CLOSE}", "xyz:HOOD", PERCEPTION, held_coins={"ADA"}
    )
    assert p["verdict"] == "PASS"
    assert p["close_guard_downgraded"] is True


# ── Non-CLOSE verdicts pass through untouched ───────────────────────────────

def test_pass_untouched():
    p = parse_verdict('{"verdict":"PASS","confidence":0.0}', "xyz:HOOD",
                      PERCEPTION, held_coins={"ADA", "PURR"})
    assert p["verdict"] == "PASS"
    assert p["close_guard_downgraded"] is False


def test_long_untouched():
    p = parse_verdict('{"verdict":"LONG","confidence":0.78,"side":"long"}',
                      "xyz:HOOD", PERCEPTION, held_coins={"ADA", "PURR"})
    assert p["verdict"] == "LONG"
    assert p["side"] == "long"
    assert p["close_guard_downgraded"] is False


def test_short_untouched():
    p = parse_verdict('{"verdict":"SHORT","confidence":0.7,"side":"short"}',
                      "ADA", PERCEPTION, held_coins={"ADA", "PURR"})
    assert p["verdict"] == "SHORT"
    assert p["side"] == "short"
    assert p["close_guard_downgraded"] is False


def test_veto_untouched():
    p = parse_verdict('{"verdict":"VETO","confidence":0.9}', "xyz:HOOD",
                      PERCEPTION, held_coins={"ADA"})
    assert p["verdict"] == "VETO"
    assert p["close_guard_downgraded"] is False


# ── Case-insensitivity of the raw token ────────────────────────────────────

def test_lowercase_close_downgraded():
    p = parse_verdict('{"verdict":"close","confidence":0.9}', "xyz:HOOD",
                      PERCEPTION, held_coins={"ADA"})
    assert p["verdict"] == "PASS"
    assert p["close_guard_downgraded"] is True


def test_lowercase_close_on_held_preserved():
    p = parse_verdict('{"verdict":"close","confidence":0.9}', "ADA",
                      PERCEPTION, held_coins={"ADA"})
    assert p["verdict"] == "CLOSE"
    assert p["close_guard_downgraded"] is False