"""t_f0c07370: signed over-extension WOULD-VETO accrual (shadow posture).

Mirror evidence (2026-10-30..11-12 cohort, live floors long 0.50 / short 0.60,
trap 0.55, BN2): ext>=10% n=307 -29.67; ext>=10 & conf>0.62 n=221 -37.03 (51
stops); ext>=10 & conf<=0.62 n=86 +7.36 — the conf leg is what separates the
poison cell from a profitable one, so the rule is conf-conditioned. The ATR leg
is NOT a gate (stop_atr1<1.0 OR atr1>=5% alone = +17.55). Both prompt fixes
against this cohort failed (arm A -22.71, arm B -14.86), which is why the
surface is post-verdict.

Modes: off (default, inert) / shadow (pass + would-block marker) / live (block).
Pins: sign convention (short mirrors), the conf bar, fail-safe on data gaps in
BOTH modes, malformed knobs, join vars on every decision while armed, the
`eval_all_gates` registration, executor wiring by AST, and the shipped posture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from hermes_trader.agents.risk_gates import (  # noqa: E402
    GateContext,
    eval_all_gates,
    overextension_gate,
)

REPO = Path(__file__).resolve().parent.parent
CFG_SHADOW = {"mode": "shadow"}
CFG_LIVE = {"mode": "live"}


def _ctx(trade_side: str = "long", daily_move_pct: object = 12.0,
         confidence: float = 0.72, mover: bool = True,
         entry_class: "str | None" = None) -> GateContext:
    return GateContext(
        confidence=confidence,
        current_positions=[],
        trade_notional_usd=10.0,
        daily_pnl=0.0,
        market_volume_24h_usd=1_000_000_000.0,
        coin="TEST",
        trade_side=trade_side,
        has_binary_news_risk=False,
        equity=100.0,
        total_open_notional=0.0,
        composite_score=35.0,
        momentum_burst_fired=False,
        daily_move_pct=cast("float | None", daily_move_pct),
        daily_mover_fired=mover,
        entry_class=entry_class,
    )


# ── (a) default posture: inert, and byte-identical (no join vars) --------------
def test_off_by_default_and_inert():
    for cfg in ({}, {"mode": "off"}):
        r = overextension_gate(_ctx(), cfg)
        assert r["pass"] is True and r["reason"] == "overextension: disabled"
        # knob-off must not grow the serialized result at all
        assert "signed_ext_pct" not in r and "overextension_trip" not in r


# ── (b) shadow trip: passes, marks, and carries the join vars ------------------
def test_shadow_trips_with_join_vars():
    r = overextension_gate(_ctx(daily_move_pct=12.0, confidence=0.72), CFG_SHADOW)
    assert r["pass"] is True
    assert r["shadow_would_block"] is True
    assert r["signed_ext_pct"] == pytest.approx(12.0)
    assert r["ext_pct"] == 10.0 and r["conf_min"] == 0.62
    assert r["daily_mover_fired"] is True and r["overextension_trip"] is True
    assert "LONG already moved +12.0%" in r["reason"]
    assert "mover_fired=true" in r["reason"]


# ── (c) live mode blocks the same cell ----------------------------------------
def test_live_mode_blocks():
    r = overextension_gate(_ctx(daily_move_pct=11.0, confidence=0.70), CFG_LIVE)
    assert r["pass"] is False and "shadow_would_block" not in r
    assert r["reason"].startswith("overextension: LONG")


# ── (d) sign convention: the SHORT side mirrors -------------------------------
def test_short_mirrors_the_sign():
    # a -12% mover is extension for a SHORT, not for a LONG
    assert overextension_gate(_ctx("short", daily_move_pct=-12.0),
                              CFG_SHADOW)["shadow_would_block"] is True
    assert overextension_gate(_ctx("long", daily_move_pct=-12.0),
                              CFG_SHADOW).get("shadow_would_block") is None
    # and a +12% mover must NOT trip a short
    r = overextension_gate(_ctx("short", daily_move_pct=12.0), CFG_SHADOW)
    assert r["pass"] is True and r["signed_ext_pct"] == pytest.approx(-12.0)
    assert r["overextension_trip"] is False


# ── (e) the conf leg is load-bearing (the +7.36 cell stays) -------------------
def test_conf_at_or_below_bar_does_not_trip():
    for conf in (0.55, 0.62):
        r = overextension_gate(_ctx(daily_move_pct=15.0, confidence=conf),
                               CFG_LIVE)
        assert r["pass"] is True, f"conf {conf} must not trip conf_min 0.62"
        assert r["overextension_trip"] is False


# ── (f) below the extension bar: no trip, join vars still ride ----------------
def test_below_bar_passes_and_still_reports():
    r = overextension_gate(_ctx(daily_move_pct=6.0, confidence=0.75), CFG_SHADOW)
    assert r["pass"] is True and r.get("shadow_would_block") is None
    assert r["signed_ext_pct"] == pytest.approx(6.0)
    assert r["overextension_trip"] is False


# ── (g) knobs are honoured ----------------------------------------------------
def test_knobs_move_the_bars():
    r = overextension_gate(_ctx(daily_move_pct=8.0, confidence=0.66),
                           {"mode": "live", "ext_pct": 7.5, "conf_min": 0.65})
    assert r["pass"] is False
    assert r["ext_pct"] == 7.5 and r["conf_min"] == 0.65


# ── (h) fail-safe: a data gap can never block, in EITHER mode -----------------
def test_missing_move_inert_in_both_modes():
    for cfg in (CFG_SHADOW, CFG_LIVE):
        for bad in (None, "n/a", float("nan")):
            r = overextension_gate(_ctx(daily_move_pct=bad), cfg)
            assert r["pass"] is True, f"{bad!r} must never block at {cfg}"
        assert r["signed_ext_pct"] is None
        assert r["overextension_trip"] is False


def test_malformed_knobs_fall_back_to_defaults():
    r = overextension_gate(_ctx(daily_move_pct=12.0, confidence=0.72),
                           {"mode": "live", "ext_pct": "wide", "conf_min": None})
    assert r["pass"] is False
    assert r["ext_pct"] == 10.0 and r["conf_min"] == pytest.approx(0.62)


def test_nan_bars_fail_closed_to_defaults():
    # float("NaN") parses cleanly and every comparison against it is False, so
    # without the guard a NaN ext_pct would make the gate silently never trip.
    nan = float("nan")
    r = overextension_gate(_ctx(daily_move_pct=12.0, confidence=0.72),
                           {"mode": "live", "ext_pct": nan, "conf_min": nan})
    assert r["pass"] is False, "NaN bars must fall back to the documented bars"
    assert r["ext_pct"] == 10.0 and r["conf_min"] == pytest.approx(0.62)
    assert r["overextension_trip"] is True


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), "Infinity",
                                 "inf", "-Infinity"])
def test_infinite_bars_fail_closed_to_defaults(bad):
    # Reviewer finding L1 (2026-11-20): the NaN-only guard left this open —
    # float("Infinity") parses, `inf != inf` is False, and EVERY comparison
    # against Infinity is False, so an Infinity bar silently never trips, even
    # at mode=live. json.loads also accepts a bare Infinity literal.
    r = overextension_gate(_ctx(daily_move_pct=999.0, confidence=0.99),
                           {"mode": "live", "ext_pct": bad, "conf_min": bad})
    assert r["pass"] is False, f"bar {bad!r} must not silently disarm the gate"
    assert r["ext_pct"] == 10.0 and r["conf_min"] == pytest.approx(0.62)
    assert r["overextension_trip"] is True


def test_infinite_move_is_a_data_gap():
    for cfg in (CFG_SHADOW, CFG_LIVE):
        r = overextension_gate(_ctx(daily_move_pct=float("inf")), cfg)
        assert r["pass"] is True and r["signed_ext_pct"] is None


def test_nan_confidence_never_fabricates_a_block():
    # Accrual loses one row; it must not manufacture a veto.
    r = overextension_gate(_ctx(daily_move_pct=12.0, confidence=float("nan")),
                           CFG_LIVE)
    assert r["pass"] is True and r["overextension_trip"] is False


@pytest.mark.parametrize("bad_conf", ["high", None, [], float("nan")])
def test_conf_confusion_never_raises_out_of_the_gate(bad_conf):
    # S1: eval_all_gates has NO per-gate try/except, so a raise here would drop
    # the whole decision in trading_loop — blocking, while in shadow mode.
    r = overextension_gate(_ctx(daily_move_pct=12.0, confidence=bad_conf),
                           CFG_SHADOW)
    assert r["pass"] is True and r.get("shadow_would_block") is None


def test_missing_composite_score_does_not_break_the_trip_reason():
    # S2: the reason string formats the score on the TRIP path only.
    ctx = _ctx(daily_move_pct=12.0, confidence=0.72)
    ctx.composite_score = cast(float, None)  # deliberately malformed producer
    r = overextension_gate(ctx, CFG_SHADOW)
    assert r["shadow_would_block"] is True and "score 0," in r["reason"]


def test_non_dict_config_block_is_inert_not_fatal():
    # S3: a config typo (`"overextension_gate": "shadow"`) must not raise.
    for bad in ("shadow", 5, None, []):
        r = overextension_gate(_ctx(daily_move_pct=12.0),
                               cast("Dict[str, Any]", bad))
        assert r["pass"] is True and r["reason"] == "overextension: disabled"


# ── (i) registered in the classic pipeline, inert without a config block ------
def test_eval_all_gates_carries_the_key():
    out = eval_all_gates(_ctx(daily_move_pct=3.0), {})
    assert "overextension" in out["results"]
    assert out["results"]["overextension"]["pass"] is True
    assert out["results"]["overextension"]["reason"] == "overextension: disabled"


def test_eval_all_gates_fires_when_configured_shadow():
    out = eval_all_gates(_ctx(daily_move_pct=13.0, confidence=0.74),
                         {"overextension_gate": {"mode": "shadow"}})
    res = out["results"]["overextension"]
    assert res["pass"] is True and res["shadow_would_block"] is True


# ── (j) executor wiring pinned by AST/source (research path can't be unit-run) -
def test_executor_shadow_line_wired():
    src = (REPO / "hermes_trader" / "agents" / "executor.py").read_text()
    assert 'gate_output["results"].get("overextension")' in src
    assert "[gate][OVEREXTENSION][SHADOW] WOULD HAVE BLOCKED" in src
    # the accrual marker family contract: the join keys off this exact phrase
    assert src.count("[gate][OVEREXTENSION][SHADOW] WOULD HAVE BLOCKED") == 1


def test_gate_registered_in_eval_all_gates_source():
    import ast
    src = (REPO / "hermes_trader" / "agents" / "risk_gates.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "eval_all_gates")
    subs = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)}
    assert "overextension_gate" in subs
    # def + exactly one call site — no second surface accruing duplicate lines
    assert src.count("overextension_gate(") == 2


# ── (k) shipped config posture: accrual, not blocking -------------------------
def test_live_config_ships_gate_in_shadow_mode():
    # The live repo-root config is deploy-only (gitignored), so an absent file
    # (a bare CI checkout) skips rather than fails; when it IS present the
    # posture must be shadow — an unreviewed flip to live goes red here.
    p = REPO / ".agent-config.json"
    if not p.exists():
        pytest.skip("live .agent-config.json not deployed in this checkout")
    block = json.loads(p.read_text()).get("overextension_gate")
    if block is None:
        pytest.skip("block not deployed yet (inert against the running image)")
    assert block.get("mode") == "shadow", block
