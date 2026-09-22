"""C.6 hard bar: `counter_regime_no_composite_escape` (LIVE ON 2026-09-21).

For trades genuinely fighting the TREND regime (regime up/down and not aligned)
the composite-score escape is closed — only confidence clears the gate. The
cohort it was built from (n=14, net −$54.83, .hermes/WATCHLIST.md §C.6) entered
through the score hatch on fresh-breakout hype scans (MET 09-07 comp 89.5 →
−$19.53 max_loss).

Pinned semantics:
  flag OFF (code default)  -> historical behaviour, byte-identical
  flag ON + counter-trend  -> composite >= 50/60 no longer passes; conf still does
  flag ON + aligned        -> UNAFFECTED (passes regardless, returns earlier)
  flag ON + neutral regime -> against-FUNDING trades KEEP the score hatch
                              (hard bar is scoped to trend disagreement)
  every block logs a loud `[gate][COUNTER-TREND-BLOCK]` WARNING whose field
  order the §C.6 review recipe greps — pinned here so refactors can't break it.
"""
import logging

from hermes_trader.agents import hyperfeed, market_regime
from hermes_trader.agents.risk_gates import GateContext, market_regime_gate


def _ctx(confidence=0.82, trade_side="long", coin="MET", composite_score=89.5,
         momentum_burst_fired=False, slow_burn_fired=False, whale_signal_fired=False):
    return GateContext(
        confidence=confidence, current_positions=[], trade_notional_usd=30.0,
        daily_pnl=0.0, market_volume_24h_usd=1e8, coin=coin,
        trade_side=trade_side, has_binary_news_risk=False, equity=1000.0,
        total_open_notional=0.0, composite_score=composite_score,
        momentum_burst_fired=momentum_burst_fired,
        slow_burn_fired=slow_burn_fired, whale_signal_fired=whale_signal_fired)


def _stub(monkeypatch, regime="down", funding="NEUTRAL"):
    monkeypatch.setattr(market_regime, "detect_regime", lambda c: regime)
    monkeypatch.setattr(hyperfeed, "market_get_funding_regime",
                        lambda: {"regime": funding,
                                 "regimes_by_class": {"crypto": funding}})
    monkeypatch.setattr(market_regime, "classify_asset", lambda c: "crypto")


# ── flag OFF = historical behaviour ──────────────────────────────────────────

def test_flag_off_counter_trend_high_composite_still_passes(monkeypatch):
    """MET 09-07 shape: regime down, conf 0.82 < bar, comp 89.5 ≥ 50 → via=composite."""
    _stub(monkeypatch, "down")
    r = market_regime_gate(_ctx(), counter_regime_min_conf=0.85)
    assert r["pass"] is True and r["via"] == "composite"


# ── flag ON closes the score hatch on counter-TREND entries ─────────────────

def test_flag_on_blocks_the_cohort_shape(monkeypatch):
    """Same MET shape, hard bar ON → blocked despite comp 89.5."""
    _stub(monkeypatch, "down")
    r = market_regime_gate(_ctx(), counter_regime_min_conf=0.85,
                           no_composite_escape=True)
    assert r["pass"] is False
    assert r["counter_trend"] is True
    # reason must not advertise the closed hatch
    assert "score >=" not in r["reason"]


def test_flag_on_blocks_soph_2_shape(monkeypatch):
    """SOPH-2 09-08 shape: conf 0.82, comp 60.2 — blocked with hard bar ON."""
    _stub(monkeypatch, "down")
    r = market_regime_gate(_ctx(composite_score=60.2), counter_regime_min_conf=0.85,
                           no_composite_escape=True)
    assert r["pass"] is False


def test_flag_on_confidence_path_still_opens(monkeypatch):
    """INJ 09-07 shape: conf 0.85 == bar 0.85 → passes via confidence even ON."""
    _stub(monkeypatch, "down")
    r = market_regime_gate(_ctx(confidence=0.85, composite_score=30.3),
                           counter_regime_min_conf=0.85, no_composite_escape=True)
    assert r["pass"] is True and r["via"] == "confidence"


def test_flag_on_aligned_unaffected(monkeypatch):
    """Regime up + long: easy pass regardless of conf/score even with flag ON."""
    _stub(monkeypatch, "up")
    r = market_regime_gate(_ctx(confidence=0.1, composite_score=0),
                           no_composite_escape=True)
    assert r["pass"] is True and r["via"] == "aligned"


def test_flag_on_neutral_regime_against_funding_keeps_score_hatch(monkeypatch):
    """Neutral TREND regime + against SHORT_CROWDED long with comp ≥ 60:
    historical own-signal discipline keeps the score escape (hard bar is scoped
    to trend disagreement — the §C.6 cohort is all regime up/down)."""
    _stub(monkeypatch, "neutral", funding="SHORT_CROWDED")
    r = market_regime_gate(_ctx(confidence=0.52, composite_score=70.0),
                           counter_regime_min_conf=0.8, no_composite_escape=True)
    assert r["pass"] is True and r["via"] == "composite"


def test_flag_on_counter_trend_short_in_up_regime_blocked(monkeypatch):
    """Symmetry: regime up + SHORT conf 0.82 comp 70 → blocked with flag ON."""
    _stub(monkeypatch, "up")
    r = market_regime_gate(_ctx(trade_side="short", composite_score=70.0),
                           counter_regime_min_conf=0.85, no_composite_escape=True)
    assert r["pass"] is False and r["counter_trend"] is True


# ── loud block line (the review recipe greps this) ───────────────────────────

def test_block_emits_loud_greppable_warning(monkeypatch, caplog):
    _stub(monkeypatch, "down")
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        market_regime_gate(_ctx(coin="MET"), counter_regime_min_conf=0.85,
                           no_composite_escape=True)
    lines = [rec.getMessage() for rec in caplog.records
             if "[gate][COUNTER-TREND-BLOCK]" in rec.getMessage()]
    assert len(lines) == 1
    msg = lines[0]
    # stable field order for the §C.6 review grep
    assert "MET long conf 0.82 score 89.5 regime down funding NEUTRAL " \
           "bar 0.85 composite_escape OFF" in msg


def test_no_warning_on_pass(monkeypatch, caplog):
    _stub(monkeypatch, "down")
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        market_regime_gate(_ctx(confidence=0.9), counter_regime_min_conf=0.8,
                           no_composite_escape=True)
    assert not [r for r in caplog.records if "COUNTER-TREND-BLOCK" in r.getMessage()]


# ── wiring: eval_all_gates passes the config key through ─────────────────────

def test_eval_all_gates_reads_config_key(monkeypatch):
    """The live call site must forward counter_regime_no_composite_escape."""
    from hermes_trader.agents import risk_gates as rg

    captured = {}
    real = rg.market_regime_gate

    def spy(ctx, *a, **kw):
        captured["args"] = a
        captured["kw"] = kw
        return real(ctx, *a, **kw)

    monkeypatch.setattr(rg, "market_regime_gate", spy)
    _stub(monkeypatch, "down")
    ctx = _ctx()
    cfg = {"counter_regime_min_conf": 0.85,
           "block_counter_trend_bypass": True,
           "crowded_with_min_conf": 0.78,
           "counter_regime_no_composite_escape": True}
    try:
        rg.eval_all_gates(ctx, config=cfg)
    except Exception:
        pass  # other gates may need more ctx; we only assert the forwarding
    assert captured.get("kw", {}).get("no_composite_escape") is True
