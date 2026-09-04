"""correlation_cap scoping (2026-09-04): the majors cap binds only MAJOR
candidates. A non-major long must pass while 2 majors are already held; a
third major must still be rejected. Regression for the over-broad gate that
rejected every new long (ZEC/LIT/CASHCAT/AZTEC/ENA/WLD) whenever the majors
book hit max_crypto_long_correlated."""


def _ctx(**kw):
    from hermes_trader.agents.risk_gates import GateContext
    base = dict(confidence=0.9, current_positions=[], trade_notional_usd=50,
                daily_pnl=0, market_volume_24h_usd=1e8, coin="BTC",
                trade_side="long", has_binary_news_risk=False, equity=1000,
                total_open_notional=0)
    base.update(kw)
    return GateContext(**base)


_TWO_MAJOR_LONGS = [
    {"coin": "UNI", "side": "long"},
    {"coin": "ADA", "side": "long"},
]


def test_non_major_long_passes_while_two_majors_held():
    from hermes_trader.agents.risk_gates import correlation_cap
    for coin in ("ZEC", "LIT", "CASHCAT", "AZTEC", "ENA", "WLD"):
        res = correlation_cap(_ctx(coin=coin, current_positions=list(_TWO_MAJOR_LONGS)), 2)
        assert res["pass"] is True, f"{coin} must not be blocked by the majors cap: {res}"


def test_major_long_still_blocked_at_cap():
    from hermes_trader.agents.risk_gates import correlation_cap
    res = correlation_cap(_ctx(coin="DOGE", current_positions=list(_TWO_MAJOR_LONGS)), 2)
    assert res["pass"] is False
    assert "correlation cap reached (2/2)" in res["reason"]


def test_short_never_capped():
    from hermes_trader.agents.risk_gates import correlation_cap
    res = correlation_cap(_ctx(coin="BTC", trade_side="short",
                               current_positions=list(_TWO_MAJOR_LONGS)), 2)
    assert res["pass"] is True


def test_below_cap_major_passes():
    from hermes_trader.agents.risk_gates import correlation_cap
    res = correlation_cap(_ctx(coin="BTC", current_positions=[{"coin": "ADA", "side": "long"}]), 2)
    assert res["pass"] is True


def test_eval_all_gates_wiring_non_major_not_blocked():
    """End-to-end through eval_all_gates: the gate's over-broad block must be
    gone from block_reasons for a non-major candidate with 2 majors held."""
    from hermes_trader.agents.risk_gates import eval_all_gates
    cfg = {"min_ai_confidence": 0.8, "max_concurrent": 5, "max_trade_notional_usd": 300,
           "min_market_volume_usd": 5e6, "max_total_notional_pct": 1.0,
           "cooldown_min": 0, "max_crypto_long_correlated": 2}
    res = eval_all_gates(_ctx(coin="ZEC", current_positions=list(_TWO_MAJOR_LONGS)), cfg)
    assert res["results"]["correlation"]["pass"] is True
    assert not any("correlation cap" in r for r in res["block_reasons"]), res["block_reasons"]
    # A major candidate under the same book is still rejected.
    res2 = eval_all_gates(_ctx(coin="DOGE", current_positions=list(_TWO_MAJOR_LONGS)), cfg)
    assert res2["results"]["correlation"]["pass"] is False
    assert any("correlation cap" in r for r in res2["block_reasons"])
