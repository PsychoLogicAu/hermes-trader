"""Regression tests for the §C.16 regime-divergence shadow annotation.

The annotation must (a) ride every market_regime gate result, pass or block,
(b) NEVER change any verdict, and (c) collapse to gap-nulls on fetch failure.
"""
from unittest.mock import patch

import pytest

from hermes_trader.agents import market_regime as mr
from hermes_trader.agents import hyperfeed
from hermes_trader.agents.risk_gates import GateContext, market_regime_gate


def _ctx(side="long", conf=0.75, score=10.0, coin="PYTH"):
    return GateContext(
        confidence=conf, current_positions=[], trade_notional_usd=30.0,
        daily_pnl=0.0, market_volume_24h_usd=1e8, coin=coin,
        trade_side=side, has_binary_news_risk=False, equity=1000.0,
        total_open_notional=0.0, composite_score=score,
    )


@pytest.fixture(autouse=True)
def _clear_caches():
    mr._regime_cache.clear()
    mr._alt_basket_regime_cache = (None, 0.0)
    mr._own_trend_shadow_cache.clear()
    yield
    mr._regime_cache.clear()
    mr._alt_basket_regime_cache = (None, 0.0)
    mr._own_trend_shadow_cache.clear()


def test_divergence_on_pass_and_block_without_changing_verdicts():
    # BTC down-regime: a long is counter-trend → block at low conf/score;
    # a short is aligned → pass. Divergence rides BOTH results.
    with patch.object(hyperfeed, "market_get_funding_regime",
                      lambda: {"regime": "NEUTRAL",
                               "regimes_by_class": {"crypto": "NEUTRAL"}}), \
         patch.object(mr, "_detect_for_proxy", return_value="down"), \
         patch.object(mr, "alt_basket_regime", return_value="up"), \
         patch.object(mr, "classify_asset", return_value="crypto"):
        blocked = market_regime_gate(_ctx("long"), counter_regime_min_conf=0.85)
        assert blocked["pass"] is False
        d = blocked["divergence"]
        assert d["btc_regime"] == "down"
        assert d["alt_basket_regime"] == "up"
        assert d["btc_vs_alt_diverged"] is True

        aligned = market_regime_gate(_ctx("short"), counter_regime_min_conf=0.85)
        assert aligned["pass"] is True          # short vs down = aligned
        assert "divergence" in aligned          # annotation on passes too


def test_coin_vs_btc_divergence_flag():
    def fake_proxy(proxy):
        return {"BTC": "down", "PYTH": "up"}.get(str(proxy).upper(), "neutral")
    with patch.object(hyperfeed, "market_get_funding_regime",
                      lambda: {"regime": "NEUTRAL",
                               "regimes_by_class": {"crypto": "NEUTRAL"}}), \
         patch.object(mr, "_detect_for_proxy", side_effect=fake_proxy), \
         patch.object(mr, "alt_basket_regime", return_value="down"), \
         patch.object(mr, "classify_asset", return_value="crypto"):
        res = market_regime_gate(_ctx("long"), counter_regime_min_conf=0.85)
    d = res["divergence"]
    assert d["coin_regime"] == "up"
    assert d["coin_vs_btc_diverged"] is True


def test_annotation_gap_never_breaks_or_blocks():
    # alt-basket read unavailable (gap) -> annotation collapses to nulls;
    # the gate verdict path is untouched.
    with patch.object(hyperfeed, "market_get_funding_regime",
                      lambda: {"regime": "NEUTRAL",
                               "regimes_by_class": {"crypto": "NEUTRAL"}}), \
         patch.object(mr, "_detect_for_proxy", return_value="neutral"), \
         patch.object(mr, "alt_basket_regime", return_value=None), \
         patch.object(mr, "classify_asset", return_value="crypto"):
        res = market_regime_gate(_ctx("long"), counter_regime_min_conf=0.85)
    d = res["divergence"]
    assert d["alt_basket_regime"] is None
    assert d["btc_vs_alt_diverged"] is False
    # regime neutral + funding neutral -> free pass, exactly as pre-annotation
    assert res["pass"] is True and res["via"] == "neutral"


def test_non_crypto_coin_gets_na_coin_regime():
    with patch.object(hyperfeed, "market_get_funding_regime",
                      lambda: {"regime": "NEUTRAL",
                               "regimes_by_class": {"equity": "NEUTRAL"}}), \
         patch.object(mr, "_detect_for_proxy", return_value="neutral"), \
         patch.object(mr, "alt_basket_regime", return_value="neutral"), \
         patch.object(mr, "classify_asset", return_value="equity"):
        res = market_regime_gate(_ctx("long", coin="xyz:MU"))
    assert res["divergence"]["coin_regime"] == "n/a"
