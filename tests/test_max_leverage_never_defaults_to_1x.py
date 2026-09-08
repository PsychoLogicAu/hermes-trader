"""A missing maxLeverage must never quietly become 1x.

Port of upstream 3a65b92a9121 (Julian-dev28/hermes-trader). The dict default
`u.get("maxLeverage", 1)` supplied 1 whenever the cached dex meta held the coin
but WITHOUT that field — a partial warm after a restart — so a 3x book silently
opened at 1x (two live xyz positions, 2026-09-08). The damage is not
risk-per-trade (a stop loses stop_pct of NOTIONAL whatever the leverage) but
margin: at 1x it equals notional, so the account fits a third of the intended
book and every sizing number derived from `leverage` is wrong while looking
right. No xyz market allows less than 3x, so 1 was never a real answer — it was
a dict default standing in for missing data.
"""
from __future__ import annotations

import os
import time

import pytest

import hermes_trader.client.exchange as EX


@pytest.fixture(autouse=True)
def _clear():
    EX._META_CACHE.clear()
    yield
    EX._META_CACHE.clear()


def test_a_partial_cache_entry_forces_a_refresh_instead_of_returning_1(monkeypatch):
    """The exact 2026-09-08 shape: the coin IS cached, but without the field."""
    calls = {"n": 0}

    def _fake(dex=None, force_refresh=False):
        calls["n"] += 1
        if not force_refresh:
            return [{"name": "xyz:FOO"}]                 # partial: no maxLeverage
        return [{"name": "xyz:FOO", "maxLeverage": 10}]  # complete after refresh

    monkeypatch.setattr(EX, "_cached_universe", _fake)
    assert EX.get_max_leverage("xyz:FOO") == 10
    assert calls["n"] >= 2, "it must actually re-fetch, not just re-read the cache"


def test_a_partial_main_dex_entry_forces_a_refresh_instead_of_returning_1(monkeypatch):
    """Same shape on the main perp dex (no HIP-3 namespace)."""
    calls = {"n": 0}

    def _fake(dex=None, force_refresh=False):
        calls["n"] += 1
        assert dex is None
        if not force_refresh:
            return [{"name": "FOO"}]
        return [{"name": "FOO", "maxLeverage": 40}]

    monkeypatch.setattr(EX, "_cached_universe", _fake)
    assert EX.get_max_leverage("FOO") == 40
    assert calls["n"] >= 2, "it must actually re-fetch, not just re-read the cache"


@pytest.mark.parametrize("bad", [None, 0, "", "abc", -1])
def test_unusable_leverage_values_never_pass_through_as_1(monkeypatch, bad):
    """Even after the forced refresh, an unusable value must RAISE, not return 1."""
    monkeypatch.setattr(
        EX, "_cached_universe",
        lambda dex=None, force_refresh=False: [
            {"name": "xyz:FOO", "maxLeverage": bad}])
    with pytest.raises(ValueError):
        EX.get_max_leverage("xyz:FOO")


def test_it_raises_rather_than_returning_a_wrong_number(monkeypatch):
    """Raising is the point: the caller does min(requested, this), so any
    plausible-but-wrong value silently mis-sizes instead of failing."""
    monkeypatch.setattr(EX, "_cached_universe",
                        lambda dex=None, force_refresh=False: [])
    with pytest.raises(ValueError):
        EX.get_max_leverage("xyz:NOPE")


def test_a_warm_entry_returns_the_real_max_leverage_without_a_refresh(monkeypatch):
    calls = {"n": 0}

    def _fake(dex=None, force_refresh=False):
        calls["n"] += 1
        return [{"name": "xyz:FOO", "maxLeverage": 20}]

    monkeypatch.setattr(EX, "_cached_universe", _fake)
    assert EX.get_max_leverage("xyz:FOO") == 20
    assert calls["n"] == 1, "a complete cache entry must not trigger a refetch"


def test_force_refresh_bypasses_a_live_cache_entry(monkeypatch):
    EX._META_CACHE["xyz"] = (time.time(), [{"name": "xyz:FOO"}])
    monkeypatch.setattr(
        EX, "_get_info",
        lambda: type("I", (), {"meta": staticmethod(
            lambda dex=None: {"universe": [
                {"name": "xyz:FOO", "maxLeverage": 5}]})})())
    assert EX._cached_universe(dex="xyz", force_refresh=True)[0]["maxLeverage"] == 5


def test_the_executor_refuses_when_leverage_cannot_be_resolved():
    """Opening at a guessed leverage is worse than not opening: the executor
    must carry the refusal (executed=False, reason `leverage_unresolved`)
    instead of letting a 1x default slip through."""
    src = os.path.join(os.path.dirname(EX.__file__), os.pardir,
                       "agents", "executor.py")
    body = open(src).read()
    assert "leverage_unresolved" in body
    i = body.index("leverage_unresolved")
    assert '"executed": False' in body[i - 300:i]
