"""HIP-3 pre-research composite pre-filter (2026-09-22).

While HIP-3 was on, the executor's long-side floor
`runner_entry_gate.min_hip3_composite` sat AFTER full LLM research: 2,176
research cycles completed and then died at that one gate (observed composite
p50 25 / p90 38 / p99 43 vs bar 50). The pre-filter re-applies the SAME rule
in perception, before research spend.

Guarantees pinned here:
  * pure verdict helper — non-HIP3 / feature-off ⇒ no-op (fail-safe);
    below-bar ⇒ drop; at/above bar ⇒ pass; short-lane trigger fired ⇒ pass
    (the executor floor only guards LONGS); shadow flag mirrored out.
  * integration via `_scan_single_market` — a HIP-3 daily-mover bypass admit
    with composite 0 is dropped pre-research when enabled, surfaced when the
    feature is off, and surfaced WITH a `[gate][SHADOW] hip3_scan_prefilter
    WOULD DROP` accrual line in shadow mode; crypto (no `dex`) never affected.

Hermetic: no network, no LLM — `_fetch_candles_sync` patched to flat candles
so the ONLY firing trigger is dailyMover (weight 0 ⇒ composite 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_trader.agents import perception  # noqa: E402
from hermes_trader.agents.config import get_config  # noqa: E402
from hermes_trader.models.types import Candle  # noqa: E402

_T0 = 1_700_000_000_000
_STEP = 900_000


def _candle(i, o, h, l, c):
    return Candle(t=_T0 + i * _STEP, o=o, h=h, l=l, c=c, v=1000.0)


# Flat 5m candles: no momentum/breakout/trend/volume trigger fires, so the
# composite is 0 and the only thing surfacing the coin is dailyMover (weight 0).
FLAT_5M = [_candle(i, 100.0, 100.05, 99.95, 100.0) for i in range(100)]

_HIP3_MARKET = {
    "coin": "xyz:TEST", "type": "perp", "dex": "xyz",
    "prevDayPx": 90.0, "dayNtlVlm": 6_000_000,   # +11% 24h, above mover floors
}
_CRYPTO_MARKET = {
    "coin": "TEST", "type": "perp",
    "prevDayPx": 90.0, "dayNtlVlm": 6_000_000,
}


def _cfg(prefilter=None, bar=35.0):
    cfg = {
        **get_config(),
        "runner_mover_surface": {"enabled": True},
        "runner_entry_gate": {"min_hip3_composite": bar},
    }
    if prefilter is not None:
        cfg["hip3_scan_prefilter"] = prefilter
    return cfg


def _hit(name, fired):
    return {"name": name, "fired": fired}


# ---------------------------------------------------------------------------
# Pure helper — every branch of the verdict matrix.
# ---------------------------------------------------------------------------

def test_verdict_non_hip3_noop():
    v = perception._hip3_prefilter_verdict(0.0, [], {"coin": "BTC"}, _cfg({"enabled": True}))
    assert v == (False, False)


def test_verdict_feature_off_noop():
    # key absent entirely, and enabled:false — both fail-safe to no-op
    assert perception._hip3_prefilter_verdict(0.0, [], _HIP3_MARKET, _cfg()) == (False, False)
    off = perception._hip3_prefilter_verdict(
        0.0, [], _HIP3_MARKET, _cfg({"enabled": False}))
    assert off == (False, False)


def test_verdict_below_bar_drops():
    drop, shadow = perception._hip3_prefilter_verdict(
        34.9, [_hit("dailyMover", True)], _HIP3_MARKET, _cfg({"enabled": True}, bar=35.0))
    assert (drop, shadow) == (True, False)


def test_verdict_at_bar_passes():
    # >= bar: the prefilter must agree exactly with the executor's `score < bar`
    assert perception._hip3_prefilter_verdict(
        35.0, [_hit("dailyMover", True)], _HIP3_MARKET,
        _cfg({"enabled": True}, bar=35.0)) == (False, False)


def test_verdict_short_lane_passes():
    # downtrendMomentum / bearishReversalCandle fired ⇒ the candidate can be
    # judged SHORT, a side the executor's hip3 floor never guards ⇒ no drop.
    for name in ("downtrendMomentum", "bearishReversalCandle"):
        assert perception._hip3_prefilter_verdict(
            0.0, [_hit(name, True)], _HIP3_MARKET,
            _cfg({"enabled": True})) == (False, False)


def test_verdict_non_fired_short_lane_still_drops():
    # a NOT-fired short-lane hit does not open the lane
    drop, _ = perception._hip3_prefilter_verdict(
        0.0, [_hit("downtrendMomentum", False), _hit("dailyMover", True)],
        _HIP3_MARKET, _cfg({"enabled": True}))
    assert drop is True


def test_verdict_shadow_flag_mirrored():
    drop, shadow = perception._hip3_prefilter_verdict(
        0.0, [_hit("dailyMover", True)], _HIP3_MARKET,
        _cfg({"enabled": True, "shadow_mode": True}))
    assert (drop, shadow) == (True, True)


def test_verdict_reads_live_bar():
    # bar moves with the config key — scan and gate can never disagree
    drop, _ = perception._hip3_prefilter_verdict(
        40.0, [_hit("dailyMover", True)], _HIP3_MARKET,
        _cfg({"enabled": True}, bar=50.0))
    assert drop is True
    drop2, _ = perception._hip3_prefilter_verdict(
        40.0, [_hit("dailyMover", True)], _HIP3_MARKET,
        _cfg({"enabled": True}, bar=35.0))
    assert drop2 is False


# ---------------------------------------------------------------------------
# Integration — `_scan_single_market` end to end (patched fetch, no network).
# ---------------------------------------------------------------------------

def _wire(monkeypatch):
    def _fetch(coin, interval, count, cache_ttl_ms, **kw):
        return FLAT_5M
    monkeypatch.setattr(perception, "_fetch_candles_sync", _fetch)


def _scan(monkeypatch, market, cfg):
    _wire(monkeypatch)
    ok, res = perception._scan_single_market(market, 100.0, cfg, 54, None, False, True)
    assert ok is True
    return res


def test_scan_hip3_low_score_dropped_when_enabled(monkeypatch):
    """daily-mover bypass admit, composite 0 < bar ⇒ dropped BEFORE research."""
    res = _scan(monkeypatch, _HIP3_MARKET, _cfg({"enabled": True}))
    assert res is None


def test_scan_drop_increments_prefilter_counter(monkeypatch):
    """Production visibility: the drop bumps the per-scan counter surfaced on
    the scan summary line (per-coin DEBUG lines are invisible at prod level)."""
    before = perception._get_prefilter_drops()
    _scan(monkeypatch, _HIP3_MARKET, _cfg({"enabled": True}))
    assert perception._get_prefilter_drops() == before + 1
    # non-drops must NOT bump it
    _scan(monkeypatch, _CRYPTO_MARKET, _cfg({"enabled": True}))
    assert perception._get_prefilter_drops() == before + 1


def test_scan_hip3_surfaced_when_feature_off(monkeypatch):
    """Feature off (the merge no-op): identical input still surfaces via the
    daily-mover bypass — pre-change behavior byte-identical."""
    res = _scan(monkeypatch, _HIP3_MARKET, _cfg())
    assert res is not None and res["coin"] == "xyz:TEST"


def test_scan_hip3_shadow_logs_accrual_and_surfaces(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.WARNING)
    res = _scan(monkeypatch, _HIP3_MARKET,
                _cfg({"enabled": True, "shadow_mode": True}))
    assert res is not None, "shadow mode must NOT change the outcome"
    lines = [r.message for r in caplog.records if "hip3_scan_prefilter" in r.message]
    assert len(lines) == 1
    assert lines[0].startswith("[gate][SHADOW] hip3_scan_prefilter WOULD DROP xyz:TEST")
    assert "composite 0.0 < bar 35" in lines[0]


def test_scan_crypto_unaffected_by_prefilter(monkeypatch):
    """Same flat/low-score crypto candidate surfaces — the prefilter is
    HIP-3-only (market has no `dex`)."""
    res = _scan(monkeypatch, _CRYPTO_MARKET, _cfg({"enabled": True}))
    assert res is not None and res["coin"] == "TEST"
