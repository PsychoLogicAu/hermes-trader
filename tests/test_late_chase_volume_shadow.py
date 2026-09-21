"""B.27 shadow accrual: volume-accumulation WOULD-RELEASE line on late-chase blocks.

Replay (.hermes/WATCHLIST.md §B.27) showed the blocked late-chase cohort keeps a
median +5-6% 24h runway at strict volume-accumulation release points, but the
candidate admission (vol_last / mean(prior 96 finalized 5m vols) >= floor, PAIRED
with trend_ride exit) is shadow-only: the accrual logs, the live block stands.

Pinned behavior:
  * flag off / ratio floor <= 0 -> NO fetch, no log (byte-identical path);
  * ratio >= floor -> loud [gate][SHADOW] ... WOULD RELEASE line, block unchanged;
  * ratio < floor -> fetch happened, silent;
  * per-coin cooldown throttles repeat probes (candleSnapshot weight is shared
    with the live scanner);
  * candle failure never raises into the gate path.
"""
import pytest

from hermes_trader.agents import executor


class _C:
    def __init__(self, v): self.v = v
    def __getitem__(self, k): return getattr(self, k)


def _mk_candles(n, vol):
    return [_C(vol) for _ in range(n)]


@pytest.fixture(autouse=True)
def _reset_throttle(monkeypatch):
    executor._LC_VA_LAST.clear()
    yield
    executor._LC_VA_LAST.clear()


def _gate(**over):
    g = {"late_chase_volume_shadow": True, "late_chase_va_ratio_min": 5.0,
         "late_chase_va_per_coin_cooldown_min": 30}
    g.update(over)
    return g


def test_flag_off_no_fetch_no_log(caplog):
    calls = []
    def fake(coin, interval, count):
        calls.append((coin, interval, count)); return _mk_candles(98, 100.0)
    executor.fetch_hl_candles = fake
    try:
        executor._late_chase_volume_accrual({"late_chase_volume_shadow": False},
                                            "JUP", "long", 0.75, 0.90)
        assert not calls
        assert "WOULD RELEASE" not in caplog.text
    finally:
        del executor.fetch_hl_candles


def test_ratio_floor_zero_inert(caplog):
    calls = []
    def fake(coin, interval, count):
        calls.append((coin, interval, count)); return _mk_candles(98, 100.0)
    executor.fetch_hl_candles = fake
    try:
        executor._late_chase_volume_accrual(
            _gate(late_chase_va_ratio_min=0.0), "JUP", "long", 0.75, 0.90)
        assert not calls
    finally:
        del executor.fetch_hl_candles


def test_high_ratio_logs_would_release(caplog):
    # 97 finalized bars of volume 100 (last-forming candle dropped) except the
    # FINALIZED last one at 800 -> ratio 8x over the prior-96 mean.
    cands = _mk_candles(97, 100.0) + [_C(1.0)]
    cands[-2].v = 800.0
    def fake(coin, interval, count):
        assert interval == "5m" and count >= 98
        return cands
    executor.fetch_hl_candles = fake
    try:
        import logging
        with caplog.at_level(logging.WARNING):
            executor._late_chase_volume_accrual(_gate(), "JUP", "long", 0.75, 0.90)
        assert "[gate][SHADOW] late_chase_volume_accumulation WOULD RELEASE JUP LONG" in caplog.text
        assert "vol_ratio 8.0x >= 5.0x" in caplog.text
    finally:
        del executor.fetch_hl_candles


def test_low_ratio_silent(caplog):
    def fake(coin, interval, count):
        return _mk_candles(98, 100.0)
    executor.fetch_hl_candles = fake
    try:
        import logging
        with caplog.at_level(logging.WARNING):
            executor._late_chase_volume_accrual(_gate(), "JUP", "long", 0.75, 0.90)
        assert "WOULD RELEASE" not in caplog.text
    finally:
        del executor.fetch_hl_candles


def test_throttle_second_call_no_fetch():
    calls = []
    def fake(coin, interval, count):
        calls.append(coin); return _mk_candles(98, 100.0)
    executor.fetch_hl_candles = fake
    try:
        executor._late_chase_volume_accrual(_gate(), "JUP", "long", 0.75, 0.90)
        executor._late_chase_volume_accrual(_gate(), "JUP", "long", 0.75, 0.90)
        assert calls == ["JUP"]
        # different coin probes independently
        executor._late_chase_volume_accrual(_gate(), "VVV", "long", 0.75, 0.90)
        assert calls == ["JUP", "VVV"]
    finally:
        del executor.fetch_hl_candles


def test_fetch_failure_never_raises():
    def boom(coin, interval, count):
        raise RuntimeError("hl down")
    executor.fetch_hl_candles = boom
    try:
        executor._late_chase_volume_accrual(_gate(), "JUP", "long", 0.75, 0.90)
    finally:
        del executor.fetch_hl_candles


def test_short_history_no_raise():
    def fake(coin, interval, count):
        return _mk_candles(10, 100.0)
    executor.fetch_hl_candles = fake
    try:
        executor._late_chase_volume_accrual(_gate(), "NEW", "long", 0.75, 0.90)
    finally:
        del executor.fetch_hl_candles
