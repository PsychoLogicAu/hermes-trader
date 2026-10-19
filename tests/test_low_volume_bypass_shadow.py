"""SHADOW mode for the low-liquidity confidence bypass (kanban t_ec2c8bbf follow-up).

runner_entry_gate.bypass_low_volume_shadow_mode:
  absent/false -> live bypass byte-identical (releases thin-volume entries at conf>=bar)
  true         -> release never fires; entry falls to the byte-identical floor-block
                  reason AND a [gate][SHADOW] low_liquidity_bypass WOULD HAVE RELEASED
                  accrual line logs conf/vol/floor per episode.
"""
import logging

import pytest

from hermes_trader.agents.risk_gates import GateContext, market_liquidity_floor

NATIVE_FLOOR = 1_000_000.0
HIP3_FLOOR = 5_000_000.0
BYPASS_CFG = {"bypass_low_volume": True, "bypass_low_volume_min_conf": 0.85}


def ctx(confidence, vol_usd, coin="STBL", side="long"):
    return GateContext(
        confidence=confidence,
        current_positions=[],
        trade_notional_usd=25.0,
        daily_pnl=0.0,
        market_volume_24h_usd=vol_usd,
        coin=coin,
        trade_side=side,
        has_binary_news_risk=False,
        equity=1000.0,
        total_open_notional=0.0,
    )


def block_reason(vol_usd, floor=NATIVE_FLOOR):
    return (f"market 24h volume ${vol_usd/1e6:.2f}M "
            f"below floor ${floor/1e6:.2f}M")


def test_key_absent_bypass_releases_byte_identical():
    """Baseline: today's live behavior — release with the exact existing via/reason."""
    r = market_liquidity_floor(ctx(0.86, 450_000), NATIVE_FLOOR, HIP3_FLOOR,
                               gate_config=dict(BYPASS_CFG))
    assert r["pass"] is True
    assert r["via"] == "bypass_low_volume"
    assert r["reason"] == "low-liquidity bypass on STBL (conf 0.86 >= 0.85)"


def test_key_false_bypass_releases_byte_identical():
    cfg = dict(BYPASS_CFG, bypass_low_volume_shadow_mode=False)
    r = market_liquidity_floor(ctx(0.88, 990_000), NATIVE_FLOOR, HIP3_FLOOR,
                               gate_config=cfg)
    assert r["pass"] is True
    assert r["via"] == "bypass_low_volume"
    assert r["reason"] == "low-liquidity bypass on STBL (conf 0.88 >= 0.85)"


def test_key_true_blocks_with_byte_identical_floor_reason(caplog):
    cfg = dict(BYPASS_CFG, bypass_low_volume_shadow_mode=True)
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        r = market_liquidity_floor(ctx(0.86, 450_000), NATIVE_FLOOR, HIP3_FLOOR,
                                   gate_config=cfg)
    assert r["pass"] is False
    assert "via" not in r
    # byte-identical to the plain below-floor block a sub-bar conf gets
    assert r["reason"] == block_reason(450_000)


def test_key_true_logs_would_have_released_accrual(caplog):
    cfg = dict(BYPASS_CFG, bypass_low_volume_shadow_mode=True)
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        market_liquidity_floor(ctx(0.86, 450_000), NATIVE_FLOOR, HIP3_FLOOR,
                               gate_config=cfg)
    shadow = [rec.getMessage() for rec in caplog.records
              if "WOULD HAVE RELEASED" in rec.getMessage()]
    assert len(shadow) == 1
    line = shadow[0]
    assert line.startswith("[gate][SHADOW] low_liquidity_bypass WOULD HAVE RELEASED STBL LONG:")
    assert "conf 0.86 >= 0.85" in line
    assert "vol $0.45M below floor $1.00M" in line
    assert "NOT releasing (shadow mode; live rule blocks)" in line


def test_key_true_sub_bar_conf_no_accrual_plain_block(caplog):
    """Conf below the bypass bar never reaches the bypass — no shadow line,
    identical to the plain floor block (mirrors mover-shadow sub-bar rule)."""
    cfg = dict(BYPASS_CFG, bypass_low_volume_shadow_mode=True)
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        r = market_liquidity_floor(ctx(0.84, 450_000), NATIVE_FLOOR, HIP3_FLOOR,
                                   gate_config=cfg)
    assert r["pass"] is False
    assert r["reason"] == block_reason(450_000)
    assert not [rec for rec in caplog.records if "WOULD HAVE RELEASED" in rec.getMessage()]


def test_key_true_above_floor_unaffected_no_accrual(caplog):
    cfg = dict(BYPASS_CFG, bypass_low_volume_shadow_mode=True)
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        r = market_liquidity_floor(ctx(0.86, 2_000_000), NATIVE_FLOOR, HIP3_FLOOR,
                                   gate_config=cfg)
    assert r["pass"] is True
    assert not [rec for rec in caplog.records if "WOULD HAVE RELEASED" in rec.getMessage()]


def test_key_true_hip3_uses_hip3_floor_in_both_reason_and_accrual(caplog):
    cfg = dict(BYPASS_CFG, bypass_low_volume_shadow_mode=True)
    with caplog.at_level(logging.WARNING, logger="hermes_trader.agents.risk_gates"):
        r = market_liquidity_floor(ctx(0.85, 4_000_000, coin="xyz:DKNG"),
                                   NATIVE_FLOOR, HIP3_FLOOR, gate_config=cfg)
    assert r["pass"] is False
    assert r["reason"] == block_reason(4_000_000, HIP3_FLOOR)
    shadow = [rec.getMessage() for rec in caplog.records
              if "WOULD HAVE RELEASED" in rec.getMessage()]
    assert len(shadow) == 1
    assert "xyz:DKNG LONG" in shadow[0]
    assert "vol $4.00M below floor $5.00M" in shadow[0]
