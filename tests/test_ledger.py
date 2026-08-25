"""hermes_trader.ledger — append-only trade ledger.

The only ledger-specific concern this suite guards: the duel join key.
OPEN and CLOSE rows must carry `perception_id` so the ledger can be mapped
directly to duel-logs/hermes-trader-duel.jsonl without the capped,
ephemeral analyses[] bridge in .agent-memory.json (see the 2026-08-25
join-mapping finding: 200-analysis cap made old rows permanently
unjoinable).

Test isolation: tests/conftest.py forces HERMES_LEDGER_FILE to a temp path
BEFORE ledger.py is imported (LEDGER_FILE is read at import time), so these
writes never touch the live /app/log/trades.jsonl.
"""

import json
import os

import pytest

from hermes_trader import ledger


@pytest.fixture
def ledger_file(tmp_path):
    """Point LEDGER_FILE at a per-test temp file (module-level, import-time)."""
    path = tmp_path / "trades.jsonl"
    old = ledger.LEDGER_FILE
    ledger.LEDGER_FILE = str(path)
    yield path
    ledger.LEDGER_FILE = old


def _read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_open_row_carries_perception_id(ledger_file):
    ledger.record_open(
        coin="SOL", side="long", entry_px=101.6, notional_usd=1000.0,
        order_id="ORD1", leverage=3, analysis_id="aid-1",
        perception_id="SOL-1787618890403-159cfd",
        config_snapshot={"regime": "uptrend"},
    )
    (row,) = _read(ledger_file)
    assert row["event"] == "OPEN"
    assert row["perception_id"] == "SOL-1787618890403-159cfd"
    assert row["analysis_id"] == "aid-1"
    # ts/epoch-ms shape unchanged
    assert isinstance(row["ts"], int) and row["ts"] > 10**12


def test_open_row_snapshot_carries_cooldown_windows(ledger_file):
    # Post-2026-08-25 cooldown incident: the OPEN row's config_snapshot must
    # record the effective cooldown windows so a future incident can be
    # reconstructed without guessing what was in force at entry time.
    ledger.record_open(
        coin="SOL", side="long", entry_px=101.6, notional_usd=1000.0,
        order_id="ORD1", leverage=3,
        config_snapshot={
            "regime": "up", "exit_policy_label": "scalp",
            "sl_atr_mult": 1.5, "tp_atr_mult": 1.0,
            "cooldown_min": 30, "loss_cooldown_min": 180,
        },
    )
    (row,) = _read(ledger_file)
    snap = row["config_snapshot"]
    assert snap["cooldown_min"] == 30
    assert snap["loss_cooldown_min"] == 180
    # pre-existing keys unchanged
    assert snap["sl_atr_mult"] == 1.5 and snap["tp_atr_mult"] == 1.0


def test_build_open_config_snapshot_reads_live_config(ledger_file, monkeypatch):
    # The snapshot must reflect the LIVE .agent-config.json at entry time
    # (hot-reloaded every cycle) — not a stale module-level copy.
    #
    # We monkeypatch executor.read_agent_config rather than writing the shared
    # session config file: several full-suite tests write that one temp file
    # concurrently, and a real read raced another test's write (a hybrid
    # cooldown_min/loss_cooldown_min read-back). A patched reader is fully
    # hermetic and still proves the helper reads fresh each call.
    from hermes_trader.agents import executor as ex

    _calls = {"n": 0}

    def _fake_read():
        _calls["n"] += 1
        # values change per call: proves the helper re-reads (no caching)
        if _calls["n"] == 1:
            return {"cooldown_min": 45, "loss_cooldown_min": 90}
        if _calls["n"] == 2:
            return {"cooldown_min": 60, "loss_cooldown_min": 120}
        return {}  # keys absent -> None

    monkeypatch.setattr(ex, "read_agent_config", _fake_read)

    snap = ex.build_open_config_snapshot("up", "scalp", 1.5, 1.0)
    assert snap == {"regime": "up", "exit_policy_label": "scalp",
                    "sl_atr_mult": 1.5, "tp_atr_mult": 1.0,
                    "cooldown_min": 45, "loss_cooldown_min": 90}
    # hot-reload: a config change after the first read must be visible next time
    snap2 = ex.build_open_config_snapshot("up", "scalp", 1.5, 1.0)
    assert snap2["cooldown_min"] == 60 and snap2["loss_cooldown_min"] == 120
    # keys absent from config -> None (never KeyError, never a fake default)
    snap3 = ex.build_open_config_snapshot("up", "scalp", 1.5, 1.0)
    assert snap3["cooldown_min"] is None and snap3["loss_cooldown_min"] is None


def test_close_row_carries_perception_id(ledger_file):
    ledger.record_close(
        coin="SOL", side="long", entry_px=101.6, exit_px=103.1,
        notional_usd=1000.0, realized_pnl_pct=4.4, realized_pnl_usd=14.0,
        spot_pct=1.48, hold_minutes=90.0, leverage=3,
        fee_usd=0.15, funding_cost_usd=0.0,
        exit_reason="take_profit", exit_type="tp",
        perception_id="SOL-1787618890403-159cfd",
    )
    (row,) = _read(ledger_file)
    assert row["event"] == "CLOSE"
    assert row["perception_id"] == "SOL-1787618890403-159cfd"


def test_perception_id_defaults_none_when_undocumented(ledger_file):
    # Pre-duelist call sites (or the duelist disabled) omit the key entirely:
    # the row must still write, with perception_id: null — never KeyError.
    ledger.record_open(
        coin="BTC", side="short", entry_px=65000.0, notional_usd=500.0,
        order_id="ORD2", leverage=1,
    )
    ledger.record_close(
        coin="BTC", side="short", entry_px=65000.0, exit_px=64800.0,
        notional_usd=500.0, realized_pnl_pct=0.6, realized_pnl_usd=3.0,
        spot_pct=0.31, hold_minutes=10.0, leverage=1,
    )
    rows = _read(ledger_file)
    assert [r["event"] for r in rows] == ["OPEN", "CLOSE"]
    assert all(r["perception_id"] is None for r in rows)


def test_join_demonstrable_end_to_end(ledger_file):
    """The point of the change: OPEN row -> duel log, no memory bridge."""
    ledger.record_open(
        coin="INJ", side="long", entry_px=0.31, notional_usd=800.0,
        order_id="ORD3", leverage=5, analysis_id="aid-3",
        perception_id="INJ-1787621234567-abc123",
    )
    # Simulated duel-log row (same shape as duel_store.record_duel writes)
    duel_row = {"coin": "INJ", "perception_id": "INJ-1787621234567-abc123",
                "primary_verdict": "LONG", "duelist_verdict": "SHORT"}
    (open_row,) = _read(ledger_file)
    assert open_row["perception_id"] == duel_row["perception_id"]