"""Append-only trade ledger — JSONL format, one line per trade event.

Separate from .agent-memory.json (which is operational state). The ledger is:
- Append-only, never truncated
- Human-readable JSONL for easy parsing/backtesting
- Written on every OPEN and CLOSE event (single chokepoints)

Usage:
  # Query with grep/jq:
  grep '"event":"CLOSE"' trades.jsonl | jq -s '.[] | {coin, realized_pnl_pct, hold_minutes}'
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

_LOG_DIR = "/app/log"
# Override with HERMES_LEDGER_FILE when the default is not the live ledger.
# Tests MUST set this (tests/conftest.py): the 2026-06-15 incident pattern —
# memory/config/DSL state got env overrides after a pytest run clobbered live
# state files — was applied to the ledger late: on 2026-08-23 a pytest run in
# the container wrote fixture OPEN/CLOSE rows (order_id OID1, ARB short @
# 0.11684) into the live /app/log/trades.jsonl because this path was the only
# state file with no test isolation.
LEDGER_FILE = os.environ.get(
    "HERMES_LEDGER_FILE",
    os.path.join(_LOG_DIR, "trades.jsonl"),
)


def _append_event(event_type: str, data: Dict[str, Any]) -> None:
    """Write a single event to the ledger file (atomic append)."""
    record = {
        "event": event_type,
        "ts": int(time.time() * 1000),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **data,
    }
    line = json.dumps(record)
    try:
        with open(LEDGER_FILE, "a") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.warning(f"[ledger] append failed (non-fatal): {e}")


def record_open(
    coin: str,
    side: str,
    entry_px: float,
    notional_usd: float,
    order_id: str | None,
    leverage: int,
    analysis_id: str | None = None,
    config_snapshot: Dict[str, Any] | None = None,
) -> None:
    """Record an opening trade."""
    _append_event("OPEN", {
        "coin": coin,
        "side": side,
        "entry_px": entry_px,
        "notional_usd": notional_usd,
        "order_id": order_id,
        "leverage": leverage,
        "analysis_id": analysis_id,
        "config_snapshot": config_snapshot,
    })


def record_close(
    coin: str,
    side: str,
    entry_px: float,
    exit_px: float,
    notional_usd: float,
    realized_pnl_pct: float,
    realized_pnl_usd: float,
    spot_pct: float,
    hold_minutes: float | None,
    leverage: int,
    fee_usd: float | None = None,
    funding_cost_usd: float | None = None,
    exit_reason: str | None = None,
    exit_type: str | None = None,
) -> None:
    """Record a closing trade with full realized details."""
    _append_event("CLOSE", {
        "coin": coin,
        "side": side,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "notional_usd": notional_usd,
        "realized_pnl_pct": realized_pnl_pct,
        "realized_pnl_usd": realized_pnl_usd,
        "spot_pct": spot_pct,
        "hold_minutes": hold_minutes,
        "leverage": leverage,
        "fee_usd": fee_usd,
        "funding_cost_usd": funding_cost_usd,
        "exit_reason": exit_reason,
        "exit_type": exit_type,
    })
