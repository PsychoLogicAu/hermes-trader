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
    perception_id: str | None = None,
    config_snapshot: Dict[str, Any] | None = None,
) -> None:
    """Record an opening trade.

    perception_id is the join key to the duel perception log
    (duel-logs/hermes-trader-duel.jsonl) — the A/B LLM evaluation artifact.
    Without it, ledger rows could only be mapped to duel rows via the capped,
    ephemeral analyses[] bridge in .agent-memory.json, which decays after
    MAX_ANALYSES rollover and makes old rows permanently unjoinable.
    """
    _append_event("OPEN", {
        "coin": coin,
        "side": side,
        "entry_px": entry_px,
        "notional_usd": notional_usd,
        "order_id": order_id,
        "leverage": leverage,
        "analysis_id": analysis_id,
        "perception_id": perception_id,
        "config_snapshot": config_snapshot,
    })


def record_scale_out(
    coin: str,
    side: str,
    entry_px: float,
    exit_px: float,
    size_coin: float,
    notional_usd: float,
    realized_pnl_pct: float,
    realized_pnl_usd: float,
    spot_pct: float,
    leverage: int,
    fee_usd: float | None = None,
    hold_minutes: float | None = None,
    fill_ts_ms: int | None = None,
    source: str = "tp_scale_out",
) -> None:
    """Record a partial (scale-out) close — the TP half taken server-side.

    The exchange-side TP trigger (`tp_scale_fraction`, executor entry bracket)
    fills WITHOUT any bot code running; the DSL tracker only notices the size
    shrink one rehydrate cycle later. Before 2026-09-22 that banked half never
    reached the ledger at all: the final CLOSE row books P/L on the *remaining*
    size only (close_position_market reads current exchange szi), so every
    TP-scaled winner was under-reported by roughly one half (39 such rows in
    the pre-fix ledger — TAO 2026-09-21 booked +$0.44 with ~+$0.54 invisible).

    Deliberately NOT a CLOSE event: the OPEN↔CLOSE stack pairing in
    `_reconcile_ledger_closes` (and every ledger analysis script) must keep
    seeing exactly one CLOSE per OPEN, so this rides as its own event type.
    Consumers summing net P/L should add SCALE_OUT.realized_pnl_usd to the
    CLOSE sum.
    """
    rec = {
        "coin": coin,
        "side": side,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "size_coin": size_coin,
        "notional_usd": notional_usd,
        "realized_pnl_pct": realized_pnl_pct,
        "realized_pnl_usd": realized_pnl_usd,
        "spot_pct": spot_pct,
        "leverage": leverage,
        "fee_usd": fee_usd,
        "hold_minutes": hold_minutes,
        "source": source,
    }
    if fill_ts_ms:
        rec["fill_ts_ms"] = fill_ts_ms
        rec["fill_ts_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(fill_ts_ms / 1000))
    _append_event("SCALE_OUT", rec)


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
    perception_id: str | None = None,
) -> None:
    """Record a closing trade with full realized details.

    perception_id carries the join key back to the duel perception log so
    closed rows can be mapped to the paired A/B verdict directly (no
    .agent-memory.json bridge).
    """
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
        "perception_id": perception_id,
    })


def _pair_key(ev: Dict[str, Any]) -> tuple | None:
    """(coin, side, entry_px) identity used to pair an OPEN with its CLOSE.

    Returns None for a row too malformed to pair — callers skip those rather
    than guessing, so a corrupt line can never close (or resurrect) a position.
    """
    try:
        coin = str(ev.get("coin") or "")
        side = str(ev.get("side") or "").strip().lower()
        px_raw = ev.get("entry_px")
        if px_raw is None:
            return None
        px = round(float(px_raw), 10)
    except (TypeError, ValueError):
        return None
    if not coin or not side:
        return None
    return coin, side, px


def count_open_with_lane(lane_tag: str) -> int:
    """OPEN rows tagged config_snapshot.lane == lane_tag with no CLOSE yet.

    This is the aligned-short lane's LIVE slot occupancy (board t_2e23804b
    experiment, owner call 2026-10-30). Read off the append-only ledger rather
    than the in-process DSL trackers so the count survives a container
    recreate — the same restart-safety discipline the daily-PnL and cooldown
    reads use. House pairing: (coin, side, entry_px), file order, each CLOSE
    consumed once. Missing file / malformed lines degrade to 0, never raise:
    a lane that cannot prove its own occupancy must not over-admit.
    """
    tag = str(lane_tag or "").strip().lower()
    if not tag:
        return 0
    opens: list[tuple] = []
    closed: set[tuple] = set()
    try:
        with open(LEDGER_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(ev, dict):
                    continue
                key = _pair_key(ev)
                if key is None:
                    continue
                event = str(ev.get("event") or "").upper()
                if event == "OPEN":
                    snap = ev.get("config_snapshot")
                    if isinstance(snap, dict) and \
                            str(snap.get("lane") or "").strip().lower() == tag:
                        opens.append(key)
                elif event == "CLOSE":
                    closed.add(key)
    except FileNotFoundError:
        return 0
    except Exception as exc:  # unreadable ledger -> assume full, not empty
        logger.warning(f"[ledger] count_open_with_lane({tag}) failed: {exc}")
        return -1
    return sum(1 for k in opens if k not in closed)
