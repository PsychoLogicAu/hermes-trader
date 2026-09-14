"""Settle exchange-side closes the moment the DSL drops a stale tracker.

A CLOSE row is written ONLY from the executor's close path
(``close_position_market`` → ``record_close``). When an exit happens WITHOUT
that path running — most importantly an exchange-side SL/TP trigger fill that
the DSL engine later drops as a "stale tracker" (``rehydrate_from_exchange``)
— the ledger keeps the position OPEN forever, no realized outcome is booked,
and NO loss cooldown arms. Observed live 2026-09-13: ZETA long stopped on
exchange at 14:31:28 UTC (−$11.92), tracker dropped as stale at 14:33:47, no
CLOSE row and no cooldown until the next process start ran
``_reconcile_ledger_closes`` — during which FIL-style re-chases on the same
names went unblocked.

The startup reconcile stays as the backstop; this module closes that gap at
the moment of detection instead of waiting for a restart:

1. ``dsl_exit.rehydrate_from_exchange`` reports each dropped tracker's entry
   context (entry px/time/size/leverage — deleted from the registry, so it
   must be captured there).
2. This module pulls the authoritative ``userFills``, attributes ONE closing
   fill to the dropped position, and books a reconcile-style CLOSE through
   the same conventions as the startup backfill (identical PnL formula,
   ``backfilled: true`` marker) plus the outcome-store record so win-rate /
   payoff stats see it immediately.
3. A losing settled close arms the loss cooldown exactly like the executor's
   stop-loss path (``stop_loss_cooldown_min`` for a max_loss-class exit).

Conservative by construction, mirroring the startup reconcile and the
executor's ``no_fill_position_already_flat`` rule:

- NO fill evidence → nothing booked, no cooldown (a close can't be told from
  a flaky position read; the startup backfill stays the catch-all).
- MORE THAN ONE closing fill in the window (split fills) → aggregate them only
  when every fill carries an exchange ``closedPnl`` and sum to a loss;
  otherwise skip booking entirely.
- Any exception → log and move on; settlement must never break the loop.

Kept as a standalone importable module (NOT inside trading_loop.py, which
runs its whole startup sequence at import) so it can be unit-tested offline.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

HL_INFO_URL = "https://api.hyperliquid.xyz/info"

# Same 2x-taker-fee estimate as executor.close_position_market and the startup
# _reconcile_ledger_closes backfill (fees_pct = 0.025 × 2 × lev) — stats treat
# all CLOSE rows identically.
FEES_PCT_PER_SIDE = 0.025


def _close_direction(side: str) -> str:
    return "Close Long" if side == "long" else "Close Short"


def _closed_pnl(f: Dict[str, Any]) -> Optional[float]:
    try:
        v = f.get("closedPnl")
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _has_closed_pnl(fills: List[Dict[str, Any]]) -> bool:
    """True iff EVERY fill carries a parseable exchange closedPnl."""
    return bool(fills) and all(_closed_pnl(f) is not None for f in fills)


def _compute_record(rec: Dict[str, float], fills: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a CLOSE record from the dropped-position context + its fills.

    Identical formula to ``record_close``/the startup backfill (verified
    against live ledger CLOSE rows): leveraged PnL net of a 2x-taker-fee
    estimate. Uses the LAST fill as exit px and sums exchange closedPnl when
    every fill carries one (split-fill case).
    """
    entry_px = float(rec["entry_px"])
    side = str(rec["side"])
    lev = int(rec.get("leverage") or 1) or 1
    size = abs(float(rec.get("size") or 0))
    last = fills[-1]
    exit_px = float(last["px"])
    ts_ms = int(last["time"])
    if size <= 0:
        # legacy tracker that never synced a size — fall back to the total
        # closed size from the fills (single fill → its size; split → sum).
        try:
            size = sum(abs(float(f.get("sz") or 0)) for f in fills)
        except (TypeError, ValueError):
            size = 0.0
    notional = round(size * entry_px, 4)

    spot_pct = ((exit_px - entry_px) if side == "long"
                else (entry_px - exit_px)) / entry_px * 100.0
    fees_pct = FEES_PCT_PER_SIDE * 2 * lev
    realized_pct = round(spot_pct * lev - fees_pct, 4)
    closed_pnls = [_closed_pnl(f) for f in fills]
    has_closed_pnl = all(c is not None for c in closed_pnls)
    net_usd = None
    if has_closed_pnl:
        # Exchange-reported PnL is the most authoritative number available.
        net_usd = round(sum(closed_pnls), 4)  # type: ignore[arg-type]
    elif notional > 0:
        gross_usd = notional * spot_pct / 100.0
        fee_usd = round(notional * (fees_pct / max(lev, 1)) / 100.0, 4)
        net_usd = round(gross_usd - fee_usd, 4)

    entry_time_s = float(rec.get("entry_time") or 0)
    hold_min = (round((ts_ms / 1000.0 - entry_time_s) / 60.0, 1)
                if entry_time_s > 0 else None)
    return {
        "coin": rec["coin"],
        "side": side,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "notional_usd": notional or None,
        "realized_pnl_pct": realized_pct,
        "realized_pnl_usd": net_usd,
        "spot_pct": round(spot_pct, 4),
        "hold_minutes": hold_min,
        "leverage": lev,
        "fee_usd": None,
        "exit_ts_ms": ts_ms,
        "exit_iso": datetime.fromtimestamp(ts_ms / 1000, timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _book_close(outcome, computed: Dict[str, Any], n_fills: int) -> None:
    """Write the CLOSE row to ledger + outcome store (reconcile-style)."""
    from hermes_trader.ledger import record_close
    exit_reason = ("exchange_close: SL/TP trigger filled on exchange; "
                   "tracker dropped as stale")
    try:
        outcome.record_close({
            "coin": computed["coin"], "side": computed["side"],
            "entry_px": computed["entry_px"], "exit_px": computed["exit_px"],
            "notional_usd": computed.get("notional_usd"),
            "spot_pct": computed["spot_pct"],
            "realized_pnl_pct": computed["realized_pnl_pct"],
            "realized_pnl_usd": computed.get("realized_pnl_usd"),
            "fee_usd": computed.get("fee_usd"),
            "leverage": computed["leverage"],
            "closed_at": computed["exit_ts_ms"],
            "exit_reason": exit_reason,
            "exit_type": "exchange_close",
            "hold_minutes": computed.get("hold_minutes"),
        })
    except Exception as e:  # outcome store must never block the ledger row
        logger.warning(f"[stale-settle] outcome-store record failed for "
                       f"{computed['coin']}: {e}")
    try:
        record_close(
            coin=computed["coin"], side=computed["side"],
            entry_px=computed["entry_px"], exit_px=computed["exit_px"],
            notional_usd=computed.get("notional_usd") or 0.0,
            realized_pnl_pct=computed["realized_pnl_pct"],
            realized_pnl_usd=computed.get("realized_pnl_usd") or 0.0,
            spot_pct=computed["spot_pct"],
            hold_minutes=computed.get("hold_minutes"),
            leverage=computed["leverage"],
            fee_usd=computed.get("fee_usd"),
            exit_reason=exit_reason,
            exit_type="exchange_close",
        )
    except Exception as e:
        logger.warning(f"[stale-settle] ledger record_close failed for "
                       f"{computed['coin']}: {e}")


def _arm_loss_cooldown(memory, read_agent_config, coin: str,
                       realized_pct: float) -> None:
    """Arm the loss cooldown exactly like executor.close_position_market.

    max_loss-class exits take the LONGER ``stop_loss_cooldown_min`` window; an
    exchange-side trigger stop is that case. Log format matches the executor's
    line so ledger-query grep patterns keep working.
    """
    try:
        cfg = read_agent_config()
        lc_min = float(cfg.get("loss_cooldown_min", 0) or 0)
        lc_min = max(lc_min, float(cfg.get("stop_loss_cooldown_min", 0) or 0))
        if lc_min > 0:
            until = int(time.time() * 1000 + lc_min * 60_000)
            memory.set_loss_cooldown(coin, until)
            logger.info(f"[executor] loss cooldown armed on {coin}: "
                        f"{lc_min:.0f}min (closed {realized_pct:.2f}%)")
    except Exception as e:
        logger.warning(f"[stale-settle] loss-cooldown arm failed for {coin}: {e}")


def _fetch_user_fills(resolve_user_address) -> List[Dict[str, Any]]:
    import requests
    user = resolve_user_address()
    if not user:
        return []
    resp = requests.post(HL_INFO_URL,
                         json={"type": "userFills", "user": user}, timeout=15)
    data = resp.json()
    return data if isinstance(data, list) else []


def settle_stale_closes(stale_recs: List[Dict[str, Any]], *,
                        fetch_fills: Optional[Callable[[], List[Dict[str, Any]]]] = None,
                        resolve_user_address=None,
                        memory=None,
                        read_agent_config=None,
                        log_event: Optional[Callable[[Dict[str, Any]], None]] = None,
                        now_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    """Book CLOSEs + arm cooldowns for trackers dropped as stale.

    ``stale_recs``: entries from ``dsl_exit.take_stale_close_records()`` —
    each {coin, side, entry_px, size, leverage, entry_time}. Returns the list
    of settled CLOSE records (empty when nothing could be attributed).
    Fail-safe: any error is logged; the caller's loop never sees an exception.
    """
    if not stale_recs:
        return []
    try:
        if fetch_fills is None:
            if resolve_user_address is None:
                from hermes_trader.client.hl_client import resolve_user_address \
                    as resolve_user_address  # type: ignore[no-redef]
            fetch_fills = lambda: _fetch_user_fills(resolve_user_address)  # noqa: E731
        if memory is None:
            from hermes_trader.agents.memory import memory as memory  # type: ignore[no-redef]
        if read_agent_config is None:
            from hermes_trader.agents.config_store import read_agent_config \
                as read_agent_config  # type: ignore[no-redef]

        try:
            fills = fetch_fills() or []
        except Exception as e:
            logger.warning(f"[stale-settle] userFills fetch failed ({e}); "
                           f"{len(stale_recs)} stale close(s) deferred to "
                           f"startup reconcile")
            return []

        settled: List[Dict[str, Any]] = []
        for rec in stale_recs:
            coin, side = str(rec["coin"]), str(rec["side"])
            want_dir = _close_direction(side)
            open_floor_ms = int(float(rec.get("entry_time") or 0) * 1000) - 1000
            upper_ms = (now_ms + 60_000) if now_ms else (time.time() * 1000 + 60_000)
            cands = [f for f in fills
                     if f.get("coin") == coin and f.get("dir") == want_dir
                     and open_floor_ms < int(f.get("time") or 0) <= upper_ms]
            # HL userFills is newest-first; sort ascending so _compute_record's
            # "last fill" is the chronologically-final exit.
            cands.sort(key=lambda f: int(f.get("time") or 0))
            if not cands:
                logger.warning(
                    f"[stale-settle] {coin}_{side} dropped as stale but no "
                    f"closing fill found since entry — nothing booked "
                    f"(startup reconcile stays the backstop)")
                continue
            computed = _compute_record(rec, cands)
            if len(cands) > 1:
                # Split fills: only trustworthy when every fill carries an
                # exchange closedPnl AND the aggregate is a loss (the gate we
                # want is anti-revenge on losers; ambiguous winners can wait).
                if not _has_closed_pnl(cands):
                    logger.warning(
                        f"[stale-settle] {coin}_{side} has {len(cands)} closing "
                        f"fills without complete closedPnl — ambiguous, skipping")
                    continue
                if (computed.get("realized_pnl_usd") or 0.0) >= 0:
                    logger.info(
                        f"[stale-settle] {coin}_{side} split fills net "
                        f"{computed['realized_pnl_usd']:+.2f} (winner) — skipping "
                        f"booking, leaving to startup reconcile")
                    continue
            _book_close(memory, computed, len(cands))
            logger.warning(
                f"[stale-settle] booked CLOSE for stale-dropped {coin}_{side}: "
                f"exit {computed['exit_px']} @ {computed['exit_iso']} "
                f"({computed['realized_pnl_pct']:+.2f}% leveraged, "
                f"{(computed.get('realized_pnl_usd') or 0.0):+.4f} USDC, "
                f"{len(cands)} fill(s))")
            if (computed["realized_pnl_pct"] < 0
                    or (computed.get("realized_pnl_usd") or 0.0) < 0):
                _arm_loss_cooldown(memory, read_agent_config, coin,
                                   computed["realized_pnl_pct"])
            settled.append(computed)
            if log_event is not None:
                try:
                    log_event({"event": "stale_close_settled", **{
                        k: v for k, v in computed.items()}})
                except Exception:
                    pass
        return settled
    except Exception as e:  # absolute last line of defence — never break the loop
        logger.warning(f"[stale-settle] unexpected failure (non-fatal): "
                       f"{type(e).__name__}: {e}")
        return []
