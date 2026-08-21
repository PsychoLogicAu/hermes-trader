"""Per-coin log attribution for the parallel research phase.

research_max_workers > 1 runs each coin's research/execute on its own worker
thread, so untagged log lines ("Verdict: ...", "Trade result: ...", LLM
failures, HL fetch errors) interleave and can't be attributed to a coin. A
contextvar carries the coin currently being processed on the current thread;
the formatter prepends "[COIN] " to every record emitted while it is set.
Cycle-level lines (heartbeat, scan, exit monitor) run with no coin and stay
bare.

Kept importable on its own (no trading-loop side effects) so the attribution
behaviour is unit-testable without starting the loop.
"""
from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager

__all__ = ["log_coin", "CoinLogFormatter", "set_coin_context"]

# Ticker of the coin currently being processed on this thread ("").
log_coin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "hermes_trader_log_coin", default=""
)


class CoinLogFormatter(logging.Formatter):
    """Prepend "[COIN] " to a record when log_coin is set on this thread."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        coin = log_coin.get()
        return f"[{coin}] {line}" if coin else line


@contextmanager
def set_coin_context(coin: str):
    """Set log_coin for the current thread's context; restore it on exit.

    In a ThreadPoolExecutor task the context is a per-task copy, so the value
    is discarded when the task ends and never leaks to other workers or the
    main thread. The explicit reset also guards the sequential (workers == 1,
    main-thread) path from leaking the last coin into cycle-level logging.
    """
    token = log_coin.set(coin)
    try:
        yield coin
    finally:
        log_coin.reset(token)