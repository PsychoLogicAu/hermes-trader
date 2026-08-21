"""Append-only JSONL activity log — the trading system's visible heartbeat.

The trading loop and the FastAPI server append events here; `status.py` and the
hourly cron report read them back. One line per event, each tagged with a `ts`
(epoch ms). Path is overridable via the `SESSION_LOG_PATH` env var.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List

SESSION_LOG_FILE = os.environ.get(
    "SESSION_LOG_PATH",
    os.path.expanduser("~/.hermes-trader-session-log.jsonl"),
)

# In-process guard: the research phase can run on multiple worker threads
# (research_max_workers > 1), and research events embed full LLM reasoning —
# a line that big is not a single atomic O_APPEND write, so two threads
# writing at once can interleave and corrupt each other's JSONL line.
_LOG_LOCK = threading.Lock()


def append(event: Dict[str, Any]) -> None:
    """Append one event as a JSONL line. A `ts` field is added automatically.

    Best-effort: a logging failure must never interrupt trading, so disk errors
    are swallowed.
    """
    record = {"ts": int(time.time() * 1000), **event}
    try:
        with _LOG_LOCK:
            with open(SESSION_LOG_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
    except OSError:
        pass


def tail(n: int = 10) -> List[Dict[str, Any]]:
    """Return the last `n` parseable events, oldest first."""
    try:
        lines = [ln for ln in open(SESSION_LOG_FILE).read().splitlines() if ln.strip()]
    except FileNotFoundError:
        return []
    out: List[Dict[str, Any]] = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out
