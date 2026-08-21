"""Worker-count clamping for the parallel research phase.

Pure + importable on its own (no heavy deps, no loop) so the knob logic is
unit-testable without importing scripts/trading_loop.py (whose body is a
module-level ``while True`` that would start trading on import).
"""

from __future__ import annotations

from typing import Any, Dict


def compute_research_workers(cfg: Dict[str, Any], n_triggers: int) -> int:
    """Clamp ``research_max_workers`` to ``[1, n_triggers]``.

    - ``1`` (default / absent / malformed) keeps the exact legacy sequential
      behavior — the safe fallback.
    - Never exceeds the number of triggers (a 4-worker pool with 2 coins is
      just 2 workers).
    - A non-numeric / missing / falsy config value degrades to 1 rather than
      raising, so a bad hot-edit can't blow up a scan cycle.
    """
    n = max(1, int(n_triggers))
    try:
        raw = int(cfg.get("research_max_workers", 1) or 1)
    except (TypeError, ValueError):
        raw = 1
    return max(1, min(raw, n))