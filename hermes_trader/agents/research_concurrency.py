"""Worker-count clamping for the parallel research phase.

Pure + importable on its own (no heavy deps, no loop) so the knob logic is
unit-testable without importing scripts/trading_loop.py (whose body is a
module-level ``while True`` that would start trading on import).
"""

from __future__ import annotations

from typing import Any, Dict, List


def order_worklist(worklist: List[Dict[str, Any]],
                   held_coins: set) -> List[Dict[str, Any]]:
    """Order the research worklist so HELD coins get the first slots.

    The dsl-fast guard covers seconds; the research verdict covers structure.
    A held position's verdict must never wait behind fresh-entry candidates,
    so held coins are front-loaded. Relative order is preserved inside each
    group (``scan_once`` returns triggers sorted by composite_score DESC, and
    ``_forced_held_reeval`` appends the quiet-held re-evals), so the held
    group runs highest-scored-first, then the fresh group.

    Pure function: no network, no side effects.
    """
    held = [p for p in worklist if p.get("coin") in held_coins]
    fresh = [p for p in worklist if p.get("coin") not in held_coins]
    return held + fresh


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