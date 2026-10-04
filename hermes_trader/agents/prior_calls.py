"""Prior-call context — stateful memory of the LLM's own recent verdicts.

r/ai_trading candidate #1 (WATCHLIST B.32, 2026-10-03): a Hyperliquid LLM bot
flipped −71%→+70% purely by making the model STATEFUL — feeding it its previous
decision + reasoning and its last completed round-trip alongside the candles.
The mechanism is architecture-level, not sample-specific: this bot's research
prompt re-litigates every candle from scratch (the only history it ever saw was
the CURRENT position's PnL annotation), so it can repeatedly enter the same
failed setup and never "remembers" judging it before.

This module renders that missing context from state the bot ALREADY persists:

  * ``memory.get_recent_analyses()`` — verdict / confidence / reasoning /
    price-at-decision / timestamp for past research calls (persisted in
    .agent-memory.json, survives restarts), filtered to this coin;
  * ``memory.last_close_for(coin)`` — the last completed round trip
    (entry→exit, realized PnL, hold time, exit reason).

Failure-PASS rows (``ai_down``) and guard-downgraded CLOSEs are EXCLUDED: a
failure is an error code, not an opinion, and feeding "you PASSed 5x" when four
of those were a dead LLM endpoint would teach the model the wrong lesson.

SHIPS DISABLED: absent/false ``prior_call_context.enabled`` → returns "" and
the prompt is byte-identical to today (same contract as every other prompt
block). The block text itself is deliberately restrained — it states the facts
(what was decided, when, at what price, how the last round trip resolved) and
one framing line; it never tells the model what its prior verdicts should imply.

Config (hot, read per call)::

    "prior_call_context": {
        "enabled": false,          # master switch (OFF = byte-identical prompt)
        "max_calls": 3,            # how many prior decisions to show
        "lookback_min": 240,       # ignore decisions older than this
        "campaign_lookback_min": 1440   # window for the last round-trip line
    }

Every path is fail-safe: any fault (memory not loaded, bad config, missing
fields) yields "" — a broken context block must never break prompt assembly.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from hermes_trader.agents.config_store import read_agent_config

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "max_calls": 3,
    "lookback_min": 240,
    "campaign_lookback_min": 1440,
}

# Reasoning snippets are the informative part but models (and prompt budgets)
# don't need the full paragraph — the last call's reasoning gets a little more.
_SNIPPET_CHARS = 180


def _cfg() -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    try:
        block = read_agent_config().get("prior_call_context")
        if isinstance(block, dict):
            merged.update(block)
    except Exception:  # noqa: BLE001 — fail-open to defaults (see docstring)
        pass
    return merged


def _fmt_px(p: Any) -> str:
    """Adaptive precision (mirrors research._build_user_message._fmt_px so a
    sub-cent coin's prior price reads the same way its current mid does)."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "?"
    if p == 0:
        return "?"
    ap = abs(p)
    if ap >= 1:
        return f"${p:.4f}"
    if ap >= 0.01:
        return f"${p:.5f}"
    if ap >= 0.0001:
        return f"${p:.6f}"
    return f"${p:.8f}"


def _ago(created_at_ms: Any, now_ms: int) -> str:
    """'27min ago' / '3.2h ago' — human-relative so the model doesn't have to
    diff epoch stamps against a clock it can't see."""
    try:
        mins = max(0, (now_ms - int(created_at_ms)) // 60_000)
    except (TypeError, ValueError):
        return "unknown time ago"
    if mins < 60:
        return f"{mins}min ago"
    return f"{mins / 60:.1f}h ago"


def _snippet(text: Any, limit: int = _SNIPPET_CHARS) -> str:
    s = " ".join(str(text or "").split())
    if not s:
        return "(no reasoning recorded)"
    if len(s) > limit:
        return s[:limit].rstrip() + "…"
    return s


def build_prior_call_block(coin: str, primary_model: Optional[str] = None) -> str:
    """The 'your recent history on this coin' prompt block, or "" when
    disabled / nothing to show / any fault (fail-safe — see module docstring).

    NOTE: the CURRENT call is never in the rendered set — memory.record_analysis
    runs AFTER the prompt is built, so no self-exclusion filter is needed.
    """
    try:
        cfg = _cfg()
        if not cfg.get("enabled", False):
            return ""
        # Import here (not at module load) so tests can redirect
        # HERMES_AGENT_MEMORY_FILE before memory freezes its path, and so this
        # module stays importable without the singleton being initialised.
        from hermes_trader.agents.memory import memory

        now_ms = int(time.time() * 1000)
        lookback_ms = float(cfg["lookback_min"]) * 60_000
        max_calls = max(1, int(cfg["max_calls"]))

        all_analyses: List[Dict[str, Any]] = memory.get_recent_analyses(limit=200)
        prior = [
            a for a in all_analyses
            if a.get("coin") == coin
            and not a.get("ai_down")               # failure-PASS ≠ opinion
            and not a.get("close_guard_downgraded")  # misread CLOSE ≠ opinion
            and (a.get("created_at") or 0) >= now_ms - lookback_ms
        ][-max_calls:]

        lines: List[str] = []
        for a in prior:
            verdict = str(a.get("verdict", "?"))
            conf = a.get("confidence")
            conf_s = f" conf {float(conf):.2f}" if conf is not None else ""
            # Name the model only when it differs from the current primary —
            # after a model swap, "PASS 3x" was arguably not you.
            am = a.get("primary_model")
            via_s = f", via {am}" if am and primary_model and am != primary_model else ""
            lines.append(
                f"  - [{_ago(a.get('created_at'), now_ms)}] {verdict}{conf_s}{via_s} "
                f"at {_fmt_px(a.get('entry_px'))} — \"{_snippet(a.get('reasoning'))}\""
            )

        campaign_line = _campaign_line(coin, memory, now_ms,
                                       float(cfg["campaign_lookback_min"]) * 60_000)

        if not lines and not campaign_line:
            return ""

        out = [
            f"Your recent history on {coin} (stateful context — you do NOT have to "
            "agree with your past self, but a new verdict that repeats an old one "
            "should say what changed; re-entering a setup you already judged and "
            "that resolved against you requires explicit new evidence):",
        ]
        out.extend(lines) if lines else out.append(
            f"  - no prior decisions on {coin} within the lookback window")
        if campaign_line:
            out.append(campaign_line)
        return "\n".join(out)
    except Exception as e:  # noqa: BLE001 — prompt assembly must never break here
        logger.debug(f"[prior_calls] context block failed for {coin}: {e}")
        return ""


def _campaign_line(coin: str, memory, now_ms: int, lookback_ms: float) -> str:
    """'Last completed round trip on COIN: …' from the realized-outcome store."""
    try:
        c = memory.last_close_for(coin)
        if not c or (c.get("closed_at") or 0) < now_ms - lookback_ms:
            return ""
        side = str(c.get("side", "?")).upper()
        pnl_pct = c.get("realized_pnl_pct")
        pnl_usd = c.get("realized_pnl_usd")
        parts = [f"{side} entry {_fmt_px(c.get('entry_px'))} → exit "
                 f"{_fmt_px(c.get('exit_px'))}"]
        if pnl_pct is not None:
            usd_s = f" ({float(pnl_usd):+.2f} USD)" if pnl_usd is not None else ""
            parts.append(f"= {float(pnl_pct):+.2f}% leveraged net of fees{usd_s}")
        hold = c.get("hold_minutes")
        if hold:
            parts.append(f"held {hold:.0f}min")
        reason = c.get("exit_reason")
        if reason:
            parts.append(f"exit reason: {reason}")
        return (f"  - Last completed round trip on {coin} "
                f"[{_ago(c.get('closed_at'), now_ms)}]: " + ", ".join(parts) + ".")
    except Exception:  # noqa: BLE001 — campaign line is optional garnish
        return ""
