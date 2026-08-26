"""Model-duel store — A/B evaluation of the primary LLM against a second
("duelist") LLM, both answering the SAME research prompt.

The duelist is a pure OBSERVER: it never executes, never gates, and never
appears in the system prompt (so it cannot know it is being compared). Its
verdict is recorded next to the primary's and, when a trade opened from the
primary's verdict later closes, the same entry-context snapshot that the
forward signal backtest uses (executor.record_entry_context -> record_close)
carries the duelist verdict into the outcome row. That join is what makes the
A/B report honest: both models are scored on the SAME realized trades, and the
duelist's column is "what would have happened if it had been live".

Persistence is an append-only JSONL (one line per paired call) — the ledger
pattern, NOT agent-memory: the duel log is an evaluation artifact that grows
unboundedly and is never truncated, and a corrupt line is one lost row, not a
wiped live state. Path is overridable via HERMES_DUEL_FILE (conftest isolates
it, like HERMES_LEDGER_FILE).

Enable with env (all three; model alone also works, inheriting url/key):
    LLM_DUEL_BASE_URL=...  LLM_DUEL_MODEL=...  LLM_DUEL_API_KEY=...
When unset the feature is fully dormant: zero extra LLM calls, zero rows, and
the primary path is byte-for-byte the old behavior (it still just calls
_call_ai, which now accepts explicit endpoint args with the same env fallbacks).

Thread-safety: research runs on a worker pool (research_max_workers > 1), so
all file appends take a module lock (the session_log pattern) and the duelist
call runs in a fresh event loop on the calling thread (the _call_ai pattern —
never a shared loop across threads).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Env var names (DUEL prefix: this is the second, observation-only model).
# Read at CALL time — mirroring _call_ai's primary LLM_* handling — so a
# test can monkeypatch the env and a running process picks up a new duelist
# model without a restart.
_DUEL_URL_VARS = ("LLM_DUEL_BASE_URL",)
_DUEL_MODEL_VARS = ("LLM_DUEL_MODEL",)
_DUEL_KEY_VARS = ("LLM_DUEL_API_KEY",)

# Overridable for tests (mirrors HERMES_LEDGER_FILE / HERMES_AGENT_MEMORY_FILE).
_DUEL_FILE = os.environ.get(
    "HERMES_DUEL_FILE",
    os.path.expanduser("~/.hermes-trader-duel.jsonl"),
)

_log_lock = threading.Lock()


def _first_env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n, "")
        if v:
            return v
    return ""


DEFAULT_MAX_TOKENS = 32768


def resolve_max_tokens(env_name: str, fallback: int = DEFAULT_MAX_TOKENS) -> int:
    """The env var's completion-token budget (fallback when unset/invalid/non-positive).

    Read at CALL time (same pattern as the LLM_* endpoint vars) so an
    operator can retune it without a rebuild. It caps the response length
    ONLY — it is NOT a prompt/context limit; the model server's ctx_size is
    the hard cap on input+output (2026-08-25 incident: the duelist returned
    200 OK at n_tokens=4191, well under the then-hardcoded 8192).
    """
    raw = os.environ.get(env_name, "").strip()
    if raw.isdigit():
        v = int(raw)
        if v > 0:
            return v
    return fallback


def duel_file() -> str:
    """Current duel-log path (read at call time so tests can redirect)."""
    return os.environ.get("HERMES_DUEL_FILE", _DUEL_FILE)


def duelist_config() -> Dict[str, Any]:
    """The duelist endpoint, resolved from env.

    Only the MODEL is duelist-specific and REQUIRED — base_url/api_key fall
    back to the PRIMARY LLM's values when unset, so pointing only
    LLM_DUEL_MODEL at a differently-named model on the same server is a
    one-line change. The model deliberately does NOT fall back to LLM_MODEL:
    a silent "duel the primary against itself" would double LLM load with no
    A/B value — the feature is dormant until a duelist model is named.
    """
    return {
        "base_url": _first_env(*_DUEL_URL_VARS)
        or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key": _first_env(*_DUEL_KEY_VARS)
        or os.environ.get("LLM_API_KEY", os.environ.get("OPENROUTER_API_KEY", "")),
        "model": _first_env(*_DUEL_MODEL_VARS),
        # LLM_DUEL_MAX_TOKENS, falling back to the primary's LLM_MAX_TOKENS,
        # then to DEFAULT_MAX_TOKENS — mirrors the base_url/api_key fallback
        # (setting one var controls both models).
        "max_tokens": resolve_max_tokens(
            "LLM_DUEL_MAX_TOKENS", resolve_max_tokens("LLM_MAX_TOKENS")
        ),
    }


def duelist_enabled() -> bool:
    return bool(duelist_config().get("model"))


# ── Store ──────────────────────────────────────────────────────────────────

def record_duel(entry: Dict[str, Any]) -> None:
    """Append one paired-call row. Best-effort: an A/B artifact must never
    interrupt trading, so disk errors are swallowed (session_log pattern)."""
    row = {"ts": int(time.time() * 1000), **entry}
    try:
        with _log_lock:
            with open(duel_file(), "a") as f:
                f.write(json.dumps(row) + "\n")
    except OSError:
        pass


def load_duels(limit: int = 5000) -> List[Dict[str, Any]]:
    """All recorded rows, oldest first, malformed lines skipped."""
    out: List[Dict[str, Any]] = []
    try:
        lines = [ln for ln in open(duel_file()).read().splitlines() if ln.strip()]
    except FileNotFoundError:
        return []
    for ln in lines[-limit:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def resolve(coin: str, perception_id: str, max_scan: int = 400) -> Optional[Dict[str, Any]]:
    """The duelist row for a perception, or None when the duelist was disabled
    or produced no row. Scans from the end (recent first) and stops at the
    matching perception_id — perception ids are process-unique uuids, so the
    first match is THE record for this research call.

    Callers must pass the ORIGINAL perception dict (not a trimmed copy) — the
    executor's record_entry_context snapshot happens in the execute path,
    where only the analysis dict survives.
    """
    if not coin or not perception_id or perception_id == "unknown":
        return None
    for row in reversed(load_duels(limit=max_scan)):
        if row.get("perception_id") == perception_id:
            return row
    return None


# ── Live call ──────────────────────────────────────────────────────────────

def call_duelist(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_message: str,
    timeout_s: float = 120.0,
    max_tokens: Optional[int] = None,
) -> str:
    """POST the SAME prompt to the duelist endpoint. Returns the raw text (""
    on any failure) and NEVER raises — a duelist outage must not cost the
    primary's verdict. Same shape as research._async_do_call (402-affordability
    retry included), with the 402 branch omitted: the duelist is a
    shadow/eval consumer, so a paid-provider credit failure degrades to a
    missing row rather than burning a shrunk call.
    """
    if not api_key:
        logger.warning("[duel] duelist LLM_API_KEY not set — skipping duelist call")
        return ""
    if max_tokens is None:
        max_tokens = resolve_max_tokens(
            "LLM_DUEL_MAX_TOKENS", resolve_max_tokens("LLM_MAX_TOKENS")
        )
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _async_duel_call(api_key, base_url, model, system_prompt, user_message,
                             timeout_s, max_tokens)
        )
    except Exception as e:  # noqa: BLE001 — the primary path must survive any duelist fault
        logger.debug(f"[duel] duelist call failed (non-fatal): {type(e).__name__}: {e}")
        return ""
    finally:
        loop.close()


async def _async_duel_call(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_message: str,
    timeout_s: float,
    max_tokens: int,
) -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        url = base_url.rstrip("/") + "/chat/completions"

        async def _post(max_toks: int):
            return await client.post(
                url,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "max_tokens": max_toks,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )

        resp = await _post(max_tokens)
        if resp.status_code == 402:
            m = re.search(r"can only afford (\d+)", resp.text or "")
            if m and int(m.group(1)) >= 500:
                resp = await _post(int(m.group(1)) - 50)
        if resp.is_success:
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                return msg.get("content") or msg.get("reasoning") or ""
            logger.error("[duel] duelist returned 200 but no choices")
            return ""
        logger.warning(f"[duel] duelist call FAILED: HTTP {resp.status_code} (non-fatal)")
    return ""


# ── Report ─────────────────────────────────────────────────────────────────

def _model_stats(pnls: List[Optional[float]]) -> Dict[str, Any]:
    vals = [p for p in pnls if p is not None]
    if not vals:
        return {"closes": 0, "wins": 0, "losses": 0, "win_rate": None,
                "realized_pnl_usd": None, "avg_pnl_usd": None,
                "avg_win_usd": None, "avg_loss_usd": None}
    wins = [p for p in vals if p > 0]
    losses = [p for p in vals if p < 0]
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = abs(sum(losses) / len(losses)) if losses else None
    return {
        "closes": len(vals),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(vals), 3),
        "realized_pnl_usd": round(sum(vals), 2),
        "avg_pnl_usd": round(sum(vals) / len(vals), 2),
        "avg_win_usd": round(avg_win, 2) if avg_win is not None else None,
        "avg_loss_usd": round(avg_loss, 2) if avg_loss is not None else None,
    }


def _duelist_pnl(close: Dict[str, Any], dl: Dict[str, Any]) -> Optional[float]:
    """The duelist's realized P&L IF its verdict had been the live one.

    - Verdict direction == live side (it concurs): identical to the live P&L.
    - Verdict direction == opposite side (it would have traded against us):
      mirrored at the same % magnitude (the exchange path is symmetric:
      leveraged spot move flips sign, fees are side-agnostic).
    - PASS/CLOSE (it would have done nothing, or it closed): 0.0 — a flat
      outcome, so it counts as a "win" only when the live trade lost (the
      comparison that matters: did the duelist avoid the loss / take the win?).
    """
    side = close.get("side")
    dl_side = dl.get("side")
    pct = close.get("realized_pnl_pct")
    live_usd = close.get("realized_pnl_usd")
    if dl_side in ("long", "short"):
        if dl_side == side:
            return live_usd
        if dl_side != side and pct is not None:
            return round(-1 * (live_usd if live_usd is not None else 0.0), 2)
    return 0.0


def _latency_stats(vals: List[Optional[float]]) -> Dict[str, Any]:
    """Mean/median/max wall time of the LLM calls that reported a latency.

    Rows written before the latency fields shipped have no *_ms and are
    excluded from the mean/median (the sample size reflects that), so
    enabling the duelist mid-run can't skew the numbers with zeros.
    """
    nums = [v for v in vals if v is not None]
    if not nums:
        return {"n": 0, "avg_ms": None, "median_ms": None, "max_ms": None}
    nums_sorted = sorted(nums)
    n = len(nums_sorted)
    mid = n // 2
    median = (nums_sorted[mid] + nums_sorted[mid - 1]) / 2 if n % 2 == 0 else float(nums_sorted[mid])
    return {
        "n": n,
        "avg_ms": round(sum(nums) / n, 1),
        "median_ms": round(median, 1),
        "max_ms": max(nums),
    }


def aggregate() -> Dict[str, Any]:
    """The A/B report: primary vs duelist, scored on the SAME realized trades.

    Join: duel rows (perception_id) -> entry context (record_entry_context)
    -> outcome store (record_close carries duelist_at_entry when it was
    snapshotted). Positions closed before this shipped (or entered while the
    duelist was disabled) have no duelist column and count toward the
    primary's stats only.
    """
    from hermes_trader.agents.memory import memory

    # Idempotent: the trading loop already loaded, but the `hermes duel` CLI
    # (and any dashboard/MCP caller) imports memory fresh — hydrate so
    # get_closes() sees the persisted outcome rows. load() is read-only here
    # (aggregate never flushes).
    memory.load()

    duels = load_duels()
    closes = memory.get_closes() or []
    by_id = {d.get("perception_id"): d for d in duels if d.get("perception_id")}

    primary_pnls: List[Optional[float]] = []
    duelist_pnls: List[Optional[float]] = []
    matched = 0
    for c in closes:
        p_pct = c.get("realized_pnl_pct")
        p_usd = c.get("realized_pnl_usd")
        primary_pnls.append(p_usd if p_usd is not None else
                            (round((p_pct or 0) / 100 * (c.get("notional_usd") or 0), 2)))
        dl = c.get("duelist_at_entry")
        if not isinstance(dl, dict) or not dl:
            continue
        matched += 1
        # Backfill the verdict from the duel row when the entry snapshot only
        # carried the side (defensive — both are written together, but the
        # report should not depend on field completeness of one write).
        row = by_id.get(c.get("perception_id")) or {}
        dl_eff = {**row, **dl}
        duelist_pnls.append(_duelist_pnl(c, dl_eff))

    # Verdict agreement on the paired calls (independent of execution), and the
    # wall time each model's LLM call took (ms, from the row's *_ms fields —
    # absent on rows written before latency tracking shipped).
    agree = 0
    splits = 0
    dl_verdicts: Dict[str, int] = {}
    primary_ms: List[Optional[float]] = []
    duelist_ms: List[Optional[float]] = []
    for d in duels:
        v = d.get("duelist_verdict")
        if v:
            dl_verdicts[v] = dl_verdicts.get(v, 0) + 1
        if v and d.get("primary_verdict"):
            if v == d.get("primary_verdict"):
                agree += 1
            else:
                splits += 1
        primary_ms.append(d.get("primary_ms"))
        duelist_ms.append(d.get("duelist_ms"))

    return {
        "duel_calls": len(duels),
        "realized_closes": len(closes),
        "closes_with_duelist": matched,
        "duelist": {"model": duelist_config().get("model") or "(disabled)"},
        "agreement": {"agree": agree, "split": splits,
                      "rate": round(agree / (agree + splits), 3) if (agree + splits) else None},
        "primary": _model_stats(primary_pnls),
        "duelist_if_live": _model_stats(duelist_pnls),
        "duelist_verdicts": dl_verdicts,
        "latency": {"primary": _latency_stats(primary_ms),
                    "duelist": _latency_stats(duelist_ms)},
    }


def _latency_line(label: str, s: Dict[str, Any]) -> str:
    if not s["n"]:
        return f"  {label:<10} latency  --  (no *_ms fields recorded yet)"
    return (f"  {label:<10} latency  avg {s['avg_ms']:>8.1f} ms   "
            f"median {s['median_ms']:>8.1f} ms   max {s['max_ms']:>9.1f} ms   "
            f"(n={s['n']})")


def print_report() -> None:
    """Human-readable A/B report for the `hermes duel` CLI command."""
    r = aggregate()
    p, d = r["primary"], r["duelist_if_live"]
    a = r["agreement"]
    print(f"Model duel — {r['duel_calls']} paired call(s), "
          f"{r['realized_closes']} realized close(s), {r['closes_with_duelist']} with duelist verdict")
    print(f"  duelist model: {r['duelist']['model']}")
    print(f"  verdict agreement: {a['agree']} agree / {a['split']} split ", end="")
    if a["rate"] is None:
        print("--")
    else:
        print(f"{a['rate'] * 100:.0f}% agree")
    print(f"  duelist verdict mix: {r['duelist_verdicts'] or '{}'}")
    print(_latency_line("primary", r["latency"]["primary"]))
    print(_latency_line("duelist", r["latency"]["duelist"]))
    if r["closes_with_duelist"] == 0:
        print("\n  No realized trades carry a duelist verdict yet — the P&L table")
        print("  fills in as trades opened since the duelist shipped get closed.")
        return

    def _line(label: str, s: Dict[str, Any]) -> str:
        wr = f"{s['win_rate'] * 100:5.1f}%" if s["win_rate"] is not None else "   --"
        pnl = f"{s['realized_pnl_usd']:+9.2f}" if s["realized_pnl_usd"] is not None else "       --"
        avg = f"{s['avg_pnl_usd']:+7.2f}" if s["avg_pnl_usd"] is not None else "     --"
        aw = f"{s['avg_win_usd']:+6.2f}" if s["avg_win_usd"] is not None else "   --"
        al = f"{s['avg_loss_usd']:+6.2f}" if s["avg_loss_usd"] is not None else "   --"
        return (f"  {label:<10} closes {s['closes']:>3}  win {s['wins']:>3}/{s['closes']:<3} "
                f"WR {wr}  PnL {pnl} USD  avg {avg}  win {aw} / loss -{al}")

    print()
    print(_line("primary", p))
    print(_line("duelist", d))
    print("  (duelist = what would have happened if its verdict had been the live one;")
    print("   PASS/CLOSE verdicts on a trade score 0 USD)")