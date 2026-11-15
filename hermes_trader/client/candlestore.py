"""Persistent candle cache — the durable, queryable HL bar store (t_18e7517a).

Disk cache ONLY by owner ruling: the live read path keeps serving from
hl_client's in-memory ``_CANDLE_WINDOWS`` rolling window unchanged; this store
never sits in the decision path. It exists because HL retention is tiered and
history past it is UNRECOVERABLE (1m ~3.5d, 5m ~17d, 15m ~51d, 1h ~208d, and
candleSnapshot silently returns [] / the recent tail for older windows), so any
counterfactual older than the horizon needs bars captured as they pass by.

Design (plan .hermes/plans/2026-11-15_014558-persistent-candle-cache.md):
  * harvest-on-pass: ``hl_client.fetch_hl_candles`` calls :func:`harvest` after
    every successful fetch — ZERO extra rate weight; the scan loop's own
    5m/1h/4h/1d traffic accumulates the store.
  * CONFIRMED BARS AT WRITE TIME: a bar is stored only once ``t + dur <= now``.
    candleSnapshot's last row is the still-forming candle; a store containing
    it poisons every replay that trusts it, so the guard lives HERE, not in
    each caller (the old scratch/candlestore left that to callers).
  * rows are immutable (INSERT OR IGNORE, PK (coin, interval, t)).
  * posture = env dir (mount point) + hot config block ``candle_store.enabled``
    (config-repo deploy flips it; env alone does NOT enable).
  * SQLite WAL: safe for the container writer + host-side reader scripts.
  * soft size cap ``max_store_mb`` (default 1024 ≈ 13.4M bars at the measured
    74.8 B/row BLOB packing): prune is value-order (1m → 15m → 5m → 1h/4h/1d),
    oldest-first. History is unrecoverable, so this trades depth for size
    deliberately.

Reader API (window semantics match the old scratch/candlestore/candles.py):
    from hermes_trader.client import candlestore
    bars  = candlestore.series(coin, "5m", t0_ms, t1_ms)   # dicts, ascending
    cov   = candlestore.coverage(coin, "5m")               # [(YYYYMMDD, n)]
    coins = candlestore.coins()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

DB_NAME = "candles.db"

# Bar duration per interval, ms. (Mirrors hl_client._MS_PER_CANDLE; kept local
# so hl_client can import this module for the harvest hook without a cycle.)
MS_PER_INTERVAL: Dict[str, int] = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

# Measured HL candleSnapshot retention tiers, days (2026-10-22 probe): windows
# older than this silently return []. Config may override via retention_days.
RETENTION_DAYS: Dict[str, float] = {
    "1m": 3.5, "5m": 17, "15m": 51, "1h": 208, "4h": 700, "1d": 3650,
}

# candleSnapshot silently truncates past this many bars per request.
MAX_BARS_PER_FETCH = 5000

# Value order for size-cap pruning: the lowest-value tier's OLDEST bars go first.
PRUNE_ORDER: Sequence[str] = ("1m", "15m", "5m", "1h", "4h", "1d")

# Rough bytes/row for the size estimate (measured 74.8 B/row, 1M-row probe).
_ROW_BYTES_EST = 75

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  coin     TEXT NOT NULL,     -- stored verbatim ('BTC', 'xyz:NVDA')
  interval TEXT NOT NULL,     -- '1m','5m','15m','1h','4h','1d'
  t        INTEGER NOT NULL,  -- bar OPEN time, epoch ms (HL 't')
  ohlcv    BLOB NOT NULL,     -- struct.pack('<5f', o,h,l,c,v)
  PRIMARY KEY (coin, interval, t)
);
"""

# ── posture ─────────────────────────────────────────────────────────────

_CFG_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "harvest": True,
    "backfill": True,
    "intervals": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "max_store_mb": 1024,
    "prune_max_days": {"1m": 7},
    "backfill_max_weight_per_min": 200,
    "backfill_interval_min": 10,
    "retention_days": {},
}

_cfg_cache: Dict[str, Any] = {"at": 0.0, "val": None}


def _cfg() -> Dict[str, Any]:
    """Hot candle_store block, ~5s cached (harvest runs ~300x/min)."""
    now = time.time()
    if _cfg_cache["val"] is not None and now - _cfg_cache["at"] < 5.0:
        return _cfg_cache["val"]
    try:
        from hermes_trader.agents.config_store import read_agent_config
        raw = read_agent_config().get("candle_store") or {}
    except Exception:
        raw = _cfg_cache["val"] or {}
    val = {**_CFG_DEFAULTS, **raw}
    _cfg_cache.update(at=now, val=val)
    return val


def store_dir() -> Optional[str]:
    return os.environ.get("HERMES_CANDLESTORE_DIR") or None


DAY_MS = 86_400_000


def horizon_days(interval: str) -> float:
    """Retention horizon for an interval (config-overridable)."""
    cfg = _cfg()
    return float((cfg.get("retention_days") or {}).get(
        interval, RETENTION_DAYS.get(interval, 17)))


def enabled() -> bool:
    """Master posture: mount env present AND config block explicitly enabled.

    Env alone does NOT enable — the deploy ships with the block absent/false
    (byte-identical fetch path), then the config repo flips
    ``candle_store.enabled`` hot without a recreate.
    """
    return store_dir() is not None and bool(_cfg().get("enabled", False))


# ── store ───────────────────────────────────────────────────────────────

class CandleStore:
    """Shared connection per process (container writer + host readers on WAL).

    Writes are short transactions behind a lock: harvest is called from scan
    worker threads, the backfill thread shares the process, and host-side
    scripts/CLIs open stores of their own — WAL + busy_timeout covers
    cross-process, the lock covers cross-thread.
    """

    def __init__(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self.path = os.path.join(dir_path, DB_NAME)
        self._lock = threading.Lock()
        self._con = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA synchronous=NORMAL")
        self._con.execute("PRAGMA busy_timeout=10000")
        self._con.executescript(_SCHEMA)
        self._con.commit()

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

    # ── write ───────────────────────────────────────────────────────────

    def upsert_bars(self, coin: str, interval: str,
                    bars: Iterable[Any], now_ms: Optional[int] = None) -> int:
        """Insert confirmed bars (INSERT OR IGNORE); returns rows inserted.

        The confirmed-bar guard lives HERE so EVERY writer (harvest, backfill,
        import) enforces it: a bar is storable only when ``t + dur <= now_ms``
        — candleSnapshot's last row is the still-forming candle and must never
        be persisted. Rows are immutable: an existing (coin,interval,t) is
        never overwritten (a stored bar is by definition already closed).
        """
        dur = MS_PER_INTERVAL.get(interval, 300_000)
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms

        def _f(x, default=None):
            if x is None:
                return default
            try:
                return float(x)
            except (TypeError, ValueError):
                return default

        rows = []
        for b in bars:
            if isinstance(b, dict):
                t, o, h, l, c, v = (b.get("t"), b.get("o"), b.get("h"),
                                    b.get("l"), b.get("c"), b.get("v"))
            else:  # Candle (pydantic) or any attr-holder
                t, o, h, l, c, v = b.t, b.o, b.h, b.l, b.c, b.v
            # Legacy scratch chunks occasionally carry None volume (and
            # string numerics); a bar with an unusable PRICE is skipped —
            # a stored bar must be complete or absent (replays read blind).
            o, h, l, c = _f(o), _f(h), _f(l), _f(c)
            v = _f(v, 0.0)
            if t is None or o is None or h is None or l is None or c is None:
                continue
            t = int(t)
            if t + dur > now_ms:
                continue  # still-forming candle — never persisted
            rows.append((coin, interval, t,
                         struct.pack("<5f", o, h, l, c, v)))
        if not rows:
            return 0
        with self._lock:
            before = self._con.total_changes
            self._con.executemany(
                "INSERT OR IGNORE INTO bars(coin,interval,t,ohlcv) VALUES(?,?,?,?)",
                rows)
            self._con.commit()
            return self._con.total_changes - before

    # ── read ────────────────────────────────────────────────────────────

    def series(self, coin: str, interval: str,
               t0_ms: Optional[int] = None,
               t1_ms: Optional[int] = None) -> List[Dict[str, Any]]:
        """Bars with t0 <= t < t1 (either end optional), ascending, dicts."""
        q = "SELECT t, ohlcv FROM bars WHERE coin=? AND interval=?"
        args: list = [coin, interval]
        if t0_ms is not None:
            q += " AND t>=?"
            args.append(int(t0_ms))
        if t1_ms is not None:
            q += " AND t<?"
            args.append(int(t1_ms))
        q += " ORDER BY t"
        out = []
        for t, blob in self._con.execute(q, args):
            o, h, l, c, v = struct.unpack("<5f", blob)
            out.append({"t": t, "o": o, "h": h, "l": l, "c": c, "v": v})
        return out

    def coverage(self, coin: str, interval: str) -> List[tuple]:
        """[(YYYYMMDD, n_bars), ...] — per-UTC-day rollup, sorted."""
        out = []
        rows = self._con.execute(
            "SELECT CAST(t/86400000 AS INTEGER) d, COUNT(*) FROM bars "
            "WHERE coin=? AND interval=? GROUP BY d ORDER BY d",
            (coin, interval)).fetchall()
        for day_epoch, n in rows:
            day = datetime.fromtimestamp(day_epoch * 86_400, tz=timezone.utc)
            out.append((day.strftime("%Y%m%d"), n))
        return out

    def coins(self) -> List[str]:
        return [r[0] for r in self._con.execute(
            "SELECT DISTINCT coin FROM bars ORDER BY coin")]

    def count_bars(self, coin: Optional[str] = None,
                   interval: Optional[str] = None) -> int:
        q = "SELECT COUNT(*) FROM bars"
        args: list = []
        if coin is not None:
            q += " WHERE coin=?"
            args.append(coin)
        if interval is not None:
            q += (" AND" if coin is not None else " WHERE") + " interval=?"
            args.append(interval)
        return self._con.execute(q, args).fetchone()[0]

    # ── gap planner (retention-honest) ─────────────────────────────────

    def gap_plan(self, coin: str, interval: str,
                 now_ms: Optional[int] = None) -> Dict[str, Any]:
        """Gap bookkeeping for one (coin, interval) at ``now_ms``.

        Never plan a fetch whose start sits inside the dead window past the
        interval horizon — candleSnapshot silently returns [] there. The part
        of the gap older than the horizon is reported ``unrecoverable``; only
        ``[max(gap_start, now-horizon), now)`` is backfillable. An empty store
        has no known gap start: its whole horizon window is the seed plan.
        """
        dur = MS_PER_INTERVAL.get(interval, 300_000)
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        horizon_ms = horizon_days(interval) * DAY_MS
        row = self._con.execute(
            "SELECT MAX(t) FROM bars WHERE coin=? AND interval=?",
            (coin, interval)).fetchone()
        last_t = row[0]
        if last_t is None:
            gap_start = now_ms - horizon_ms
            unrecoverable = 0  # pre-store span; seed covers the horizon
        else:
            gap_start = last_t + dur
            unrecoverable = max(0, (now_ms - horizon_ms) - gap_start)
        gap_ms = max(0, now_ms - gap_start)
        lo = max(gap_start, now_ms - horizon_ms)
        return {
            "coin": coin, "interval": interval, "last_t": last_t,
            "gap_ms": gap_ms,
            "unrecoverable_ms": unrecoverable,
            "backfillable_ms": max(0, now_ms - lo),
        }

    def backfill_plans(self, coins: Sequence[str], intervals: Sequence[str],
                       now_ms: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch windows <= MAX_BARS_PER_FETCH, newest-first per (coin, iv).

        Older ends of long gaps are silently truncated past the 5000-bar cap,
        so a gap wider than the cap splits into multiple windows.
        """
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        plans: List[Dict[str, Any]] = []
        for coin in coins:
            for iv in intervals:
                dur = MS_PER_INTERVAL.get(iv, 300_000)
                g = self.gap_plan(coin, iv, now_ms=now_ms)
                if g["backfillable_ms"] <= 0:
                    continue
                lo = now_ms - g["backfillable_ms"]
                hi = now_ms
                while hi > lo:
                    start = max(lo, hi - MAX_BARS_PER_FETCH * dur)
                    plans.append({"coin": coin, "interval": iv,
                                  "start_ms": int(start), "end_ms": int(hi)})
                    hi = start
        return plans

    # ── pruning (deliberate: history is unrecoverable) ─────────────────

    def prune_hard_caps(self, now_ms: int,
                        prune_max_days: Dict[str, float]) -> int:
        """Hard per-interval day caps (e.g. 1m@7d) — runs every cycle."""
        removed = 0
        with self._lock:
            for iv, days in (prune_max_days or {}).items():
                cur = self._con.execute(
                    "DELETE FROM bars WHERE interval=? AND t < ?",
                    (iv, now_ms - float(days) * DAY_MS))
                removed += cur.rowcount
            self._con.commit()
        return removed

    def prune_to_size(self, max_bytes: float, batch_rows: int = 50_000,
                      now_ms: Optional[int] = None) -> int:
        """Enforce the soft size cap: VALUE-ORDER tiers, oldest-first.

        Lowest-value tier's oldest bars go first (1m -> 15m -> 5m -> 1h/4h/1d);
        one tier before spilling into the next, <= batch_rows deleted per
        call (the trickle re-enters next cycle if still over).
        """
        if self.est_bytes() <= max_bytes:
            return 0
        removed = 0
        with self._lock:
            for tier in PRUNE_ORDER:
                if removed >= batch_rows:
                    break
                cur = self._con.execute(
                    "DELETE FROM bars WHERE interval=? AND rowid IN "
                    "(SELECT rowid FROM bars WHERE interval=? ORDER BY t "
                    "LIMIT ?)", (tier, tier, batch_rows - removed))
                removed += cur.rowcount
            self._con.commit()
        if removed:
            logger.info(f"[candlestore] pruned {removed} rows "
                        f"to stay under max_store_mb")
        return removed

    # ── size accounting ─────────────────────────────────────────────────

    def est_bytes(self) -> int:
        """Effective db bytes: (page_count - freelist_count) * page_size.

        Uses the freelist so DELETEs (which park pages on the freelist) count
        as reclaimed without forcing an immediate VACUUM.
        """
        pc = self._con.execute("PRAGMA page_count").fetchone()[0]
        fl = self._con.execute("PRAGMA freelist_count").fetchone()[0]
        ps = self._con.execute("PRAGMA page_size").fetchone()[0]
        return max(0, pc - fl) * ps


# ── module-level store handle ───────────────────────────────────────────

_store_lock = threading.Lock()
_store: Optional[CandleStore] = None
_store_path: Optional[str] = None


def get_store() -> Optional[CandleStore]:
    """Store singleton for HERMES_CANDLESTORE_DIR; None when unset."""
    global _store, _store_path
    d = store_dir()
    if d is None:
        return None
    with _store_lock:
        if _store is None or _store_path != d:
            if _store is not None:
                _store.close()
            _store = CandleStore(d)
            _store_path = d
        return _store


def reset_store() -> None:
    """Close + drop the singleton (tests; env changes)."""
    global _store, _store_path
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None
        _store_path = None
        _cfg_cache.update(at=0.0, val=None)


# ── harvest (the zero-weight accumulation engine) ───────────────────────

_harvest_err_state = {"count": 0, "last_msg": None}


def harvest(coin: str, interval: str, candles: Iterable[Any],
            now_ms: Optional[int] = None) -> None:
    """Persist confirmed bars from a live fetch. NEVER raises.

    Called by hl_client.fetch_hl_candles after every successful
    candleSnapshot — no extra API weight, decision path untouched (the caller
    returns exactly what it always returned). Fail-open with a once-warned
    error: a broken store must never affect a live scan.
    """
    try:
        if not enabled() or not _cfg().get("harvest", True):
            return
        store = get_store()
        if store is None:
            return
        store.upsert_bars(coin, interval, candles, now_ms=now_ms)
    except Exception as e:
        _harvest_err_state["count"] += 1
        msg = str(e)
        if _harvest_err_state["last_msg"] != msg:
            logger.warning(
                f"[candlestore] harvest error (fail-open, live fetch "
                f"unaffected): {e}")
            _harvest_err_state["last_msg"] = msg


# Convenience module-level readers (scripts/counterfactuals)

def series(coin: str, interval: str, t0_ms: Optional[int] = None,
           t1_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    store = get_store()
    return store.series(coin, interval, t0_ms, t1_ms) if store else []


def coverage(coin: str, interval: str) -> List[tuple]:
    store = get_store()
    return store.coverage(coin, interval) if store else []


def coins() -> List[str]:
    store = get_store()
    return store.coins() if store else []


# ── backfill runner (paced; shares the live HL rate limiter) ────────────

def _default_fetch(coin: str, interval: str,
                   start_ms: int, end_ms: int) -> Optional[list]:
    """Raw candleSnapshot via hl_client._http_post — SAME limiter/session as
    the live loop (per-IP budget). Returns list (possibly empty) or None on
    rate-budget exhaustion / transient failure."""
    from hermes_trader.client import hl_client
    payload = {"type": "candleSnapshot",
               "req": {"coin": coin, "interval": interval,
                       "startTime": int(start_ms), "endTime": int(end_ms)}}
    return hl_client._http_post("/info", payload)


def backfill_once(store: CandleStore, plans: Sequence[Dict[str, Any]],
                  fetch_fn=None, now_ms: Optional[int] = None,
                  max_fetches: Optional[int] = None) -> Dict[str, Any]:
    """Execute planned windows; NOTHING is written for []/None responses.

    None (rate budget / transient) = skip, retried next cycle. [] = legitimate
    no-history (delisted / pre-listing) — counted, nothing stored. The
    confirmed-bar guard inside upsert_bars protects the newest window edge.
    """
    fetch_fn = fetch_fn or _default_fetch
    stats = {"fetches_used": 0, "fetched": 0, "stored": 0,
             "empty": 0, "failed": 0, "deferred": 0}
    for i, p in enumerate(plans):
        if max_fetches is not None and stats["fetches_used"] >= max_fetches:
            stats["deferred"] = len(plans) - i
            break
        stats["fetches_used"] += 1
        raw = fetch_fn(p["coin"], p["interval"], p["start_ms"], p["end_ms"])
        if not isinstance(raw, list):
            stats["failed"] += 1
            continue
        stats["fetched"] += 1
        if not raw:
            stats["empty"] += 1
            continue
        stats["stored"] += store.upsert_bars(
            p["coin"], p["interval"], raw, now_ms=now_ms)
    return stats


def target_coins(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Decision A (owner 2026-11-15): BTC + every coin ever in the ledger +
    the live scan universe (top-N by 24h volume, HIP-3 included). Not all HL
    perps — cold coins accrue via harvest only if some consumer reads them.
    """
    out: List[str] = ["BTC"]
    seen = {"BTC"}
    try:
        from hermes_trader.ledger import LEDGER_FILE
        with open(LEDGER_FILE) as f:
            for ln in f:
                try:
                    c = json.loads(ln).get("coin")
                except Exception:
                    continue
                if c and c not in seen:
                    seen.add(c)
                    out.append(c)
    except Exception:
        pass  # no ledger yet (fresh install)
    try:
        from hermes_trader.client.universe import get_universe
        limit = int(os.environ.get("HERMES_MAX_MARKETS", "60"))
        for m in get_universe(include_hip3=True)[:limit]:
            c = m.get("coin")
            if c and c not in seen:
                seen.add(c)
                out.append(c)
    except Exception:
        pass  # universe fetch failed; ledger set still useful
    return out


def run_backfill_cycle(fetch_fn=None,
                       coins: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """One maintenance pass: hard day caps -> gap backfill (paced) -> size cap.

    Returns stats, or None when posture is off (master switch / no mount).
    ``fetch_fn``/``coins`` injectable for tests and one-shot seed runs.
    """
    cfg = _cfg()
    if not enabled() or not cfg.get("backfill", True):
        return None
    store = get_store()
    if store is None:
        return None
    now_ms = int(time.time() * 1000)
    hard = store.prune_hard_caps(
        now_ms=now_ms, prune_max_days=cfg.get("prune_max_days") or {})
    coin_list = list(coins) if coins is not None else target_coins(cfg)
    plans = store.backfill_plans(
        coin_list, cfg.get("intervals") or list(MS_PER_INTERVAL),
        now_ms=now_ms)
    # Sustained-rate bound: weight/min * cycle-minutes / 20 weight per fetch.
    max_fetches = int(float(cfg.get("backfill_max_weight_per_min", 200))
                      * float(cfg.get("backfill_interval_min", 10)) / 20.0)
    stats = backfill_once(store, plans, fetch_fn=fetch_fn,
                          now_ms=now_ms, max_fetches=max_fetches)
    pruned = 0
    max_bytes = float(cfg.get("max_store_mb", 1024)) * 1_000_000
    if store.est_bytes() > max_bytes:
        pruned = store.prune_to_size(max_bytes=max_bytes,
                                     batch_rows=50_000, now_ms=now_ms)
    stats.update(plans=len(plans), coins=len(coin_list),
                 hard_pruned=hard, pruned=pruned,
                 est_mb=round(store.est_bytes() / 1e6, 1))
    logger.info(
        f"[candlestore] cycle: coins={stats['coins']} "
        f"plans={stats['plans']} fetched={stats['fetched']} "
        f"stored={stats['stored']} empty={stats['empty']} "
        f"failed={stats['failed']} deferred={stats['deferred']} "
        f"hard_pruned={hard} pruned={pruned} est_mb={stats['est_mb']}")
    return stats


_thread_state: Dict[str, Any] = {"thread": None, "stop": None}


def start_backfill_thread() -> Optional[threading.Thread]:
    """Daemon thread: first cycle after ~90 s (post startup-grace), then every
    backfill_interval_min. Returns None (no thread) when posture is off."""
    if store_dir() is None or not enabled():
        return None
    if _thread_state["thread"] is not None and _thread_state["thread"].is_alive():
        return _thread_state["thread"]
    stop = threading.Event()

    def _loop():
        wait = 90.0
        while not stop.wait(wait):
            wait = float(_cfg().get("backfill_interval_min", 10)) * 60.0
            try:
                run_backfill_cycle()
            except Exception as e:
                logger.warning(f"[candlestore] backfill cycle failed "
                               f"(non-fatal, retried next cycle): {e}")

    th = threading.Thread(target=_loop, name="candlestore-backfill",
                          daemon=True)
    _thread_state.update(thread=th, stop=stop)
    th.start()
    return th


def stop_backfill_thread(timeout: float = 2.0) -> None:
    st = _thread_state.get("stop")
    if st is not None:
        st.set()