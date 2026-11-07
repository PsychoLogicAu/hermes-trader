"""Log-only open-interest (OI) accrual — the positioning feed the decision seat never had.

Board origin: hermes-trader kanban **t_9f660914** (2026-11-07), follow-up to the seat-state
block gauntlet (t_13846e4b). That gauntlet's `positioning` arm was unpriceable: the only
positioning field reaching the archive (`[shadow-signals] whale=`) is `balanced` on 95.8% of
scans, and the seat's own OI-based whale field
(`whale_index.oi_surge_accumulation` → `perception.whale_signal`) has NEVER run in production —
its fetch is gated on whale *action* knobs (`whale_force_execute` / `whale_regime_bypass` /
`whale_size_multiplier`) that are all off, and its snapshot is a single-step overwrite at repo
root (container-local, and fed by a 24 h-cached universe, so its "since last scan" is really
daily granularity).

What this module does, once per scan:

* ONE `metaAndAssetCtxs` POST (HL weight 20, covers all ~235 perps) and appends the snapshot as
  an **append-only JSONL line** on the log volume (`/app/log/oi-series.jsonl`, i.e. host
  `trader-logs/`). That series is what a future positioning block needs: ΔOI reconstructible at
  15m/1h/4h offline, in COIN units (price-independent position growth) alongside price.
* A compact `[oi-shadow]` line for each coin whose OI moved while price stayed flat — the
  "loading quietly" tell, the same shape `oi_surge_accumulation` describes but at multi-window
  resolution.
* A per-scan `[oi-accrual] snap N coins` heartbeat, because zero flag lines are otherwise
  indistinguishable from a dead read path.

ZERO DECISION SURFACE — deliberate, and pinned by `tests/test_oi_accrual.py`: nothing in
`research.py`, `decision_voice.py`, `executor.py` or `risk_gates.py` reads this module. It cannot
change a verdict, a state field, a gate or a size. Wiring OI into the seat is a SEPARATE, later
decision that needs its own gauntlet (the owner's standing rule: accrue log-only first).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_LOG_DIR = "/app/log"
# Read per call (never frozen at import) so operators and tests can redirect it —
# same rule as ledger.py / prompt_log.py: append-only state must live on the
# volume-mounted log dir, and tests MUST set this (tests/conftest.py) or a pytest
# run writes fixture snapshots into the live series.
_SERIES_ENV = "HERMES_OI_SERIES_FILE"

# Defaults mirror whale_index.oi_surge_accumulation's constants so the two OI views read as one
# scale; every value is a config knob under the `oi_accrual` block (hot-reload, no rebuild).
_DEFAULTS: Dict[str, Any] = {
    "enabled": False,                 # inert unless the config block says otherwise
    "windows_min": [15, 60, 240],     # ΔOI lookbacks for the log line + the tell
    "min_oi_usd": 5e6,                # size gate on OI notional (OI is in COIN units)
    "min_doi_pct": 3.0,               # flag when |ΔOI| (coin units) >= this in ANY window
    "max_px_move_pct": 4.0,           # the flat-price tell; over this, the move already ran
    "min_interval_s": 60.0,           # throttle: one snapshot per scan, never per candidate
    "max_in_memory": 4000,            # rolling history window (≈10 days at one scan/3.5 min)
}

# Rolling history for the delta computation: [(ts_s, {coin: (oi_coins, px)})], oldest first.
_HISTORY: deque = deque(maxlen=int(_DEFAULTS["max_in_memory"]))
_LOCK = threading.Lock()
_last_snap_ts = 0.0
_seeded = False


def _series_path() -> str:
    return os.environ.get(_SERIES_ENV) or os.path.join(_LOG_DIR, "oi-series.jsonl")


def reset_history() -> None:
    """Clear the in-memory rolling history (tests; and after a series-file rotation)."""
    global _last_snap_ts, _seeded
    with _LOCK:
        _HISTORY.clear()
        _last_snap_ts = 0.0
        _seeded = False


def accrual_enabled(cfg: Optional[Dict[str, Any]]) -> bool:
    """Master switch, read fresh from the agent config each scan (hot-reloadable).

    Absent block / malformed value = OFF (zero fetches, zero lines) — the deployed posture
    until the owner flips it.
    """
    block = (cfg or {}).get("oi_accrual")
    if not isinstance(block, dict):
        return False
    return bool(block.get("enabled", False))


def _cfg_value(cfg: Optional[Dict[str, Any]], key: str) -> Any:
    block = (cfg or {}).get("oi_accrual")
    if isinstance(block, dict) and key in block:
        return block[key]
    return _DEFAULTS[key]


# ── Pure pieces (unit-testable, no network, no clock) ────────────────────────

def parse_ctx_rows(payload: Any) -> Dict[str, Tuple[float, float]]:
    """`metaAndAssetCtxs` payload → {coin: (oi_coins, px)}.

    Payload shape: [meta, assetCtxs] where assetCtxs is index-aligned with
    meta["universe"]. OI is in COIN UNITS (whale_index 2026-06-02 audit) — never treat it as
    USD; notional is OI × price. Malformed rows are skipped, never fatal.
    """
    out: Dict[str, Tuple[float, float]] = {}
    try:
        meta, ctxs = payload[0], payload[1]
        universe = meta.get("universe") or []
    except (TypeError, IndexError, AttributeError):
        return out
    for entry, ctx in zip(universe, ctxs):
        coin = entry.get("name") if isinstance(entry, dict) else None
        if not coin or not isinstance(ctx, dict):
            continue
        try:
            oi = float(ctx.get("openInterest") or 0.0)
            px = float(ctx.get("midPx") or ctx.get("markPx") or 0.0)
        except (TypeError, ValueError):
            continue
        if oi > 0 and px > 0:
            out[coin] = (oi, px)
    return out


def _pick_reference(history: List[Tuple[float, Dict[str, Tuple[float, float]]]],
                    now: float, window_min: float) -> Optional[Dict[str, Tuple[float, float]]]:
    """Newest snapshot at least `window_min` minutes old (None = window not covered yet)."""
    target = now - window_min * 60.0
    for ts, snap in reversed(history):
        if ts <= target:
            return snap
    return None


def compute_flags(
    history: List[Tuple[float, Dict[str, Tuple[float, float]]]],
    now: float,
    current: Dict[str, Tuple[float, float]],
    windows_min: List[float],
    min_oi_usd: float,
    min_doi_pct: float,
    max_px_move_pct: float,
) -> List[Dict[str, Any]]:
    """ΔOI per coin across the configured windows, flagged on the flat-price tell.

    ΔOI is computed on COIN units: a price rise must not masquerade as position building.
    A coin flags when |ΔOI| >= `min_doi_pct` in ANY window whose price move stayed inside
    `max_px_move_pct` — i.e. positions moved quietly. Windows with no reference snapshot yet
    report None (early accrual) rather than a fake zero.
    """
    flags: List[Dict[str, Any]] = []
    refs = {w: _pick_reference(history, now, w) for w in windows_min}
    for coin, (oi, px) in current.items():
        if oi * px < min_oi_usd:
            continue
        dois: List[Optional[float]] = []
        moves: List[Optional[float]] = []
        for w in windows_min:
            ref = refs.get(w)
            if not ref or coin not in ref:
                dois.append(None)
                moves.append(None)
                continue
            p_oi, p_px = ref[coin]
            if p_oi <= 0 or p_px <= 0:
                dois.append(None)
                moves.append(None)
                continue
            dois.append((oi - p_oi) / p_oi * 100.0)
            moves.append((px - p_px) / p_px * 100.0)
        hit = None
        for w, d, mv in zip(windows_min, dois, moves):
            if d is None or mv is None:
                continue
            if abs(d) >= min_doi_pct and abs(mv) <= max_px_move_pct:
                hit = (w, d, mv)
                break
        if hit is None:
            continue
        flags.append({
            "coin": coin,
            "windows_min": list(windows_min),
            "doi_pct": [None if d is None else round(d, 2) for d in dois],
            "px_move_pct": [None if m is None else round(m, 2) for m in moves],
            "oi_usd": round(oi * px, 0),
            "oi_coins": oi,
            "px": px,
            "hit_window_min": hit[0],
            "hit_doi_pct": round(hit[1], 2),
            "hit_px_move_pct": round(hit[2], 2),
        })
    # Biggest quiet position move first.
    return sorted(flags, key=lambda f: -abs(f["hit_doi_pct"]))


def format_flag_line(f: Dict[str, Any]) -> str:
    """One greppable line carrying the raw decision numbers for the offline join."""
    wins = ",".join(f"{int(w)}m" for w in f["windows_min"])
    dois = ",".join("na" if d is None else f"{d:+.1f}" for d in f["doi_pct"])
    moves = ",".join("na" if m is None else f"{m:+.1f}" for m in f["px_move_pct"])
    return (f"[oi-shadow] {f['coin']} dOI({wins})={dois}% px({moves})% "
            f"oi=${f['oi_usd'] / 1e6:.1f}M hit={int(f['hit_window_min'])}m")


# ── Fetch indirection (tests monkeypatch THIS, like the other HL readers) ────

def _fetch_ctx_rows() -> Any:
    """One HL `metaAndAssetCtxs` call through the shared rate-budgeted client.

    `_http_post` returns None on budget exhaustion or HTTP failure (never raises) — the caller
    treats that as a skipped snapshot, so a saturated scan costs nothing extra here.
    """
    from hermes_trader.client.hl_client import _http_post
    return _http_post("/info", {"type": "metaAndAssetCtxs"}, timeout=10)


def _seed_from_file(max_bytes: int = 400_000) -> None:
    """Cold-start: load the tail of the series file so windows survive a container recreate."""
    global _seeded
    if _seeded:
        return
    _seeded = True
    path = _series_path()
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()                     # drop the partial first line
            raw = fh.read()
    except OSError:
        return
    loaded = 0
    for line in raw.splitlines():
        try:
            rec = json.loads(line)
            ts = float(rec["ts"])
            snap = {c: (float(v[0]), float(v[1])) for c, v in rec["m"].items()}
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if snap:
            _HISTORY.append((ts, snap))
            loaded += 1
    if loaded:
        logger.info(f"[oi-accrual] seeded {loaded} snapshots from {_series_path()}")


def _append_series(ts: float, current: Dict[str, Tuple[float, float]]) -> None:
    """Append one snapshot line (append-only; never rewrites history)."""
    path = _series_path()
    rec = {"ts": int(ts), "m": {c: [round(v[0], 6), round(v[1], 6)] for c, v in current.items()}}
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except OSError as e:
        logger.warning(f"[oi-accrual] series append failed: {e}")


# ── Entry point (called once per scan from perception.scan_markets) ──────────

def snapshot_and_log(
    cfg: Optional[Dict[str, Any]] = None,
    now: Optional[float] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Snapshot OI, append it to the series, log the flat-price flags. Never raises.

    Returns the flag list (also logged). A disabled switch, a rate-budget skip or any error
    yields [] and costs the scan nothing.
    """
    global _last_snap_ts
    if not accrual_enabled(cfg):
        return []
    now = time.time() if now is None else float(now)
    min_interval = float(_cfg_value(cfg, "min_interval_s"))
    with _LOCK:
        if not force and _last_snap_ts and now - _last_snap_ts < min_interval:
            return []
    try:
        payload = _fetch_ctx_rows()
        if not payload:
            logger.debug("[oi-accrual] skipped: no payload (rate budget or HL error)")
            return []
        current = parse_ctx_rows(payload)
        if not current:
            logger.debug("[oi-accrual] skipped: no usable OI rows in payload")
            return []
        _seed_from_file()
        with _LOCK:
            history = list(_HISTORY)
            _HISTORY.append((now, dict(current)))
            _last_snap_ts = now
        _append_series(now, current)
        flags = compute_flags(
            history, now, current,
            list(_cfg_value(cfg, "windows_min")),
            float(_cfg_value(cfg, "min_oi_usd")),
            float(_cfg_value(cfg, "min_doi_pct")),
            float(_cfg_value(cfg, "max_px_move_pct")),
        )
        for f in flags:
            logger.info(format_flag_line(f))
        logger.info(f"[oi-accrual] snap {len(current)} coins flags={len(flags)} "
                    f"hist={len(history) + 1} file={_series_path()}")
        return flags
    except Exception as e:                                    # never reach the scan
        logger.debug(f"[oi-accrual] failed: {e}")
        return []
