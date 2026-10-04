"""Prompt/response archive — zstd-chunked, rotation-bounded record of every
LLM research call.

Why: the prompt text and raw model responses are the ground truth for any
later analysis of decision quality (statefulness experiments, model A/Bs,
prompt regressions). trader.log only keeps the last 800 chars of each reply
(parse_verdict's debug line) and none of the prompts; the session log carries
verdicts but no reasoning text. This module writes ONE record per research
call — full system prompt, full user message, raw response, parsed verdict,
price context, latencies — plus a second ``role="duelist"`` line when the
A/B duelist answered (its prompts are byte-identical to the primary's, so it
references them instead of duplicating ~20 KB per call).

Storage model (bounded by design — storage size must never be a reason to
turn the archive off):

  <dir>/calls.jsonl                  active chunk, plaintext append
  <dir>/calls-<UTCstamp>.jsonl.zst   rotated chunks (gzip fallback: .jsonl.gz)
  <dir>/index.jsonl                  append-only manifest of rotations/deletions

Rotation triggers (checked at append time): the active chunk exceeds
``max_chunk_raw_mb`` OR the UTC day rolled (when ``rotate_daily``). Rotation
compresses the WHOLE active file into one zstd frame-append (.zst frames are
concatenable — one frame per write batch, decodable by any zstd tool), then
truncates the active file. Retention prunes rotated chunks older than
``retention_days`` or beyond ``max_total_stored_mb`` (newest-first keep).

Crash semantics: compress-then-truncate means a crash mid-rotation can only
duplicate records in the next chunk (dedupe by ts+perception_id at analysis
time), never lose them. The active file is fsynced... not fsynced per line —
this is an analysis archive, not a ledger; a lost tail on host power failure
is acceptable and matches every other log here.

Compression backend: the ``zstandard`` package (requirements.txt) → stdlib
``compression.zstd`` (Python 3.14+) → gzip fallback (.gz suffix + one LOUD
warning). Data is never silently dropped because a codec is missing.

Never raises into the caller: a full disk / read-only mount / bad config
degrades to at most one warning per issue, exactly like session_log/duel_store.

Config (hot, read per rotation check)::

    "prompt_log": {
        "enabled": true,             # OFF = zero writes, zero stats syscalls
        "dir": null,                 # null -> /app/log/prompt-log (env: HERMES_PROMPT_LOG_DIR)
        "max_chunk_raw_mb": 32,      # rotate when the active chunk exceeds this
        "rotate_daily": true,        # also rotate on UTC day roll
        "retention_days": 30,        # 0 = keep forever (subject to size cap)
        "max_total_stored_mb": 512,  # 0 = no size cap
        "zstd_level": 3,             # zstandard/stdlib level; gzip uses 6
        "log_duelist": true,         # also record the duelist's raw response
        "max_prompt_chars": 40000,   # per-field truncation guards
        "max_response_chars": 20000
    }

Path resolution mirrors ledger.py: default ``/app/log/prompt-log`` (the
host-mounted volume — container-local paths are lost on recreate), overridable
via ``HERMES_PROMPT_LOG_DIR`` at import time for tests.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from hermes_trader.agents.config_store import read_agent_config

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "/app/log/prompt-log"

DEFAULTS: Dict[str, Any] = {
    "enabled": False,          # ships dark; flip via config repo like every gate
    "dir": None,
    "max_chunk_raw_mb": 32.0,
    "rotate_daily": True,
    "retention_days": 30,
    "max_total_stored_mb": 512.0,
    "zstd_level": 3,
    "log_duelist": True,
    "max_prompt_chars": 40000,
    "max_response_chars": 20000,
}

_lock = threading.Lock()          # research runs on a worker pool (session_log pattern)
_state: Dict[str, Any] = {}      # {dir, day, bytes, warned: set} — lazily initialised


# ── Compression backend (zstandard -> stdlib 3.14 -> gzip) ────────────────

def _backend() -> Tuple[Any, str]:
    """(backend, suffix). Cached after first resolution — the configured
    zstd_level is captured at first use (a level change takes effect on the
    next process start; the codec choice itself never changes mid-run)."""
    if "backend" in _state:
        return _state["backend"], _state["suffix"]
    level = int(_cfg().get("zstd_level", DEFAULTS["zstd_level"]) or 3)
    backend: Any
    try:
        import zstandard  # type: ignore

        class _ZstdPkg:
            name = "zstandard"

            def __init__(self, level: int):
                self._c = zstandard.ZstdCompressor(level=level)
                self._d = zstandard.ZstdDecompressor()

            def compress(self, data: bytes) -> bytes:
                return self._c.compress(data)

            def decompress_all(self, blob: bytes) -> bytes:
                # stream_reader transparently reads concatenated frames.
                import io
                with self._d.stream_reader(io.BytesIO(blob)) as r:
                    return r.read()

        backend, suffix = _ZstdPkg(level), ".jsonl.zst"
    except ImportError:
        try:
            from compression import zstd as _szstd  # Python 3.14+ stdlib

            class _ZstdStdlib:
                name = "compression.zstd"

                def __init__(self, level: int):
                    self._c = _szstd.ZstdCompressor(level=level)
                    self._d = _szstd.ZstdDecompressor()

                def compress(self, data: bytes) -> bytes:
                    return self._c.compress(data)

                def decompress_all(self, blob: bytes) -> bytes:
                    import io
                    with self._d.stream_reader(io.BytesIO(blob)) as r:
                        return r.read()

            backend, suffix = _ZstdStdlib(level), ".jsonl.zst"
        except ImportError:
            import gzip

            class _Gzip:
                name = "gzip (fallback — zstandard not installed)"

                def __init__(self, level: int):
                    pass

                def compress(self, data: bytes) -> bytes:
                    return gzip.compress(data, 6)

                def decompress_all(self, blob: bytes) -> bytes:
                    # gzip.decompress reads multi-member streams natively.
                    return gzip.decompress(blob)

            backend, suffix = _Gzip(0), ".jsonl.gz"
    _state["backend"], _state["suffix"] = backend, suffix
    return backend, suffix


def decompress_blob(blob: bytes) -> bytes:
    """Decompress a chunk written by this module (.zst or .gz). Public helper
    for offline analysis scripts."""
    if blob[:2] == b"\x1f\x8b":  # gzip magic
        import gzip
        return gzip.decompress(blob)
    backend, _ = _backend()
    return backend.decompress_all(blob)


# ── Config / paths ─────────────────────────────────────────────────────────

def _cfg() -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    try:
        block = read_agent_config().get("prompt_log")
        if isinstance(block, dict):
            merged.update(block)
    except Exception:  # noqa: BLE001 — fail-open (archive must never break trading)
        pass
    return merged


def log_dir(cfg: Optional[Dict[str, Any]] = None) -> str:
    """Archive directory: config ``prompt_log.dir`` → HERMES_PROMPT_LOG_DIR →
    /app/log/prompt-log. Read per call so tests (and operators) can redirect."""
    cfg = cfg or _cfg()
    d = cfg.get("dir")
    if isinstance(d, str) and d.strip():
        return d.strip()
    return os.environ.get("HERMES_PROMPT_LOG_DIR", _DEFAULT_DIR)


def _warn_once(key: str, msg: str) -> None:
    warned = _state.setdefault("warned", set())
    if key not in warned:
        warned.add(key)
        logger.warning(msg)


# ── Active-chunk state / rotation ──────────────────────────────────────────

def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _init_state(d: str) -> None:
    path = os.path.join(d, "calls.jsonl")
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    # Attribute mtimes to UTC days; a fresh file counts as today.
    try:
        day = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).strftime("%Y%m%d")
    except OSError:
        day = _utc_day()
    _state["dir"] = d
    _state["bytes"] = size
    _state["day"] = day


def _rotated_chunks(d: str) -> List[str]:
    try:
        names = [n for n in os.listdir(d)
                 if n.startswith("calls-") and (n.endswith(".jsonl.zst") or n.endswith(".jsonl.gz"))]
    except OSError:
        return []
    return sorted(names)  # UTC-stamp prefix -> lexicographic == chronological


def _index_append(d: str, event: Dict[str, Any]) -> None:
    try:
        with open(os.path.join(d, "index.jsonl"), "a") as f:
            f.write(json.dumps({"ts": int(time.time() * 1000), **event}) + "\n")
    except OSError:
        pass


def _rotate(d: str, cfg: Dict[str, Any]) -> None:
    """Compress the active chunk into one timestamped archive file, truncate,
    enforce retention. Called with _lock held. Best-effort: any failure leaves
    the active file intact (compress-then-truncate ordering)."""
    path = os.path.join(d, "calls.jsonl")
    try:
        raw = open(path, "rb").read()
    except OSError:
        raw = b""
    if not raw:
        _state["bytes"] = 0
        _state["day"] = _utc_day()
        return
    backend, suffix = _backend()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Zero-padded rotation counter keeps lexicographic sort == chronological
    # even for multiple rotations inside one second (a bare "-1" suffix would
    # sort BEFORE the un-suffixed name: '-' < '.').
    n = 0
    name = f"calls-{stamp}-{n:02d}{suffix}"
    while os.path.exists(os.path.join(d, name)):
        n += 1
        name = f"calls-{stamp}-{n:02d}{suffix}"
    tmp = os.path.join(d, name + ".tmp")
    try:
        blob = backend.compress(raw)
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, os.path.join(d, name))
        # Truncate the active file only after the archive is durable-ish.
        with open(path, "w"):
            pass
        _state["bytes"] = 0
        _state["day"] = _utc_day()
        first_ts = last_ts = None
        for line in raw.split(b"\n"):
            if not line.strip():
                continue
            try:
                ts = json.loads(line).get("ts")
            except Exception:
                continue
            if isinstance(ts, int):
                first_ts = ts if first_ts is None else min(first_ts, ts)
                last_ts = ts if last_ts is None else max(last_ts, ts)
        _index_append(d, {"event": "rotated", "chunk": name, "backend": backend.name,
                          "n_bytes_raw": len(raw), "n_bytes_stored": len(blob),
                          "first_ts": first_ts, "last_ts": last_ts})
        logger.info(
            f"[prompt-log] rotated -> {name} ({len(raw)} raw / {len(blob)} stored bytes, "
            f"backend={backend.name})")
        _enforce_retention(d, cfg)
    except Exception as e:  # noqa: BLE001 — rotation failure must not lose data
        try:
            os.unlink(tmp)
        except OSError:
            pass
        _warn_once("rotate", f"[prompt-log] rotation failed (active chunk kept intact): {e!r}")


def _enforce_retention(d: str, cfg: Dict[str, Any]) -> None:
    """Delete rotated chunks past retention_days or beyond max_total_stored_mb
    (oldest first). 0/absent disables each rule."""
    try:
        days = float(cfg.get("retention_days", DEFAULTS["retention_days"]) or 0)
        cap_mb = float(cfg.get("max_total_stored_mb", DEFAULTS["max_total_stored_mb"]) or 0)
    except (TypeError, ValueError):
        days, cap_mb = DEFAULTS["retention_days"], DEFAULTS["max_total_stored_mb"]
    chunks = _rotated_chunks(d)
    now = time.time()
    kept: List[Tuple[str, int]] = []
    for name in chunks:
        p = os.path.join(d, name)
        try:
            st = os.stat(p)
        except OSError:
            continue
        if days > 0 and (now - st.st_mtime) > days * 86400:
            _delete_chunk(d, p, name, "retention_days")
            continue
        kept.append((name, st.st_size))
    if cap_mb > 0:
        total = sum(sz for _, sz in kept)
        limit = cap_mb * 1024 * 1024
        i = 0
        while total > limit and i < len(kept):
            name, sz = kept[i]
            _delete_chunk(d, os.path.join(d, name), name, "max_total_stored_mb")
            total -= sz
            i += 1


def _delete_chunk(d: str, path: str, name: str, why: str) -> None:
    try:
        os.unlink(path)
        _index_append(d, {"event": "deleted", "chunk": name, "why": why})
        logger.info(f"[prompt-log] pruned {name} ({why})")
    except OSError as e:
        _warn_once("prune", f"[prompt-log] prune failed for {name}: {e!r}")


# ── Public API ─────────────────────────────────────────────────────────────

def record_call(rec: Dict[str, Any]) -> None:
    """Append one call record. NEVER raises (session_log pattern): the archive
    is an analysis artifact, trading must not care whether it works.

    Required keys are by convention, not enforced — see _build_record in
    research.py for the canonical shape."""
    try:
        cfg = _cfg()
        if not cfg.get("enabled", False):
            return
        d = log_dir(cfg)
        with _lock:
            if _state.get("dir") != d:
                os.makedirs(d, exist_ok=True)
                _init_state(d)
            # Rotation check BEFORE append (size cap + UTC day roll).
            max_bytes = float(cfg.get("max_chunk_raw_mb", DEFAULTS["max_chunk_raw_mb"]) or 0) * 1024 * 1024
            if (_state["bytes"] > 0 and max_bytes > 0
                    and _state["bytes"] >= max_bytes):
                _rotate(d, cfg)
            elif (cfg.get("rotate_daily", True) and _utc_day() != _state["day"]
                  and _state["bytes"] > 0):
                _rotate(d, cfg)

            line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
            # Sanity ceiling for a SINGLE record (drop rather than poison the
            # chunk with a megabyte-scale runaway). Deliberately NOT tied to
            # max_chunk_raw_mb — that's the rotation threshold and may be set
            # small; field truncation in _archive_record already bounds size.
            if len(line.encode()) > 8 * 1024 * 1024:
                return
            try:
                with open(os.path.join(d, "calls.jsonl"), "a") as f:
                    f.write(line)
                _state["bytes"] += len(line.encode())
            except OSError as e:
                _warn_once("write", f"[prompt-log] write failed (archive degraded): {e!r}")
    except Exception as e:  # noqa: BLE001 — belt and braces: never into research()
        _warn_once("record", f"[prompt-log] record_call failed: {e!r}")


def iter_records(d: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Yield every readable record: rotated chunks oldest-first, then the live
    active chunk. Malformed lines are skipped. For offline analysis and tests."""
    cfg = _cfg()
    d = d or log_dir(cfg)
    for name in _rotated_chunks(d):
        try:
            blob = decompress_blob(open(os.path.join(d, name), "rb").read())
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[prompt-log] cannot read chunk {name}: {e!r}")
            continue
        for line in blob.decode("utf-8", errors="replace").splitlines():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass
    try:
        with open(os.path.join(d, "calls.jsonl")) as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        return


def stats(d: Optional[str] = None) -> Dict[str, Any]:
    """Archive footprint snapshot for the dashboard/CLI."""
    cfg = _cfg()
    d = d or log_dir(cfg)
    out: Dict[str, Any] = {"dir": d, "enabled": bool(cfg.get("enabled", False)),
                           "backend": _backend()[0].name}
    try:
        active = os.path.join(d, "calls.jsonl")
        out["active_bytes"] = os.path.getsize(active)
        with open(active) as f:
            out["active_records"] = sum(1 for ln in f if ln.strip())
    except OSError:
        out["active_bytes"] = 0
        out["active_records"] = 0
    chunks = _rotated_chunks(d)
    sizes = [os.path.getsize(os.path.join(d, c)) for c in chunks]
    out["chunks"] = len(chunks)
    out["stored_bytes"] = sum(sizes) + out["active_bytes"]
    return out
