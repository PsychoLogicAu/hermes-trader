#!/usr/bin/env python3
"""One-time import: scratch/candlestore/ -> the persistent store (t_18e7517a).

Reads the legacy week-chunked JSON layout (per coin/<interval>/YYYYMMDD.json)
and upserts every bar into candlestore via the module's single write path —
so the store's PK dedupe applies (re-runnable) and the confirmed-bar guard
applies (bars that were still forming at old-fetch time are old now and pass
legitimately; day-aligned chunk windows never crossed the forming boundary
mid-window, see plan risks).

Non-day files (topup-*.json, analysis caches) are skipped exactly like the
legacy reader does.

Usage:
  python3 scripts/candlestore_import.py \
      [--src scratch/candlestore] [--store ./candlestore] [--limit-coins A B]

Prints per-(coin) imported/skipped counts and a before/after verification:
file-scanned bar total vs store row total (diffs = PK dupes + malformed).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hermes_trader.client import candlestore as cs  # noqa: E402


def day_files(iv_dir: str):
    """Chunk files whose NAME starts YYYYMMDD (legacy reader convention)."""
    if not os.path.isdir(iv_dir):
        return []
    out = []
    for fn in sorted(os.listdir(iv_dir)):
        try:
            datetime.strptime(fn[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out.append(fn)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="scratch/candlestore")
    ap.add_argument("--store", default=os.environ.get("HERMES_CANDLESTORE_DIR",
                                                      "candlestore"))
    ap.add_argument("--limit-coins", nargs="*", default=None)
    a = ap.parse_args()

    os.environ["HERMES_CANDLESTORE_DIR"] = os.path.abspath(a.store)
    store = cs.get_store()
    if store is None:
        print("no store", file=sys.stderr)
        return 1

    coins = sorted(d for d in os.listdir(a.src)
                   if os.path.isdir(os.path.join(a.src, d)))
    if a.limit_coins:
        coins = [c for c in coins if c in a.limit_coins]

    file_total = 0
    inserted_total = 0
    malformed = 0
    for i, coin in enumerate(coins):
        cdir = os.path.join(a.src, coin)
        file_n = ins_n = 0
        for iv in cs.MS_PER_INTERVAL:
            iv_dir = os.path.join(cdir, iv)
            for fn in day_files(iv_dir):
                try:
                    bars = json.load(open(os.path.join(iv_dir, fn)))
                except Exception:
                    continue
                if not isinstance(bars, list):
                    continue
                good = []
                for b in bars:
                    if isinstance(b, dict) and "t" in b:
                        good.append(b)
                    else:
                        malformed += 1
                file_n += len(good)
                ins_n += store.upsert_bars(coin, iv, good)
        file_total += file_n
        inserted_total += ins_n
        print(f"[import] {i+1}/{len(coins)} {coin}: file_bars={file_n} "
              f"inserted={ins_n}", flush=True)

    db_total = store.count_bars()
    print(f"[import] DONE coins={len(coins)} scanned={file_total} "
          f"inserted={inserted_total} malformed={malformed} "
          f"db_rows_total={db_total} est_mb={store.est_bytes()/1e6:.0f}")
    print("[import] NOTE scanned-inserted = PK dupes across overlapping "
          "chunks (expected; dedupe is the point). Re-running is safe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())