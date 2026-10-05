#!/usr/bin/env python3
"""B.22 timesfm_tail_trigger LIVE forward-check — pass 1: census.

Arm ts: 2026-09-17 23:59 UTC (config flip; container recreated 23:57).
Cohort source: `Trade result:` lines whose blocked_by contains a
timesfm_tail_trigger reason (live blocks; shadow accrual went quiet at arm).

Outputs:
  * per-day live-block counts (vs ~2.4/day pre-promote baseline)
  * double-veto share: of scans where timesfm gate failed, how many ALSO
    had chronos_tail_trigger fail (both / t-only / c-only)
  * SOLE-BLOCKER cohort: timesfm is the only failing gate -> the
    incremental set the gate costs/releases OOS.
"""
import ast
import datetime as dt
import glob
import json
import re
from collections import Counter, defaultdict

LOGS = sorted(glob.glob("/home/oknight/src/hermes-trader/trader-logs/trader.log*"))
ARM = dt.datetime(2026, 9, 17, 23, 59)
PFX = re.compile(r"^\[(?P<coin>.+?)\] (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ ")
TRADE_RE = re.compile(r"INFO:trading_loop:Trade result: (\{.*)$")

def is_ttt(reason):
    return reason.startswith("timesfm_tail_trigger")

def is_ctt(reason):
    return reason.startswith("chronos_tail_trigger")

day_blocks = Counter()
ttt_fail_scans = 0          # gate failed (in blocked_by) regardless of co-blocks
both = t_only = 0
sole = []                   # sole-blocker cohort
co_blockers = Counter()
tail_depths = []            # |tail| of ttt trips (band check)
scans_total = 0
executed_since_arm = 0

for path in LOGS:
    for line in open(path, errors="ignore"):
        pm = PFX.match(line)
        if not pm:
            continue
        ts = dt.datetime.strptime(pm["ts"], "%Y-%m-%d %H:%M:%S")
        if ts < ARM:
            continue
        tm = TRADE_RE.search(line)
        if not tm:
            continue
        try:
            d = ast.literal_eval(tm.group(1))
        except Exception:
            continue
        scans_total += 1
        if d.get("executed"):
            executed_since_arm += 1
        bb = d.get("blocked_by") or []
        ttt = [r for r in bb if is_ttt(r)]
        if not ttt:
            continue
        ttt_fail_scans += 1
        ctt = [r for r in bb if is_ctt(r)]
        if ctt:
            both += 1
        else:
            t_only += 1
        m = re.search(r"= -?([0-9.]+)% beyond", ttt[0])
        if m:
            tail_depths.append(float(m.group(1)))
        day_blocks[ts.strftime("%m-%d")] += 1
        others = [r for r in bb if not is_ttt(r)]
        for r in others:
            co_blockers[r.split("(")[0].split(":")[0].strip()[:40]] += 1
        if not others:
            gr = d.get("gate_results") or {}
            sole.append(dict(
                coin=pm["coin"], ts=ts.isoformat(),
                side=("long" if "long entry" in ttt[0] else "short"),
                reason=ttt[0],
                tail=float(m.group(1)) if m else None,
                conf=(d.get("llm_confidence")),
                comp=(d.get("composite_score")),
                executed=bool(d.get("executed")),
            ))

print(f"scans (Trade result lines) since arm: {scans_total}; executed: {executed_since_arm}")
print(f"timesfm_tail live blocks: {ttt_fail_scans}  "
      f"(chronos co-fail {both}, timesfm-only {t_only})")
print(f"per-day blocks: {dict(sorted(day_blocks.items()))}")
if tail_depths:
    b1 = sum(1 for x in tail_depths if x <= 2.5)
    print(f"tail depth: n={len(tail_depths)} shallow(-2.0..-2.5)={b1} "
          f"deep(<-2.5)={len(tail_depths)-b1} max={max(tail_depths)}")
print(f"\nco-blockers on ttt trips (top): {co_blockers.most_common(12)}")
print(f"\nSOLE-BLOCKER cohort n={len(sole)}")
for s in sole:
    print(f"  {s['ts']} {s['coin']:8s} {s['side']:5s} tail -{s['tail']} conf {s['conf']} comp {s['comp']}")
json.dump(sole, open("/home/oknight/src/hermes-trader/scripts/analysis/b22_timesfm_tail_forward/_b22_sole.json", "w"), indent=1)
