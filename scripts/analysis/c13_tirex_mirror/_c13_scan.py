#!/usr/bin/env python3
"""C.13 tirex_tail_trigger mirror accrual — pass 1: census + agreement.

Mirror armed ~2026-09-17 05:17 UTC (bdc67c2 deploy; watchlist says ~09-17).
Arms:
  * tirex  = SHADOW  -> would-blocks appear as gate_results.tirex_tail_trigger
              shadow_would_block:True (plus [gate][SHADOW] WOULD HAVE BLOCKED
              lines from ~10-03 onward; wording changed eras, so parse the dict).
  * chronos = LIVE   -> blocks appear in blocked_by (chronos_tail_trigger reason).

Outputs:
  * per-arm trip counts (scans) vs the >=15/arm bar
  * agreement on gated entries: both / only-tirex / only-chronos
    (offline baseline 45/10/10 at X=2.5)
  * tirex EXECUTED cohort: would-block scans that executed anyway (shadow) ->
    real ledger closes = the filter counterfactual for the tirex arm.
  * chronos live-block cohort (deduped candidates) -> priced in pass 2.
"""
import ast
import datetime as dt
import glob
import json
import re
from collections import Counter

LOGS = sorted(glob.glob("/home/oknight/src/hermes-trader/trader-logs/trader.log*"))
ARM = dt.datetime(2026, 9, 17, 5, 17)  # bdc67c2 mirror deploy (UTC)
PFX = re.compile(r"^\[(?P<coin>.+?)\] (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ ")
TRADE_RE = re.compile(r"INFO:trading_loop:Trade result: (\{.*)$")

def is_ctt(reason):
    return reason.startswith("chronos_tail_trigger")

day_t = Counter()
day_c = Counter()
both = t_only = c_only = 0
tirex_would = []   # shadow trips (any scan)
chronos_blocks = []  # live blocks
scans_total = 0
executed_total = 0

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
            executed_total += 1
        gr = d.get("gate_results") or {}
        tt = gr.get("tirex_tail_trigger") or {}
        ct = gr.get("chronos_tail_trigger") or {}
        t_trip = bool(tt.get("shadow_would_block")) or (tt.get("pass") is False)
        # chronos live block = reason in blocked_by (pass:False there too)
        bb = d.get("blocked_by") or []
        c_trip = any(is_ctt(r) for r in bb) or (ct.get("pass") is False)
        if t_trip:
            day_t[ts.strftime("%m-%d")] += 1
        if c_trip:
            day_c[ts.strftime("%m-%d")] += 1
        if t_trip and c_trip:
            both += 1
        elif t_trip:
            t_only += 1
        elif c_trip:
            c_only += 1
        m = re.search(r"= -?([0-9.]+)% beyond", str(tt.get("reason", "")))
        if t_trip:
            tirex_would.append(dict(
                coin=pm["coin"], ts=ts.isoformat(),
                side=("long" if "long entry" in str(tt.get("reason", "")) else "short"),
                tail=float(m.group(1)) if m else None,
                conf=d.get("llm_confidence"), comp=d.get("composite_score"),
                executed=bool(d.get("executed")),
                chronos_trip=c_trip,
                other_blockers=[r.split("(")[0].split(":")[0].strip()[:40]
                                for r in bb if not is_ctt(r)],
            ))
        if c_trip:
            reason = ct.get("reason") or next((r for r in bb if is_ctt(r)), "")
            mc = re.search(r"= -?([0-9.]+)% beyond", reason)
            chronos_blocks.append(dict(
                coin=pm["coin"], ts=ts.isoformat(),
                side=("long" if "long entry" in reason else "short"),
                tail=float(mc.group(1)) if mc else None,
                conf=d.get("llm_confidence"), comp=d.get("composite_score"),
                tirex_trip=t_trip,
                other_blockers=[r.split("(")[0].split(":")[0].strip()[:40]
                                for r in bb if not is_ctt(r)],
            ))

print(f"scans since mirror arm: {scans_total}; executed: {executed_total}")
print(f"tirex shadow trips (scans): {len(tirex_would)}")
print(f"chronos live blocks (scans): {len(chronos_blocks)}")
print(f"agreement on trips: both {both} / tirex-only {t_only} / chronos-only {c_only}")
print(f"per-day tirex: {dict(sorted(day_t.items()))}")
print(f"per-day chronos: {dict(sorted(day_c.items()))}")

tw_exec = [s for s in tirex_would if s["executed"]]
print(f"\ntirex would-blocks that EXECUTED (real closes): {len(tw_exec)}")
depths = [s["tail"] for s in tirex_would if s["tail"] is not None]
if depths:
    b1 = sum(1 for x in depths if x <= 2.5)
    print(f"tirex trip depth: n={len(depths)} at-band(<=2.5)={b1} deep(>2.5)={len(depths)-b1} max={max(depths)}")

json.dump(dict(tirex_would=tirex_would, chronos_blocks=chronos_blocks),
          open("/home/oknight/src/hermes-trader/scripts/analysis/c13_tirex_mirror/_c13_census.json", "w"),
          indent=1)
