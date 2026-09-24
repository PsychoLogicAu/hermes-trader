#!/usr/bin/env python3
"""Tail-trigger gate audit — pass 1: extract blocks + verdicts (fixed prefix format).

Log line format: "[COIN] YYYY-MM-DD HH:MM:SS,mmm LEVEL:logger:message"
blocked_by entries are full reason strings; gate name = text up to first space/'('.
"""
import re, ast, json, glob, os
from datetime import datetime, timezone

ROOT = "/home/oknight/src/hermes-trader"
LOGS = sorted(glob.glob(os.path.join(ROOT, "trader-logs/trader.log*")))
OUT = os.path.join(ROOT, ".hermes/scratch/tail_blocks.json")

line_re = re.compile(r"^\[(?P<coin>[A-Z0-9$\-\.]+)\] (?P<ts>\S+ \S+?),\d+ (?P<rest>.*)$")
TRADE_RES = "INFO:trading_loop:Trade result: "
VERDICT_HDR = "INFO:hermes_trader.agents.research:[parse_verdict] "
VLINE = re.compile(r"INFO:trading_loop:Verdict: (LONG|SHORT|PASS), Confidence: ([\d.]+)")

TAIL_GATES = ("chronos_tail_trigger", "timesfm_tail_trigger", "tirex_tail_trigger")

def ts(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
def gate_name(reason): return reason.split(" ")[0].split("(")[0].strip()

blocks, verdicts, vlines = [], [], []

for f in LOGS:
    with open(f, errors="replace") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        m = line_re.match(line)
        if not m: continue
        coin, tstamp, rest = m.group("coin"), m.group("ts"), m.group("rest")
        idx = rest.find(TRADE_RES)
        if idx >= 0:
            try: r = ast.literal_eval(rest[idx+len(TRADE_RES):].strip())
            except Exception: continue
            if r.get("executed"): continue
            gb = r.get("blocked_by") or []
            tail_hits = sorted({gate_name(g) for g in gb if gate_name(g) in TAIL_GATES})
            if not tail_hits: continue
            gr = r.get("gate_results") or {}
            rec = {"ts": tstamp, "coin": coin, "blocked_by": gb,
                   "tail_hits": tail_hits,
                   "other_blockers": [gate_name(g) for g in gb if gate_name(g) not in TAIL_GATES],
                   "action": r.get("action"), "verdict_word": r.get("verdict"),
                   "side": (r.get("analysis") or {}).get("side") if isinstance(r.get("analysis"), dict) else None}
            for g in TAIL_GATES:
                info = gr.get(g)
                if isinstance(info, dict):
                    rec[g] = {k: info.get(k) for k in ("pass","reason","tail_pct","shadow_would_block")}
            blocks.append(rec)
            continue
        if VERDICT_HDR in rest:
            cm = re.search(r"\[parse_verdict\] (\S+) raw AI text", rest)
            c2 = cm.group(1) if cm else coin
            # verdict JSON may be single-line OR pretty-printed multi-line
            buf, started = [], False
            for j in range(i+1, min(i+25, len(lines))):
                lm = line_re.match(lines[j])
                s = lm.group("rest") if lm else lines[j]
                s = s.strip()
                if not started:
                    if s.startswith("{") and '"verdict"' in s:
                        started = True; buf = [s]
                        if s.endswith("}"):
                            try: verdicts.append({"ts": tstamp, "coin": c2, **json.loads(s)})
                            except Exception: pass
                            break
                else:
                    buf.append(s)
                    if s.endswith("}"):
                        try: verdicts.append({"ts": tstamp, "coin": c2, **json.loads("\n".join(buf))})
                        except Exception: pass
                        break
            continue
        vm = VLINE.search(rest)
        if vm and rest.startswith("INFO:trading_loop:Verdict:"):
            vlines.append({"ts": tstamp, "coin": coin, "side": vm.group(1).lower(), "conf": float(vm.group(2))})

def join(target, pool, dtmax):
    t = ts(target["ts"]); best = None
    for c in reversed(pool):
        d = (t - ts(c["ts"])).total_seconds()
        if d > dtmax: break
        if d < -5: continue
        if c.get("coin") and c["coin"] != target.get("coin"): continue
        best = c; break
    return best

verdicts.sort(key=lambda x: x["ts"])
vlines.sort(key=lambda x: x["ts"])

# dedupe rotated-log overlap by (coin, ts)
seen = set(); uniq = []
for b in sorted(blocks, key=lambda x: x["ts"]):
    k = (b["coin"], b["ts"])
    if k in seen: continue
    seen.add(k); uniq.append(b)
blocks = uniq

for b in blocks:
    v = join(b, verdicts, 300)
    vl = join(b, vlines, 120)
    if v:
        b["v_entry"], b["v_stop"], b["v_tp"] = v.get("entryPx"), v.get("stopPx"), v.get("tpPx")
        b["v_side"] = (v.get("side") or v.get("verdict") or "").lower()
    if vl: b["conf"], b["vline_side"] = vl["conf"], vl["side"]

# tail_pct / side parsed from reason strings for each hit gate
for b in blocks:
    for g in b["tail_hits"]:
        info = b.get(g) or {}
        rs = info.get("reason") or next((x for x in b["blocked_by"] if x.startswith(g)), "")
        mm = re.search(r"\((\w+) entry.*?= ([+-][\d.]+)%", rs)
        if mm:
            b.setdefault("sides", {})[g] = mm.group(1)
            b.setdefault("tails", {})[g] = float(mm.group(2))

stats = {"log_files": [os.path.basename(x) for x in LOGS],
         "window_first": min(b["ts"] for b in blocks) if blocks else None,
         "window_last": max(b["ts"] for b in blocks) if blocks else None,
         "blocks": len(blocks),
         "with_verdict_json": sum(1 for b in blocks if "v_entry" in b),
         "with_conf": sum(1 for b in blocks if "conf" in b)}
json.dump({"stats": stats, "blocks": blocks}, open(OUT, "w"), indent=1)
print(json.dumps(stats, indent=1))

from collections import Counter
sole = Counter(b["tail_hits"][0] if len(b["tail_hits"])==1 and not b["other_blockers"] else "?" for b in blocks)
co_pairs = Counter()
for b in blocks:
    if not b["other_blockers"]: co_pairs[tuple(b["tail_hits"])] += 1
    else:
        for g in b["tail_hits"]:
            for o in set(b["other_blockers"]): co_pairs[(g, "+", o)] += 1
print("tail-hit combos (no other blocker) / co-blocks:")
for k,v in co_pairs.most_common(15): print(" ", k, v)
