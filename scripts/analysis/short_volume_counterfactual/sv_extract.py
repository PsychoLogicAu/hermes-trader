#!/usr/bin/env python3
"""min_short_volume_usd audit — pass 1: extract short-thin-market blocks + verdicts.

Log line format: "[COIN] YYYY-MM-DD HH:MM:SS,mmm LEVEL:logger:message".
Gate reason substring: "short on thin market: 24h vol $X.XM < short floor $20M".
Records per block: ts, coin, logged 24h volume ($M), full blocked_by list,
other blockers (gate-name level), verdict JSON (entryPx etc.), Verdict-line
side+conf. Dedupe rotated-log overlap by (coin, ts); caller does 30-min
same-coin dedupe downstream.
"""
import re, ast, json, glob, os
from datetime import datetime, timezone

ROOT = "/home/oknight/src/hermes-trader"
LOGS = sorted(glob.glob(os.path.join(ROOT, "trader-logs/trader.log*")))
OUT = os.environ.get("SV_SCRATCH", os.path.join(ROOT, ".hermes/scratch"))
os.makedirs(OUT, exist_ok=True)
OUTF = os.path.join(OUT, "sv_blocks.json")

line_re = re.compile(r"^\[(?P<coin>[A-Z0-9$\-\.:]+)\] (?P<ts>\S+ \S+?),\d+ (?P<rest>.*)$")
TRADE_RES = "INFO:trading_loop:Trade result: "
VERDICT_HDR = "INFO:hermes_trader.agents.research:[parse_verdict] "
VLINE = re.compile(r"INFO:trading_loop:Verdict: (LONG|SHORT|PASS), Confidence: ([\d.]+)")
SV_REASON = "short on thin market"
VOL_RE = re.compile(r"24h vol \$([\d.]+)M")

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
            sv = next((g for g in gb if SV_REASON in g), None)
            if not sv: continue
            vm = VOL_RE.search(sv)
            rec = {"ts": tstamp, "coin": coin,
                   "vol_musd": float(vm.group(1)) if vm else None,
                   "blocked_by": gb,
                   "other_blockers": [gate_name(g) for g in gb if SV_REASON not in g],
                   "mode": r.get("mode")}
            blocks.append(rec)
            continue
        if VERDICT_HDR in rest:
            cm = re.search(r"\[parse_verdict\] (\S+) raw AI text", rest)
            c2 = cm.group(1) if cm else coin
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

stats = {"log_files": [os.path.basename(x) for x in LOGS],
         "window_first": min(b["ts"] for b in blocks) if blocks else None,
         "window_last": max(b["ts"] for b in blocks) if blocks else None,
         "blocks": len(blocks),
         "sole_blocker": sum(1 for b in blocks if not b["other_blockers"]),
         "with_verdict_json": sum(1 for b in blocks if "v_entry" in b),
         "with_conf": sum(1 for b in blocks if "conf" in b)}
json.dump({"stats": stats, "blocks": blocks}, open(OUTF, "w"), indent=1)
print(json.dumps(stats, indent=1))

from collections import Counter
co = Counter()
for b in blocks:
    if not b["other_blockers"]: co[("SOLE",)] += 1
    else:
        for o in set(b["other_blockers"]): co[(o,)] += 1
print("co-blocker profile (gate -> count with short-thin):")
for k, v in co.most_common(20): print(" ", k[0], v)
vols = sorted(b["vol_musd"] for b in blocks if b["vol_musd"] is not None)
print("volume dist ($M):", vols)
