#!/usr/bin/env python3
"""Tail-trigger counterfactual — occupancy-aware replay of sole-tail-gate blocks.

Releases ONLY blocks whose sole blocker set is tail gates (chronos/timesfm);
every other gate stays in force. Sim book shares the 5-slot cap with real
ledger intervals; no same-coin re-entry while held (real or sim); stale_flat
fires only when contention (real+sim open) >= 3, mirroring live code.

Exit policy mirrors live dsl_exit.py on 5m bars:
  entry = open of first bar at/after decision ts
  max_loss: bar low/high touch of stop -> fill at stop px
  floor: phase1 = entry*(1∓MAXLOSS); phase2 arms when peak >= protect(1%),
         floor = entry ± range*(1-retrace), tier by PEAK (8%->0.35,15%->0.40,
         default .25); breakeven ratchet peak>=2.5% -> lock 0.05%; ratchets only.
  breach: 2 consecutive bar CLOSES beyond floor; first-90s grace (bar 1 close
          at ~5min is past grace, so grace effectively no-op on 5m bars).
  stale_flat: elapsed>=240min AND peak<protect AND contention>=3 -> exit at close
  hard_timeout: 1800min. Fees 2.5 bps/side on notional. lev 5x, $35 notional.
"""
import json, os, time, urllib.request, bisect
from datetime import datetime, timezone
from collections import Counter

ROOT = "/home/oknight/src/hermes-trader"
SCRATCH = os.environ.get("TAIL_AUDIT_SCRATCH", os.path.join(ROOT, ".hermes/scratch"))
CACHE = os.path.join(SCRATCH, "hl_candles_tail.json")
DATA = json.load(open(os.path.join(SCRATCH, "tail_blocks.json")))

NOTIONAL = 35.0; LEV = 5; FEE = 0.00025
MAXLOSS = min(5.0, 30.0/LEV); PROTECT = 1.0
TIERS = [(15.0, 0.40), (8.0, 0.35)]; DEFAULT_RETRACE = 0.25
BE_TRIG, BE_LOCK = 2.5, 0.05
STALE_MIN, HARD_MIN, CONSEC = 240, 1800, 2
BAR = 300_000

def ets(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()*1000

def fetch(coin, start_ms, end_ms):
    req = urllib.request.Request("https://api.hyperliquid.xyz/info",
        data=json.dumps({"type":"candleSnapshot","req":{"coin":coin,"interval":"5m",
            "startTime":int(start_ms),"endTime":int(end_ms)}}).encode(),
        headers={"Content-Type":"application/json"})
    for a in range(5):
        try: return json.loads(urllib.request.urlopen(req, timeout=25).read())
        except Exception: time.sleep(3*(a+1))
    return []

blocks = DATA["blocks"]
rel = [b for b in blocks if not b["other_blockers"]]
need = sorted({b["coin"] for b in rel})
lo = min(ets(b["ts"]) for b in rel) - 2*BAR
hi = max(ets(b["ts"]) for b in rel) + (HARD_MIN+10)*60_000
print(f"releasable (sole tail-gate) blocks: {len(rel)} of {len(blocks)}; coins: {len(need)}")

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
missing = [c for c in need if c not in cache]
for i, c in enumerate(missing):
    cd = fetch(c, lo, hi)
    cache[c] = [[r["t"], float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"])] for r in cd]
    time.sleep(0.15)
    if (i+1) % 15 == 0:
        print(f"  fetched {i+1}/{len(missing)}"); json.dump(cache, open(CACHE,"w"))
json.dump(cache, open(CACHE,"w"))

# real ledger intervals
opens = {}; real_intervals = []
def lts(e): return float(e["ts"])  # ledger ts is epoch ms
for line in open(os.path.join(ROOT, "trader-logs/trades.jsonl"), errors="replace"):
    try: e = json.loads(line)
    except Exception: continue
    ev = e.get("event")
    if ev == "OPEN": opens[(e.get("coin"), e.get("side"))] = lts(e)
    elif ev == "CLOSE":
        k = (e.get("coin"), e.get("side"))
        if k in opens: real_intervals.append((opens.pop(k), lts(e)))
for st in opens.values(): real_intervals.append((st, float("inf")))
real_intervals.sort()

def pnl(spot_pct): return NOTIONAL * (spot_pct/100.0) * LEV - NOTIONAL*2*FEE

def simulate(bars, t0_ms, side, stale_ok):
    """Return (pnl, exit_reason, exit_ts_ms) or (None, reason, None)."""
    if not bars: return None, "no_candles", None
    ts_list = [b[0] for b in bars]
    i = bisect.bisect_left(ts_list, t0_ms)
    if i >= len(bars)-1: return None, "no_forward_data", None
    entry = bars[i][1]
    if entry <= 0: return None, "bad_entry", None
    is_long = side == "long"
    peak = entry
    floor = entry*(1-MAXLOSS/100) if is_long else entry*(1+MAXLOSS/100)
    consec = 0
    for n in range(1, HARD_MIN//5):
        j = i+n
        if j >= len(bars): 
            c = bars[-1][4]
            upct = ((c-entry)/entry*100) if is_long else ((entry-c)/entry*100)
            return pnl(upct), "end_of_data", bars[-1][0]
        _, o, h, l, c = bars[j]
        el_min = (bars[j][0]-bars[i][0])/60000
        peak = max(peak, h) if is_long else min(peak, l)
        pp = ((peak-entry)/entry*100) if is_long else ((entry-peak)/entry*100)
        upct = ((c-entry)/entry*100) if is_long else ((entry-c)/entry*100)
        stop_px = entry*(1-MAXLOSS/100) if is_long else entry*(1+MAXLOSS/100)
        if (is_long and l <= stop_px) or (not is_long and h >= stop_px):
            return pnl(-MAXLOSS), "max_loss", bars[j][0]
        if pp >= PROTECT:
            retrace = DEFAULT_RETRACE
            for tp, tr in TIERS:
                if pp >= tp: retrace = tr; break
            f2 = entry + (peak-entry)*(1-retrace) if is_long else entry - (entry-peak)*(1-retrace)
        else:
            f2 = floor
        if pp >= BE_TRIG:
            f2 = max(f2, entry*(1+BE_LOCK/100)) if is_long else min(f2, entry*(1-BE_LOCK/100))
        floor = max(floor, f2) if is_long else min(floor, f2)
        breached = (c < floor) if is_long else (c > floor)
        consec = consec+1 if breached else 0
        if consec >= CONSEC:
            return pnl(upct), "floor_breach", bars[j][0]
        if el_min >= STALE_MIN and pp < PROTECT and stale_ok():
            return pnl(upct), "stale_flat", bars[j][0]
        if el_min >= HARD_MIN:
            return pnl(upct), "hard_timeout", bars[j][0]
    c = bars[min(i+HARD_MIN//5, len(bars)-1)][4]
    upct = ((c-entry)/entry*100) if is_long else ((entry-c)/entry*100)
    return pnl(upct), "end_of_data", bars[min(i+HARD_MIN//5, len(bars)-1)][0]

def real_open_at(t): return sum(1 for a,z in real_intervals if a <= t <= z)
real_by_coin = {}
opens2 = {}; order = []
for line in open(os.path.join(ROOT, "trader-logs/trades.jsonl"), errors="replace"):
    try: e = json.loads(line)
    except Exception: continue
    ev = e.get("event")
    if ev == "OPEN": opens2[e.get("coin")] = lts(e)
    elif ev == "CLOSE" and e.get("coin") in opens2:
        real_by_coin.setdefault(e["coin"], []).append((opens2.pop(e["coin"]), lts(e)))
for c,st in opens2.items(): real_by_coin.setdefault(c, []).append((st, float("inf")))

def coin_held_real(coin, t):
    return any(a <= t <= z for a,z in real_by_coin.get(coin, []))

events = sorted(rel, key=lambda b: ets(b["ts"]))

def replay(occupancy):
    sim_open = {}   # coin -> [open_ts, close_ts] while open (list of closed too)
    sim_intervals = []
    out, skips = [], Counter()
    for b in events:
        t0 = ets(b["ts"])
        side = b.get("vline_side") or (b.get("sides", {}) or {}).get(b["tail_hits"][0])
        if side not in ("long","short"): skips["no_side"] += 1; continue
        # close expired sims
        for c in list(sim_open):
            pass
        open_sims = [(c,(a,z)) for c,(a,z) in sim_open.items() if z == "open" or t0 <= z]
        held_sim = any(c == b["coin"] and (z == "open" or a <= t0 <= z) for c,(a,z) in sim_open.items())
        if occupancy:
            if coin_held_real(b["coin"], t0): skips["coin_held_real"] += 1; continue
            if held_sim: skips["coin_held_sim"] += 1; continue
            n_open = len(open_sims) + real_open_at(t0)
            if n_open >= 5: skips["slots_full"] += 1; continue
        def stale_ok(t=t0, sims=open_sims):
            return (len(sims) + real_open_at(t)) >= 3
        res, why, tex = simulate(cache.get(b["coin"], []), t0, side, stale_ok)
        if res is None: skips[why] += 1; continue
        sim_open[b["coin"]] = (t0, tex)
        out.append({"ts": b["ts"], "coin": b["coin"], "side": side, "pnl": round(res,4),
                    "exit": why, "conf": b.get("conf"), "tails": b.get("tails"),
                    "exit_ts": datetime.fromtimestamp(tex/1000, timezone.utc).strftime("%Y-%m-%d %H:%M") if tex else None})
    return out, dict(skips)

def summarize(rs, label):
    tot = sum(r["pnl"] for r in rs); w = sum(1 for r in rs if r["pnl"] > 0)
    print(f"\n{label}: n={len(rs)} net=${tot:+.2f}  win {w}/{len(rs)}"
          + (f"  avg ${tot/len(rs):+.2f}" if rs else ""))
    print("  exits:", dict(Counter(r["exit"] for r in rs)))
    by = {}
    for r in rs:
        tl = "+".join(t.replace("_tail_trigger","") for t in (r.get("tails") or {}).keys()) or "?"
        by.setdefault(tl, [0,0.0]); by[tl][0]+=1; by[tl][1]+=r["pnl"]
    for g,(n,p) in sorted(by.items(), key=lambda x:-x[1][1]): print(f"  {g:16s} n={n:3d} ${p:+8.2f}")
    bys = {}
    for r in rs:
        bys.setdefault(r["side"], [0,0.0]); bys[r["side"]][0]+=1; bys[r["side"]][1]+=r["pnl"]
    print("  by side:", {k:(v[0],round(v[1],2)) for k,v in bys.items()})
    return tot

go, go_skips = replay(occupancy=False)
ex, ex_skips = replay(occupancy=True)
summarize(go, "GATE-ONLY (no occupancy)")
summarize(ex, "EXECUTABLE (occupancy-aware, 5-slot shared w/ real book)")
print("\nskips gate-only:", go_skips)
print("skips executable:", ex_skips)

# conf-banded view on executable
bands = {}
for r in ex:
    c = r.get("conf") or 0
    band = "<0.75" if c < 0.75 else ("0.75-0.82" if c <= 0.82 else ">0.82")
    bands.setdefault(band,[0,0.0]); bands[band][0]+=1; bands[band][1]+=r["pnl"]
print("executable by conf band:", {k:(v[0],round(v[1],2)) for k,v in sorted(bands.items())})

json.dump({"gate_only": go, "executable": ex, "skips_exec": ex_skips},
          open(os.path.join(SCRATCH,"tail_sim_results.json"),"w"), indent=1)
print("\nsaved tail_sim_results.json")
