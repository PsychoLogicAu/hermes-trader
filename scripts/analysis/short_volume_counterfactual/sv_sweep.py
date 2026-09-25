#!/usr/bin/env python3
"""min_short_volume_usd counterfactual — floor sweep with occupancy-aware replay.

Releases sole-short-thin blocks whose logged 24h vol clears a candidate floor.
Sweep: 20 (live) -> 15 -> 10 -> 5 -> 2 ($M). A block is released at floor X iff
X <= vol < 20 and it has NO other blocker (all other gates stay in force).
30-min same-coin dedupe. Occupancy: 5-slot book shared with real ledger
intervals, no same-coin re-entry while held (real or sim), stale_flat only when
contention >= 3 (mirrors live code).

Exit policy mirrors live dsl_exit.py on 5m bars (same as tail_audit_sim):
  entry = open of first bar at/after decision ts
  max_loss: stop = entry*(1+MAXLOSS) for shorts, MAXLOSS = min(5, 30/lev)=5
  floor: phase1 = entry*(1+MAXLOSS/100); phase2 arms when peak >= protect(1%),
         floor = entry - range*(1-retrace), tier by PEAK (8%->0.35, 15%->0.40,
         default .25); breakeven ratchet peak>=2.5% -> lock 0.05%; ratchets only.
  breach: 2 consecutive bar CLOSES beyond floor.
  stale_flat: elapsed>=240min AND peak<protect AND contention>=3.
  hard_timeout: 1800min. Fees 2.5 bps/side. lev 5x, $35 notional.
"""
import json, os, time, urllib.request, bisect
from datetime import datetime, timezone
from collections import Counter

ROOT = "/home/oknight/src/hermes-trader"
SCRATCH = os.environ.get("SV_SCRATCH", os.path.join(ROOT, ".hermes/scratch"))
CACHE = os.path.join(SCRATCH, "hl_candles_sv.json")
DATA = json.load(open(os.path.join(SCRATCH, "sv_blocks.json")))

NOTIONAL = 35.0; LEV = 5; FEE = 0.00025
MAXLOSS = min(5.0, 30.0/LEV); PROTECT = 1.0
TIERS = [(15.0, 0.40), (8.0, 0.35)]; DEFAULT_RETRACE = 0.25
BE_TRIG, BE_LOCK = 2.5, 0.05
STALE_MIN, HARD_MIN, CONSEC = 240, 1800, 2
BAR = 300_000
FLOORS = [15.0, 10.0, 5.0, 2.0]   # candidate floors ($M); live = 20

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
# 30-min same-coin dedupe (keep first of each burst)
seen_pair = {}
deduped = []
for b in sorted(blocks, key=lambda x: x["ts"]):
    t = ets(b["ts"])
    k = b["coin"]
    if k in seen_pair and t - seen_pair[k] < 30*60_000: continue
    seen_pair[k] = t
    deduped.append(b)
rel_all = [b for b in deduped if not b["other_blockers"] and b.get("vol_musd")]
print(f"blocks {len(blocks)} -> 30min-deduped {len(deduped)}; sole-blocker releasable pool: {len(rel_all)}")

need = sorted({b["coin"] for b in rel_all})
lo = min(ets(b["ts"]) for b in rel_all) - 2*BAR
hi = max(ets(b["ts"]) for b in rel_all) + (HARD_MIN+10)*60_000

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
def lts(e): return float(e["ts"])
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
opens2 = {}
for line in open(os.path.join(ROOT, "trades.jsonl" if False else "trader-logs/trades.jsonl"), errors="replace"):
    try: e = json.loads(line)
    except Exception: continue
    ev = e.get("event")
    if ev == "OPEN": opens2[e.get("coin")] = lts(e)
    elif ev == "CLOSE" and e.get("coin") in opens2:
        real_by_coin.setdefault(e["coin"], []).append((opens2.pop(e["coin"]), lts(e)))
for c,st in opens2.items(): real_by_coin.setdefault(c, []).append((st, float("inf")))

def coin_held_real(coin, t):
    return any(a <= t <= z for a,z in real_by_coin.get(coin, []))

def replay(events, occupancy=True):
    events = sorted(events, key=lambda b: ets(b["ts"]))
    sim_open = {}
    out, skips = [], Counter()
    for b in events:
        t0 = ets(b["ts"])
        side = "short"  # gate only ever fires on shorts
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
                    "exit": why, "conf": b.get("conf"), "vol_musd": b.get("vol_musd"),
                    "exit_ts": datetime.fromtimestamp(tex/1000, timezone.utc).strftime("%Y-%m-%d %H:%M") if tex else None})
    return out, dict(skips)

def summarize(rs, label):
    if not rs:
        print(f"\n{label}: n=0"); return 0.0
    tot = sum(r["pnl"] for r in rs); w = sum(1 for r in rs if r["pnl"] > 0)
    print(f"\n{label}: n={len(rs)} net=${tot:+.2f}  win {w}/{len(rs)} ({100*w/len(rs):.0f}%)  avg ${tot/len(rs):+.2f}")
    print("  exits:", dict(Counter(r["exit"] for r in rs)))
    return tot

results = {}
print("\n=== INCREMENTAL COHORTS (vol band between floors, gate-only) ===")
for f in FLOORS:
    incr = [b for b in rel_all if f <= b["vol_musd"] < (FLOORS[FLOORS.index(f)-1] if FLOORS.index(f) else 20.0)]
    gi, ski = replay(incr, occupancy=False)
    summarize(gi, f"FLOOR {f:.0f}M — incremental band [{f:.0f},{(FLOORS[FLOORS.index(f)-1] if FLOORS.index(f) else 20):.0f})M gate-only")
    ex, skx = replay(incr, occupancy=True)
    summarize(ex, f"FLOOR {f:.0f}M — incremental band occupancy-aware")
    results[str(f)] = {"incremental_gate_only": gi, "incremental_exec": ex, "skips_exec": skx}

print("\n=== CUMULATIVE (release everything >= floor X) ===")
for f in FLOORS:
    cum = [b for b in rel_all if b["vol_musd"] >= f]
    goc, _ = replay(cum, occupancy=False)
    exc, skc = replay(cum, occupancy=True)
    summarize(goc, f"FLOOR {f:.0f}M cumulative gate-only (n pool {len(cum)})")
    summarize(exc, f"FLOOR {f:.0f}M cumulative occupancy-aware")
    results[str(f)]["cumulative_gate_only"] = goc
    results[str(f)]["cumulative_exec"] = exc
    results[str(f)]["cum_skips"] = skc

# per-coin breakdown on the 10M cumulative executable set
print("\n=== per-coin (floor 10M cumulative, occupancy-aware) ===")
byc = {}
for r in results["10.0"]["cumulative_exec"]:
    byc.setdefault(r["coin"], [0,0.0]); byc[r["coin"]][0]+=1; byc[r["coin"]][1]+=r["pnl"]
for c,(n,p) in sorted(byc.items(), key=lambda x:x[1][1]):
    print(f"  {c:12s} n={n:2d} ${p:+7.2f}")

# conf bands
print("\n=== conf bands (floor 10M cumulative, occupancy-aware) ===")
bands = {}
for r in results["10.0"]["cumulative_exec"]:
    c = r.get("conf") or 0
    band = "<0.75" if c < 0.75 else ("0.75-0.82" if c <= 0.82 else ">0.82")
    bands.setdefault(band,[0,0.0]); bands[band][0]+=1; bands[band][1]+=r["pnl"]
print({k:(v[0],round(v[1],2)) for k,v in sorted(bands.items())})

json.dump(results, open(os.path.join(SCRATCH,"sv_sim_results.json"),"w"), indent=1)
print("\nsaved sv_sim_results.json")
