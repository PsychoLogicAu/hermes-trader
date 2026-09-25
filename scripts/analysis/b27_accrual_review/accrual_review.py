#!/usr/bin/env python3
"""B.27 WOULD-RELEASE accrual review (~10-05 target, run early at n=51, 2026-09-24).

Population: every `late_chase_volume_accumulation WOULD RELEASE` line in
trader-logs/trader.log* 09-21 11:36 -> 09-24 22:21 UTC (scratch/_b27_accrual_rows.json,
51 lines, all LONG, deduped per coin/30min — no collisions).

Per review recipe (WATCHLIST §B.27):
  - price each accrual through BOTH exit mirrors on candlestore 5m:
      scalp       protect 1.0 / retrace .25 / tiers [(8,.35),(15,.40)]
      trend_ride  protect 3.0 / retrace .55 / tiers [(3,.55),(8,.45),(15,.4)]
    (validated LONG-only mirror from _lc_grind_levers.py / _trendride_scope_ab.py;
     +~1.7pp harness inflation — deltas usable, absolutes not)
  - SPLIT by BTC regime at scan (up vs neutral+down), replicated via
    _trendride_scope_ab._regime_at (xyz:* -> SP500 proxy); pass-without-top-trade
    bar must hold in EACH subcohort.
  - test vol_ratio CEILING band (release 5-12x) alongside floor, per day-1 VR note.
  - report net, win, max_loss, top-trade share; promote only if trend_ride cohort
    positive WITHOUT its top trade (in each regime subcohort).
  - sequential occupancy-aware dedupe (win 30m / loss 180m cooldown) as realism check.
"""
import json, os, sys, datetime as dt, statistics as st
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "scratch", "candlestore"))
from candles import series

FEE_PCT, NOTIONAL = 0.10, 35.0
TIERS_SCALP = [(8.0, 0.35), (15.0, 0.40)]
TIERS_TRIDE = [(3.0, 0.55), (8.0, 0.45), (15.0, 0.40)]
CONSEC_REQ, BE_TRIG, BE_LOCK = 2, 2.5, 0.05
MAX_LOSS_SPOT, GRACE_S = 5.0, 90.0
HARD_TIMEOUT_MIN = 1800.0
COOLDOWN_WIN_MS, COOLDOWN_LOSS_MS = 30*60_000, 180*60_000

ROWS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_b27_accrual_rows.json")))

_cache5 = {}
def m5(coin):
    if coin not in _cache5:
        _cache5[coin] = series(coin, "5m")
    return _cache5[coin]

def bis_left(rows, t_ms):
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi) // 2
        if rows[mid]["t"] < t_ms: lo = mid + 1
        else: hi = mid
    return lo

def ema(vals, n):
    if not vals: return []
    k = 2/(n+1); out=[vals[0]]
    for v in vals[1:]:
        out.append(v*k + out[-1]*(1-k))
    return out

def _regime_at(rows, t_ms):
    closes = [r["c"] for r in rows if r["t"] + 3_600_000 <= t_ms]
    if len(closes) < 51: return "neutral"
    fast = ema(closes, 20); slow = ema(closes, 50)
    f_now, s_now = fast[-1], slow[-1]
    f_prev = fast[-9]
    if f_prev == 0: return "neutral"
    slope = (f_now - f_prev)/abs(f_prev)
    if f_now > s_now and slope > 0.001: return "up"
    if f_now < s_now and slope < -0.001: return "down"
    return "neutral"

_btc1h = _sp5 = None
def regime_for(coin, t_ms):
    global _btc1h, _sp5
    if coin.startswith("xyz:") and coin != "xyz:SP500": pass
    if _btc1h is None: _btc1h = series("BTC", "1h")
    if _sp5 is None: _sp5 = series("xyz:SP500", "1h")
    rows = _sp5 if coin.startswith("xyz:") else _btc1h
    return _regime_at(rows, t_ms)

def replay(coin, t_ms, protect, retrace, tiers):
    """LONG-only DSL mirror (identical engine to _trendride_scope_ab.replay)."""
    rows = m5(coin)
    if not rows: return None
    i = bis_left(rows, t_ms)
    ent_rows = [r for r in rows[:i] if r["t"] + 300_000 <= t_ms]
    if not ent_rows: return None
    j0 = len(ent_rows)-1
    fut = rows[j0+1:]
    if len(fut) < 4: return None
    entry = ent_rows[-1]["c"]; t0 = ent_rows[-1]["t"]
    peak, floor, breaches = entry, None, 0
    exit_px = exit_type = None; hold = None; max_adv = 0.0
    for r in fut:
        age_s = (r["t"]-t0)/1000
        stop = entry*(1-MAX_LOSS_SPOT/100)
        adv = (1-r["l"]/entry)*100
        if adv > max_adv: max_adv = adv
        if r["l"] <= stop:
            exit_px, exit_type = min(r["o"], stop) if r["o"] < stop else stop, "max_loss"
        elif floor is not None and age_s >= GRACE_S and r["l"] <= floor:
            breaches += 1
            if breaches >= CONSEC_REQ:
                exit_px, exit_type = min(r["o"], floor), "floor_breach"
        else:
            breaches = 0
        if exit_px is None:
            peak = max(peak, r["h"])
            pp = (peak/entry-1)*100
            if pp >= protect:
                rt = retrace
                for thr, v in tiers:
                    if pp >= thr: rt = v
                floor = max(floor or -1e18, entry + (peak-entry)*(1-rt))
            if pp >= BE_TRIG:
                floor = max(floor or -1e18, entry*(1+BE_LOCK/100))
        else:
            hold = (r["t"]-t0)/60_000; break
        if (r["t"]-t0)/60_000 >= HARD_TIMEOUT_MIN:
            exit_px, exit_type, hold = r["c"], "hard_timeout", (r["t"]-t0)/60_000
            break
    if exit_px is None:
        exit_px, exit_type = fut[-1]["c"], "stream_end"
        hold = (fut[-1]["t"]-t0)/60_000
    gross = (exit_px/entry-1)*100
    return {"pnl_pct": gross-FEE_PCT, "usd": (gross-FEE_PCT)/100*NOTIONAL,
            "exit": exit_type, "hold_min": round(hold), "max_adv_pct": round(max_adv,2)}

# ── price every accrual under both policies + regime at scan ─────────────────
priced = []
for e in sorted(ROWS, key=lambda r: r["ms"]):
    s  = replay(e["coin"], e["ms"], 1.0, 0.25, TIERS_SCALP)
    tr = replay(e["coin"], e["ms"], 3.0, 0.55, TIERS_TRIDE)
    if not s or not tr:
        print("NO CANDLES:", e["coin"], e["ts"]); continue
    reg = regime_for(e["coin"], e["ms"])
    priced.append({**e, "regime": reg, "scalp": s, "tride": tr})
print(f"priced {len(priced)}/{len(ROWS)} accruals  (window {ROWS[0]['ts']} -> {ROWS[-1]['ts']} UTC)")

def summarize(label, rows):
    if not rows:
        print(f"{label}: n=0"); return
    for pol, key in (("scalp", "scalp"), ("trend_ride", "tride")):
        sel = [r[key] for r in rows]
        net = sum(x["usd"] for x in sel)
        w   = sum(1 for x in sel if x["pnl_pct"] > 0)
        ml  = sum(1 for x in sel if x["exit"] == "max_loss")
        top_row = max(rows, key=lambda r: r[key]["usd"])
        top = top_row[key]
        ex_top = net - top["usd"]
        print(f"{label} | {pol:10s} n={len(sel):3d} net ${net:+7.2f} win {w/len(sel)*100:3.0f}% "
              f"maxL {ml:2d} | top +${top['usd']:.2f} ({top_row['coin']} {top_row['ts'][5:16]}) "
              f"w/o-top ${ex_top:+7.2f}")

def sequential(rows):
    """occupancy-aware dedupe: one sim per coin, cooldown win30/loss180."""
    out = {"scalp": [], "tride": []}
    for pol in ("scalp", "tride"):
        last_free = {}
        for r in sorted(rows, key=lambda x: x["ms"]):
            if r["ms"] < last_free.get(r["coin"], 0): continue
            out[pol].append(r)
            cd = COOLDOWN_WIN_MS if r[pol]["pnl_pct"] > 0 else COOLDOWN_LOSS_MS
            last_free[r["coin"]] = r["ms"] + max(r[pol]["hold_min"]*60_000, 0) + cd
    return out

print("\n═══ POOLED (all 51, per-event) ═══")
summarize("ALL          ", priced)
print("\n═══ REGIME SPLIT at scan (pass-without-top must hold in EACH) ═══")
up   = [r for r in priced if r["regime"] == "up"]
nond = [r for r in priced if r["regime"] != "up"]
summarize(f"regime=up    ", up)
summarize(f"regime!=up   ", nond)

print("\n═══ VR-BAND TEST (ceiling per day-1 observation) ═══")
for lo, hi in ((5.0, 99.0), (5.0, 12.0), (5.0, 8.0), (8.0, 99.0), (12.0, 99.0)):
    band = [r for r in priced if lo <= r["vr"] < hi]
    summarize(f"vr {lo:>4}-{hi:<4}", band)

print("\n═══ SEQUENTIAL OCCUPANCY-AWARE (win30/loss180 cooldowns) ═══")
seq = sequential(priced)
summarize("SEQ ALL      ", seq["scalp"])
summarize("SEQ vr5-12   ", [r for r in seq["scalp"] if r["vr"] < 12])

print("\n═══ PER-EVENT DETAIL (sorted by trend_ride pnl) ═══")
for r in sorted(priced, key=lambda x: x["tride"]["usd"]):
    ts = r["ts"][5:16]
    print(f"  {ts} {r['coin']:10s} vr {r['vr']:5.1f} conf {r['conf']:.2f}/{r['bar']:.2f} "
          f"{r['regime']:7s} | scalp {r['scalp']['pnl_pct']:+6.2f}% ({r['scalp']['exit']:11s}"
          f" {r['scalp']['hold_min']:4d}m) | TR {r['tride']['pnl_pct']:+6.2f}% "
          f"({r['tride']['exit']:11s} {r['tride']['hold_min']:4d}m)")

json.dump([{k: v for k, v in r.items() if k not in ("scalp", "tride")} |
           {"scalp": r["scalp"], "tride": r["tride"]} for r in priced],
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_b27_accrual_priced.json"), "w"), indent=1)
print("\nsaved scratch/_b27_accrual_priced.json")
