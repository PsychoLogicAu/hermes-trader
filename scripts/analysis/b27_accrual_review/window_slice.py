#!/usr/bin/env python3
"""B.27 follow-up 2 — is the vr BASELINE WINDOW (96×5m = 8h) too long for meme tape?

Owner question 2026-09-25: gate fires on last-5m-vol >= 5x mean(prior 96 bars). Maybe
8h of history is stale for fast-decay coins — try shorter baselines (e.g. 8x15min = 2h,
i.e. mean of prior 24x5m) which demand RECENT expansion instead of expansion-vs-yesterday.

Method: recompute vr_w = last bar / mean(prior w bars) for w in {8,16,24,48,96} at each
of the 51 scan times (same confirmed-bars discipline). Two questions:
  1. SELECTIVITY — how many of the 51 survive vr_w >= 5 as w shrinks? (if it stays ~51,
     the window change doesn't alter what fires; if it drops, it's a real filter)
  2. SEPARATION — do the survivors at short windows price better under both mirrors?
Also: events that FAIL vr24 but pass vr96 = "expansion vs stale baseline only" class —
price them separately (that's the class the user suspects is junk).

CAVEAT: same n=51, post-hoc slices — hypothesis generator, not promote evidence.
"""
import json, os, sys, statistics as st
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "scratch", "candlestore"))
from candles import series

PRICED = json.load(open(os.path.join(_HERE, "_b27_accrual_priced.json")))
_c5 = {}
def m5(coin):
    if coin not in _c5:
        _c5[coin] = series(coin, "5m")
    return _c5[coin]

def bis_left(rows, t_ms):
    lo, hi = 0, len(rows)
    while lo < hi:
        mid = (lo + hi)//2
        if rows[mid]["t"] < t_ms: lo = mid+1
        else: hi = mid
    return lo

def vr_w(coin, t_ms, w):
    """last finalized 5m vol / mean(prior w finalized bars) — same convention as the gate."""
    rows = m5(coin)
    i = bis_left(rows, t_ms)
    conf = [r for r in rows[:i] if r["t"] + 300_000 <= t_ms]
    if len(conf) < w + 1: return None
    vols = [r["v"] for r in conf]
    mu = sum(vols[-(w+1):-1]) / w
    return vols[-1] / mu if mu > 0 else None

WINDOWS = (8, 16, 24, 48, 96)   # 40m, 80m, 2h, 4h, 8h baselines
for r in PRICED:
    for w in WINDOWS:
        r[f"vr{w}"] = vr_w(r["coin"], r["ms"], w)

def summarize(label, rows):
    if not rows:
        print(f"{label:46s} n= 0"); return
    s = sum(r["scalp"]["usd"] for r in rows); t = sum(r["tride"]["usd"] for r in rows)
    w_ = sum(1 for r in rows if r["scalp"]["pnl_pct"] > 0)
    ml = sum(1 for r in rows if r["scalp"]["exit"] == "max_loss")
    print(f"{label:46s} n={len(rows):2d} scalp ${s:+6.2f} TR ${t:+6.2f} win {w_/len(rows)*100:3.0f}% maxL {ml}")

print("Q1 — SELECTIVITY: how many of the 51 still fire at vr>=5 per baseline window")
for w in WINDOWS:
    n = sum(1 for r in PRICED if (r[f"vr{w}"] or 0) >= 5)
    hrs = w*5/60
    print(f"  prior {w:2d} bars ({hrs:4.1f}h baseline): vr>={5:>4} fires on {n}/51")

print("\nQ2 — SEPARATION: cohort priced, sliced by the SHORT-baseline ratio")
for w in (8, 16, 24):
    lo = [r for r in PRICED if (r[f"vr{w}"] or 0) >= 5]
    hi_ = [r for r in PRICED if (r[f"vr{w}"] or 0) < 5]
    summarize(f"vr{w} >= 5  (recent expansion agrees)", lo)
    summarize(f"vr{w} <   5  (only passes vs stale 8h mean)", hi_)
    print()

print("── same idea as a threshold shift on the CURRENT window (fair comparison) ──")
for thr in (5, 8, 12):
    summarize(f"vr96 >= {thr} (today's gate at higher bar)", [r for r in PRICED if r["vr"] >= thr])

print("\n── best-case union rules the owner might mean ──")
summarize("vr24>=5 AND vr96>=5 (both windows agree)",
          [r for r in PRICED if (r["vr24"] or 0) >= 5 and r["vr"] >= 5])
summarize("vr8>=5 AND vr24<15 (recent but not paroxysm)",
          [r for r in PRICED if (r["vr8"] or 0) >= 5 and (r["vr24"] or 0) < 15])

print("\n── per-event short-window ratios (sorted by vr24) ──")
for r in sorted(PRICED, key=lambda x: -(x["vr24"] or 0)):
    f = lambda k: f"{r[k]:6.1f}" if r[k] is not None else "   n/a"
    print(f"  {r['ts'][5:16]} {r['coin']:10s} vr8 {f('vr8')} vr16 {f('vr16')} vr24 {f('vr24')} "
          f"vr48 {f('vr48')} vr96 {f('vr')} | scalp {r['scalp']['pnl_pct']:+6.2f}% | TR {r['tride']['pnl_pct']:+6.2f}%")

json.dump(PRICED, open(os.path.join(_HERE, "_b27_accrual_priced_vrw.json"), "w"), indent=1)
print("\nsaved _b27_accrual_priced_vrw.json")
