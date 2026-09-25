#!/usr/bin/env python3
"""B.27 follow-up — does a SUSTAINED volume RAMP (vs single-bar spike) separate the cohort?

Owner question 2026-09-25: lever B died on vr = last-5m-bar / mean(prior 96). But that
feature can't tell "one fat bar" from "volume stepping up over a window while price grinds
up". Re-slice the same 51 WOULD-RELEASE accruals with ramp features computed at scan time:

  slope12   : OLS slope of ln(volume) over last 12 finalized 5m bars (per-bar growth)
  sust12    : mean(last 12 vols) / mean(prior 96 excl those 12)   — hour-scale accumulation
  spike_dom : last bar vol / mean(last 12 vols)                    — ~1 = broad, >>1 = one-bar spike
  grind24   : price return over prior 24 finalized bars            — "slow grinding upwards"

Slice the cohort on these and report both exit mirrors. CAVEAT stated up front: n=51,
post-hoc slicing — a positive cell here is a HYPOTHESIS for a new accrual, not a promote.
"""
import json, os, sys, math, datetime as dt, statistics as st
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

def feats(coin, t_ms):
    rows = m5(coin)
    i = bis_left(rows, t_ms)
    conf = [r for r in rows[:i] if r["t"] + 300_000 <= t_ms]
    if len(conf) < 110: return None
    vols = [r["v"] for r in conf]
    last12 = vols[-12:]
    prior96 = vols[-108:-12]
    mu_prior = sum(prior96)/len(prior96)
    mu12 = sum(last12)/len(last12)
    # OLS slope of ln(vol) over last 12 bars (guard zero vols with tiny epsilon)
    xs = list(range(12))
    ys = [math.log(max(v, 1e-9)) for v in last12]
    mx, my = sum(xs)/12, sum(ys)/12
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = sum((x-mx)**2 for x in xs)
    slope12 = num/den                       # per-bar log growth over the hour
    sust12 = mu12/mu_prior if mu_prior > 0 else None
    spike_dom = vols[-1]/mu12 if mu12 > 0 else None
    closes = [r["c"] for r in conf]
    grind24 = (closes[-1]/closes[-25] - 1)*100
    return {"slope12": slope12, "sust12": sust12, "spike_dom": spike_dom, "grind24": grind24}

for r in PRICED:
    f = feats(r["coin"], r["ms"])
    if f is None:
        print("feature fail:", r["coin"], r["ts"]); continue
    r.update(f)

def summarize(label, rows):
    if not rows:
        print(f"{label:44s} n=0"); return
    s = sum(r["scalp"]["usd"] for r in rows); t = sum(r["tride"]["usd"] for r in rows)
    w = sum(1 for r in rows if r["scalp"]["pnl_pct"] > 0)
    ml = sum(1 for r in rows if r["scalp"]["exit"] == "max_loss")
    print(f"{label:44s} n={len(rows):2d} scalp ${s:+6.2f} TR ${t:+6.2f} win {w/len(rows)*100:3.0f}% maxL {ml}")

print("distribution of features over the 51:")
for k in ("slope12","sust12","spike_dom","grind24"):
    vals = sorted(r[k] for r in PRICED if r.get(k) is not None)
    q = lambda p: vals[min(len(vals)-1, int(p*len(vals)))]
    print(f"  {k:9s} min {vals[0]:+.3f} p25 {q(.25):+.3f} med {q(.5):+.3f} p75 {q(.75):+.3f} max {vals[-1]:+.3f}")

print("\n── ramp vs spike slices (ramp = sustained hour-scale accumulation, not one bar) ──")
summarize("ALL (baseline)", PRICED)
summarize("slope12 > 0 (vol rising over the hour)", [r for r in PRICED if r["slope12"] > 0])
summarize("slope12 <= 0 (flat/fading)", [r for r in PRICED if r["slope12"] <= 0])
summarize("sust12 >= 2 (hour mean 2x prior day-mean)", [r for r in PRICED if r["sust12"] >= 2])
summarize("sust12 < 2 (vr5 driven by single bar)", [r for r in PRICED if r["sust12"] < 2])
summarize("spike_dom <= 2 (broad, no one-bar monster)", [r for r in PRICED if r["spike_dom"] <= 2])
summarize("spike_dom > 2 (one bar dominates)", [r for r in PRICED if r["spike_dom"] > 2])

print("\n── + grind filter ('slow grinding upwards' measured over prior 2h) ──")
summarize("slope12>0 AND grind24>0", [r for r in PRICED if r["slope12"] > 0 and r["grind24"] > 0])
summarize("sust12>=2 AND grind24>0", [r for r in PRICED if r["sust12"] >= 2 and r["grind24"] > 0])
summarize("sust12>=2 AND spike_dom<=2", [r for r in PRICED if r["sust12"] >= 2 and r["spike_dom"] <= 2])
summarize("sust12>=2 AND spike_dom<=2 AND grind24>0",
          [r for r in PRICED if r["sust12"] >= 2 and r["spike_dom"] <= 2 and r["grind24"] > 0])
summarize("slope12>=0.10 (steep ramp) AND grind24>0",
          [r for r in PRICED if r["slope12"] >= 0.10 and r["grind24"] > 0])

print("\n── per-event with features (sorted by slope12) ──")
for r in sorted(PRICED, key=lambda x: -x["slope12"]):
    print(f"  {r['ts'][5:16]} {r['coin']:10s} vr {r['vr']:5.1f} slope {r['slope12']:+.3f} "
          f"sust {r['sust12']:5.2f} dom {r['spike_dom']:4.1f} grind24 {r['grind24']:+5.1f}% "
          f"| scalp {r['scalp']['pnl_pct']:+6.2f}% | TR {r['tride']['pnl_pct']:+6.2f}%")

json.dump(PRICED, open(os.path.join(_HERE, "_b27_accrual_priced_feat.json"), "w"), indent=1)
print("\nsaved _b27_accrual_priced_feat.json")
