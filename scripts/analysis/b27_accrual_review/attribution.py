#!/usr/bin/env python3
"""B.27 follow-up 3 — OUTCOME-DRIVEN attribution: what separates the winners from losers?

Owner question 2026-09-25: instead of hand-picked slices, rank every feature at scan time
by how cleanly it separates the cohort's best vs worst outcomes. Features = everything
computed so far (vr8..vr96, slope12, sust12, spike_dom, grind24h=grind24, regime, conf)
PLUS new ones never examined: ext24 (price return over prior 24h), dist_24h_high (entry
below trailing 24h high), atr_pct (realised vol), bars_since_big_vr (how long ago the
volume event started), hour_utc.

Outcome groups (primary = scalp mirror, since that's the live exit):
  LOSERS   : scalp max_loss (the -5.1% tail)      vs all others -> Mann-Whitney U per feature
  and a graded view: top-tercile vs bottom-tercile of scalp pnl.

Multiple-comparison honesty: ~12 features x 2 tests on n=51 — anything "found" is a
HYPOTHESIS for fresh accrual, not promote evidence (same bar as ramp/window follow-ups).
"""
import json, os, sys, math, statistics as st
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "scratch", "candlestore"))
from candles import series

PRICED = json.load(open(os.path.join(_HERE, "_b27_accrual_priced_vrw.json")))
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

def new_feats(coin, t_ms):
    rows = m5(coin)
    i = bis_left(rows, t_ms)
    conf = [r for r in rows[:i] if r["t"] + 300_000 <= t_ms]
    if len(conf) < 300: return None
    closes = [r["c"] for r in conf]; highs = [r["h"] for r in conf]; lows = [r["l"] for r in conf]
    vols = [r["v"] for r in conf]
    px = closes[-1]
    ext24 = (px/closes[-289] - 1)*100 if len(closes) >= 289 else None      # 24h move into entry
    hi24 = max(highs[-288:]); dist_hi = (px/hi24 - 1)*100                  # <=0, how far below 24h high
    trs = [max(highs[k]-lows[k], abs(highs[k]-closes[k-1]), abs(lows[k]-closes[k-1]))
           for k in range(-28, -1)]
    atr_pct = (sum(trs)/28 / px) * 100                                     # 5m ATR as % of px
    mu96 = sum(vols[-97:-1])/96
    bsib = None                                                             # bars since last bar >=5x mu96
    for k in range(1, min(289, len(vols))):
        if vols[-k] >= 5*mu96: bsib = k-1; break
    return {"ext24": ext24, "dist_hi": dist_hi, "atr_pct": atr_pct, "bsib": bsib,
            "hour_utc": (t_ms // 3_600_000) % 24}

for r in PRICED:
    nf = new_feats(r["coin"], r["ms"])
    if nf: r.update(nf)

FEATS = ["vr","vr8","vr16","vr24","slope12","sust12","spike_dom","grind24",
         "ext24","dist_hi","atr_pct","bsib","conf","hour_utc"]

def mwu(a, b):
    """Mann-Whitney U + normal-approx z (ties: midranks). Returns (z, p_two_sided)."""
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3: return None
    allv = sorted(list(a) + list(b))
    def rank_of(v):
        lo = allv.index(v); hi2 = lo
        while hi2+1 < len(allv) and allv[hi2+1] == v: hi2 += 1
        return (lo + hi2)/2 + 1
    R1 = sum(rank_of(v) for v in a)
    U = R1 - n1*(n1+1)/2
    mu = n1*n2/2
    sd = math.sqrt(n1*n2*(n1+n2+1)/12)
    if sd == 0: return None
    z = (U - mu)/sd
    p = 2*(1 - 0.5*(1 + math.erf(abs(z)/math.sqrt(2))))
    return z, p

print("═══ A) scalp max_loss (n=11) vs rest — Mann-Whitney separation ═══")
losers = [r for r in PRICED if r["scalp"]["exit"] == "max_loss"]
winners = [r for r in PRICED if r["scalp"]["exit"] != "max_loss"]
rows_a = []
for f in FEATS:
    a = [r[f] for r in losers if r.get(f) is not None]
    b = [r[f] for r in winners if r.get(f) is not None]
    res = mwu(a, b)
    if res:
        z, p = res
        rows_a.append((abs(z), f, st.median(a), st.median(b), z, p))
for _, f, ma, mb, z, p in sorted(rows_a, reverse=True):
    flag = "  <-- candidate" if p < 0.05 else ""
    print(f"  {f:9s} loser-med {ma:+8.2f}  rest-med {mb:+8.2f}   z={z:+5.2f} p={p:.3f}{flag}")

print("\n═══ B) graded: bottom vs top tercile of scalp pnl ═══")
sp = sorted(PRICED, key=lambda r: r["scalp"]["pnl_pct"])
k = 17
bot, top = sp[:k], sp[-k:]
for f in FEATS:
    a = [r[f] for r in bot if r.get(f) is not None]
    b = [r[f] for r in top if r.get(f) is not None]
    res = mwu(a, b)
    if res:
        z, p = res
        flag = "  <-- candidate" if p < 0.05 else ""
        print(f"  {f:9s} worst-med {st.median(a):+8.2f}  best-med {st.median(b):+8.2f}   z={z:+5.2f} p={p:.3f}{flag}")

print("\n═══ C) eyeball table: every event, outcome-ordered ═══")
for r in sorted(PRICED, key=lambda x: x["scalp"]["usd"]):
    def g(f, w=6, d=2):
        v = r.get(f)
        return (f"{v:+{w}.{d}f}" if isinstance(v, float) else f"{str(v):>{w}}")
    print(f"  {r['ts'][5:16]} {r['coin']:9s} {g('ext24'):>7} ext {g('dist_hi'):>6} hi "
          f"{g('atr_pct',5)} atr {g('bsib',4,0)} bsib vr{r['vr']:5.1f} "
          f"reg {r['regime'][:3]} | scalp {r['scalp']['pnl_pct']:+6.2f}% ({r['scalp']['exit']})")

# D) the only structural question left: does ANY simple rule on the strongest separators
# produce a positive cell? (report even if none — foreclosure)
print("\n═══ D) rules built from top separators ═══")
def summarize(label, rows):
    if not rows: print(f"  {label:44s} n= 0"); return
    s_ = sum(r["scalp"]["usd"] for r in rows); t = sum(r["tride"]["usd"] for r in rows)
    w_ = sum(1 for r in rows if r["scalp"]["pnl_pct"] > 0)
    ml = sum(1 for r in rows if r["scalp"]["exit"] == "max_loss")
    print(f"  {label:44s} n={len(rows):2d} scalp ${s_:+6.2f} TR ${t:+6.2f} win {w_/len(rows)*100:3.0f}% maxL {ml}")
summarize("dist_hi <= -5% (NOT at the highs)", [r for r in PRICED if r["dist_hi"] <= -5])
summarize("dist_hi > -2% (at/near 24h high)", [r for r in PRICED if r["dist_hi"] > -2])
summarize("ext24 < +10% (not stretched on day scale)", [r for r in PRICED if r["ext24"] is not None and r["ext24"] < 10])
summarize("atr_pct <= median (calm vol)", [r for r in PRICED if r["atr_pct"] <= st.median(x["atr_pct"] for x in PRICED)])
summarize("dist_hi<=-5 AND ext24<10", [r for r in PRICED if r["dist_hi"] <= -5 and (r["ext24"] or 99) < 10])
