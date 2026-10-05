#!/usr/bin/env python3
"""B.22 pass 2: price the timesfm_tail_trigger SOLE-BLOCKER cohort (live era).

Cohort: scripts/analysis/b22_timesfm_tail_forward/_b22_sole.json (from _b22_scan.py)
= scans since arm (2026-09-17 23:59Z) where timesfm_tail_trigger is the ONLY
failing gate -> the incremental set the live gate costs/saves.

Two counterfactuals, both occupancy-aware (real ledger intervals + sim book,
5 slots, no same-coin re-entry while held), live DSL exit mirror
(.agent-config.json dsl_exit @ 2026-10-05):
  A) FILTERED: candidate never enters. pnl = 0.
  B) RE-TIMED: entry at first same-coin scan >= block where the timesfm gate
     passes again (scan stream from Trade result lines); if none before the
     sim close, blocked-through-close (entry at close ts).
Baseline = entry at the block scan itself (what the world without the gate does).
pnl = notional * spot_pct - fees(2.5bps/side). Entry = open of first 5m bar
at/after decision ts. Entry-bar stop: close-side only; full OHLC from bar 1.
30-min same-coin dedupe of scans (one candidate, many scans).
"""
import ast, glob, json, re, time, urllib.request
import datetime as dt
from bisect import bisect_left

D = "/home/oknight/src/hermes-trader/scripts/analysis/b22_timesfm_tail_forward"
LOGS = sorted(glob.glob("/home/oknight/src/hermes-trader/trader-logs/trader.log*"))
LEDGER = "/home/oknight/src/hermes-trader/trader-logs/trades.jsonl"
CANDLE_CACHE = D + "/_candles.json"
ARM = dt.datetime(2026, 9, 17, 23, 59)  # naive UTC (log-line strptime is naive)
PFX = re.compile(r"^\[(?P<coin>.+?)\] (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ ")
TRADE_RE = re.compile(r"INFO:trading_loop:Trade result: (\{.*)$")

# live dsl_exit params
LEV = 5
STOP_PCT = min(5.0, 30.0 / LEV)          # 5.0 spot
PROTECT = 1.0
RETRACE = 0.25
TIERS = [(8.0, 0.35), (15.0, 0.40)]
BE_TRIG, BE_LOCK = 2.5, 0.05
STALE_MIN, HARD_MIN = 240, 1800
FEE = 2.5e-4
NOTIONAL = 33.33                        # live cap; median executed ~ same
BAR = 300_000

def ms_of(naive):
    return int(naive.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

# ── scan stream: per coin, (ts_ms, timesfm_failed) ──────────────────────────
scans = {}
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
        bb = d.get("blocked_by") or []
        failed = any(r.startswith("timesfm_tail_trigger") for r in bb)
        scans.setdefault(pm["coin"], []).append((ms_of(ts), failed))
for v in scans.values():
    v.sort()

# ── cohort: dedupe sole-blocker scans (30-min same-coin) ────────────────────
sole = json.load(open(D + "/_b22_sole.json"))
sole.sort(key=lambda s: s["ts"])
deduped, last = [], {}
for s in sole:
    t = ms_of(dt.datetime.fromisoformat(s["ts"]))
    if s["coin"] in last and t - last[s["coin"]] < 30 * 60_000:
        continue
    last[s["coin"]] = t
    deduped.append(dict(coin=s["coin"], t=t, side=s["side"]))
print(f"sole-blocker scans {len(sole)} -> deduped candidates {len(deduped)}")

# ── real ledger intervals (occupancy) ───────────────────────────────────────
rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
real_iv = []
open_map = {}
for r in rows:
    if r.get("event") == "OPEN":
        open_map[(r["coin"], r.get("side"))] = ms_of(dt.datetime.fromisoformat(r["ts_iso"].replace("Z","+00:00")))
    elif r.get("event") == "CLOSE":
        k = (r["coin"], r.get("side"))
        if k in open_map:
            real_iv.append((open_map.pop(k), ms_of(dt.datetime.fromisoformat(r["ts_iso"].replace("Z","+00:00"))), r["coin"]))
for k, t in open_map.items():
    real_iv.append((t, 2**62, k[0]))

# ── candles ─────────────────────────────────────────────────────────────────
try:
    CACHE = json.load(open(CANDLE_CACHE))
except Exception:
    CACHE = {}

def fetch(coin, t0, t1):
    key = coin
    got = CACHE.get(key, [])
    if got and got[0][0] <= t0 - BAR and got[-1][0] >= min(t1, time.time()*1000*1000) - BAR:
        return got
    req = urllib.request.Request("https://api.hyperliquid.xyz/info",
        data=json.dumps({"type": "candleSnapshot", "req": {
            "coin": coin, "interval": "5m",
            "startTime": t0 - 3600_000, "endTime": int(time.time()*1000)}}).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            arr = json.loads(urllib.request.urlopen(req, timeout=30).read())
            break
        except Exception:
            time.sleep(3 * (attempt + 1))
    else:
        arr = []
    bars = [(int(k["t"]), float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]))
            for k in arr if k.get("c") is not None]
    bars.sort()
    CACHE[key] = bars
    return bars

def sim_exit(coin, entry_t, entry_px, side):
    """Live DSL mirror. Returns (exit_px, exit_t, exit_type)."""
    bars = fetch(coin, entry_t, entry_t + HARD_MIN * 60_000)
    i = bisect_left([b[0] for b in bars], entry_t)
    if i >= len(bars):
        return None, None, "no_candles"
    sgn = 1 if side == "long" else -1
    stop = entry_px * (1 - sgn * STOP_PCT / 100)
    floor = stop
    peak = entry_px
    armed = False
    for j in range(i, min(i + HARD_MIN // 5 + 1, len(bars))):
        t, o, h, l, c = bars[j]
        if j == i:  # entry bar straddles fill: close-side only
            adv = c
        else:
            adv = None
        el = (t - entry_t) / 60_000
        # update peak with bar extremes (peak after entry bar open)
        hi, lo = (h, l) if j > i else (max(o, c), min(o, c))
        peak = max(peak, hi) if sgn == 1 else min(peak, lo)
        up = sgn * (peak - entry_px) / entry_px * 100
        # trailing floor
        if up >= PROTECT:
            armed = True
            rt = RETRACE
            for tp, rv in TIERS:
                if up >= tp:
                    rt = rv
            f = entry_px + sgn * (peak - entry_px) * (1 - rt)
            floor = max(floor, f) if sgn == 1 else min(floor, f)
        if up >= BE_TRIG:
            be = entry_px * (1 + sgn * BE_LOCK / 100)
            floor = max(floor, be) if sgn == 1 else min(floor, be)
        # exit checks (SL first on dual touch)
        if j == i:
            hit_sl = (c <= stop) if sgn == 1 else (c >= stop)
            hit_fl = (c < floor) if sgn == 1 else (c > floor)
        else:
            hit_sl = (l <= stop) if sgn == 1 else (h >= stop)
            hit_fl = (l < floor) if sgn == 1 else (h > floor)
        if hit_sl:
            return stop, t, "max_loss"
        if armed and hit_fl:
            return floor, t, "floor_breach"
        if el >= STALE_MIN and up < PROTECT:
            return c, t, "stale_flat"
        if el >= HARD_MIN:
            return c, t, "hard_timeout"
    last = bars[min(i + HARD_MIN // 5, len(bars) - 1)]
    return last[4], last[0], "window_end"

def pnl_of(entry_t, entry_px, side, exit_px):
    sgn = 1 if side == "long" else -1
    spot = sgn * (exit_px / entry_px - 1) * 100
    return NOTIONAL * spot / 100 - 2 * FEE * NOTIONAL

def next_clear(coin, t):
    for s_ms, failed in scans.get(coin, []):
        if s_ms >= t and not failed:
            return s_ms
    return None

# ── occupancy-aware replay, time-ordered ────────────────────────────────────
sim_iv = []
def occupied(t, coin):
    n = sum(1 for a, b, _ in real_iv if a <= t < b) + sum(1 for a, b, _ in sim_iv if a <= t < b)
    if n >= 5:
        return "slots"
    if any((a <= t < b) and c == coin for a, b, c in real_iv + sim_iv):
        return "coin"
    return None

results = []
for cand in deduped:
    coin, t, side = cand["coin"], cand["t"], cand["side"]
    skip = occupied(t, coin)
    bars = fetch(coin, t, t + HARD_MIN * 60_000)
    i = bisect_left([b[0] for b in bars], t)
    if i >= len(bars):
        results.append(dict(coin=coin, ts=t, side=side, status="no_candles"))
        continue
    entry_px = bars[i][1]
    exit_px, exit_t, etype = sim_exit(coin, t, entry_px, side)
    base = pnl_of(t, entry_px, side, exit_px) if exit_px else None
    # re-timed path
    tc = next_clear(coin, t + 1)
    if tc is None or (exit_t and tc > exit_t):
        rt_entry_t, rt_basis = (exit_t or t), "blocked-through-close"
    else:
        rt_entry_t, rt_basis = tc, f"cleared+{(tc-t)//60000}m"
    j = bisect_left([b[0] for b in bars], rt_entry_t)
    j = min(j, len(bars) - 1)
    rt_px = bars[j][1]
    ex2, et2, ty2 = sim_exit(coin, rt_entry_t, rt_px, side)
    rtp = pnl_of(rt_entry_t, rt_px, side, ex2) if ex2 else None
    results.append(dict(coin=coin, ts=t, side=side, entry_px=entry_px,
                        exit_type=etype, base_pnl=base, rt_basis=rt_basis,
                        rt_pnl=rtp, rt_exit_type=ty2, skip=skip))
    if base is not None:
        sim_iv.append((t, exit_t, coin))

json.dump(results, open(D + "/_b22_priced.json", "w"), indent=1)
json.dump(CACHE, open(CANDLE_CACHE, "w"))

# ── report ──────────────────────────────────────────────────────────────────
priced = [r for r in results if r.get("base_pnl") is not None]
gate_only = [r for r in priced if not r.get("skip")]
print(f"\npriced n={len(priced)}  executable(occupancy) n={len(gate_only)}")
for label, grp in (("GATE-ONLY", priced), ("OCCUPANCY", gate_only)):
    nb = sum(r["base_pnl"] for r in grp)
    nr = sum(r["rt_pnl"] for r in grp if r.get("rt_pnl") is not None)
    w = sum(1 for r in grp if r["base_pnl"] > 0)
    ml = sum(1 for r in grp if r["exit_type"] == "max_loss")
    print(f"{label}: n={len(grp)}  baseline(no-gate) net {nb:+.2f}  wins {w}/{len(grp)}  max_loss {ml}")
    print(f"          re-timed net {nr:+.2f}   -> gate SAVES (filter) {-nb:+.2f} / (re-timed) {nb-nr:+.2f}")
grp = priced
grp.sort(key=lambda r: r["base_pnl"] if r["base_pnl"] is not None else 0)
print("\nworst 6 baseline:", [(r['coin'], round(r['base_pnl'],2), r['exit_type']) for r in grp[:6]])
print("best 6 baseline: ", [(r['coin'], round(r['base_pnl'],2), r['exit_type']) for r in grp[-6:]])
top = max(abs(r["base_pnl"]) for r in grp)
tot = sum(r["base_pnl"] for r in grp)
print(f"top-trade share of |net|: {top/max(tot,1e-9):.2f} of net {tot:+.2f}; "
      f"ex-worst {tot-grp[0]['base_pnl']:+.2f}, ex-best {tot-grp[-1]['base_pnl']:+.2f}")
import collections
print("exit mix:", collections.Counter(r["exit_type"] for r in grp))
print("re-time basis:", collections.Counter(r["rt_basis"].split("+")[0] for r in grp))
