#!/usr/bin/env python3
"""C.13 pass 2: price both arms.

TIREX arm (shadow): would-block scans that EXECUTED anyway -> join real ledger
CLOSE (±15s, coin+side) -> actual P/L of the entries the gate would have
filtered. Net-negative executed cohort => tirex filter saves money.
Also: tirex-ONLY trips (chronos passed) that executed -> the incremental
cohort a swap would newly block.

CHRONOS arm (live): live blocks -> deduped candidates (30-min same-coin),
sole-blocker subset (chronos the only failing gate) -> occupancy-aware
replay (real ledger intervals + sim book, 5 slots) with live DSL mirror,
baseline = entry at block scan (world without the gate). Re-timing caveat:
next same-coin scan where chronos gate passes again.
"""
import ast, glob, json, re, time, urllib.request
import datetime as dt
from bisect import bisect_left
from collections import Counter

D = "/home/oknight/src/hermes-trader/scripts/analysis/c13_tirex_mirror"
LOGS = sorted(glob.glob("/home/oknight/src/hermes-trader/trader-logs/trader.log*"))
LEDGER = "/home/oknight/src/hermes-trader/trader-logs/trades.jsonl"
CANDLE_CACHE = D + "/_candles.json"
ARM = dt.datetime(2026, 9, 17, 5, 17)
PFX = re.compile(r"^\[(?P<coin>.+?)\] (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ ")
TRADE_RE = re.compile(r"INFO:trading_loop:Trade result: (\{.*)$")

LEV = 5
STOP_PCT = min(5.0, 30.0 / LEV)
PROTECT, RETRACE = 1.0, 0.25
TIERS = [(8.0, 0.35), (15.0, 0.40)]
BE_TRIG, BE_LOCK = 2.5, 0.05
STALE_MIN, HARD_MIN = 240, 1800
FEE = 2.5e-4
NOTIONAL = 33.33
BAR = 300_000

def ms_of(naive):
    return int(naive.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

# ── scan stream: per coin, (ts_ms, chronos_failed) ──────────────────────────
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
        failed = any(r.startswith("chronos_tail_trigger") for r in bb)
        scans.setdefault(pm["coin"], []).append((ms_of(ts), failed))
for v in scans.values():
    v.sort()

# ── ledger closes for the tirex executed cohort ─────────────────────────────
rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
closes = {}  # (coin, side) -> list of (open_ms, close_ms, spot_pct, exit_type)
open_map = {}
for r in rows:
    if r.get("event") == "OPEN":
        open_map[(r["coin"], r.get("side"))] = ms_of(dt.datetime.fromisoformat(r["ts_iso"].replace("Z", "+00:00")))
    elif r.get("event") == "CLOSE":
        k = (r["coin"], r.get("side"))
        if k in open_map:
            closes.setdefault(k, []).append((open_map.pop(k), ms_of(dt.datetime.fromisoformat(r["ts_iso"].replace("Z", "+00:00"))), r.get("spot_pct"), r.get("exit_type")))
real_iv = [(a, b, k[0]) for k, v in closes.items() for (a, b, *_r) in v]
for k, t in open_map.items():
    real_iv.append((t, 2**62, k[0]))

census = json.load(open(D + "/_c13_census.json"))
tirex_would = census["tirex_would"]
chronos_blocks = census["chronos_blocks"]

def pnl_from_spot(spot_pct, notional=NOTIONAL):
    return notional * spot_pct / 100 - 2 * FEE * notional

# ── TIREX arm: executed would-blocks -> real closes ─────────────────────────
tw_exec = [s for s in tirex_would if s["executed"]]
# dedupe 30-min same coin+side
tw_exec.sort(key=lambda s: s["ts"])
tw_deduped = []
tw_last = {}
for s in tw_exec:
    t = ms_of(dt.datetime.fromisoformat(s["ts"]))
    k = (s["coin"], s["side"])
    if k in tw_last and t - tw_last[k] < 30 * 60_000:
        continue
    tw_last[k] = t
    tw_deduped.append(dict(s, t=t))
tw_ded = tw_deduped
print(f"tirex executed would-blocks: {len(tw_exec)} scans -> deduped {len(tw_ded)}")
tw_priced = []
for s in tw_ded:
    k = (s["coin"], s["side"])
    hit = None
    for (a, b, spot, et) in closes.get(k, []):
        if abs(a - s["t"]) <= 15_000:
            hit = (a, b, spot, et)
            break
    if hit is None:  # widen: any open within 90s
        for (a, b, spot, et) in closes.get(k, []):
            if abs(a - s["t"]) <= 90_000:
                hit = (a, b, spot, et)
                break
    if hit:
        a, b, spot, et = hit
        tw_priced.append(dict(coin=s["coin"], side=s["side"], ts=s["ts"],
                              spot=spot, exit_type=et, pnl=pnl_from_spot(spot),
                              chronos_trip=s["chronos_trip"], tail=s["tail"]))
    else:
        tw_priced.append(dict(coin=s["coin"], side=s["side"], ts=s["ts"],
                              spot=None, exit_type="no_close_join",
                              pnl=None, chronos_trip=s["chronos_trip"], tail=s["tail"]))
print("\nTIREX executed-would-block cohort (what the shadow gate would have filtered):")
for r in tw_priced:
    print(f"  {r['ts']} {r['coin']:9s} {r['side']:5s} tail -{r['tail']} chronos_trip={r['chronos_trip']} spot {r['spot']} {r['exit_type']} pnl {r['pnl'] if r['pnl'] is None else round(r['pnl'],2)}")
ok = [r for r in tw_priced if r["pnl"] is not None]
net = sum(r["pnl"] for r in ok)
inc = [r for r in ok if not r["chronos_trip"]]
net_inc = sum(r["pnl"] for r in inc)
print(f"joined n={len(ok)}/{len(tw_priced)}  net {net:+.2f}  wins {sum(1 for r in ok if r['pnl']>0)}/{len(ok)}")
print(f"  tirex-ONLY incremental executed n={len(inc)} net {net_inc:+.2f}")

# ── CHRONOS arm: live blocks -> deduped, occupancy-aware replay ─────────────
chronos_blocks.sort(key=lambda s: s["ts"])
ded, last = [], {}
for s in chronos_blocks:
    t = ms_of(dt.datetime.fromisoformat(s["ts"]))
    if s["coin"] in last and t - last[s["coin"]] < 30 * 60_000:
        continue
    last[s["coin"]] = t
    ded.append(dict(coin=s["coin"], t=t, side=s["side"], tirex_trip=s["tirex_trip"],
                    other_blockers=s["other_blockers"], tail=s["tail"]))
sole = [c for c in ded if not c["other_blockers"]]
print(f"\nchronos live-block scans {len(chronos_blocks)} -> deduped {len(ded)}; sole-blocker {len(sole)}")

try:
    CACHE = json.load(open(CANDLE_CACHE))
except Exception:
    CACHE = {}

def fetch(coin, t0):
    got = CACHE.get(coin, [])
    if got and got[0][0] <= t0 - BAR and got[-1][0] >= min(time.time() * 1000 * 1000, t0 + HARD_MIN * 60_000) - BAR:
        return got
    req = urllib.request.Request("https://api.hyperliquid.xyz/info",
        data=json.dumps({"type": "candleSnapshot", "req": {
            "coin": coin, "interval": "5m",
            "startTime": t0 - 3600_000, "endTime": int(time.time() * 1000)}}).encode(),
        headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            arr = json.loads(urllib.request.urlopen(req, timeout=30).read())
            break
        except Exception:
            time.sleep(3 * (attempt + 1))
    else:
        arr = []
    bars = sorted((int(k["t"]), float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]))
                  for k in arr if k.get("c") is not None)
    CACHE[coin] = bars
    return bars

def sim_exit(coin, entry_t, entry_px, side):
    bars = fetch(coin, entry_t)
    i = bisect_left([b[0] for b in bars], entry_t)
    if i >= len(bars):
        return None, None, "no_candles"
    sgn = 1 if side == "long" else -1
    stop = entry_px * (1 - sgn * STOP_PCT / 100)
    floor, peak, armed = stop, entry_px, False
    for j in range(i, min(i + HARD_MIN // 5 + 1, len(bars))):
        t, o, h, l, c = bars[j]
        hi, lo = (h, l) if j > i else (max(o, c), min(o, c))
        peak = max(peak, hi) if sgn == 1 else min(peak, lo)
        up = sgn * (peak - entry_px) / entry_px * 100
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
        el = (t - entry_t) / 60_000
        if el >= STALE_MIN and up < PROTECT:
            return c, t, "stale_flat"
        if el >= HARD_MIN:
            return c, t, "hard_timeout"
    last_bar = bars[min(i + HARD_MIN // 5, len(bars) - 1)]
    return last_bar[4], last_bar[0], "window_end"

def pnl_of(entry_t, entry_px, side, exit_px):
    sgn = 1 if side == "long" else -1
    spot = sgn * (exit_px / entry_px - 1) * 100
    return NOTIONAL * spot / 100 - 2 * FEE * NOTIONAL

def next_clear(coin, t):
    for s_ms, failed in scans.get(coin, []):
        if s_ms >= t and not failed:
            return s_ms
    return None

sim_iv = []
def occupied(t, coin):
    n = sum(1 for a, b, _ in real_iv if a <= t < b) + sum(1 for a, b, _ in sim_iv if a <= t < b)
    if n >= 5:
        return "slots"
    if any((a <= t < b) and c == coin for a, b, c in real_iv + sim_iv):
        return "coin"
    return None

def replay(cohort, label):
    global sim_iv
    sim_iv = []
    res = []
    for cand in cohort:
        coin, t, side = cand["coin"], cand["t"], cand["side"]
        skip = occupied(t, coin)
        bars = fetch(coin, t)
        i = bisect_left([b[0] for b in bars], t)
        if i >= len(bars):
            res.append(dict(coin=coin, ts=t, side=side, status="no_candles"))
            continue
        entry_px = bars[i][1]
        exit_px, exit_t, etype = sim_exit(coin, t, entry_px, side)
        base = pnl_of(t, entry_px, side, exit_px) if exit_px else None
        tc = next_clear(coin, t + 1)
        if tc is None or (exit_t and tc > exit_t):
            rt_entry_t, rt_basis = (exit_t or t), "blocked-through-close"
        else:
            rt_entry_t, rt_basis = tc, f"cleared+{(tc-t)//60000}m"
        j = min(bisect_left([b[0] for b in bars], rt_entry_t), len(bars) - 1)
        rt_px = bars[j][1]
        ex2, et2, ty2 = sim_exit(coin, rt_entry_t, rt_px, side)
        rtp = pnl_of(rt_entry_t, rt_px, side, ex2) if ex2 else None
        res.append(dict(coin=coin, ts=t, side=side, entry_px=entry_px,
                        exit_type=etype, base_pnl=base, rt_basis=rt_basis,
                        rt_pnl=rtp, skip=skip, tirex_trip=cand.get("tirex_trip")))
        if base is not None:
            sim_iv.append((t, exit_t, coin))
    json.dump(res, open(D + f"/_c13_{label}.json", "w"), indent=1)
    priced = [r for r in res if r.get("base_pnl") is not None]
    gate_only = [r for r in priced if not r.get("skip")]
    print(f"\n== CHRONOS {label}: priced n={len(priced)} executable n={len(gate_only)}")
    for lbl, grp in (("GATE-ONLY", priced), ("OCCUPANCY", gate_only)):
        nb = sum(r["base_pnl"] for r in grp)
        nr = sum(r["rt_pnl"] for r in grp if r.get("rt_pnl") is not None)
        w = sum(1 for r in grp if r["base_pnl"] > 0)
        ml = sum(1 for r in grp if r["exit_type"] == "max_loss")
        print(f"{lbl}: n={len(grp)} baseline(no-gate) net {nb:+.2f} wins {w}/{len(grp)} max_loss {ml}")
        print(f"         re-timed net {nr:+.2f} -> gate SAVES (filter) {-nb:+.2f} / (re-timed) {nb-nr:+.2f}")
    return priced

replay(ded, "chronos_all")
replay(sole, "chronos_sole")

json.dump(CACHE, open(CANDLE_CACHE, "w"))
print("\nexit mix (chronos_all):", Counter(r["exit_type"] for r in json.load(open(D + "/_c13_chronos_all.json")) if r.get("base_pnl") is not None))
