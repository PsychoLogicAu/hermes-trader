#!/usr/bin/env python3
"""Band-snapback parameter-space explorer with in-sample/out-of-sample splits.

Purpose: find (if it exists) a robust positive-expectancy configuration of
the band_snapback trigger + production DSL exit stack — i.e. check whether
the indicator can be turned into a "money printer" without selection bias.

Method
------
Fetch N bars per coin (paged, cached to scratch/grid_cache/), then evaluate a
grid of trigger parameters (band_span x min_poke_atr) over a SPLIT timeline.
band_span is the band's SINGLE window — edges, drift/direction, and ATR all
run over it (the old separate `window`/drift-reference lookback was retired),
so there is no independent window axis to sweep:

  [warmup] |---- IN-SAMPLE phase ----| gap |---- OUT-OF-SAMPLE phase ----|
              --span x poke grid--            (same length)

* Signals are classified by entry bar: IS entries exit ONLY within the IS
  phase, OOS entries only within the OOS phase (the DSL sim runs on the
  sliced candle window, so the real DSLTracker clock/timers never see the
  other half — no leakage).
* Exit model: the production DSLTracker/ExitPolicy (.agent-config.json),
  server SL/TP + round-trip fees — identical to scripts/band_snapback_backtest.py.
* --scale-timeouts keeps the wall-clock timers' BAR-COUNT meaning when the
  interval differs from the one they were tuned on (4h: 480min=2bars is
  meaningless; x4.0 keeps the intended ~8-bar stale-flat cutoff).

Read the output
---------------
* IS = the window you would have used to PICK the config (selection window).
* OOS = the unseen window a real edge must still survive.
* A top IS config whose OOS collapses (or is negative) is an overfit spike,
  not an edge. A config sitting on a "plateau" (neighbouring grid cells
  also positive) is far more likely to be robust than an isolated spike.
* dsl_mean = per-signal average net P/L (fees incl) — the money-printer
  metric. hold_mean = fixed-K-bar baseline, gross (no fees).

Usage (project venv):
  .venv/bin/python scripts/band_snapback_grid.py --interval 1h
  .venv/bin/python scripts/band_snapback_grid.py --interval 4h \
      --spans 8,16,32 --scale-timeouts 4.0
  (--primary-window / --windows are deprecated no-ops: band_span IS the
   window, and phase A already sweeps every span in --spans.)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from band_snapback_backtest import (  # noqa: E402
    EXIT_WIN_MS, apply_dsl_exits, build_live_policy, fetch_range,
    plot_event, plot_summary, replay, _load_agent_config, _ts_ms,
)
from hermes_trader.indicators.triggers import _band_ma, _project_band_edge  # noqa: E402
from hermes_trader.models.types import Candle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "scratch", "grid_cache")
OUT_DIR = os.path.join(ROOT, "scratch", "grid_sweeps")
CHART_DIR = os.path.join(ROOT, "scratch", "band_snapback_charts")

# Default grids
SPANS = [6, 8, 12, 16, 24, 32]
POKES = [0.30, 0.50, 0.75]


# ── data ─────────────────────────────────────────────────────────────────

def load_candles(coin: str, interval: str, bars: int, refresh: bool = False):
    tag = hashlib.md5(f"{coin}|{interval}|{bars}".encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"candles_{tag}.json")
    if not refresh and os.path.exists(path):
        try:
            with open(path) as f:
                obj = json.load(f)
            if (obj["coin"], obj["interval"], obj["bars"]) == (coin, interval, bars):
                cds = [Candle(t=c["t"], o=c["o"], h=c["h"], l=c["l"],
                              c=c["c"], v=c["v"]) for c in obj["candles"]]
                return cds, True
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    cds = fetch_range(coin, interval, bars)
    if cds and len(cds) >= bars * 0.9:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"coin": coin, "interval": interval, "bars": bars,
                       "fetched_utc": datetime.now(tz=timezone.utc).isoformat(),
                       "candles": [c.__dict__ for c in cds]}, f)
    return cds, False


# ── metrics ──────────────────────────────────────────────────────────────

def _stats(pnl_list: list[float]) -> dict:
    n = len(pnl_list)
    if n == 0:
        return {"n": 0, "wins": 0, "net": 0.0, "mean": float("nan"),
                "winrate": float("nan"), "worst": float("nan"),
                "best": float("nan")}
    wins = sum(1 for p in pnl_list if p > 0)
    return {"n": n, "wins": wins, "net": sum(pnl_list), "mean": sum(pnl_list) / n,
            "winrate": wins / n, "worst": min(pnl_list), "best": max(pnl_list)}


def evaluate_cell(coin: str, candles: list[Candle], span: int,
                  poke: float, hold: int, ma_type: str, drift: float,
                  is_start: int, is_end: int, oos_start: int, oos_end: int,
                  cfg: dict, time_scale: float, leverage: int,
                  sl_mult: float, tp_frac: float, hard_bars: int) -> dict:
    """Replay the trigger once over the FULL timeline (band/ATR history at
    every bar is identical to a production run, no slicing), classify
    signals into IS/OOS by entry bar, and sim the DSL exit on the FULL
    candle list with the walk censored at the phase end (stop_at=b).

    Censoring rule: an entry is only eligible for a phase if its MAXIMUM
    possible hold (the DSL hard timeout, hard_bars bars) fits inside the
    phase (entry_i + hard_bars <= phase_end). Entries that cannot fit are
    dropped from that phase (counted under 'censored') so no position's P/L
    can straddle the inter-phase gap. A position still open at the boundary
    is marked at the boundary bar's close (reason 'data_end').

    Charting: events keep full-timeline indices and the candle list passed
    to plot_* is the full list, so no index shifting and band geometry
    stays intact.
    """
    all_ev = replay(candles, drift, poke, hold, ma_type, span)
    ev_is = [e for e in all_ev if is_start <= e["entry_i"] < is_end
             and e["entry_i"] + hard_bars <= is_end]
    ev_oos = [e for e in all_ev if oos_start <= e["entry_i"] < oos_end
              and e["entry_i"] + hard_bars <= oos_end]

    out = {"params": {"span": span, "poke": poke},
           "censored": len(all_ev) - len(ev_is) - len(ev_oos)}

    def phase(tag: str, evs: list[dict], a: int, b: int):
        if not evs:
            return {"n": 0, "dsl": {"n": 0}, "hold": {"n": 0},
                    "reasons": {}, "events": [], "candles": None}
        # Full-timeline candles + walk censored at the phase end: band/ATR
        # context and chart geometry stay identical to the production
        # replay; no position can bleed P/L across the phase gap.
        apply_dsl_exits(coin, candles, evs, cfg, time_scale=time_scale,
                        stop_at=b)
        dsl = _stats([e["pnl_pct"] for e in evs])
        hold_st = _stats([e["hold_pnl_pct"] for e in evs])
        reasons: dict[str, int] = {}
        for e in evs:
            k = e["reason"].split(" (")[0]
            reasons[k] = reasons.get(k, 0) + 1
        return {"n": len(evs), "dsl": dsl, "hold": hold_st,
                "reasons": reasons, "events": evs, "candles": candles}

    # apply_dsl_exits mutates evs (adds hold_pnl_pct etc). IS first, then OOS.
    res_is = phase("is", ev_is, is_start, is_end)
    res_oos = phase("oos", ev_oos, oos_start, oos_end)
    return {**out, "is": res_is, "oos": res_oos}


# ── driver ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default="BTC,ETH,SOL")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--bars", type=int, default=1000)
    ap.add_argument("--spans", default=",".join(str(s) for s in SPANS))
    ap.add_argument("--pokes", default=",".join(str(p) for p in POKES))
    ap.add_argument("--drift", type=float, default=1.5)
    ap.add_argument("--hold", type=int, default=24,
                    help="fixed-bar baseline hold (both intervals; the DSL "
                         "exit is what actually matters)")
    ap.add_argument("--ma-type", dest="ma_type", default="ema",
                    choices=["ema", "sma"])
    ap.add_argument("--scale-timeouts", dest="scale_timeouts", type=float,
                    default=1.0)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the candle cache and refetch")
    ap.add_argument("--plot-top", type=int, default=3,
                    help="phase: 'oos' charts for the top-N OOS configs (0 = off)")
    # Deprecated no-ops (the separate drift-reference window was retired —
    # band_span is the band's single window). Kept so old command lines
    # from notes/skills don't crash.
    ap.add_argument("--primary-window", dest="primary_window", type=int,
                    default=None, help=argparse.SUPPRESS)
    ap.add_argument("--windows", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--top", type=int, default=6, help=argparse.SUPPRESS)
    args = ap.parse_args()

    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    spans = sorted({s for s in (int(x) for x in args.spans.split(","))})
    pokes = sorted({float(x) for x in args.pokes.split(",")})
    if args.primary_window is not None or args.windows:
        print("NOTE: --primary-window/--windows are deprecated no-ops — "
              "band_span is the band's single window and the grid sweeps "
              "every span in --spans.", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = _load_agent_config()
    pol = build_live_policy(cfg, time_scale=args.scale_timeouts)
    leverage = int(cfg.get("leverage", 1) or 1)
    sl_mult = float(cfg.get("sl_atr_mult", 1.5))
    tp_frac = float(cfg.get("tp_scale_fraction", 0.5))

    # Split geometry is computed AFTER the fetch (needs the actual candle
    # counts). Placeholder here, filled in below.
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
    # Disambiguate concurrent runs of the same interval (different coin pools
    # would otherwise collide on the minute-resolution stamp and clobber each
    # other's CSV). Short hash of the pool keeps the filename readable.
    import hashlib as _hl
    pooltag = _hl.md5(",".join(coins).encode()).hexdigest()[:6]
    stamp = f"{stamp}_{pooltag}"
    t0 = time.time()

    bar_min = EXIT_WIN_MS[args.interval] / 60000.0
    stale_bars = pol.stale_flat_timeout_minutes / bar_min
    hard_bars = int(math.ceil(pol.hard_timeout_minutes / bar_min))
    print(f"interval={args.interval} coins={','.join(coins)} bars={args.bars} "
          f"hold={args.hold} drift={args.drift} ma={args.ma_type}", flush=True)
    print(f"dsl: leverage={leverage}x sl={sl_mult}xATR(4h) tp={tp_frac:.0%}@1.0x "
          f"fees5bps stale={pol.stale_flat_timeout_minutes:g}min "
          f"hard={pol.hard_timeout_minutes:g}min (x{args.scale_timeouts:g} -> "
          f"stale={stale_bars:.1f} bars hard={hard_bars} bars)", flush=True)

    data = {}
    for coin in coins:
        print(f"[{coin}] fetching {args.bars} x {args.interval} candles ...", flush=True)
        cds, cached = load_candles(coin, args.interval, args.bars, args.refresh)
        data[coin] = cds
        d0 = datetime.fromtimestamp(_ts_ms(cds[0].t), tz=timezone.utc)
        d1 = datetime.fromtimestamp(_ts_ms(cds[-1].t), tz=timezone.utc)
        print(f"[{coin}] {len(cds)} candles ({'cache' if cached else 'fetched'}) "
              f"{d0:%Y-%m-%d} -> {d1:%Y-%m-%d %H:%M} UTC", flush=True)

    # Split geometry — sized off the SHORTEST coin so the phases are identical
    # across the pool (per-signal pooling only works on equal timelines).
    min_len = min(len(cds) for cds in data.values())
    max_lookback = 2 * max(spans)
    is_start = max(100, max_lookback)
    phase_len = (min_len - is_start - 40 - 20) // 2   # 20 bars of OOS tail slack
    if phase_len < 50:
        raise SystemExit(
            f"not enough data for an IS/OOS split on this coin pool: "
            f"min {min_len} bars, need >= {is_start + 40 + 20 + 100}; "
            f"lower --bars or drop the short-history coins")
    is_end = is_start + phase_len
    oos_start = is_end + 40
    oos_end = oos_start + phase_len
    if oos_end + 20 > min_len:
        oos_end = min_len - 5
    for coin, cds in data.items():
        if len(cds) < oos_end + 20:
            data[coin] = cds[:oos_end + 20]
    print(f"split: IS bars {is_start}-{is_end-1} ({phase_len} bars), "
          f"OOS bars {oos_start}-{oos_end-1} ({phase_len} bars), "
          f"hard-timeout censor={hard_bars} bars before each phase end", flush=True)

    # ── Phase A: full grid (span x poke — band_span IS the window) ─────────
    cells = [(s, p) for s in spans for p in pokes]
    results = {}   # (s,p) -> {coin: evaluate_cell(...)}
    for (s, p) in cells:
        tag = f"s{s}_p{p:.2f}"
        print(f"\n== cell {tag} (span{s} poke{p:.2f}) ==", flush=True)
        results[(s, p)] = {
            coin: evaluate_cell(coin, data[coin], s, p, args.hold,
                                args.ma_type, args.drift,
                                is_start, is_end, oos_start, oos_end,
                                cfg, args.scale_timeouts, leverage,
                                sl_mult, tp_frac, hard_bars)
            for coin in coins
        }

    # pooled IS/OOS stats per cell
    def attach_per100(cs: dict) -> None:
        # "money rate": pooled net P/L per 100 bars of phase (throughput x
        # edge). This is the figure a money printer needs to beat.
        for ph in ("is", "oos"):
            st = cs[ph]
            cs[ph + "_per100"] = (st["net"] / phase_len * 100) if st["n"] else 0.0

    cell_stats = {}
    for key, per_coin in results.items():
        is_pnl, oos_pnl, is_hold, oos_hold = [], [], [], []
        for coin in coins:
            r = per_coin[coin]
            is_pnl += [e["pnl_pct"] for e in r["is"].get("events", [])]
            oos_pnl += [e["pnl_pct"] for e in r["oos"].get("events", [])]
            is_hold += [e["hold_pnl_pct"] for e in r["is"].get("events", [])]
            oos_hold += [e["hold_pnl_pct"] for e in r["oos"].get("events", [])]
        cell_stats[key] = {
            "is": _stats(is_pnl), "oos": _stats(oos_pnl),
            "is_hold": _stats(is_hold), "oos_hold": _stats(oos_hold),
        }
        attach_per100(cell_stats[key])

    # plateau: among the 4 nearest grid neighbours (in span/poke space), how
    # many also have positive IS per-signal mean?
    def plateau(key):
        s, p = key
        idx_s = spans.index(s) if s in spans else None
        idx_p = pokes.index(p) if p in pokes else None
        if idx_s is None or idx_p is None:
            return float("nan")
        nb = []
        if idx_s > 0:
            nb.append((spans[idx_s - 1], p))
        if idx_s < len(spans) - 1:
            nb.append((spans[idx_s + 1], p))
        if idx_p > 0:
            nb.append((s, pokes[idx_p - 1]))
        if idx_p < len(pokes) - 1:
            nb.append((s, pokes[idx_p + 1]))
        pos = sum(1 for k in nb if k in cell_stats
                  and cell_stats[k]["is"]["n"] >= 5
                  and cell_stats[k]["is"]["mean"] > 0)
        return f"{pos}/{len(nb)}"

    ranked = sorted(cell_stats,
                    key=lambda k: (cell_stats[k]["is"]["n"] > 0,
                                   cell_stats[k]["is"]["mean"] if cell_stats[k]["is"]["n"] else -99),
                    reverse=True)

    def fmt(s: dict) -> str:
        if s["n"] == 0:
            return "   --   (no signals)"
        return (f"  n={s['n']:3d} wr={s['winrate']:4.0%} "
                f"mean={s['mean']:+6.3f}% net={s['net']:+8.2f}% "
                f"worst={s['worst']:+7.2f}%")

    print("\n" + "=" * 118)
    print(f"PHASE A — grid over spans {spans} x pokes {pokes}, IS vs OOS "
          f"(per-signal averages, DSL exit, fees incl; per100 = net P/L per 100 bars)")
    print("=" * 118)
    print(f"{'cell':<14} | {'IN-SAMPLE':<44} | {'OOS':<44} | nb+ | per100 IS / OOS")
    for k in ranked:
        s, p = k
        cs = cell_stats[k]
        tag = f"s{s}_p{p:.2f}"
        print(f"{tag:<14} | {fmt(cs['is'])} | {fmt(cs['oos'])} | {plateau(k)}"
              f" | {cs['is_per100']:+7.3f} / {cs['oos_per100']:+7.3f}")
    for k in ranked[:6]:
        s, p = k
        cs = cell_stats[k]
        print(f"  s{s}_p{p:.2f}: IS {cs['is_hold']['mean']:+.3f}% "
              f"OOS {cs['oos_hold']['mean']:+.3f}% (hold-{args.hold} gross, "
              f"n={cs['is_hold']['n']}/{cs['oos_hold']['n']})")

    # CSV dump (phase A)
    csv_a = os.path.join(OUT_DIR, f"grid_{args.interval}_{stamp}_A.csv")
    with open(csv_a, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cell", "span", "poke", "phase", "coin", "n",
                     "wins", "net_pct", "mean_pct", "winrate", "worst", "best",
                     "reasons"])
        for k in ranked:
            s, p = k
            for coin in coins:
                for ph in ("is", "oos"):
                    r = results[(s, p)][coin][ph]
                    d = r.get("dsl") or {"n": 0}
                    if d.get("n", 0):
                        wr.writerow([f"s{s}_p{p:.2f}", s, f"{p:.2f}",
                                     ph, coin, d["n"], d["wins"],
                                     f"{d['net']:.3f}", f"{d['mean']:.3f}",
                                     f"{d['winrate']:.3f}", f"{d['worst']:.3f}",
                                     f"{d['best']:.3f}",
                                     json.dumps(r.get("reasons", {}))])
                    else:
                        wr.writerow([f"s{s}_p{p:.2f}", s, f"{p:.2f}",
                                     ph, coin, 0, 0, "", "", "", "", "", ""])
    print(f"\nCSV: {csv_a}")

    # ── Charts for the best OOS configs ─────────────────────────────────
    if args.plot_top > 0:
        # rank all cells by OOS mean (min 5 OOS signals)
        oos_ranked = [k for k in cell_stats if cell_stats[k]["oos"]["n"] >= 5]
        oos_ranked.sort(key=lambda k: cell_stats[k]["oos"]["mean"], reverse=True)
        for k in oos_ranked[:args.plot_top]:
            s, p = k
            tag = f"s{s}_p{p:.2f}"
            for coin in coins:
                r = results[k][coin]
                for ph in ("is", "oos"):
                    evs = r[ph].get("events", [])
                    if not evs:
                        continue
                    sl = r[ph]["candles"]
                    cs = cell_stats[k]
                    title = (f"{coin} {args.interval} {tag} [{ph.upper()}] "
                             f"IS {cs['is']['mean']:+.3f}%/sig OOS "
                             f"{cs['oos']['mean']:+.3f}%/sig "
                             f"({cs['oos']['winrate']:.0%} wr)")
                    plot_summary(coin, evs,
                                 os.path.join(CHART_DIR,
                                              f"GRID_{coin}_{stamp}_{args.interval}_{tag}_{ph}_summary.png"),
                                 args.hold, title)
                    for n, ev in enumerate(evs[:6]):
                        d = datetime.fromtimestamp(_ts_ms(ev["t"]), tz=timezone.utc)
                        plot_event(coin, sl, ev, args.hold,
                                   os.path.join(CHART_DIR,
                                                f"GRID_{coin}_{stamp}_{args.interval}_{tag}_{ph}_{n}_{d:%m%d_%H%M}_{ev['side']}_pnl{ev['pnl_pct']:+.2f}.png"),
                                   args.interval)
        print(f"\ncharts for top-{args.plot_top} OOS cells in {CHART_DIR}")

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()