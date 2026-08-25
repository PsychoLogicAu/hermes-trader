#!/usr/bin/env python3
"""Sweep the loss-cooldown window (loss_cooldown_min) with the validated
reentry_backtest engine.

Answers: "is 180 min the right loss-cooldown length, or do 30/60/90 min
windows do better?" Replays the SAME entry/exit engine (heuristic entries,
DSL two-phase exits, one open position per coin) under BLOCK policy with
cooldown_bars set from each candidate window, over a longer window of
history than the ledger counterfactual can cover.

15m bars so 30/60/90/180 min resolve to 2/4/6/12 bars (1h bars would
collapse 30 and 60 into the same resolution).

NOT modelled (caveats, same as scripts/reentry_backtest.py):
  - the standard 30-min OPEN-based `cooldown_min` gate (bar-resolution
    artifact; at 15m it is 2 bars — short next to any losing close anyway)
  - the LLM research step (deterministic heuristic stands in)
  - funding, compounding (equity held constant)
  - the momentum-reentry bypass (enabled=false live)

Usage: python3 scripts/cooldown_sweep.py [--days 15] [--coins 20] [--windows 30,60,90,180]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reentry_backtest import simulate, _stats  # noqa: E402  (validated engine)
from backtest import get_config  # noqa: E402
from hermes_trader.agents.config_store import read_agent_config as _live_cfg  # noqa: E402
from hermes_trader.client.hl_client import fetch_hl_candles  # noqa: E402
from hermes_trader.client.universe import get_universe  # noqa: E402

BARS_PER_DAY = {"5m": 288, "15m": 96, "1h": 24, "4h": 6}
WARMUP = 100


def _churn_loss(trades, within_bars):
    """Losses closed within `within_bars` of entry — the revenge-re-entry
    signature (stop out almost immediately). Returns (count, pnl)."""
    n = 0
    p = 0.0
    for t in trades:
        hold = t.get("exit_bar", t["entry_bar"] + 999) - t["entry_bar"]
        if t["pnl_usd"] < 0 and hold <= within_bars:
            n += 1
            p += t["pnl_usd"]
    return n, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=15)
    ap.add_argument("--coins", type=int, default=20)
    ap.add_argument("--interval", default="15m", choices=["5m", "15m", "1h", "4h"])
    ap.add_argument("--windows", default="30,60,90,180",
                    help="comma-separated loss_cooldown_min candidates")
    args = ap.parse_args()
    windows = [int(w) for w in args.windows.split(",") if w.strip()]

    live = _live_cfg()
    live_dsl = live.get("dsl_exit", {}) or {}
    equity_fraction = float(live.get("equity_fraction_per_trade", 0.12))
    leverage_ceiling = int(live.get("leverage", 8))
    max_loss = float(live_dsl.get("max_loss_pct", 0.75))
    protect = float(live_dsl.get("protect_pct", 1.5))
    retrace = float(live_dsl.get("retrace_threshold", 0.30))
    reclaim_pct = float((live.get("momentum_reentry") or {}).get("reclaim_pct", 1.0))
    min_comp = float((live.get("momentum_reentry") or {}).get("min_composite", 30))

    bpd = BARS_PER_DAY[args.interval]
    total_bars = args.days * bpd + WARMUP
    ms_per_bar = 1440 * 60_000 / bpd

    cfg = get_config()
    perps = [m for m in get_universe() if m["type"] == "perp" and not m["coin"].startswith("@")]
    coins = sorted(perps, key=lambda m: m.get("dayNtlVlm", 0), reverse=True)[: args.coins]

    print("=== LOSS-COOLDOWN WINDOW SWEEP (BLOCK policy, validated engine) ===")
    print(f"period {args.days}d  interval {args.interval}  top-{len(coins)}  "
          f"lev<= {leverage_ceiling}x  equity $180 (constant)\n")

    # fetch once, simulate every window per coin
    candle_cache = []
    for m in coins:
        coin, max_lev = m["coin"], int(m.get("maxLeverage", 5))
        try:
            candles = fetch_hl_candles(coin, args.interval, total_bars)
        except Exception as e:
            print(f"  {coin}: fetch fail ({e})")
            continue
        if len(candles) < WARMUP + 50:
            print(f"  {coin}: only {len(candles)} candles, skip")
            continue
        candle_cache.append((coin, max_lev, candles))
        print(f"  {coin}: {len(candles)} bars", flush=True)

    runs = {w: [] for w in windows}
    for coin, max_lev, candles in candle_cache:
        for w in windows:
            cooldown_bars = max(1, round(w / (1440 / bpd)))
            trades = simulate(
                coin, candles, max_lev,
                policy="BLOCK", cfg=cfg, equity=180.0,
                equity_fraction=equity_fraction, lev_ceiling=leverage_ceiling,
                max_loss_pct=max_loss, protect_pct=protect,
                retrace_threshold=retrace, cooldown_bars=cooldown_bars,
                reclaim_pct=reclaim_pct, min_composite=min_comp,
                warmup=WARMUP, exit_mode="fixed")
            runs[w].extend(trades)

    # churn window in bars: 4 bars = 1h at 15m — "stopped out within an hour"
    churn_bars = max(2, round(60 / (1440 / bpd)))

    print(f"\n=== RESULT ({len(candle_cache)} coins x {args.days}d) ===")
    print(f"{'window':>9} | {'cooldown bars':>13} | {'trades':>6} | {'win%':>5} | "
          f"{'PnL':>9} | {'avg/trade':>9} | {'churn-loss*':>11}")
    for w in windows:
        tr = runs[w]
        n, win_pct, pnl = _stats(tr)
        cb = max(1, round(w / (1440 / bpd)))
        cn, cp = _churn_loss(tr, churn_bars)
        avg = pnl / n if n else 0.0
        print(f"{w:>7} min | {cb:>13} | {n:>6} | {win_pct:>4.1f}% | "
              f"${pnl:>+8.2f} | ${avg:>+8.3f} | {cn:>4d} ${cp:>+6.2f}")
    print("\n* churn-loss = losing trades closed within "
          f"{churn_bars} bars ({churn_bars * (1440 / bpd):.0f} min) of entry — the "
          "revenge re-entry signature (stop out almost immediately).")
    print("Caveats: heuristic entries (no LLM); 1 open pos/coin; constant "
          "equity; standard 30-min open-based cooldown not modelled; "
          "momentum-reentry bypass OFF (as live); past != future.")


if __name__ == "__main__":
    main()