#!/usr/bin/env python3
"""Manual backtest for the band_snapback trigger, with annotated charts.

Replays the trigger bar-by-bar over a window of 5m candles fetched from the
Hyperliquid public API (same endpoint the live scanner uses), measuring what
happened AFTER each fired poke+snapback — entry at the poke bar's close, exit
K bars later, vs a hold baseline. No lookahead: the signal only sees bars up
to the poke bar.

Usage (project venv):
  .venv/bin/python scripts/band_snapback_backtest.py [--coins BTC,ETH,SOL]
      [--window 48] [--drift 1.5] [--min-poke 0.5] [--hold 12] [--bars 400]
      [--chart-dir scratch/band_snapback_charts]

Charts: one PNG per fired signal (annotated with the fitted band, the poke
bar, entry/exit and P/L) plus one summary chart. Output PNGs land in
--chart-dir (scratch/ by default, git-ignored).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_trader.client.hl_client import fetch_hl_candles  # noqa: E402
from hermes_trader.indicators.triggers import band_snapback, _band_ma, _project_band_edge  # noqa: E402
from hermes_trader.models.types import Candle  # noqa: E402

UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"
BAND_COLOR = "#42a5f5"
ENTRY_COLOR = "#ffee58"
EXIT_WIN_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000,
               "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}


def fetch_range(coin: str, interval: str, total_bars: int,
                retries: int = 5) -> list[Candle]:
    """Fetch `total_bars` candles (paged: HL candleSnapshot max ~500/req).

    Retries with backoff on 429 / empty — the live bot shares this host's
    IP and its scan cadence exhausts the HL weight budget, so we bypass the
    shared HL_LIMITER and manage our own tiny budget here.
    """
    import requests
    sess = requests.Session()
    out: list[Candle] = []
    remaining = total_bars
    end_ms = int(time.time() * 1000)
    page = 500
    while remaining > 0:
        n = min(page, remaining)
        start_ms = end_ms - EXIT_WIN_MS.get(interval, 300_000) * n
        payload = {
            "type": "candleSnapshot",
            "req": {"coin": coin, "interval": interval,
                    "startTime": start_ms, "endTime": end_ms},
        }
        raw = None
        for attempt in range(retries):
            try:
                resp = sess.post("https://api.hyperliquid.xyz/info",
                                 json=payload, timeout=10)
                resp.raise_for_status()
                raw = resp.json()
            except Exception as e:
                print(f"[{coin}] request error: {e}", flush=True)
                raw = None
            if isinstance(raw, list) and raw:
                break
            wait = 5.0 * (attempt + 1)
            print(f"[{coin}] 429/empty on page, retry {attempt + 1}/{retries} "
                  f"in {wait:.0f}s ...", flush=True)
            time.sleep(wait)
        if not isinstance(raw, list) or not raw:
            break
        page_candles = [
            Candle(t=c["t"], o=float(c["o"]), h=float(c["h"]),
                   l=float(c["l"]), c=float(c["c"]), v=float(c.get("v", "0")))
            for c in raw
        ]
        # keep the NEWEST n bars and page backwards from just before the oldest
        page_candles = page_candles[-n:]
        out = page_candles + out
        end_ms = page_candles[0].t - 1
        remaining -= len(page_candles)
        if len(page_candles) == 0:
            break
        time.sleep(1.5)  # gentle pace: we're sharing an IP with the live bot
    return out[-total_bars:]


# Hyperliquid perp taker fee model (same as scripts/backtest.py): 2.5 bps per
# side → 5 bps round trip, applied to the full notional.
ROUND_TRIP_FEE_BPS = 5.0
# executor.py hardcodes TP_ATR_MULT = 1.0 (not config-driven).
TP_ATR_MULT = 1.0


# ── DSL exit simulation ─────────────────────────────────────────────────
# Replays the PRODUCTION exit stack (hermes_trader/agents/dsl_exit.py + the
# server-side backup SL / TP scale-out orders from executor.py) bar-by-bar,
# instead of the dumb fixed-K-bar exit.
#
# Faithfulness notes:
#   * Runs the real DSLTracker/ExitPolicy — same code the live bot runs.
#   * Policy is built from .agent-config.json exactly like executor.py:1074
#     (including the phase2_tiers override). trailing_tp is deliberately NOT
#     wired in — the executor's policy builder never sets it, so it is dead
#     in live too.
#   * entry ATR mirrors get_hl_atr("4h", 14): Wilder ATR on 4h candles,
#     including the PARTIAL 4h candle the entry bar sits in (what live sees).
#   * Intra-bar: SL (1.5x ATR4h, full size, server-side = instant) is checked
#     BEFORE TP (50% scale-out at 1.0x ATR4h) — both touched in one bar →
#     the SL fills first (conservative). The DSL floor (60s live cadence) is
#     checked at the bar's adverse extreme and fills at the floor price.
#   * Isolation: trackers are constructed directly (global registry untouched),
#     _save_state is no-opped, and time.time is patched to the replay clock
#     only for the duration of each simulation (restored in `finally`).


def _load_agent_config() -> dict:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(root, ".agent-config.json")) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[dsl] WARNING: could not read .agent-config.json ({e}); "
              f"falling back to ExitPolicy defaults", flush=True)
        return {}


def build_live_policy(cfg: dict, time_scale: float = 1.0):
    """Build the ExitPolicy exactly like executor.py:1074-1091 does.

    time_scale: backtest-only multiplier applied to the wall-clock timeout
    fields (stale_flat / hard_timeout). Live trading is interval-agnostic:
    a stale_flat of 480 min is 480 min in real time no matter the candle
    size. But the live values were *tuned on 1h bars* (480 min = 8 bars,
    1800 min = 30 bars), so replaying them on a 4h chart means the DSL
    cuts every position after ~2 bars — before a 4h snapback can develop.
    Pass time_scale = interval_minutes/60 to keep the intended *bar count*
    constant across intervals (4h -> 4.0). 1h and below need no scaling.
    """
    from hermes_trader.agents.dsl_exit import ExitPolicy, RetraceTier
    dsl_cfg = cfg.get("dsl_exit") or {}
    tiers_raw = dsl_cfg.get("phase2_tiers")
    tiers = [RetraceTier(**t) for t in tiers_raw] if tiers_raw else None
    atr_cfg = dsl_cfg.get("atr_stop", {}) or {}
    noise_cfg = dsl_cfg.get("noise_band", {}) or {}
    return ExitPolicy(
        max_loss_pct=dsl_cfg.get("max_loss_pct", 2.5),
        max_loss_roe_pct=dsl_cfg.get("max_loss_roe_pct", 50.0),
        protect_pct=dsl_cfg.get("protect_pct", 1.5),
        retrace_threshold=dsl_cfg.get("retrace_threshold", 0.30),
        hard_timeout_minutes=dsl_cfg.get("hard_timeout_minutes", 180.0)
                            * time_scale,
        breakeven_trigger_pct=dsl_cfg.get("breakeven_trigger_pct", 0.0),
        breakeven_lock_pct=dsl_cfg.get("breakeven_lock_pct", 0.0),
        atr_stop_enabled=bool(atr_cfg.get("enabled", False)),
        atr_stop_mult=float(atr_cfg.get("atr_mult", 1.5)),
        atr_stop_floor_pct=float(atr_cfg.get("floor_pct", 1.0)),
        atr_stop_ceiling_pct=float(atr_cfg.get("ceiling_pct", 4.0)),
        stale_flat_timeout_minutes=float(
            dsl_cfg.get("stale_flat_timeout_minutes", 0.0) or 0.0)
            * time_scale,
        consecutive_breaches_required=int(
            dsl_cfg.get("consecutive_breaches_required", 1) or 1),
        noise_band_enabled=bool(noise_cfg.get("enabled", False)),
        noise_band_atr_mult=float(noise_cfg.get("atr_mult", 1.0)),
        phase2_tiers=tiers if tiers else ExitPolicy().phase2_tiers,
    )


def atr4h_at_entry(candles: list[Candle], entry_i: int,
                   period: int = 14) -> float:
    """Wilder ATR(14) on 4h candles as of the entry bar — the same series
    get_hl_atr("4h", 14) would return live: the 4h bucket the entry bar sits
    in is PARTIAL (o/h/l/c through the entry bar), which is what the executor
    sees at order time. No lookahead: only candles <= entry_i."""
    BUCKET = 4 * 3_600_000
    buckets: dict[int, list[float]] = {}
    order: list[int] = []
    for c in candles[: entry_i + 1]:
        b = c.t // BUCKET * BUCKET
        if b not in buckets:
            buckets[b] = [c.o, c.h, c.l, c.c]
            order.append(b)
        else:
            s = buckets[b]
            s[1] = max(s[1], c.h)
            s[2] = min(s[2], c.l)
            s[3] = c.c
    series = [Candle(t=b, o=buckets[b][0], h=buckets[b][1],
                     l=buckets[b][2], c=buckets[b][3], v=0.0)
              for b in order]
    if len(series) < period + 1:
        return 0.0
    # Identical formula to get_hl_atr: SMA seed over first `period` TRs,
    # then Wilder smoothing.
    tr = []
    for i in range(1, len(series)):
        cur, pc = series[i], series[i - 1]
        tr.append(max(cur.h - cur.l, abs(cur.h - pc.c), abs(cur.l - pc.c)))
    if len(tr) < period:
        return 0.0
    a = sum(tr[:period]) / period
    for i in range(period, len(tr)):
        a = (a * (period - 1) + tr[i]) / period
    return a


class _ReplayClock:
    """Stand-in for the `time` module: .time() returns the current replay
    bar's timestamp (seconds), so DSL elapsed-time logic (stale-flat 480min,
    hard-timeout 1800min) runs on HISTORICAL time."""

    def __init__(self, now_s: float) -> None:
        self._now = now_s

    def time(self) -> float:
        return self._now


def simulate_dsl_exit(coin: str, side: str, entry_i: int, entry_px: float,
                      candles: list[Candle], policy, leverage: int,
                      sl_mult: float = 1.5, tp_fraction: float = 0.5,
                      stop_at: int | None = None,
                      fee_pct: float = ROUND_TRIP_FEE_BPS / 10000.0) -> dict:
    """Replay one position through the production DSL + server-side orders.

    Entry = candles[entry_i].c (the poke bar's close — same entry the hold
    baseline uses, so the two exit modes are comparable). Returns a dict with
    the DSL outcome; `pnl_pct` is spot-move % net of the round-trip taker fee.

    stop_at: exclusive end index for the walk. Used by the grid explorer to
    censor positions at the IS/OOS phase boundary (a position that is still
    open at the boundary is closed at that bar's close and marked
    `data_end`). Default None = walk to the end of the candle list.
    """
    import hermes_trader.agents.dsl_exit as dsl_exit

    entry_t_s = candles[entry_i].t / 1000.0
    sign = 1 if side == "long" else -1

    clock = _ReplayClock(entry_t_s)
    real_time = dsl_exit.time
    real_save = dsl_exit._save_state
    # Redirect persistence + freeze the clock BEFORE any tracker call.
    dsl_exit.time = clock
    dsl_exit._save_state = lambda: None
    try:
        atr4 = atr4h_at_entry(candles, entry_i)
        entry_atr_pct = (atr4 / entry_px * 100) if atr4 > 0 and entry_px > 0 else 0.0
        if atr4 > 0:
            sl_px = entry_px - atr4 * sl_mult if side == "long" else entry_px + atr4 * sl_mult
            tp_px = entry_px + atr4 * TP_ATR_MULT if side == "long" else entry_px - atr4 * TP_ATR_MULT
        else:
            sl_px = None
            tp_px = None

        tracker = dsl_exit.DSLTracker(coin, side, entry_px, entry_t_s, policy,
                                      leverage=leverage,
                                      entry_atr_pct=entry_atr_pct)
        tracker.current_tp_px = tp_px  # what the live loop would have placed

        def move(px: float) -> float:
            return sign * (px / entry_px - 1) * 100

        remaining = 1.0
        realized = 0.0          # notional-weighted spot % of closed legs
        tp_armed = tp_px is not None and 0 < tp_fraction <= 1.0
        exit_reason = ""
        exit_i = len(candles) - 1
        exit_px = candles[-1].c
        bar_end = stop_at if stop_at is not None else len(candles)

        for j in range(entry_i + 1, bar_end):
            bar = candles[j]
            clock._now = bar.t / 1000.0

            # 1) Server-side backup SL — full remaining size, fires instantly
            #    at the stop price (checked BEFORE TP: conservative when a bar
            #    spans both levels).
            if sl_px is not None and remaining > 0 and (
                    (side == "long" and bar.l <= sl_px)
                    or (side == "short" and bar.h >= sl_px)):
                realized += remaining * move(sl_px)
                exit_i, exit_px, remaining = j, sl_px, 0.0
                exit_reason = f"server_sl ({sl_mult:g}x ATR4h)"
                break

            # 2) Server-side TP scale-out — banks `tp_fraction` at tp_px.
            if tp_armed and (
                    (side == "long" and bar.h >= tp_px)
                    or (side == "short" and bar.l <= tp_px)):
                frac = tp_fraction
                realized += frac * move(tp_px)
                remaining -= frac
                tp_armed = False
                # If the bar ALSO ran the full DSL stop distance after the TP
                # (it didn't touch the SL above), the remainder keeps riding.

            if remaining <= 0:
                exit_i, exit_px = j, tp_px
                exit_reason = "tp_scaleout_full"  # only if fraction was 1.0
                break

            # 3) DSL floor — live cadence is 60s marks; approximate by
            #    sampling the bar's FAVORABLE extreme first (peak tracking,
            #    as the 60s ticks would have seen it) then the ADVERSE one.
            #    A floor breach fills at the floor price (the DSL close is a
            #    market order at the mid when the tick crossed it).
            fav = bar.h if side == "long" else bar.l
            adv = bar.l if side == "long" else bar.h
            v1 = tracker.check(fav)
            if not v1.exit:
                v1 = tracker.check(adv)
            if v1.exit:
                if v1.reason.startswith(("stale_flat_timeout", "hard_timeout")):
                    px = bar.c  # timeout closes at the current mark
                else:
                    px = v1.floor_price if v1.floor_price else adv
                realized += remaining * move(px)
                exit_i, exit_px, remaining = j, px, 0.0
                exit_reason = v1.reason
                break

        if remaining > 0:
            # Walk ended (data end, or the stop_at phase boundary) with the
            # position still open — mark at the boundary bar's close.
            last = bar_end - 1
            exit_i, exit_px = last, candles[last].c
            realized += remaining * move(exit_px)
            exit_reason = "data_end (marked at last close)"

        pnl_pct = realized - (fee_pct * 100)  # round-trip taker fee, full notional
        return {
            "exit_i": exit_i, "exit_px": exit_px, "pnl_pct": pnl_pct,
            "reason": exit_reason, "bars_held": exit_i - entry_i,
            "atr4": atr4, "sl_px": sl_px, "tp_px": tp_px,
            "entry_atr_pct": entry_atr_pct, "leverage": leverage,
        }
    finally:
        dsl_exit.time = real_time
        dsl_exit._save_state = real_save


def apply_dsl_exits(coin: str, candles: list[Candle], events: list[dict],
                    cfg: dict, time_scale: float = 1.0,
                    stop_at: int | None = None) -> dict:
    """Run the DSL sim over every fired event (mutating the event dicts so
    the existing charts/summary use the DSL exit) and return a summary.

    stop_at: exclusive end index capping each position's walk (see
    simulate_dsl_exit). Used by the grid explorer for IS/OOS phase
    boundaries. Default None = walk to the end of the data.
    """
    policy = build_live_policy(cfg, time_scale=time_scale)
    leverage = int(cfg.get("leverage", 1) or 1)
    sl_mult = float(cfg.get("sl_atr_mult", 1.5))
    tp_frac = float(cfg.get("tp_scale_fraction", 0.5))
    for ev in events:
        res = simulate_dsl_exit(coin, ev["side"], ev["entry_i"], ev["entry_px"],
                                candles, policy, leverage,
                                sl_mult=sl_mult, tp_fraction=tp_frac,
                                stop_at=stop_at)
        # Preserve the hold-baseline P/L and the original trigger reason
        # (ev.update(res) below overwrites pnl_pct/exit_i/exit_px/reason).
        ev["hold_pnl_pct"] = ev["pnl_pct"]
        ev["trigger_reason"] = ev["reason"]
        ev["exit_mode"] = "dsl"
        ev.update(res)
        ev["exit_label"] = f"{res['reason'].split(' (')[0]} · {res['bars_held']} bars"
    wins = sum(1 for e in events if e["pnl_pct"] > 0)
    net = sum(e["pnl_pct"] for e in events)
    reasons: dict[str, int] = {}
    for e in events:
        key = e["reason"].split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1
    return {"wins": wins, "net": net, "reasons": reasons,
            "leverage": leverage, "sl_mult": sl_mult, "tp_frac": tp_frac}


def replay(candles: list[Candle], window: int, max_drift_pct: float,
           min_poke_atr: float, hold_bars: int, ma_type: str = "ema",
           band_span=None):
    """Bar-by-bar replay. Returns list of event dicts for fired signals."""
    events = []
    span = window if band_span is None else max(2, int(band_span))
    # The trigger needs 2*window bars BEFORE the poke (window for the band
    # edges + window for the drift reference), so the first eligible poke is
    # at index 2*window.
    for i in range(2 * window, len(candles)):
        history = candles[: i + 1]          # up to and incl. poke bar i (closed)
        hit = band_snapback(history, window=window, max_drift_pct=max_drift_pct,
                            min_poke_atr=min_poke_atr, ma_type=ma_type,
                            band_span=band_span, include_partial=False)
        if not hit.get("fired"):
            continue
        entry_i = i
        entry_px = candles[i].c
        exit_i = min(i + hold_bars, len(candles) - 1)
        exit_px = candles[exit_i].c
        side = "long" if "lower" in hit["reason"] else "short"
        sign = 1 if side == "long" else -1
        pnl_pct = sign * (exit_px / entry_px - 1) * 100
        baseline_pct = sign * ((candles[exit_i].c / entry_px - 1) * 100)
        # Curved MA band over the `span` bars ending at i-1 (the bar before
        # the poke) — the same band the trigger judged the poke against,
        # plus its one-bar linear projection onto the poke bar (dashed).
        fit = candles[i - 2 * window: i]
        up_ma, lo_ma = _band_ma(fit, span, ma_type)
        up_fit = up_ma[-span:]              # curved upper edge (MA of highs)
        lo_fit = lo_ma[-span:]              # curved lower edge (MA of lows)
        band = candles[i - span: i]         # the span those edges cover
        up_proj = _project_band_edge(up_ma)  # projected edge at the poke bar
        lo_proj = _project_band_edge(lo_ma)
        events.append({
            "i": i, "t": candles[i].t, "side": side,
            "score": hit["score"], "reason": hit["reason"],
            "entry_i": entry_i, "entry_px": entry_px,
            "exit_i": exit_i, "exit_px": exit_px,
            "pnl_pct": pnl_pct, "baseline_pct": baseline_pct,
            "up_band": up_fit, "lo_band": lo_fit,
            "up_proj": up_proj, "lo_proj": lo_proj,
            "band": band, "ma_type": ma_type, "span": span,
        })
    return events


def _ts_ms(ms: int) -> float:
    return ms / 1000.0


def plot_event(coin: str, candles: list[Candle], ev: dict, hold_bars: int,
               path: str, interval: str = "15m") -> None:
    # Chart window: from the band-window start through the exit.
    start_i = ev["i"] - len(ev["band"])
    end_i = ev["exit_i"]
    view = candles[start_i: end_i + 1]
    # x-axis in bar-index (unit = one candle) so any interval plots cleanly.
    xs = list(range(len(view)))

    fig, ax = plt.subplots(figsize=(12.5, 6.4), dpi=130)
    fig.patch.set_facecolor("#101418")
    ax.set_facecolor("#101418")
    grid = ax.grid(True, color="#2a3140", linewidth=0.6)
    for sp in ax.spines.values():
        sp.set_color("#2a3140")

    # Candles
    for c, x in zip(view, xs):
        up = c.c >= c.o
        col = UP_COLOR if up else DOWN_COLOR
        ax.plot([x, x], [c.l, c.h], color=col, linewidth=0.9, zorder=2)
        body_lo, body_hi = sorted([c.o, c.c])
        ax.add_patch(Rectangle((x - 0.4, body_lo), 0.8,
                               max(body_hi - body_lo, 1e-9),
                               facecolor=col, edgecolor=col, zorder=3))

    # Curved MA band: the rolling MA of highs (upper) / lows (lower) over the
    # window ending at the bar before the poke.
    band = ev["band"]
    up_fit = ev["up_band"]
    lo_fit = ev["lo_band"]
    # up_fit[k] is the band edge at candle candles[i-span+k]; view[0] ==
    # candles[i-span], so the band belongs at the LEFTMOST len(band) slots,
    # ending one bar before the poke marker (at x = span). The old
    # rightmost-slot placement shifted the whole band right by (len(view)-span)
    # candle widths — the x-offset the band "floating" on the far right.
    band_xs = xs[:len(band)]
    poke_x = xs[ev["i"] - start_i]
    band_label = f"MA band — {ev.get('ma_type', 'ema').upper()} of highs/lows"
    ax.plot(band_xs, up_fit, color=BAND_COLOR, linewidth=1.7, zorder=4,
            label=band_label)
    ax.plot(band_xs, lo_fit, color=BAND_COLOR, linewidth=1.7, zorder=4)
    # One-bar linear projection of the last true edge onto the POKING bar
    # (dashed) — the de-lagged edge the poke is actually judged against.
    proj_last_x = band_xs[-1]
    ax.plot([proj_last_x, poke_x], [up_fit[-1], ev["up_proj"]],
            color=BAND_COLOR, linewidth=1.4, linestyle="--", zorder=4,
            label="projected 1 bar (poke judged here)")
    ax.plot([proj_last_x, poke_x], [lo_fit[-1], ev["lo_proj"]],
            color=BAND_COLOR, linewidth=1.4, linestyle="--", zorder=4)

    # Poke bar marker
    poke = candles[ev["i"]]
    # Annotation offsets in BAR units (x-axis is bar-index). A fixed
    # bar-count keeps text inside the view at any span/hold.
    ann_off = max(2, len(xs) // 5)
    ax.plot([poke_x], [poke.h if ev["side"] == "short" else poke.l],
            marker="v" if ev["side"] == "short" else "^", color="#ff9800",
            markersize=13, zorder=6)
    ax.annotate(f"wick poke\n({ev.get('trigger_reason', ev['reason']).split(' (')[0]})",
                xy=(poke_x, poke.h if ev["side"] == "short" else poke.l),
                xytext=(max(0, poke_x - ann_off),
                        poke.h * 1.004 if ev["side"] == "short"
                        else poke.l * 0.996),
                color="#ff9800", fontsize=9, ha="right",
                arrowprops=dict(arrowstyle="->", color="#ff9800"))

    # Entry/exit (x in bar-index; entry_i == poke_i)
    entry_x = xs[ev["entry_i"] - start_i]
    exit_x = xs[ev["exit_i"] - start_i]
    def _tl(idx_in_view):
        d = datetime.fromtimestamp(_ts_ms(candles[start_i + idx_in_view].t),
                                   tz=timezone.utc)
        return d.strftime("%m-%d %H:%M") if interval in ("1h", "4h", "1d") else d.strftime("%H:%M")
    ax.axhline(ev["entry_px"], color=ENTRY_COLOR, linewidth=1.1,
               linestyle=":", xmin=0.55, zorder=5)
    # DSL mode: show the server-side orders the live executor would have
    # placed (backup SL + TP scale-out), so the exit marker's provenance is
    # visible on the chart.
    if ev.get("exit_mode") == "dsl":
        # NOTE: labels use pure DATA coordinates (not xaxis_transform) —
        # mixed-transform text breaks under the tight_layout() below
        # (FreeType raster overflow crash on some charts).
        _lbl_x = 0.55 * (len(view) - 1)  # mid-view data-x (x≈0 sat under legend)
        if ev.get("sl_px"):
            ax.axhline(ev["sl_px"], color="#ef5350", linewidth=0.9,
                       linestyle="--", xmin=0.55, zorder=4)
            ax.text(_lbl_x, ev["sl_px"], f" SL@{ev['sl_px']:.6g} ",
                    color="#ef5350", fontsize=7, va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#1b2129",
                              edgecolor="#ef5350"))
        if ev.get("tp_px"):
            ax.axhline(ev["tp_px"], color="#66bb6a", linewidth=0.9,
                       linestyle="--", xmin=0.55, zorder=4)
            ax.text(_lbl_x, ev["tp_px"], f" TP@{ev['tp_px']:.6g} ",
                    color="#66bb6a", fontsize=7, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#1b2129",
                              edgecolor="#66bb6a"))
        # axhline doesn't extend autoscale — widen the y-limits so the SL/TP
        # lines (and their labels) stay inside the frame.
        lo, hi = ax.get_ylim()
        lo = min(lo, ev.get("sl_px") or lo)
        hi = max(hi, ev.get("tp_px") or hi)
        ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))
    ax.plot([entry_x], [ev["entry_px"]],
            marker="o", color=ENTRY_COLOR, markersize=8, zorder=6)
    ax.annotate(f"entry {ev['entry_px']:.6g}",
                xy=(entry_x, ev["entry_px"]),
                xytext=(max(0, entry_x - ann_off), ev["entry_px"]),
                color=ENTRY_COLOR, fontsize=9, ha="right")
    ax.plot([exit_x], [ev["exit_px"]],
            marker="x", color="#e040fb", markersize=9, zorder=6)
    pnl_col = UP_COLOR if ev["pnl_pct"] >= 0 else DOWN_COLOR
    exit_lbl = ev.get("exit_label") or f"+{hold_bars} bars"
    if "hold_pnl_pct" in ev:
        _pl_line = f"P/L {ev['pnl_pct']:+.2f}% (hold {ev['hold_pnl_pct']:+.2f}%)"
    else:
        _pl_line = f"P/L {ev['pnl_pct']:+.2f}%"
    ax.annotate(f"exit {exit_lbl} @ {ev['exit_px']:.6g}\n{_pl_line}",
                xy=(exit_x, ev["exit_px"]),
                xytext=(max(0, exit_x - ann_off),
                        ev["exit_px"] + (0.002 if ev["side"] == "long" else -0.002)
                        * (view[-1].h - view[0].l)),
                color=pnl_col, fontsize=9, ha="right",
                arrowprops=dict(arrowstyle="->", color=pnl_col))

    title = (f"{coin} — band snapback {ev['side'].upper()} @ "
             f"{datetime.fromtimestamp(_ts_ms(ev['t']), tz=timezone.utc):%Y-%m-%d %H:%M} UTC "
             f"({interval})")
    ax.set_title(title, color="#eceff1", fontsize=13, loc="left")
    step = max(1, len(xs) // 10)
    tick_idxs = list(range(0, len(xs), step))[:11]
    ax.set_xticks([xs[j] for j in tick_idxs])
    ax.set_xticklabels([_tl(j) for j in tick_idxs],
                       color="#90a4ae", fontsize=8)
    ax.tick_params(axis="y", colors="#90a4ae", labelsize=8)
    legend_items = [
        Line2D([], [], color=BAND_COLOR, linewidth=1.7,
               label=f"MA band ({ev.get('ma_type', 'ema').upper()} of highs/low)"),
        Line2D([], [], color="#ff9800", marker="^", linestyle="", label="wick poke"),
        Line2D([], [], color=ENTRY_COLOR, marker="o", linestyle=":", label="entry (poke close)"),
        Line2D([], [], color="#e040fb", marker="x", linestyle="",
               label="exit (DSL)" if ev.get("exit_mode") == "dsl"
               else f"exit (+{hold_bars} bars)"),
    ]
    ax.legend(handles=legend_items, loc="upper left", facecolor="#1b2129",
              edgecolor="#2a3140", labelcolor="#eceff1", fontsize=8)
    ax.set_xlim(xs[0] - 1.5, xs[-1] + 1.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_summary(coin: str, events: list[dict], path: str, hold_bars: int,
                 exit_title: str = "hold") -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.2), dpi=130,
                                   gridspec_kw={"height_ratios": [3, 1.2]})
    fig.patch.set_facecolor("#101418")
    for ax in (ax1, ax2):
        ax.set_facecolor("#101418")
        ax.grid(True, color="#2a3140", linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color("#2a3140")
        ax.tick_params(colors="#90a4ae", labelsize=8)

    xs = list(range(len(events)))
    pnls = [e["pnl_pct"] for e in events]
    cols = [UP_COLOR if p >= 0 else DOWN_COLOR for p in pnls]
    ax1.bar(xs, pnls, color=cols, width=0.7, zorder=3)
    ax1.axhline(0, color="#546e7a", linewidth=0.9)
    ax1.axhline(sum(pnls) / len(pnls), color="#ffee58", linewidth=1.1, linestyle="--",
                label=f"avg {sum(pnls) / len(pnls):+.2f}%")
    ax1.set_ylabel("P/L, % (fees incl)" if "dsl" in exit_title.lower() else "P/L vs hold baseline, %",
               color="#90a4ae", fontsize=9)
    ax1.legend(facecolor="#1b2129", edgecolor="#2a3140", labelcolor="#eceff1", fontsize=8)
    ax1.set_title(f"{coin} — band snapback: {len(events)} signals, {exit_title}, {events[0]['ma_type'].upper()} band",
                  color="#eceff1", fontsize=12, loc="left")

    for i, e in enumerate(events):
        d = datetime.fromtimestamp(_ts_ms(e["t"]), tz=timezone.utc)
        ax1.text(i, e["pnl_pct"] + (0.35 if e["pnl_pct"] >= 0 else -0.6),
                 f"{e['side'][0].upper()}\n{d:%m-%d %H:%M}", ha="center",
                 color="#b0bec5", fontsize=7)

    ax2.axis("off")
    wins = sum(1 for p in pnls if p > 0)
    text = (f"signals {len(events)}  ·  wins {wins} ({wins / len(events):.0%})  ·  "
            f"net {sum(pnls):+.2f}%  ·  best {max(pnls):+.2f}%  ·  worst {min(pnls):+.2f}%")
    ax2.text(0.02, 0.5, text, color="#eceff1", fontsize=10, va="center",
             family="monospace")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", default="BTC,ETH,SOL")
    ap.add_argument("--interval", default="15m",
                    help="candle timeframe: 1m 3m 5m 15m 1h 4h (default 15m)")
    ap.add_argument("--bars", type=int, default=500)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--ma-type", dest="ma_type", default="ema",
                    choices=["ema", "sma"],
                    help="band-edge moving average (default ema)")
    ap.add_argument("--band-span", dest="band_span", type=int, default=None,
                    help="band-edge MA span (bars); LAG dial. Default: = --window. "
                         "Smaller = tighter/faster band, e.g. 16")
    ap.add_argument("--drift", type=float, default=1.5)
    ap.add_argument("--min-poke", dest="min_poke", type=float, default=0.5)
    ap.add_argument("--hold", type=int, default=12, help="bars to hold after entry")
    ap.add_argument("--max-events", type=int, default=40)
    ap.add_argument("--exit", dest="exit_mode", default="dsl",
                    choices=["hold", "dsl"],
                    help="exit model: 'hold' = fixed --hold bars (baseline), "
                         "'dsl' = full live DSL exit stack (production DSLTracker "
                         "+ 4h-ATR server SL/TP scale-out + fees, .agent-config.json) "
                         "(default dsl)")
    ap.add_argument("--scale-timeouts", dest="scale_timeouts", type=float,
                    default=1.0,
                    help="backtest-only: multiply the wall-clock DSL timeouts "
                         "(stale_flat / hard_timeout) by this factor. Live timers "
                         "are tuned in real minutes; on a 4h chart 480min = 2 bars "
                         "so pass 4.0 to keep the intended ~8-bar/~30-bar hold in "
                         "bar-count terms. 1h and below use 1.0.")
    ap.add_argument("--chart-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scratch", "band_snapback_charts"))
    args = ap.parse_args()

    os.makedirs(args.chart_dir, exist_ok=True)
    coins = [c.strip() for c in args.coins.split(",") if c.strip()]
    span = args.window if args.band_span is None else max(2, args.band_span)
    lag_bars = (span - 1) / 2
    dsl_mode = args.exit_mode == "dsl"
    time_scale = args.scale_timeouts
    cfg = _load_agent_config() if dsl_mode else {}
    if dsl_mode:
        pol = build_live_policy(cfg, time_scale=time_scale)
        _scale_note = (f" (×{time_scale:g} → {pol.stale_flat_timeout_minutes:g}min stale / "
                       f"{pol.hard_timeout_minutes:g}min hard)" if time_scale != 1.0 else "")
        print(f"exit: DSL (production stack) — leverage={cfg.get('leverage', 1)}x, "
              f"sl={cfg.get('sl_atr_mult', 1.5):g}×ATR(4h), "
              f"tp={cfg.get('tp_scale_fraction', 0.5):.0%}@{TP_ATR_MULT:g}×ATR(4h), "
              f"fees {ROUND_TRIP_FEE_BPS:g}bps rt, "
              f"dsl max_loss={pol.max_loss_pct:g}% protect={pol.protect_pct:g}% "
              f"retrace={pol.retrace_threshold:g} "
              f"stale_flat={pol.stale_flat_timeout_minutes:g}min "
              f"hard_timeout={pol.hard_timeout_minutes:g}min{_scale_note}\n", flush=True)
    else:
        print(f"exit: hold-{args.hold} bars (fixed, gross — no fees)\n", flush=True)

    all_events = []
    for coin in coins:
        print(f"[{coin}] fetching {args.bars} x {args.interval} candles ...", flush=True)
        candles = fetch_range(coin, args.interval, args.bars)
        if len(candles) < 2 * args.window + 2:
            print(f"[{coin}] not enough candles ({len(candles)} < {2 * args.window + 2} = 2*window+2), skipping")
            continue
        events = replay(candles, args.window, args.drift, args.min_poke,
                        args.hold, args.ma_type, args.band_span)
        print(f"[{coin}] {len(candles)} candles, {len(events)} fired signals "
              f"({datetime.fromtimestamp(_ts_ms(candles[0].t), tz=timezone.utc):%Y-%m-%d} → "
              f"{datetime.fromtimestamp(_ts_ms(candles[-1].t), tz=timezone.utc):%Y-%m-%d %H:%M})")
        if events:
            hold_wins = sum(1 for e in events if e["pnl_pct"] > 0)
            hold_net = sum(e["pnl_pct"] for e in events)
            print(f"[{coin}] hold-{args.hold}bars: wins {hold_wins}/{len(events)} "
                  f"({hold_wins / len(events):.0%}), net {hold_net:+.2f}%, "
                  f"avg {hold_net / len(events):+.3f}%")
            dsl_summary = None
            if dsl_mode:
                dsl_summary = apply_dsl_exits(coin, candles, events, cfg,
                                              time_scale=time_scale)
                d_wins = dsl_summary["wins"]
                d_net = dsl_summary["net"]
                print(f"[{coin}] dsl:       wins {d_wins}/{len(events)} "
                      f"({d_wins / len(events):.0%}), net {d_net:+.2f}% (fees incl), "
                      f"avg {d_net / len(events):+.3f}%")
                _reasons = ", ".join(f"{k}×{v}" for k, v in
                                     sorted(dsl_summary["reasons"].items(),
                                            key=lambda kv: -kv[1]))
                print(f"[{coin}]   exits: {_reasons}")
            exit_title = "DSL exit" if dsl_mode else f"hold {args.hold} bars"
            stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
            plot_summary(coin, events,
                         os.path.join(args.chart_dir, f"{coin}_{stamp}_summary.png"),
                         args.hold, exit_title)
            for n, ev in enumerate(events[:args.max_events]):
                d = datetime.fromtimestamp(_ts_ms(ev["t"]), tz=timezone.utc)
                if dsl_mode:
                    print(f"  #{n + 1} {ev['side']:5s} {d:%m-%d %H:%M} "
                          f"exit={ev.get('exit_label', '?'):18s} "
                          f"in={ev['entry_px']:.6g} out={ev['exit_px']:.6g} "
                          f"pnl {ev['pnl_pct']:+.2f}% (hold-{args.hold} "
                          f"{ev.get('hold_pnl_pct', 0):+.2f}%)")
                plot_event(coin, candles, ev, args.hold,
                           os.path.join(args.chart_dir,
                                        f"{coin}_{stamp}_{args.interval}_{d:%m%d_%H%M}_{ev['side']}_pnl{ev['pnl_pct']:+.2f}.png"),
                           args.interval)
            all_events.extend(events)
        time.sleep(0.5)

    if all_events:
        hold_wins = sum(1 for e in all_events if e["pnl_pct"] > 0) or \
                    sum(1 for e in all_events if e.get("hold_pnl_pct", 0) > 0)
        print(f"\n=== TOTAL {len(all_events)} signals across {len(coins)} coins ===")
        if dsl_mode:
            hold_wins = sum(1 for e in all_events if e.get("hold_pnl_pct", 0) > 0)
            hold_net = sum(e.get("hold_pnl_pct", 0) for e in all_events)
            dsl_wins = sum(1 for e in all_events if e["pnl_pct"] > 0)
            dsl_net = sum(e["pnl_pct"] for e in all_events)
            reasons: dict = {}
            for e in all_events:
                k = e["reason"].split(" (")[0]
                reasons[k] = reasons.get(k, 0) + 1
            _r = ", ".join(f"{k}×{v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
            print(f"hold-{args.hold}bars: wins {hold_wins}/{len(all_events)} ({hold_wins / len(all_events):.0%}), "
                  f"net {hold_net:+.2f}% (gross)")
            print(f"dsl:          wins {dsl_wins}/{len(all_events)} ({dsl_wins / len(all_events):.0%}), "
                  f"net {dsl_net:+.2f}% (fees incl)")
            print(f"exits: {_r}")
        else:
            net = sum(e["pnl_pct"] for e in all_events)
            print(f"hold-{args.hold}bars: wins {hold_wins}/{len(all_events)} ({hold_wins / len(all_events):.0%}), "
                  f"net {net:+.2f}%")
        print(f"charts: {args.chart_dir}")
    else:
        print("\nNo signals fired in this window — try a wider window, larger --bars, or more coins.")


if __name__ == "__main__":
    main()