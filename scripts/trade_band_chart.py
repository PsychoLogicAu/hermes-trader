#!/usr/bin/env python3
"""Per-trade annotated band chart for hermes-trader trades.

Given a trade (coin + approximate entry time, taken from the ledger or a
log line), fetches the candles the live bot would have seen and renders one
annotated PNG:

  * candles on the band's CONFIG interval (band_snapback.interval, with the
    per-coin override — 1h base, 15m for PUMP/CASHCAT/ZEC/LIT)
  * the MA band the trigger actually runs (rolling `ma_type` of highs/lows,
    `band_span` bars) as computed by triggers._band_ma — NOT a re-derivation
  * the trigger's verdict at the entry bar (what band_snapback() returned
    with include_partial=False on history through the entry bar)
  * entry (yellow dot), exit (magenta x) with the ledger's exit reason and P/L
  * the server-side SL / TP levels the executor would have placed
    (sl_atr_mult x ATR4h stop, 1.0x ATR4h TP scale-out — the same 4h Wilder
    ATR the live executor reads, partial 4h bucket included)

Usage (project venv):
  .venv/bin/python scripts/trade_band_chart.py --coin CASHCAT --at 2026-08-24T00:54:27Z
  .venv/bin/python scripts/trade_band_chart.py --coin CASHCAT --at "2026-08-24 00:54"   # fuzzy ok
  .venv/bin/python scripts/trade_band_chart.py --coin CASHCAT --at 2026-08-24T00:54:27Z \
      --coin PURR --at 2026-08-23T14:30:34Z          # multiple trades
  .venv/bin/python scripts/trade_band_chart.py --coin CASHCAT --at ... --out /tmp/x.png

Trade identification: OPEN rows in trader-logs/trades.jsonl; --at is matched
to the OPEN whose ts is closest (within 30 min) and the matching CLOSE (same
coin+side+entry_px, earliest after entry). If a matching CLOSE is missing the
chart is still drawn for an open position (P/L shown vs the last fetched
close).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_trader.models.types import Candle, TriggerHit  # noqa: E402
from hermes_trader.indicators.triggers import band_snapback, _band_ma, atr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "trader-logs", "trades.jsonl")
CONFIG = os.path.join(ROOT, ".agent-config.json")
HL_API = "https://api.hyperliquid.xyz/info"

WIN_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
          "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"
BAND_COLOR = "#42a5f5"
ENTRY_COLOR = "#ffee58"
EXIT_COLOR = "#e040fb"
SL_COLOR = "#ef5350"
TP_COLOR = "#66bb6a"

# executor.py hardcodes TP_ATR_MULT = 1.0 (not config-driven).
TP_ATR_MULT = 1.0
TP_SCALE_FRACTION = 0.5


# ── inputs ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _parse_at(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise SystemExit(f"unrecognised --at timestamp: {s!r} "
                     "(use e.g. 2026-08-24T00:54:27Z or '2026-08-24 00:54')")


def locate_trade(rows: list[dict], coin: str, at: datetime) -> tuple[dict, dict | None]:
    """Find the OPEN nearest to `at` (<=30 min) and its matching CLOSE."""
    opens = [r for r in rows if r.get("event") == "OPEN" and r.get("coin") == coin]
    if not opens:
        raise SystemExit(f"no OPEN for {coin} in the ledger")
    def ts(e):
        return datetime.fromtimestamp(e["ts"] / 1000, tz=timezone.utc)
    op = min(opens, key=lambda e: abs((ts(e) - at).total_seconds()))
    gap = abs((ts(op) - at).total_seconds())
    if gap > 30 * 60:
        best = ts(op).isoformat()
        raise SystemExit(f"no {coin} OPEN within 30 min of {at.isoformat()} "
                         f"(nearest: {best}, {gap / 60:.0f} min off)")
    entry_px = float(op.get("entry_px") or 0)
    closes = [r for r in rows if r.get("event") == "CLOSE" and r.get("coin") == coin
              and r.get("side") == op.get("side")
              and abs(float(r.get("entry_px") or 0) - entry_px) < 1e-12
              and r["ts"] >= op["ts"]]
    cl = min(closes, key=lambda r: r["ts"]) if closes else None
    return op, cl


def load_band_params(cfg: dict, coin: str) -> dict:
    bs = cfg.get("band_snapback") or {}
    ov = (bs.get("overrides") or {}).get(coin) or {}
    return {
        "interval": ov.get("interval", bs.get("interval", "1h")),
        "window": int(bs.get("window", 48)),
        "ma_type": bs.get("ma_type", "ema"),
        "band_span": int(ov.get("band_span", bs.get("band_span", 16))),
        "max_drift_pct": float(bs.get("max_drift_pct", 1.5)),
        "min_poke_atr": float(bs.get("min_poke_atr", 0.75)),
        "sl_atr_mult": float(cfg.get("sl_atr_mult", 1.5)),
    }


# ── data ───────────────────────────────────────────────────────────────

def fetch_range(coin: str, interval: str, start_ms: int, end_ms: int,
                retries: int = 5) -> list[Candle]:
    """Paged candleSnapshot fetch [start_ms, end_ms]. Own session + pacing:
    the live bot shares this host's IP and its scan cadence exhausts the HL
    weight budget, so we don't touch the shared HL_LIMITER (same approach as
    scripts/band_snapback_backtest.py)."""
    import requests
    sess = requests.Session()
    out: list[Candle] = []
    end = end_ms
    step = WIN_MS.get(interval, 300_000)
    while end > start_ms:
        start = max(start_ms, end - step * 500)
        payload = {"type": "candleSnapshot",
                   "req": {"coin": coin, "interval": interval,
                           "startTime": start, "endTime": end}}
        raw = None
        for attempt in range(retries):
            try:
                resp = sess.post(HL_API, json=payload, timeout=10)
                resp.raise_for_status()
                raw = resp.json()
            except Exception as e:
                print(f"  [{coin} {interval}] request error: {e}", flush=True)
                raw = None
            if isinstance(raw, list):
                break
            wait = 5.0 * (attempt + 1)
            print(f"  [{coin} {interval}] 429/empty, retry {attempt + 1}/{retries} "
                  f"in {wait:.0f}s ...", flush=True)
            time.sleep(wait)
        if not isinstance(raw, list):
            break
        page = [Candle(t=c["t"], o=float(c["o"]), h=float(c["h"]),
                       l=float(c["l"]), c=float(c["c"]), v=float(c.get("v", "0")))
                for c in raw]
        out = page + out
        if len(page) == 0:
            break
        end = page[0].t - 1
        time.sleep(1.5)  # gentle pace: sharing an IP with the live bot
    out.sort(key=lambda c: c.t)
    return [c for c in out if start_ms <= c.t <= end_ms]


def atr4h_at_entry(candles: list[Candle], entry_i: int, period: int = 14) -> float:
    """Wilder ATR(14) on 4h candles as of the entry bar — mirrors the backtest
    helper: the 4h bucket the entry bar sits in is PARTIAL (through entry),
    which is what the live executor sees at order time. No lookahead."""
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
                     l=buckets[b][2], c=buckets[b][3], v=0.0) for b in order]
    if len(series) < period + 1:
        return 0.0
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


# ── rendering ──────────────────────────────────────────────────────────

def plot_trade(coin: str, side: str, op: dict, cl: dict | None,
               candles: list[Candle], interval: str, bp: dict,
               state: dict, sl_px: float | None, tp_px: float | None,
               atr4: float, path: str) -> None:
    win = bp["window"]
    span = bp["band_span"]
    entry_px = float(op["entry_px"])
    entry_ms = op["ts"]

    # Entry/exit bar indices (bar whose timestamp is <= the event).
    entry_i = max(i for i, c in enumerate(candles) if c.t <= entry_ms)
    if cl:
        exit_px, exit_ms = float(cl["exit_px"]), cl["ts"]
        exit_i = max(i for i, c in enumerate(candles) if c.t <= exit_ms)
        hold = (exit_ms - entry_ms) / 60000.0
    else:
        exit_px, exit_ms = candles[-1].c, candles[-1].t
        exit_i = len(candles) - 1
        hold = (exit_ms - entry_ms) / 60000.0

    sign = 1 if side == "long" else -1
    spot = sign * (exit_px / entry_px - 1) * 100 if entry_px else 0.0
    lev = op.get("leverage") or 1
    pnl_pct = spot * lev

    # Chart window: band window + a little before the entry through the exit.
    before = win + span
    start_i = max(0, min(entry_i - before, len(candles) - 2))
    end_i = max(entry_i + 6, min(exit_i + 3, len(candles) - 1))
    view = candles[start_i: end_i + 1]
    xs = list(range(len(view)))

    fig, ax = plt.subplots(figsize=(12.5, 6.6), dpi=130)
    fig.patch.set_facecolor("#101418")
    ax.set_facecolor("#101418")
    ax.grid(True, color="#2a3140", linewidth=0.6)
    for sp in ax.spines.values():
        sp.set_color("#2a3140")

    # Candles.
    for c, x in zip(view, xs):
        up = c.c >= c.o
        col = UP_COLOR if up else DOWN_COLOR
        ax.plot([x, x], [c.l, c.h], color=col, linewidth=0.9, zorder=2)
        body_lo, body_hi = sorted([c.o, c.c])
        ax.add_patch(Rectangle((x - 0.4, body_lo), 0.8,
                               max(body_hi - body_lo, 1e-9),
                               facecolor=col, edgecolor=col, zorder=3))

    # MA band (rolling MA of highs/lows), drawn over the whole view.
    up_ma, lo_ma = _band_ma(view, span, bp["ma_type"])
    band_xs = [x for x, (u, l) in zip(xs, zip(up_ma, lo_ma)) if u == u and l == l]
    if band_xs:
        ax.plot(band_xs, [u for u in up_ma if u == u], color=BAND_COLOR,
                linewidth=1.7, zorder=4,
                label=f"band — {bp['ma_type'].upper()} of highs/lows ({span} bars)")
        ax.plot(band_xs, [l for l in lo_ma if l == l], color=BAND_COLOR,
                linewidth=1.7, zorder=4)

    def _tl(idx: int) -> str:
        d = datetime.fromtimestamp(view[idx].t / 1000.0, tz=timezone.utc)
        return d.strftime("%m-%d %H:%M") if interval in ("1h", "4h", "1d") else d.strftime("%H:%M")

    # Entry.
    e_x = xs[entry_i - start_i]
    ax.plot([e_x], [entry_px], marker="o", color=ENTRY_COLOR, markersize=9, zorder=6)
    ann_off = max(2, len(xs) // 6)
    ax.annotate(f"entry {entry_px:.6g}  ({lev}x)",
                xy=(e_x, entry_px),
                xytext=(max(0, e_x - ann_off), entry_px),
                color=ENTRY_COLOR, fontsize=9, ha="right",
                arrowprops=dict(arrowstyle="->", color=ENTRY_COLOR))

    # Exit. (annotation sits LEFT of the marker, at ~exit price level —
    # inside the view, the same way the backtest charts place it.)
    x_x = xs[exit_i - start_i]
    reason = (cl or {}).get("exit_reason") or (cl or {}).get("exit_type") or "open"
    reason = re.sub(r"\(.*\)$", "", reason).strip() or "open"
    pnl_col = UP_COLOR if spot >= 0 else DOWN_COLOR
    ax.plot([x_x], [exit_px], marker="x", color=EXIT_COLOR, markersize=10, zorder=6)
    usd = ""
    if cl and cl.get("realized_pnl_usd") is not None:
        usd = f" / {cl['realized_pnl_usd']:+.2f} USD"
    lbl = (f"exit {reason} @ {exit_px:.6g}\n"
           f"spot {spot:+.2f}% · ROE {pnl_pct:+.1f}%{usd} · held {hold:.0f} min")
    # True view price range (never assume the first/last candle is the
    # min/max — a downtrend would otherwise flip the sign of the offset).
    v_lo, v_hi = min(c.l for c in view), max(c.h for c in view)
    ax.annotate(lbl,
                xy=(x_x, exit_px),
                xytext=(max(0, x_x - ann_off),
                        exit_px + (0.003 if side == "long" else -0.003)
                        * (v_hi - v_lo)),
                color=pnl_col, fontsize=9, ha="right",
                arrowprops=dict(arrowstyle="->", color=pnl_col))

    # Server-side SL / TP levels the executor placed (from entry rightwards).
    # Label placement mirrors the proven plot_event() in
    # band_snapback_backtest.py: pure DATA coordinates at mid-view x
    # (mixed xaxis_transform text breaks tight_layout), SL label below its
    # line (va="bottom"), TP label above its (va="top"), then widen ylim so
    # both lines and labels stay inside the frame.
    # Base y-range: the TRUE min/max of the view candles — view[0].l /
    # view[-1].h would invert the axis when the trend runs from top to
    # bottom (first low > last high), which also flings the SL/TP labels
    # out of the frame and breaks tight_layout.
    lo, hi = v_lo, v_hi
    _lbl_x = 0.55 * (len(view) - 1)  # mid-view data-x (x≈0 sat under legend)
    # Label side of its line: away from the price action (longs: SL label
    # below its line, TP label above it; shorts are mirrored).
    sl_va = "bottom" if side == "long" else "top"
    tp_va = "top" if side == "long" else "bottom"
    levels = [p for p in (sl_px, tp_px) if p]
    if sl_px:
        ax.axhline(sl_px, color=SL_COLOR, linewidth=0.9, linestyle="--",
                   xmin=e_x / max(len(xs) - 1, 1), zorder=4)
        ax.text(_lbl_x, sl_px, f" SL {bp['sl_atr_mult']:g}×ATR4h @{sl_px:.6g} ",
                color=SL_COLOR, fontsize=7, va=sl_va, ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#1b2129",
                          edgecolor=SL_COLOR))
    if tp_px:
        ax.axhline(tp_px, color=TP_COLOR, linewidth=0.9, linestyle="--",
                   xmin=e_x / max(len(xs) - 1, 1), zorder=4)
        ax.text(_lbl_x, tp_px, f" TP {TP_SCALE_FRACTION:.0%}@{TP_ATR_MULT:g}×ATR4h @{tp_px:.6g} ",
                color=TP_COLOR, fontsize=7, va=tp_va, ha="left",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#1b2129",
                          edgecolor=TP_COLOR))
    if levels:
        lo, hi = min(lo, *levels), max(hi, *levels)
    ax.set_ylim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))

    # Title + trigger verdict at entry (second line; tight_layout accounts
    # for the 2-line title's extent, unlike transAxes text which breaks it).
    e_dt = datetime.fromtimestamp(entry_ms / 1000.0, tz=timezone.utc)
    fired = state.get("fired")
    verdict = state.get("reason", "?")
    vtxt = (f"band @ entry: {verdict}" if state else
            f"band @ entry: insufficient history")
    if fired:
        vtxt = f"* {vtxt} (TRIGGER FIRED)"
    ax.set_title(f"{coin} {side.upper()} — opened {e_dt:%Y-%m-%d %H:%M} UTC ({interval} chart)\n{vtxt}",
                 color="#eceff1", fontsize=12, loc="left")

    step = max(1, len(xs) // 10)
    tick_idxs = list(range(0, len(xs), step))[:11]
    ax.set_xticks([xs[j] for j in tick_idxs])
    ax.set_xticklabels([_tl(j) for j in tick_idxs], color="#90a4ae", fontsize=8)
    ax.tick_params(axis="y", colors="#90a4ae", labelsize=8)
    legend_items = [
        Line2D([], [], color=BAND_COLOR, linewidth=1.7,
               label=f"MA band ({bp['ma_type'].upper()}/{span})"),
        Line2D([], [], color=ENTRY_COLOR, marker="o", linestyle="", label="entry"),
        Line2D([], [], color=EXIT_COLOR, marker="x", linestyle="",
               label=f"exit ({reason})"),
    ]
    if sl_px:
        legend_items.append(Line2D([], [], color=SL_COLOR, linestyle="--", label="SL"))
    if tp_px:
        legend_items.append(Line2D([], [], color=TP_COLOR, linestyle="--", label="TP"))
    ax.legend(handles=legend_items, loc="upper left", facecolor="#1b2129",
              edgecolor="#2a3140", labelcolor="#eceff1", fontsize=8)
    ax.set_xlim(xs[0] - 1.5, xs[-1] + 1.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────

def render_one(rows: list[dict], cfg: dict, coin: str, at: datetime,
               out_dir: str, force_path: str | None = None) -> str | None:
    op, cl = locate_trade(rows, coin, at)
    entry_ms = op["ts"]
    side = op.get("side", "long")
    bp = load_band_params(cfg, coin)
    interval, win, span = bp["interval"], bp["window"], bp["band_span"]
    step = WIN_MS[interval]
    print(f"[{coin}] {side} entry {float(op['entry_px']):.6g} @ "
          f"{datetime.fromtimestamp(entry_ms / 1000.0, tz=timezone.utc):%Y-%m-%d %H:%M} UTC"
          f" (matched {abs((at.timestamp() * 1000 - entry_ms) / 1000.0 / 60.0):.1f} min off)")
    if cl:
        print(f"[{coin}] close {float(cl['exit_px']):.6g} @ "
              f"{datetime.fromtimestamp(cl['ts'] / 1000.0, tz=timezone.utc):%Y-%m-%d %H:%M} UTC — "
              f"{(cl.get('exit_reason') or '')[:70]}")
    else:
        print(f"[{coin}] no CLOSE found in ledger — charting as still-open")

    # Chart-interval candles: 2*window+margin before entry (enough for the
    # trigger verdict, which needs 2*window bars of history) → now.
    hold_ms = (cl["ts"] - entry_ms) if cl else 6 * 3_600_000
    start_ms = entry_ms - (2 * win + 10) * step
    end_ms = max(entry_ms + hold_ms + 10 * step, int(time.time() * 1000))
    print(f"[{coin}] fetching {interval} candles [{datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc):%Y-%m-%d %H:%M} → now] ...", flush=True)
    candles = fetch_range(coin, interval, start_ms, end_ms)
    if len(candles) < 2 * win + 4:
        print(f"[{coin}] only {len(candles)} candles fetched "
              f"(need {2 * win + 4} for the trigger verdict) — verdict may "
              f"read 'insufficient_history'", flush=True)

    entry_i = max(i for i, c in enumerate(candles) if c.t <= entry_ms)

    # Trigger verdict at the entry bar — the real trigger, history through
    # entry, all closed bars (include_partial=False).
    hist = candles[: entry_i + 1]
    state = band_snapback(hist, window=win, max_drift_pct=bp["max_drift_pct"],
                          min_poke_atr=bp["min_poke_atr"], ma_type=bp["ma_type"],
                          band_span=span, include_partial=False)

    # 4h ATR at entry (partial bucket included) → SL/TP the executor placed.
    a4 = fetch_range(coin, "4h", entry_ms - 30 * WIN_MS["4h"], entry_ms)
    if len(a4):
        a4_i = max(i for i, c in enumerate(a4) if c.t <= entry_ms)
        atr4 = atr4h_at_entry(a4, a4_i)
    else:
        atr4 = 0.0
    sign = 1 if side == "long" else -1
    sl_px = (float(op["entry_px"]) - sign * bp["sl_atr_mult"] * atr4) if atr4 > 0 else None
    tp_px = (float(op["entry_px"]) + sign * TP_ATR_MULT * atr4) if atr4 > 0 else None

    e_dt = datetime.fromtimestamp(entry_ms / 1000.0, tz=timezone.utc)
    name = f"{coin}_{e_dt:%Y%m%d_%H%M}_{side}.png"
    path = force_path or os.path.join(out_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plot_trade(coin, side, op, cl, candles, interval, bp, state, sl_px, tp_px, atr4, path)
    print(f"[{coin}] band verdict @ entry: {state.get('reason')}")
    if atr4 > 0:
        print(f"[{coin}] ATR4h @{entry_dt(e_dt)} {atr4:.6g} — SL {sl_px:.6g}, TP {tp_px:.6g}")
    print(f"[{coin}] chart: {path}")
    return path


def entry_dt(dt: datetime) -> str:
    return f"{dt:%H:%M}Z"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-trade annotated band chart for hermes-trader trades.")
    ap.add_argument("--coin", action="append", required=True)
    ap.add_argument("--at", action="append", required=True,
                    help="trade time (UTC), e.g. 2026-08-24T00:54:27Z or '2026-08-24 00:54'")
    ap.add_argument("--out", default=None, help="output PNG path (single trade only)")
    ap.add_argument("--chart-dir", default=os.path.join(ROOT, "scratch", "trade_charts"))
    ap.add_argument("--band-span", type=int, default=None, metavar="N",
                    help="override base band_snapback.band_span (per-coin overrides still win)")
    ap.add_argument("--interval", default=None, metavar="IV",
                    help="override base band_snapback.interval, e.g. 1m/5m/15m/1h/4h")
    args = ap.parse_args()
    if args.out and (len(args.coin) > 1):
        ap.error("--out is only valid for a single --coin/--at pair")
    if len(args.coin) != len(args.at):
        ap.error("every --coin needs a matching --at (same order)")

    rows = load_jsonl(LEDGER)
    try:
        with open(CONFIG) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"WARNING: could not read {CONFIG} ({e}); using trigger defaults",
              file=sys.stderr)
        cfg = {}

    if args.band_span is not None or args.interval:
        bs = cfg.setdefault("band_snapback", {})
        if args.band_span is not None:
            bs["band_span"] = args.band_span
        if args.interval:
            bs["interval"] = args.interval
        print(f"NOTE: base band_snapback overridden for this render only — "
              f"band_span={bs.get('band_span')}, interval={bs.get('interval')} "
              f"(per-coin overrides still win)", file=sys.stderr)

    paths = []
    for coin, ats in zip(args.coin, args.at):
        p = render_one(rows, cfg, coin, _parse_at(ats), args.chart_dir,
                       force_path=args.out)
        if p:
            paths.append(p)
    if not paths:
        raise SystemExit("no charts produced")


if __name__ == "__main__":
    main()