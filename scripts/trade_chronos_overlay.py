"""TRUMP 2026-08-28 06:27:45Z LONG — decision state + chronos-2 forecast overlay.

HL-interface aligned:
  * x = minutes since 23:00 UTC 2026-08-27 (single epoch basis, no offsets)
  * 1h bodies centred on their start-time hour line (body straddles the tick,
    wick on the line)
  * context bars 23:00..05:00 (closed), the 06:00 entry bar frozen at the
    2.8413 fill (06:27:47), and the 07:00 bar live to the last 5m close
  * chronos-2 quantile fan (12x5m = 60m) anchored on the 06:25 close
  * ground-truth 5m actuals from the anchor to now

EPOCHS verified: 06:25:00 UTC = 1787898300, 23:00 UTC (08-27) = 1787871600.
"""
import json
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UP, DOWN = "#26a69a", "#ef5350"
ENTRY = "#ffee58"
Q10, Q50, Q90, GT = "#64b5f6", "#4dd0e1", "#ffcc80", "#e0e0e0"
BG, GRID, BOX = "#101418", "#2a3140", "#1b2129"

OUT = "scratch/trade_charts/TRUMP_20260828_0627_long_chronos.png"

T2300 = 1787871600000  # 2026-08-27 23:00 UTC (chart origin, verified)
T0600 = 1787896800000  # 06:00
T0625 = 1787898300000  # 06:25
T0700 = 1787900400000  # 07:00
ENTRY_TS = 1787898467000  # 06:27:47 (ledger OPEN ts 1787898467210 ms)
ENTRY_PX = 2.8413
DECISION_TS = 1787898465000  # 06:27:45 (gate log line)


def post(payload):
    req = urllib.request.Request(
        "https://api.hyperliquid.xyz/info",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def x_of(ts_ms):
    return (ts_ms - T2300) / 60000.0  # minutes since 23:00


fc = json.load(open("/tmp/cf3_trump_chronos.json"))
anchor_close = fc["anchor_close"]
paths = fc["paths"]

# ── candles ─────────────────────────────────────────────────────────────────
h1 = post({"type": "candleSnapshot", "req": {
    "coin": "TRUMP", "interval": "1h",
    "startTime": T2300, "endTime": int(time.time() * 1000)}})
h1 = {c["t"]: c for c in h1}
m5 = post({"type": "candleSnapshot", "req": {
    "coin": "TRUMP", "interval": "5m",
    "startTime": T0600, "endTime": int(time.time() * 1000)}})
m5 = {c["t"]: c for c in m5}

bars = []  # (x0, x1, o, h, l, c)
t_last5 = max(m5.keys())  # latest 5m candle open
hour_live = (t_last5 // 3600000) * 3600000  # hour containing the latest data
# closed 1h bars 23:00 .. hour before the live one (the 06:00 entry bar is
# drawn separately, frozen at the 06:27:47 fill)
for t in range(T2300, hour_live, 3600000):
    if t == T0600:
        continue
    c = h1[t]
    o, h, l, cl = (float(c[k]) for k in "ohlc")
    bars.append((x_of(t) - 29, x_of(t) + 29, o, h, l, cl))

# 06:00 entry bar, FROZEN at the 06:27:47 fill: o from the 1h bar, h/l from
# the closed 5m bars 06:00..06:20 plus the entry tick, c = entry price
o6 = float(h1[T0600]["o"])
hs = [float(m5[t]["h"]) for t in range(T0600, T0625, 300000)]  # 06:00..06:20
ls = [float(m5[t]["l"]) for t in range(T0600, T0625, 300000)]
bars.append((x_of(T0600) - 29, x_of(ENTRY_TS),
             o6, max(hs + [ENTRY_PX]), min(ls + [ENTRY_PX]), ENTRY_PX))

# live-hour bar, partial to the last 5m close (may still be forming)
hs7 = [float(m5[t]["h"]) for t in m5 if t >= hour_live]
ls7 = [float(m5[t]["l"]) for t in m5 if t >= hour_live]
c7 = float(m5[t_last5]["c"])
bars.append((x_of(hour_live) - 29, x_of(t_last5 + 300000),
             float(m5[hour_live]["o"]), max(hs7), min(ls7), c7))

# ground-truth actuals since the 06:25 anchor
gt = sorted((t, float(m5[t]["c"])) for t in m5 if t >= T0625 and t <= t_last5)
gx = [x_of(t) for t, _ in gt]
gy = [p for _, p in gt]

# chronos fan
step_x = [x_of(T0625 + k * 300000) for k in range(fc["horizon_steps"])]
fan = {q: [anchor_close * (1 + p["pct"][k] / 100) for k in range(fc["horizon_steps"])]
       for q, p in paths.items()}

# ── plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14.5, 8.0), dpi=130)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.grid(True, color=GRID, linewidth=0.6)

for (x0, x1, o, h, l, cl) in bars:
    col = UP if cl >= o else DOWN
    wick_x = (x0 + x1) / 2
    ax.plot([wick_x, wick_x], [l, h], color=col, linewidth=0.9, zorder=3)
    ax.add_patch(plt.Rectangle((x0, min(o, cl)), x1 - x0, max(abs(cl - o), 0.0015),
                               facecolor=col, edgecolor=col, zorder=4))

ax.plot(gx, gy, color=GT, linewidth=1.6, linestyle=(0, (4, 2)), zorder=5,
        label="actual 5m (06:25 → now)")

ax.fill_between(step_x, fan["q10"], fan["q90"], color=Q50, alpha=0.15, zorder=2)
ax.plot(step_x, fan["q90"], color=Q90, linewidth=1.4, zorder=5, label="chronos q90 path")
ax.plot(step_x, fan["q50"], color=Q50, linewidth=1.9, zorder=5, label="chronos median path")
ax.plot(step_x, fan["q10"], color=Q10, linewidth=1.9, zorder=5, label="chronos q10 (adverse)")
k_trip = fc["q10_first6_argmin_idx"]
ax.plot([step_x[k_trip]], [fan["q10"][k_trip]], marker="v", color=DOWN,
        markersize=9, zorder=6)
ax.annotate(f"q10 first-6 min {fc['q10_first6_min_pct']:.2f}%\n"
            f"(gate threshold: −3.0%; gate logged −3.80%)",
            xy=(step_x[k_trip], fan["q10"][k_trip]),
            xytext=(step_x[k_trip] - 42, fan["q10"][k_trip] - 0.05),
            color=DOWN, fontsize=8.5, ha="center",
            arrowprops=dict(arrowstyle="->", color=DOWN),
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BOX, edgecolor=GRID))

x_dec = x_of(DECISION_TS)
ax.axvline(x_dec, color="#ff8a65", linewidth=1.0, linestyle=":", zorder=4)

ax.plot([x_of(ENTRY_TS)], [ENTRY_PX], marker="o", color=ENTRY, markersize=9, zorder=7)
ax.annotate("entry 2.8413 (long, $34, 5×)", xy=(x_of(ENTRY_TS), ENTRY_PX),
            xytext=(x_of(ENTRY_TS) - 155, 2.935), color=ENTRY, fontsize=8.5, ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BOX, edgecolor=GRID,
                      alpha=0.95))
ax.plot([x_of(T0625)], [anchor_close], marker="^", color="#90a4ae",
        markersize=7, zorder=6)

live_lbl = time.strftime("%H:%M", time.gmtime(hour_live / 1000))
txt = ("STATE AT 06:27 UTC\n"
       "band 1h: trending UP, 14.2% drift · px +4.4% vs upper edge\n"
       f"anchor 5m close 06:25: {anchor_close:.4f} (entry chased +1.7% above)\n"
       "trigger 4.3 · verdict LONG conf 0.76\n"
       f"chronos: median {fc['median_pct']:+.2f}% · spread {fc['spread_pct']:.1f}% · "
       f"q10 first-6 min {fc['q10_first6_min_pct']:.2f}%\n"
       "gates: chronos_tail_trigger SHADOW would-block (conf<0.90, comp<60)\n"
       "       → executed anyway (shadow mode) · squeeze: inside channel (pos 0.79)\n"
       f"bars: 23:00–05:00 closed · 06:00 frozen at entry · {live_lbl} live")
ax.text(0.008, 0.988, txt, transform=ax.transAxes, fontsize=8.2, color="#eceff1",
        va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=BOX, edgecolor=GRID,
                  alpha=0.92))

XMIN_P = -34
XMAX_P = max(530, x_of(t_last5 + 300000) + 8)
ax.set_xlim(XMIN_P, XMAX_P)
lo = min(min(b[4] for b in bars), min(fan["q10"]), 2.594 * 0.985)
hi = max(max(b[3] for b in bars), max(fan["q90"]), max(gy), 3.006 * 1.006) + 0.04
ax.set_ylim(lo, hi)
FR = (x_dec - XMIN_P) / (XMAX_P - XMIN_P)
ax.axhline(2.594, color=DOWN, linewidth=0.9, linestyle="--", alpha=0.7, xmin=FR)
ax.axhline(3.006, color=UP, linewidth=0.9, linestyle="--", alpha=0.7, xmin=FR)
ax.text(XMAX_P - 4, 2.594, "SL 2.594", color=DOWN, fontsize=7.5, ha="right", va="bottom")
ax.text(XMAX_P - 4, 3.006, "TP 3.006", color=UP, fontsize=7.5, ha="right", va="bottom")

# hourly ticks (x = minutes since 23:00) — each 1h body is centred on its
# start-time line; generate dynamically so the live hour gets a line too
ticks = list(range(T2300, hour_live + 3600000, 3600000))
ax.set_xticks([x_of(t) for t in ticks])
labels = []
for t in ticks:
    hh = time.gmtime(t / 1000).tm_hour
    labels.append(f"{hh:02d}:00")
ax.set_xticklabels(labels, color="#90a4ae", fontsize=8)
ax.text(x_dec - 3, hi - 0.012, "gate eval 06:27:45", color="#ff8a65", fontsize=8,
        ha="right", va="top")
ax.tick_params(axis="y", colors="#90a4ae", labelsize=8)
ax.set_title("TRUMP LONG 2026-08-28 06:27:45Z — decision state + chronos-2 forecast "
             f"(1h bodies centred on the hour line · 23:00–05:00 closed · 06:00 frozen at entry · "
             f"{live_lbl} live · fan 12×5m)",
             color="#eceff1", fontsize=11, loc="left")
ax.legend(loc="lower center", ncol=2, facecolor=BOX, edgecolor=GRID,
          labelcolor="#eceff1", fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT)
print("chart:", OUT, "| y-range", round(lo, 4), round(hi, 4),
      "| last actual", time.strftime('%H:%M', time.gmtime(gt[-1][0] / 1000)),
      gt[-1][1])