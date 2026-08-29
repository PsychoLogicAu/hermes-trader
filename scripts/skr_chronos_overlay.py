"""SKR 2026-08-28 12:57:19Z LONG — decision state + chronos-2 forecast overlay.

HL-interface aligned:
  * x = minutes-of-day (zero epoch offsets; verified below)
  * 1h bodies centred on their start-time hour line
  * context bars 07:00..11:00 (closed), 12:00 entry bar frozen at the
    12:57:19 fill, 13:00 bar live to the 13:45:19 exit (max_loss)
  * chronos-2 quantile fan (12x5m = 60m) anchored on the 12:50 close
    (the signal the 12:57:17 gate line read: computed 12:54:25, cache hit)
  * ground-truth 5m actuals from the anchor to the exit

EPOCHS verified: 12:50:00 UTC = 1787921400, 07:00 UTC = 1787900400.
"""
import json
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

UP, DOWN = "#26a69a", "#ef5350"
ENTRY, EXIT = "#ffee58", "#ff80ab"
Q10, Q50, Q90, GT = "#64b5f6", "#4dd0e1", "#ffcc80", "#e0e0e0"
BG, GRID, BOX = "#101418", "#2a3140", "#1b2129"

OUT = "scratch/trade_charts/SKR_20260828_1257_long_chronos.png"

T0700 = 1787900400000   # 2026-08-28 07:00 UTC (chart origin)
T1200 = 1787918400000   # 12:00
T1250 = 1787921400000   # anchor 5m bar open
ENTRY_TS = 1787921839783  # 12:57:19.783 (ledger OPEN ts)
ENTRY_PX = 0.009861
EXIT_TS = 1787924719547   # 13:45:19.547 (ledger CLOSE ts, max_loss)
EXIT_PX = 0.009348
DECISION_TS = 1787921837000  # 12:57:17 (gate log line)
SL_PX, TP_PX = 0.00897578678151055, 0.010451142145659634


def post(payload):
    req = urllib.request.Request(
        "https://api.hyperliquid.xyz/info",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def x_of(ts_ms):
    return (ts_ms - T0700) / 60000.0  # minutes since 07:00


fc = json.load(open("/tmp/cf3_skr_chronos.json"))
anchor_close = fc["anchor_close"]
paths = fc["paths"]

# ── candles ─────────────────────────────────────────────────────────────────
h1 = post({"type": "candleSnapshot", "req": {
    "coin": "SKR", "interval": "1h",
    "startTime": T0700, "endTime": int(time.time() * 1000)}})
h1 = {c["t"]: c for c in h1}
m5 = post({"type": "candleSnapshot", "req": {
    "coin": "SKR", "interval": "5m",
    "startTime": T1200, "endTime": EXIT_TS + 300_000}})
m5 = {c["t"]: c for c in m5}

bars = []  # (x0, x1, o, h, l, c)
# closed 1h bars 07:00 .. 11:00
for t in range(T0700, T1200, 3600000):
    c = h1[t]
    o, h, l, cl = (float(c[k]) for k in "ohlc")
    bars.append((x_of(t) - 29, x_of(t) + 29, o, h, l, cl))

# 12:00 entry bar, FROZEN at the 12:57:19 fill: o from the 1h bar, h/l from
# the closed 5m bars 12:00..12:45 plus the entry tick, c = entry price
o12 = float(h1[T1200]["o"])
hs = [float(m5[t]["h"]) for t in range(T1200, T1250, 300000)]  # 12:00..12:45
ls = [float(m5[t]["l"]) for t in range(T1200, T1250, 300000)]
bars.append((x_of(T1200) - 29, x_of(ENTRY_TS),
             o12, max(hs + [ENTRY_PX]), min(ls + [ENTRY_PX]), ENTRY_PX))

# 13:00 bar, partial up to the exit tick (o from 5m, h/l/c from 5m closes +
# the exit tick)
hs3 = [float(m5[t]["h"]) for t in m5 if t >= 1787922000000 and t <= 1787924400000]
ls3 = [float(m5[t]["l"]) for t in m5 if t >= 1787922000000 and t <= 1787924400000]
bars.append((x_of(1787922000000) - 29, x_of(EXIT_TS),
             float(m5[1787922000000]["o"]),
             max(hs3 + [EXIT_PX]), min(ls3 + [EXIT_PX]), EXIT_PX))

# ground-truth actuals since the 12:50 anchor, to the exit
gt = sorted((t, float(m5[t]["c"])) for t in m5 if T1250 <= t <= 1787924400000)
gt.append((EXIT_TS, EXIT_PX))
gx = [x_of(t) for t, _ in gt]
gy = [p for _, p in gt]

# chronos fan
step_x = [x_of(T1250 + k * 300000) for k in range(fc["horizon_steps"])]
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
    ax.add_patch(plt.Rectangle((x0, min(o, cl)), x1 - x0, max(abs(cl - o), 0.000015),
                               facecolor=col, edgecolor=col, zorder=4))

ax.plot(gx, gy, color=GT, linewidth=1.6, linestyle=(0, (4, 2)), zorder=5,
        label="actual 5m (12:50 → exit)")

ax.fill_between(step_x, fan["q10"], fan["q90"], color=Q50, alpha=0.15, zorder=2)
ax.plot(step_x, fan["q90"], color=Q90, linewidth=1.4, zorder=5, label="chronos q90 path")
ax.plot(step_x, fan["q50"], color=Q50, linewidth=1.9, zorder=5, label="chronos median path")
ax.plot(step_x, fan["q10"], color=Q10, linewidth=1.9, zorder=5, label="chronos q10 (adverse)")
k_trip = fc["q10_first6_argmin_idx"]
ax.plot([step_x[k_trip]], [fan["q10"][k_trip]], marker="v", color=DOWN,
        markersize=9, zorder=6)
ax.annotate(f"q10 first-6 min {fc['q10_first6_min_pct']:.2f}% "
            f"(gate logged −4.70%; threshold −3.0%)",
            xy=(step_x[k_trip], fan["q10"][k_trip]),
            xytext=(step_x[k_trip] - 150, fan["q10"][k_trip] + 0.00025),
            color=DOWN, fontsize=8.5, ha="center",
            arrowprops=dict(arrowstyle="->", color=DOWN),
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BOX, edgecolor=GRID))

x_dec = x_of(DECISION_TS)
ax.axvline(x_dec, color="#ff8a65", linewidth=1.0, linestyle=":", zorder=4)

ax.plot([x_of(ENTRY_TS)], [ENTRY_PX], marker="o", color=ENTRY, markersize=9, zorder=7)
ax.annotate("entry 0.009861 (long, $34, 3×)", xy=(x_of(ENTRY_TS), ENTRY_PX),
            xytext=(x_of(ENTRY_TS) + 8, ENTRY_PX + 0.00022), color=ENTRY, fontsize=8.5,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BOX, edgecolor=GRID,
                      alpha=0.95))
ax.plot([x_of(EXIT_TS)], [EXIT_PX], marker="x", color=EXIT, markersize=11,
        markeredgewidth=2.2, zorder=7)
ax.annotate("exit 0.009348 · max_loss (−5.2% spot, −$1.80, 48m)",
            xy=(x_of(EXIT_TS), EXIT_PX),
            xytext=(x_of(EXIT_TS) - 185, EXIT_PX - 0.00042), color=EXIT, fontsize=8.5,
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BOX, edgecolor=GRID,
                      alpha=0.95))
ax.plot([x_of(T1250)], [anchor_close], marker="^", color="#90a4ae",
        markersize=7, zorder=6)

txt = ("STATE AT 12:57 UTC (22:57 AEST)\n"
       "band 1h EMA16: breakout trigger 32.9 · 15.3σ volume spike\n"
       f"anchor 5m close 12:50: {anchor_close:.6f} (entry +0.34% above)\n"
       "verdict SPLIT — primary LONG conf 0.78 vs duelist PASS conf 0.00\n"
       f"chronos (cache 12:54, median {fc['median_pct']:+.2f}%): q10 first-6 min "
       f"{fc['q10_first6_min_pct']:.2f}% vs logged −4.70%\n"
       "gates: chronos_tail_trigger SHADOW would-block (conf 0.78 < 0.90, comp 32.9 < 60)\n"
       "       → executed anyway (shadow mode) · squeeze: inside channel (pos 0.57)\n"
       "bars: 07:00–11:00 closed · 12:00 frozen at entry · 13:00 partial to exit")
ax.text(0.008, 0.988, txt, transform=ax.transAxes, fontsize=8.2, color="#eceff1",
        va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=BOX, edgecolor=GRID,
                  alpha=0.92))

XMIN_P, XMAX_P = -34, x_of(EXIT_TS) + 35
ax.set_xlim(XMIN_P, XMAX_P)
lo = min(min(b[4] for b in bars), min(fan["q10"]), SL_PX) * 0.999
# ~19% headroom above the data so the top-left state box (axis 0.85–0.99)
# sits in empty space — the 07:00–11:00 candles run ~0.0108, i.e. at the top
# of the data range, and would otherwise overlap the box.
data_hi = max(max(b[3] for b in bars), max(fan["q90"]), max(gy), TP_PX)
hi = lo + (data_hi - lo) / 0.81
ax.set_ylim(lo, hi)
FR = (x_dec - XMIN_P) / (XMAX_P - XMIN_P)
ax.axhline(SL_PX, color=DOWN, linewidth=0.9, linestyle="--", alpha=0.7, xmin=FR)
ax.axhline(TP_PX, color=UP, linewidth=0.9, linestyle="--", alpha=0.7, xmin=FR)
ax.text(XMAX_P - 4, SL_PX, f"SL {SL_PX:.6f}", color=DOWN, fontsize=7.5, ha="right", va="bottom")
ax.text(XMAX_P - 4, TP_PX, f"TP {TP_PX:.6f}", color=UP, fontsize=7.5, ha="right", va="bottom")

ticks = list(range(T0700, 1787925600000, 3600000))  # 07:00 .. 14:00
ax.set_xticks([x_of(t) for t in ticks])
labels = []
for t in ticks:
    hh = time.gmtime(t / 1000).tm_hour
    labels.append(f"{hh:02d}:00")
ax.set_xticklabels(labels, color="#90a4ae", fontsize=8)
ax.text(x_dec - 3, hi * 0.998, "gate eval 12:57:17", color="#ff8a65", fontsize=8,
        ha="right", va="top")
ax.text(x_of(EXIT_TS) + 3, hi * 0.998, "exit 13:45", color=EXIT, fontsize=8,
        ha="left", va="top")

ax.set_title("SKR LONG 2026-08-28 12:57 UTC — chronos-2 forecast at decision vs actual\n"
             "band breakout 32.9 · SPLIT (primary LONG 0.78 / duelist PASS 0.00) · "
             "chronos_tail_trigger would-block — trade ran anyway (shadow) → max_loss −5.2%",
             color="#eceff1", fontsize=10.5)
ax.set_ylabel("price (USD)", color="#90a4ae", fontsize=9)
ax.tick_params(colors="#90a4ae", labelsize=8)
ax.legend(loc="lower right", fontsize=8, facecolor=BOX, edgecolor=GRID,
          labelcolor="#eceff1", framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT, facecolor=BG)
print("wrote", OUT)