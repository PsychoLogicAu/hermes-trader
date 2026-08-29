"""SKR 2026-08-28 12:57:17Z LONG: replay the chronos-2 forecast on the
decision-time context (last 100 closed 5m bars up to the 12:55 bar),
exactly as the shadow gate computed it. Container-only (torch CPU).
Writes /app/cf3_skr_chronos.json for the host-side plotter.

Validated against the logged gate line: q10 first-6 min = -4.70%,
median +0.42% (conf 0.78, composite 32.9).
"""
import json
import sys
import time
import urllib.request
from calendar import timegm
from datetime import datetime, timezone

sys.path.insert(0, "/app")
import torch  # noqa: E402
from hermes_trader.agents.chronos_signal import (  # noqa: E402
    _find_quantile_index,
    _get_chronos_config,
    _get_pipeline,
)

COIN = "SKR"
# The gate at 12:57:17 was a cache hit on the signal COMPUTED at 12:54:25
# (the [chronos] INFO line: median +0.42%, matching the gate's logged median
# 0.4%; TTL 300s still valid at 12:57:17). At 12:54:25 the forming 5m bar's
# open time was 12:50, so the anchor = the 12:50 bar's (finalized) close.
DECISION_UTC = (2026, 8, 28, 12, 50, 0)   # anchor bar open time
ANCHOR_TS = int(timegm(DECISION_UTC)) * 1000


def fetch_5m(n=130):
    req = urllib.request.Request(
        "https://api.hyperliquid.xyz/info",
        data=json.dumps({"type": "candleSnapshot",
                         "req": {"coin": COIN, "interval": "5m",
                                 "startTime": ANCHOR_TS - n * 300_000,
                                 "endTime": ANCHOR_TS + 300_000}}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=20))


candles = [c for c in fetch_5m(150) if c["t"] <= ANCHOR_TS]
candles = candles[-100:]
closes = [float(c["c"]) for c in candles]
last_close = closes[-1]
assert candles[-1]["t"] == ANCHOR_TS, f"anchor bar missing: {candles[-1]['t']}"

cfg = _get_chronos_config()
horizon = int(cfg.get("forecast_horizon", 48))
pipeline = _get_pipeline()
mq = pipeline.quantiles
i_low, i_med, i_high = (
    _find_quantile_index(mq, q) for q in (0.1, 0.5, 0.9))

t0 = time.time()
fc = pipeline.predict([torch.tensor(closes, dtype=torch.float32)],
                      prediction_length=horizon)[0]
ms = (time.time() - t0) * 1000

paths = {}
for name, idx in (("q10", i_low), ("q50", i_med), ("q90", i_high)):
    p = [float(v) for v in fc[0, idx, :].tolist()]
    paths[name] = {
        "price": p,
        "pct": [((v - last_close) / last_close * 100) for v in p],
    }

out = {
    "coin": COIN,
    "anchor_bar_utc": "2026-08-28T12:50:00Z",
    "anchor_close": last_close,
    "decision_utc": "2026-08-28T12:57:17Z (gate line; signal computed 12:54:25, cache hit)",
    "horizon_steps": horizon,
    "inference_ms": round(ms, 1),
    "median_pct": (sum(paths["q50"]["price"]) / horizon - last_close) / last_close * 100,
    "spread_pct": (sum(paths["q90"]["price"]) / horizon - sum(paths["q10"]["price"]) / horizon) / last_close * 100,
    "q10_first6_min_pct": min(paths["q10"]["pct"][:6]),
    "q10_first6_argmin_idx": paths["q10"]["pct"][:6].index(min(paths["q10"]["pct"][:6])),
    "paths": paths,
    "logged_gate_values": {"tail_pct": -4.7, "median_pct": 0.4,
                           "conf": 0.78, "composite": 32.9},
}
json.dump(out, open("/app/cf3_skr_chronos.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "paths"}, indent=1))