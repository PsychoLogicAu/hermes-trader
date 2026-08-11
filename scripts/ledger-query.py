#!/usr/bin/env python3
"""Build trade table with Chronos signals, LLM verdict/confidence, runner gate info.

Usage:
  python scripts/ledger-query.py            # uses default log path
  python scripts/ledger-query.py /path/to/log

Filters trades to the period since Chronos went live (Aug 10 2026 ~23:48 UTC).
"""
import re
import sys
from datetime import datetime

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "/home/oknight/src/hermes-trader/trader-logs/trader.log"
CUTOFF = datetime(2026, 8, 10, 23, 0, 0)  # before Chronos started

def ts(line):
    return datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")

with open(LOG_FILE) as f:
    lines = f.readlines()

# Step 1: Parse executed trades since Chronos started
executions = []
for i, line in enumerate(lines):
    if "'executed': True" not in line or "mode': 'LIVE'" not in line:
        continue
    t = ts(line)
    if t < CUTOFF:
        continue

    aid = re.search(r"analysis_id': '([a-f0-9-]+)'", line)
    if not aid:
        continue

    # Coin from liquidity bypass or research call
    coin = None
    for j in range(i, max(0, i-200), -1):
        bm = re.search(r"low-liquidity bypass on (\w+)", lines[j])
        if bm:
            coin = bm.group(1)
            break
        cm = re.search(r"Researching (\w+)\s*\(", lines[j])
        if cm:
            coin = cm.group(1)
            break
    if not coin:
        continue

    via = re.search(r"'via': '(\w+)'", line)
    executions.append({
        "ts": t,
        "ts_str": line[:19],
        "analysis_id": aid.group(1),
        "coin": coin,
        "regime_via": via.group(1) if via else None,
        "log_idx": i,
    })

# Step 2: LLM verdict/confidence from Verdict line (~1-2 lines before execution)
llm_info = {}
for exc in executions:
    start = max(0, exc["log_idx"] - 20)
    end = exc["log_idx"]
    verdict = None
    confidence = None
    reasoning = None
    structural_override = False

    for line in lines[start:end]:
        v_m = re.search(r"Verdict:\s*(\w+),\s*Confidence:\s*([\d\.]+)", line)
        if v_m:
            verdict = v_m.group(1)
            confidence = float(v_m.group(2))

    if verdict and confidence is not None:
        start2 = max(0, exc["log_idx"] - 100)
        for line in lines[start2:end]:
            r_m = re.search(r"'reasoning'\s*:\s*'((?:[^'\\]|\\.)+)'", line)
            if r_m:
                reasoning = r_m.group(1).replace("\\n", " ")[:100]
                if "[structural override]" in reasoning:
                    structural_override = True
                break

    llm_info[exc["analysis_id"]] = {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "structural_override": structural_override,
    }

# Step 3: Closed trades (DSL floor_breach/max_loss + loss cooldown armed)
closes = []
for line in lines:
    # DSL closes
    m = re.search(r"\[dsl\] Closing (\w+) (\w+) \(\d+x\): (.+?) \(margin ([+-]?\d+\.\d+)% · spot ([+-]?\d+\.\d+)%\)", line)
    if m:
        closes.append({
            "ts": ts(line),
            "ts_str": line[:19],
            "coin": m.group(1),
            "side": m.group(2),
            "reason": m.group(3),
            "margin_pct": float(m.group(4)),
            "spot_pct": float(m.group(5)),
        })

# Also add loss cooldown closes (fallback when DSL close not found)
for line in lines:
    m = re.search(r"loss cooldown armed on (\w+):\s+\d+min\s+\(closed\s+([+-]?\d+\.\d+)%\)", line)
    if m:
        coin = m.group(1)
        pnl = float(m.group(2))
        t = ts(line)
        # Only add if no DSL close for this coin around this time
        existing = [c for c in closes if c["coin"] == coin and abs((c["ts"] - t).total_seconds()) < 60]
        if not existing:
            closes.append({
                "ts": t,
                "ts_str": line[:19],
                "coin": coin,
                "side": "?",
                "reason": "loss_cooldown",
                "margin_pct": pnl,
                "spot_pct": pnl / 5.0,  # rough estimate
            })

# Step 4: Chronos signals near each trade entry
chronos_by_trade = {}
for exc in executions:
    coin = exc["coin"]
    entry_ts = exc["ts"]
    signals = []
    for line in lines:
        try:
            line_ts = ts(line)
        except:
            continue
        delta = (line_ts - entry_ts).total_seconds()
        if delta < 0 or delta > 3600:
            continue
        cm = re.search(r"\[chronos\]\s+(\w+)\s+\((long|short)\)\s+median=([\d\.]+)\s+\(([+-]?\d+\.\d+)%\)", line)
        if cm and cm.group(1) == coin:
            status_m = re.search(r"(↑ ALIGN|↓ MISMATCH)", line)
            if status_m:
                signals.append({
                    "ts": line_ts,
                    "median_change": float(cm.group(4)),
                    "status": status_m.group(1),
                })
    chronos_by_trade[exc["analysis_id"]] = sorted(signals, key=lambda s: s["ts"])

# Step 5: Output table
conf_fmt = lambda c: "%.2f" % c if c is not None else "N/A"
print(f"{'Coin':>8} | {'Open Time':>15} | {'Close Time':>15} | {'Verdict':>8} | {'Conf':>6} | {'Chronos @ Entry':>16} | {'Regime Via':>12} | {'Struct Override':>15} | {'P/L (margin)':>14} | {'Close Reason':<20}")
print("-" * 240)

for exc in executions:
    aid = exc["analysis_id"]
    llm = llm_info.get(aid, {})
    chronos = chronos_by_trade.get(aid, [])

    chrono_status = "-"
    chrono_detail = ""
    if chronos:
        first = chronos[0]
        chrono_status = first["status"]
        chrono_detail = " (%+.1f)" % first["median_change"]

    pnl_str = "-"
    close_time = "-"
    close_reason = "-"
    # Match to the first close >= this execution's timestamp
    for c in closes:
        if c["coin"] == exc["coin"] and c["ts"] >= exc["ts"]:
            pnl_str = f"{c['margin_pct']:+6.1f}%"
            close_time = c["ts_str"]
            close_reason = c["reason"][:20]
            break

    regime = exc["regime_via"] or "-"
    struct_ovr = "YES" if llm.get("structural_override") else "NO"
    verdict_str = llm.get("verdict") or "?"
    conf_str = conf_fmt(llm.get("confidence"))

    print("%8s | %15s | %15s | %8s | %6s | %8s%8s | %12s | %15s | %14s | %20s" % (
        exc["coin"],
        exc["ts_str"],
        close_time,
        verdict_str,
        conf_str,
        chrono_status,
        chrono_detail,
        regime,
        struct_ovr,
        pnl_str,
        close_reason,
    ))
