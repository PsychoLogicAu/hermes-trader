#!/usr/bin/env python3
"""B.32 prior-call block A/B replay (systemone seat).

Arms (same archived request, the block is the ONLY delta):
  live  = the archived parsed verdict (what the bot did; no call needed)
  A     = replay with the state EXACTLY as archived (determinism self-check)
  B     = replay with state["recent_history"] = prior_block_text (the
          statefulness block the seat did not see)

Source rows: prompt-log role="primary" rows carrying prior_block_text
(written by the observability change, PR #101; rows before the deploy lack
the field and are skipped -- clean era clock starts at the deploy).

Verdict mapping reuses the LIVE path (decision_voice.verdict_from_answers)
so arms are shaped exactly like the bot's own verdicts. Raw answers are
archived so every threshold variant stays priceable offline.

Inertness caveat (server-side): if the decision-model prompt template does
not render the injected key, arm B == arm A on ~100% of rows. The script
prints agreement stats every 25 rows -- a==live ~100% (systemone is
near-deterministic) AND b==a ~100% means the injection never reached the
template, NOT that the block has no effect. Verify before trusting a null.

Run:  uv run python scripts/prior_block_ab_replay.py [--limit N]
          [--since YYYY-MM-DD] [--coins A,B] [--selfcheck] [--out FILE]
Output: JSONL rows -> trader-logs/prior_block_ab_rows.jsonl (append;
        dedupe key = perception_id + arm).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time

import httpx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from hermes_trader.agents.decision_voice import verdict_from_answers  # noqa: E402

PLOG = os.path.join(REPO, "trader-logs", "prompt-log")
LEDGER = os.path.join(REPO, "trader-logs", "trades.jsonl")
PROFILE = None  # resolved in main(): --profile or the LIVE llm.primary_engine seat
DEFAULT_OUT = os.path.join(REPO, "trader-logs", "prior_block_ab_rows.jsonl")


def resolve_profile(profile_arg: str | None) -> str:
    if profile_arg:
        return os.path.join(REPO, "config-models", f"{profile_arg}.json")
    seat = json.load(open(os.path.join(REPO, ".agent-config.json")))[
        "llm"]["primary_engine"]["model_profile"]
    return os.path.join(REPO, "config-models", f"{seat}.json")


def resolve_base_url(profile_path: str) -> str:
    prof = json.load(open(profile_path))
    base = prof["base_url"]  # e.g. http://d3-27b-serve:8090 (compose-internal)
    host = base.split("//", 1)[1].split(":")[0]
    if not all(c.isdigit() for c in host.split(".")):
        try:  # compose-internal hostname -> bridge IP (works from the host;
            # re-resolve every run — IPs change on restart)
            ip = subprocess.run(
                ["docker", "inspect", "-f",
                 "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                 host],
                capture_output=True, text=True, timeout=10).stdout.strip()
            if ip and not ip.startswith("invalid"):
                port = base.split(":")[-1]
                return f"http://{ip}:{port}"
        except Exception:
            pass
    return base.rstrip("/")


def load_rows(since_ms: int, coins: set[str] | None, limit: int, model: str):
    files = [os.path.join(PLOG, "calls.jsonl")] + sorted(
        glob.glob(os.path.join(PLOG, "calls-*.jsonl.zst")))
    out, seen = [], set()
    for f in files:
        opener = (lambda p: subprocess.run(["unzstd", "-c", p],
                                           capture_output=True, text=True).stdout.splitlines()
                  if f.endswith(".zst") else open(f, errors="replace"))
        for line in opener(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("role") != "primary" or not r.get("prior_block_text"):
                continue
            if model and r.get("model") != model:
                continue
            if r.get("ts", 0) < since_ms:
                continue
            if coins and r.get("coin") not in coins:
                continue
            k = r.get("perception_id")
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
    out.sort(key=lambda r: r["ts"])
    return out[:limit] if limit else out


def load_close_index():
    closes = {}
    try:
        for line in open(LEDGER):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") == "CLOSE":
                pid = r.get("perception_id") or r.get("analysis_id")
                if pid:
                    closes.setdefault(pid, r)
    except FileNotFoundError:
        pass
    return closes


def call_systemone(base, model, state, questions, timeout_s=30.0):
    resp = httpx.post(base + "/v1/systemone",
                      json={"model": model, "state": state, "questions": questions},
                      timeout=timeout_s)
    if not resp.is_success:
        return None
    ans = resp.json().get("answers")
    return ans if isinstance(ans, dict) and ans else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--since", default="")
    ap.add_argument("--coins", default="")
    ap.add_argument("--selfcheck", action="store_true",
                    help="arm A only (replay determinism check, no B calls)")
    ap.add_argument("--executed-only", action="store_true",
                    help="replay only rows whose perception_id has a ledger OPEN")
    ap.add_argument("--sample", type=int, default=0,
                    help="random sample of N rows AFTER filtering (seeded)")
    ap.add_argument("--skip-arm-a", action="store_true",
                    help="arm B only (arm A determinism already established; halves "
                         "load on the live seat) — for entry-add screening")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between rows (seat-share throttle on live endpoint)")
    ap.add_argument("--profile", default=None,
                    help="config-models seat (default: LIVE llm.primary_engine.model_profile)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    profile_path = resolve_profile(args.profile)
    prof = json.load(open(profile_path))
    model = prof["model"]

    since_ms = 0
    if args.since:
        since_ms = int(dt.datetime.strptime(args.since, "%Y-%m-%d")
                       .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    coins = {c.strip() for c in args.coins.split(",") if c.strip()} or None

    rows = load_rows(since_ms, coins, args.limit, model)
    done = set()
    if args.skip_existing and os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["perception_id"])
            except Exception:
                pass
    rows = [r for r in rows if r["perception_id"] not in done]
    if args.executed_only:
        exec_pids = set()
        for line in open(LEDGER):
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("event") == "OPEN":
                exec_pids.add(o.get("perception_id") or o.get("analysis_id"))
        rows = [r for r in rows if r["perception_id"] in exec_pids]
    if args.sample and len(rows) > args.sample:
        import random
        random.Random(42).shuffle(rows)
        rows = rows[:args.sample]
        rows.sort(key=lambda r: r["ts"])
    if not rows:
        print("no eligible rows (need prior_block_text — deploy PR #101 first?)")
        return 0

    base = resolve_base_url(profile_path)
    closes = load_close_index()
    print(f"{len(rows)} rows | target {base} model={model} | out {args.out}")

    n = a_match = b_match_a = conf_moved = 0
    with open(args.out, "a") as out:
        for r in rows:
            try:
                req = json.loads(r["user_message"])
                state, questions = req["state"], req["questions"]
            except Exception:
                continue
            pid = r["perception_id"]
            perc = {"id": pid, "mid": r.get("mid"),
                    "coin": r["coin"]}  # verdict_from_answers uses mid/coin only

            if args.skip_arm_a:
                ans_a, v_a = None, None
            else:
                ans_a = call_systemone(base, model, dict(state), questions)
                v_a = verdict_from_answers(ans_a, r["coin"], perc) if ans_a else None
                if v_a is None:
                    print(f"  arm A call failed {r['coin']} {pid}; continuing")
                    continue

            rec = {
                "ts": r["ts"], "coin": r["coin"], "perception_id": pid,
                "model": r.get("model"),
                "live": r.get("parsed"),
                "arm_a": None if v_a is None else {
                    "verdict": v_a["verdict"], "confidence": v_a["confidence"],
                    "answers": ans_a},
            }
            live_v = (r.get("parsed") or {}).get("verdict")
            live_c = (r.get("parsed") or {}).get("confidence")
            n += 1
            if v_a is not None:
                a_match += (v_a["verdict"] == live_v)

            if not args.selfcheck:
                st_b = dict(state)
                st_b["recent_history"] = r["prior_block_text"]
                ans_b = call_systemone(base, model, st_b, questions)
                if ans_b:
                    v_b = verdict_from_answers(ans_b, r["coin"], perc)
                    rec["arm_b"] = {"verdict": v_b["verdict"],
                                    "confidence": v_b["confidence"],
                                    "answers": ans_b}
                    # In --skip-arm-a the live verdict/confidence stand in for
                    # arm A (byte-exact determinism established on the
                    # executed cohort).
                    ref_v = v_a["verdict"] if v_a else live_v
                    ref_c = v_a["confidence"] if v_a else (live_c or 0.0)
                    b_match_a += (v_b["verdict"] == ref_v)
                    conf_moved += abs(v_b["confidence"] - ref_c) >= 0.005
                else:
                    rec["arm_b"] = None

            c = closes.get(pid)
            if c is not None:
                rec["executed_pnl_usd"] = c.get("realized_pnl_usd")
                rec["exit_reason"] = c.get("exit_reason") or c.get("exit_type")

            out.write(json.dumps(rec) + "\n")
            if args.sleep:
                time.sleep(args.sleep)
            if n % 25 == 0:
                print(f"  n={n} a==live {a_match}/{n}  b==ref {b_match_a}/{n}  "
                      f"conf-moved {conf_moved}/{n}")
    print(f"DONE n={n} a==live {a_match}/{n} b==ref {b_match_a}/{max(n,1)} "
          f"conf-moved {conf_moved}/{max(n,1)}")
    if n and b_match_a == n and conf_moved == 0 and not args.selfcheck:
        print("WARNING: arm B identical to reference on every row AND zero conf "
              "movement — injection may be INERT server-side. Do not read as null effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
