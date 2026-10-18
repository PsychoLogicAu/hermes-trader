#!/usr/bin/env python3
"""Run the A/B replay over the random-100 sample, first N rows only.

Same monkeypatch as ab_sample100.py (COHORT -> sample file) but passes
--limit N to cmd_replay so an arm only runs the first N todo rows.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import llm_ab_replay as ab  # noqa: E402

ab.COHORT = REPO / "scratch" / "_ab_cohort_sample100.jsonl"

ap = argparse.ArgumentParser()
ap.add_argument("--arms", required=True, help="comma list")
ap.add_argument("--limit", type=int, default=25)
ap.add_argument("--concurrency", type=int, default=2)
args = ap.parse_args()

for arm in args.arms.split(","):
    # systemone-branch flags (added to cmd_replay after this driver was
    # written): default OFF — plain model arm.
    ns = argparse.Namespace(arm=arm, limit=args.limit, concurrency=args.concurrency,
                            inject_vwap=False, fade_mode=None, strip_fields=None,
                            only_vwap_episodes=False, stride=1)
    print(f"\n===== SAMPLE{args.limit} arm={arm} =====", flush=True)
    ab.cmd_replay(ns)
    ab.cmd_audit(argparse.Namespace(arm=arm))
