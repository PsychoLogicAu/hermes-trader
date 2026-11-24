#!/usr/bin/env python3
"""Config guard: the confidence-gate PAIRING RULE (t_5f7a5b5b).

Prevents one-sided short/long bar retunes: an edit to a seat profile's
side-specific bar (entry_min_short_confidence / entry_min_confidence) that
leaves the CLASSIC confidence gate (min_ai_confidence) or the runner gate's
general bar free to re-veto — or silently bypass — the admits the profile
intends. Pairing rule origin: hermes-trader-config commit 5ea1d4c ("classic
min_ai_confidence AND runner min_confidence moved together so one gate can't
re-block the other's admits"); the cost of violating it was measured by the
t_552a0b2c audit: 1,211 deduped `confidence 0.6x < 0.65` classic-gate
re-vetoes of 0.60-0.64 SHORTs the mammoth-1 seat bar admits, 67 sole-blocker
victims.

WHAT IS CHECKED (per answering seat actually referenced by the config's
`llm` slots, matched by profile file stem or profile "model" key):

  MC  = effective classic floor      = gate_bars.min_ai_confidence (profile
                                       overlay beats top-level, per
                                       apply_seat_gate_bars) else top-level
                                       min_ai_confidence else default 0.8
  GEN = effective runner general bar = gate_bars.runner_entry_gate
                                       .min_confidence else top-level else 0.70
  SB_s / SB_l = profile entry_min_short_confidence / entry_min_confidence
                                       (the seat's INTENT for each side)

DEPLOYED semantics (running image predates the conf-floor-precedence code
cc86fe00 — the classic gate reads the side-agnostic MC; the runner seat-bar
branch fires only while conf < GEN):
  SHORT: runner gate has its own short-confidence check at SB_s, so the
         system short floor is max(MC, SB_s).
         [SHORT-PAIRING]     error iff MC > SB_s  (classic re-vetoes [SB_s, MC))
  LONG:  no side-specific check exists for longs; the system long floor is
         max(MC, min(GEN, SB_l)).
         [LONG-PAIRING]      error iff MC > SB_l  (classic re-vetoes [SB_l, MC))
         [LONG-ENFORCEMENT]  error iff max(MC, min(GEN, SB_l)) < SB_l, i.e.
                             GEN and MC both sit BELOW the seat's long floor —
                             the profile bar goes inert and sub-floor LONGs
                             pass BOTH gates.

PRECEDENCE semantics (image contains effective_conf_floor / cc86fe00+): the
classic and runner floors resolve side-aware with the profile bar winning, so
the system floor EQUALS SB per side by construction — the veto/bypass checks
are structurally satisfied and only [BAR-SANITY] applies. (Once cc86fe00 is
deployed the audit end-state GEN=0.60 becomes the SAFE shape; that is why the
LONG-ENFORCEMENT rule is mode-aware instead of absolute.)

[BAR-SANITY] applies in every mode: an explicit side bar that is
non-numeric or outside (0, 1] is silently fail-opened by the resolver — the
bar would be inert without an error anywhere.

Override: an explicit runner_entry_gate.cohort_admissions key for THIS seat
("<seat-model>/<cohort-tag>") waives the [SHORT-PAIRING]/[LONG-PAIRING]
veto rules for it (the cohort lane admits around the bars by design).

Usage (the config repo's scripts/deploy.sh runs this as a pre-deploy gate;
check_drift.sh runs it against the live files):
  python3 gate_pairing_check.py --config .agent-config.json \
      --models config-models [--mode auto|deployed|precedence] \
      [--container hermes-trader] [--all-profiles]
Exit: 0 = OK, 1 = pairing violations, 2 = usage/IO error.

Pure stdlib: importable from tests and callable from the config repo without
the hermes_trader package on sys.path (the overlay/algebra here mirrors
hermes_trader/agents/config_store.py; tests/test_gate_pairing_guard.py pins
that mirror).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

EPS = 1e-9
DEFAULT_CLASSIC_FLOOR = 0.8   # risk_gates eval_all_gates default
DEFAULT_RUNNER_GENERAL = 0.70  # executor _runner_entry_block_reason default


def _num(raw: Any) -> Optional[float]:
    """Same numeric contract as config_store.effective_conf_floor._num."""
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if 0.0 < val <= 1.0 else None


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def overlay_gate_bars(config: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror of config_store.apply_seat_gate_bars (sub-key merge for dict
    values, scalar replace otherwise)."""
    bars = profile.get("gate_bars")
    if not isinstance(bars, dict) or not bars:
        return config
    out = dict(config)
    for key, bar in bars.items():
        base = out.get(key)
        if isinstance(base, dict) and isinstance(bar, dict):
            out[key] = {**base, **bar}
        else:
            out[key] = bar
    return out


def referenced_seats(config: Dict[str, Any]) -> List[str]:
    """Model/profile names referenced by the config's llm slots (recursive:
    primary, duelist+fallback, decision_voice, ...). Matches profile FILE
    stems and profile 'model' values case-insensitively, like
    read_model_profile_for_model."""
    names: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("model_profile", "model") and isinstance(v, str) and v.strip():
                    names.append(v.strip().lower())
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(config.get("llm"))
    return names


def check_seat(stem: str, profile: Dict[str, Any], cfg: Dict[str, Any],
               mode: str) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    notes: List[str] = []
    model = str(profile.get("model") or stem).strip().lower()
    src = f"config-models/{stem}.json"

    sb_s_raw, sb_l_raw = profile.get("entry_min_short_confidence"), profile.get("entry_min_confidence")
    if sb_s_raw is None and sb_l_raw is None:
        return errors, notes  # no side-specific intent on this seat: nothing to pair

    # [BAR-SANITY] — explicit side bars must actually bind (resolver fail-opens
    # silently on malformed/out-of-range values).
    for label, raw in (("entry_min_short_confidence", sb_s_raw), ("entry_min_confidence", sb_l_raw)):
        if raw is not None and _num(raw) is None:
            errors.append(
                f"[gate-pairing][BAR-SANITY] seat '{model}' ({src}): {label}={raw!r} is "
                f"not a number in (0, 1] — the gate resolvers fail OPEN past it, so the "
                f"bar is inert and the general bars bind instead. Fix the value or delete the key."
            )
    sb_s, sb_l = _num(sb_s_raw), _num(sb_l_raw)

    eff = overlay_gate_bars(cfg, profile)
    mc = _num(eff.get("min_ai_confidence"))
    if mc is None:
        mc = DEFAULT_CLASSIC_FLOOR
    gate = eff.get("runner_entry_gate")
    gate = gate if isinstance(gate, dict) else {}
    gen = _num(gate.get("min_confidence"))
    if gen is None:
        gen = DEFAULT_RUNNER_GENERAL

    adm = gate.get("cohort_admissions") or {}
    cohort_override = any(
        "/" in str(k) and str(k).split("/", 1)[0].strip().lower() in (stem.lower(), model)
        for k in adm
    )
    if cohort_override:
        notes.append(
            f"[gate-pairing] seat '{model}': explicit cohort_admissions override present "
            f"({', '.join(sorted(str(k) for k in adm if str(k).split('/', 1)[0].strip().lower() in (stem.lower(), model)))}) "
            f"— pairing VETO rules waived for this seat"
        )

    rule = ("PAIRING RULE (config-repo 5ea1d4c + t_552a0b2c): the classic gate "
            "(min_ai_confidence) and the runner gate must not sit above a seat's "
            "side bar, or they silently re-veto every admit the profile intends")

    if mode == "deployed":
        if sb_s is not None and mc > sb_s + EPS:
            msg = (f"[gate-pairing][SHORT-PAIRING] seat '{model}' ({src}): effective classic floor "
                   f"min_ai_confidence={mc:.2f} (after profile gate_bars overlay) > profile "
                   f"entry_min_short_confidence={sb_s:.2f} — the CLASSIC gate re-vetoes every SHORT "
                   f"in [{sb_s:.2f}, {mc:.2f}) the seat bar admits ({rule}). "
                   f"Fix: lower min_ai_confidence in BOTH .agent-config.json and the profile's "
                   f"gate_bars to <= {sb_s:.2f}, or add an explicit "
                   f"runner_entry_gate.cohort_admissions override for this seat.")
            if cohort_override:
                notes.append(msg.replace("[gate-pairing][SHORT-PAIRING]", "[gate-pairing][SHORT-PAIRING][WAIVED]"))
            else:
                errors.append(msg)
        if sb_l is not None and mc > sb_l + EPS:
            msg = (f"[gate-pairing][LONG-PAIRING] seat '{model}' ({src}): effective classic floor "
                   f"min_ai_confidence={mc:.2f} > profile entry_min_confidence={sb_l:.2f} — the "
                   f"CLASSIC gate re-vetoes every LONG in [{sb_l:.2f}, {mc:.2f}) the seat admits "
                   f"({rule}). Fix: lower min_ai_confidence (top-level AND profile gate_bars) to "
                   f"<= {sb_l:.2f}, or add an explicit cohort_admissions override for this seat.")
            if cohort_override:
                notes.append(msg.replace("[gate-pairing][LONG-PAIRING]", "[gate-pairing][LONG-PAIRING][WAIVED]"))
            else:
                errors.append(msg)
        if sb_l is not None:
            floor_eff = max(mc, min(gen, sb_l))
            if floor_eff < sb_l - EPS:
                errors.append(
                    f"[gate-pairing][LONG-ENFORCEMENT] seat '{model}' ({src}): profile "
                    f"entry_min_confidence={sb_l:.2f} is INERT under deployed code — both the "
                    f"effective classic floor min_ai_confidence={mc:.2f} and the runner general "
                    f"bar runner_entry_gate.min_confidence={gen:.2f} sit BELOW it, so LONGs in "
                    f"[{floor_eff:.2f}, {sb_l:.2f}) pass BOTH gates (the runner seat-bar branch "
                    f"only fires while min_confidence >= the seat bar). Fix: raise "
                    f"runner_entry_gate.min_confidence (top-level AND profile gate_bars) to >= "
                    f"{sb_l:.2f}, or deploy the conf-floor-precedence image (cc86fe00+) and "
                    f"validate with --mode precedence."
                )
    # mode == "precedence": effective_conf_floor makes the side bars bind at
    # BOTH gates by construction (profile side bar > config side bar > general),
    # so veto/bypass are structurally impossible — only BAR-SANITY applies.

    notes.append(
        f"[gate-pairing] seat '{model}': MC(classic)={mc:.2f} GEN(runner general)={gen:.2f} "
        f"SB_short={sb_s if sb_s is not None else '-'} SB_long={sb_l if sb_l is not None else '-'}"
    )
    return errors, notes


def resolve_mode(args: argparse.Namespace) -> Tuple[str, str]:
    if args.mode != "auto":
        return args.mode, f"mode={args.mode} (explicit)"
    try:
        rc = subprocess.run(
            ["docker", "exec", args.container, "grep", "-q", "effective_conf_floor",
             "/app/hermes_trader/agents/config_store.py"],
            capture_output=True, timeout=15,
        ).returncode
        if rc == 0:
            return "precedence", f"mode=precedence (container '{args.container}' HAS effective_conf_floor)"
        if rc == 1:
            return "deployed", f"mode=deployed (container '{args.container}' LACKS effective_conf_floor)"
        raise RuntimeError(f"grep rc={rc}")
    except Exception as e:  # noqa: BLE001
        return "deployed", (f"mode=deployed (default — could not probe container "
                            f"'{args.container}': {e}; pass --mode explicitly if the "
                            f"conf-floor-precedence code is deployed)")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="confidence-gate pairing-rule config guard (t_5f7a5b5b)")
    ap.add_argument("--config", default=".agent-config.json")
    ap.add_argument("--models", default="config-models")
    ap.add_argument("--mode", choices=("auto", "deployed", "precedence"), default="auto")
    ap.add_argument("--container", default="hermes-trader")
    ap.add_argument("--all-profiles", action="store_true",
                    help="check every profile file, not just seats referenced by the llm slots")
    ap.add_argument("--quiet", action="store_true", help="only print violations + summary")
    args = ap.parse_args(argv)

    try:
        cfg = load_json(args.config)
    except Exception as e:  # noqa: BLE001
        print(f"gate-pairing: cannot read {args.config}: {e}", file=sys.stderr)
        return 2
    if not isinstance(cfg, dict):
        print(f"gate-pairing: {args.config} is not an object", file=sys.stderr)
        return 2

    try:
        files = sorted(os.listdir(args.models))
    except OSError as e:
        print(f"gate-pairing: cannot read models dir {args.models}: {e}", file=sys.stderr)
        return 2

    names = set(referenced_seats(cfg))
    checked = 0
    errors: List[str] = []
    notes: List[str] = []
    mode, mode_note = resolve_mode(args)
    for fn in files:
        if not fn.endswith(".json"):
            continue
        stem = fn[:-5]
        try:
            prof = load_json(os.path.join(args.models, fn))
        except Exception as e:  # noqa: BLE001
            errors.append(f"[gate-pairing][BAR-SANITY] profile {fn} unreadable: {e}")
            continue
        if not isinstance(prof, dict):
            continue
        if not args.all_profiles:
            prof_model = str(prof.get("model") or "").strip().lower()
            if stem.lower() not in names and prof_model not in names:
                continue
        checked += 1
        e, n = check_seat(stem, prof, cfg, mode)
        errors += e
        notes += n

    for line in ([] if args.quiet else notes):
        print(line)
    print(mode_note)
    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        print(f"REFUSE: {len(errors)} gate-pairing violation(s) — the pairing rule "
              f"(5ea1d4c: gates move together) is violated; fix the bars or use an "
              f"explicit override.", file=sys.stderr)
        return 1
    print(f"gate-pairing OK: {checked} referenced seat(s) checked, no pairing violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
