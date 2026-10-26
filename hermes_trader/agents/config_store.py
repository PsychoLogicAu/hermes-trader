"""Read/write the agent config at .agent-config.json."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Use absolute path based on this file's location (hermes-trader project root)
# __file__ = .../hermes-trader/hermes_trader/agents/config_store.py
# Go up 3 levels: agents/ → hermes_trader/ → hermes-trader/
# Override with HERMES_AGENT_CONFIG_FILE when deploying behind a mounted volume.
_CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.environ.get(
    "HERMES_AGENT_CONFIG_FILE",
    os.path.join(_CONFIG_DIR, ".agent-config.json"),
)

# ── Model profiles (config-repo split 2026-10-14) ────────────────────────
# Model-SPECIFIC tuning (model id, sampling, per-engine trap thresholds,
# fade-shape choice, per-model endpoint) lives in one file per model under
# MODEL_CONFIG_DIR, referenced from the main JSON by name:
#
#   "llm": { "primary_engine": {"engine": "systemone",
#                               "model_profile": "decision2-lux-9b", ...} }
#
# and config-models/decision2-lux-9b.json holds the tuning. Merge rule:
# the profile file is the BASE, inline slot keys OVERRIDE it (so an
# emergency one-off override never requires editing the profile). A slot
# without "model_profile" is returned untouched — fully back-compat.
# Fail-open: unreadable/corrupt profile logs LOUD and degrades to the
# inline keys only (a seat with no model key then behaves per its own
# absent-config policy, e.g. primary_engine_config stays "llm").
MODEL_CONFIG_DIR = os.environ.get(
    "MODEL_CONFIG_DIR",
    os.path.join(os.path.dirname(CONFIG_PATH) or ".", "config-models"),
)


def read_model_profile(name: str) -> Dict[str, Any]:
    """Read config-models/<name>.json. Fail-open: {} on any problem, with a
    LOUD log — callers merge it as the base layer."""
    name = str(name or "").strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        logger.warning(f"[config] model_profile name {name!r} invalid — ignored")
        return {}
    path = os.path.join(MODEL_CONFIG_DIR, f"{name}.json")
    try:
        with open(path, "r") as f:
            profile = json.load(f)
        if isinstance(profile, dict):
            return profile
        logger.warning(f"[config] model profile {path} is not an object — ignored")
    except FileNotFoundError:
        logger.warning(
            f"[config] model_profile '{name}' not found at {path} — using inline "
            f"slot keys ONLY (a seat needing model/base_url goes to its absent-config policy)"
        )
    except Exception as e:  # noqa: BLE001 — fail-open by contract
        logger.warning(f"[config] model profile {path} unreadable ({e}) — ignored")
    return {}


def read_model_profile_for_model(model_name: str) -> Dict[str, Any]:
    """Resolve a seat's ANSWERING model name to its profile file.

    The analysis dict carries the model name (engine slot "model" /
    llm.primary.model), which may differ from the profile FILE name (e.g.
    "Qwen3.8-27B-..." lives in qwen3.8-27b-iq2xs.json). Match order: exact
    file name, then any profile whose "model" key matches
    case-insensitively. Fail-open: {} when nothing matches (silent — this
    runs on every analysis; a seat simply having no profile is normal).
    """
    name = str(model_name or "").strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return {}
    if os.path.exists(os.path.join(MODEL_CONFIG_DIR, f"{name}.json")):
        return read_model_profile(name)
    low = name.lower()
    try:
        entries = sorted(os.listdir(MODEL_CONFIG_DIR))
    except OSError:
        return {}
    for fn in entries:
        if not fn.endswith(".json"):
            continue
        prof = read_model_profile(fn[:-5])
        if str(prof.get("model") or "").strip().lower() == low:
            return prof
    return {}


def resolve_model_slot(slot: Any) -> Any:
    """Expand a slot's ``model_profile`` reference (hot, no cache).

    Returns the slot unchanged when it isn't a dict or names no profile.
    Otherwise: {**profile_file, **inline_keys_minus_model_profile}. Never
    raises — a missing/corrupt profile logs and returns the inline keys.
    """
    if not isinstance(slot, dict) or "model_profile" not in slot:
        return slot
    inline = {k: v for k, v in slot.items() if k != "model_profile"}
    merged = dict(read_model_profile(slot.get("model_profile") or ""))
    merged.update(inline)
    return merged


def apply_seat_gate_bars(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Overlay a seat's OWN gate bars from its model profile (t_623a4dcf).

    A profile file (config-models/<seat>.json) may carry a ``gate_bars``
    block keyed EXACTLY like the main config's gate keys:

        "gate_bars": {
            "min_ai_confidence": 0.65,
            "runner_entry_gate": {"min_short_confidence": 0.65},
            "duelist_veto_gate": {"min_conf": 0.90}
        }

    Precedence: the profile's gate_bars OVERRIDE the config block; the
    config block stays the fallback for unprofiled seats — byte-identical
    behavior when nothing matches. Dict values merge at SUB-KEY level so
    non-bar siblings (mode, shadow_mode, drift_confirmed_release,
    cohort_admissions...) survive untouched; scalars replace.

    The lookup keys on the ANSWERING model whose conf the bar is compared
    against (same weld as the state_admission clause in risk_gates) — a bar
    that compares against a DIFFERENT seat's conf belongs in THAT seat's
    profile (e.g. duelist_veto_gate's conviction escape is compared against
    the primary's conf, so its bar lives in the primary-engine profile).
    Distinct from the profile's entry_min_confidence /
    entry_min_short_confidence seat bars (runner-gate sub-floor admission,
    2026-10-13/16): gate_bars sets the EFFECTIVE bar itself.
    Hot (no cache, per-decision), fail-open: absent/unreadable profile or a
    seat with no gate_bars returns the SAME config object untouched.
    """
    if not isinstance(config, dict):
        return config
    name = str(model_name or "").strip()
    if not name:
        return config
    bars = read_model_profile_for_model(name).get("gate_bars")
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


def seat_gate_cfg(config: Dict[str, Any], key: str, model_name: str,
                  default: Any = None) -> Any:
    """Single-gate-key convenience over apply_seat_gate_bars (same
    contract: hot, fail-open, profile overrides the config block)."""
    merged = apply_seat_gate_bars(config, model_name)
    return merged.get(key, default) if isinstance(merged, dict) else default

DEFAULT_CONFIG: Dict[str, Any] = {"mode": "OFF"}

# Documented block shape for the LLM slots (hot-read at call time by the
# duel_store/research helpers — model swap + settings are a same-inode config
# flip, no recreate). NESTED per-slot shape; the flat keys remain as legacy
# per-key fallbacks:
#
#   "llm": {
#       "primary": {                     # env fallbacks: LLM_MODEL / LLM_MAX_TOKENS
#           "model": "<primary model id>",
#           "sampling": {...},           # primary POST-body sampling
#           "max_tokens": 8192,          # primary completion budget
#           "chat_template_kwargs": {...}  # optional; passed to the model
#                                         # server verbatim (e.g.
#                                         # {"enable_thinking": false})
#       },
#       "duelist": {                     # env fallbacks: LLM_DUEL_MODEL /
#                                         # LLM_DUEL_MAX_TOKENS; absent
#                                         # everywhere = duelist off
#           "model": "<duelist id>",
#           "sampling": {...},           # duelist POST-body sampling
#           "max_tokens": 8192           # duelist completion budget (inherits
#                                        # the primary's resolved value)
#       }
#   }
#
# Legacy flat keys (still honored, per-key, below the nested slot keys):
# llm.model, llm.duelist_model, llm.sampling, llm.duelist_sampling,
# llm.max_tokens, llm.duelist_max_tokens.
#
# Endpoint/key stay in .env.local (LLM_BASE_URL, LLM_API_KEY,
# LLM_DUEL_BASE_URL, LLM_DUEL_API_KEY). The max-tokens env vars remain as
# FALLBACKS — the config keys above win when present.


def read_agent_config() -> Dict[str, Any]:
    """Read the agent config from .agent-config.json."""
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG)


def write_agent_config(cfg: Dict[str, Any]) -> None:
    """Write the agent config to .agent-config.json (atomic replace)."""
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, CONFIG_PATH)
    logger.info(f"[config] written {len(cfg)} keys to {CONFIG_PATH}")
