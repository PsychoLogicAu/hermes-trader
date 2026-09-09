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
