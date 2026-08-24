"""Pytest bootstrap — load .env.local before the test session starts.

hermes_trader.client.exchange reads Hyperliquid credentials at import time, so
the environment must be populated before any test imports the package. This is
a no-op when .env.local is absent (e.g. CI without secrets).

Also isolates ALL persistent state files to a throwaway temp dir, BEFORE any
hermes module import. The 2026-06-15 incident (pytest wiped live memory)
added this for memory/config/DSL state in tests/conftest.py only — but the
production image EXCLUDES tests/ (see .dockerignore) and ships root-level test
files, so in-container pytest runs only ever saw THIS root conftest, with no
isolation. On 2026-08-23 that let a pytest run write fixture OPEN/CLOSE rows
into the live /app/log/trades.jsonl. The ledger is now env-overridable
(HERMES_LEDGER_FILE) and must be set here, before collection.
"""
import os
import pathlib
import tempfile

_ENV_FILE = pathlib.Path(__file__).parent / ".env.local"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# State-file isolation (mirror of tests/conftest.py — the two must stay in
# sync). Force, not setdefault: tests must never touch live state, even if the
# dev shell or container env exports these for the running bot.
_tmp = tempfile.mkdtemp(prefix="hermes-test-state-")
os.environ["HERMES_AGENT_MEMORY_FILE"] = os.path.join(_tmp, ".agent-memory.json")
os.environ["HERMES_AGENT_CONFIG_FILE"] = os.path.join(_tmp, ".agent-config.json")
os.environ["HERMES_DSL_STATE_FILE"] = os.path.join(_tmp, ".dsl-state.json")
os.environ["HERMES_LEDGER_FILE"] = os.path.join(_tmp, "trades.jsonl")
