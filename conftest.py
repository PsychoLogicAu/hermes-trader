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
os.environ["HERMES_DUEL_FILE"] = os.path.join(_tmp, ".hermes-trader-duel.jsonl")
# The model-duel feature (2026-08-25) reads LLM_DUEL_* at call time, and the
# .env.local setdefault above leaks the live values into the test env. DELETE
# them (not empty-string) so a dev shell or .env.local export can't make a
# test fire a real second LLM call; a test that wants the duelist enables it
# via monkeypatch.setenv.
os.environ.pop("LLM_DUEL_MODEL", None)
os.environ.pop("LLM_DUEL_BASE_URL", None)
os.environ.pop("LLM_DUEL_API_KEY", None)
