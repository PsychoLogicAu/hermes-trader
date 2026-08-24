"""Test isolation: redirect agent state files to a throwaway temp dir BEFORE any
hermes module imports, so a test can never read or truncate the live
.agent-memory.json / .agent-config.json (a pytest run wiped live trading state
on 2026-06-15). This runs at conftest import — before test modules are collected,
hence before memory.py / config_store.py freeze their module-level paths.

2026-08-23: the ledger got the same treatment late. A pytest run in the
container (root-level test files, root conftest without this isolation) wrote
fixture OPEN/CLOSE rows — order_id OID1, ARB short @ 0.11684, +$11.5616 × 2 —
into the live /app/log/trades.jsonl (bind-mounted rw to trader-logs/),
poisoning the daily report's realized-PnL by ~$23. The tests mock the exchange
and the outcome store but NOT the ledger module, so every un-mocked
record_open/record_close hit the real file. HERMES_LEDGER_FILE is read at
ledger.py import time, so it must be set here, before collection.
"""

import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="hermes-test-state-")
# Force (not setdefault): even if the dev shell exports these, tests must use
# disposable paths.
os.environ["HERMES_AGENT_MEMORY_FILE"] = os.path.join(_tmp, ".agent-memory.json")
os.environ["HERMES_AGENT_CONFIG_FILE"] = os.path.join(_tmp, ".agent-config.json")
os.environ["HERMES_DSL_STATE_FILE"] = os.path.join(_tmp, ".dsl-state.json")
os.environ["HERMES_LEDGER_FILE"] = os.path.join(_tmp, "trades.jsonl")
