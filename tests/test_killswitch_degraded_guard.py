"""Heartbeat partial-dex guard covers IDLE-FUND dexes (2026-10-03 killswitch incident).

The 2026-10-03 20:09 UTC first-ever HARD killswitch fire was a phantom: a HIP-3
dex query 429'd during a rate-limit storm, dropping ~$948 of IDLE funds (no
open position on that dex) from the aggregate equity read. The old guard only
checked dexes backing OPEN positions (`held_dexes`), so the degraded $6.19
read sailed into memory -> daily PnL -948.16 -> flatten on fiction.

Fix: hl_client.fetch_account_state now reports `failed_dexes`; the heartbeat
treats ANY failed dex query as a degraded aggregate read (same
preserve-last-known-good path as equity<=0).

trading_loop.py is NOT importable (module-level `while True`), so — mirroring
tests/test_cooldown_research_skip.py — we AST-extract `_sync_account_state`
and exec it in a stubbed namespace. NO network, NO live state files.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

_tmpdir = tempfile.mkdtemp(prefix="hermes-test-killguard-")
os.environ["HERMES_AGENT_MEMORY_FILE"] = os.path.join(_tmpdir, ".agent-memory.json")
os.environ["HERMES_AGENT_CONFIG_FILE"] = os.path.join(_tmpdir, ".agent-config.json")
os.environ["HERMES_DSL_STATE_FILE"] = os.path.join(_tmpdir, ".dsl-state.json")
os.environ["HERMES_LEDGER_FILE"] = os.path.join(_tmpdir, "trades.jsonl")

LOOP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "trading_loop.py")

_loop_src = open(LOOP_PATH).read()
_tree = ast.parse(_loop_src)
_FUNCS = {n.name: n for n in _tree.body if isinstance(n, ast.FunctionDef)}


def _extract(name):
    assert name in _FUNCS, f"{name} not found at module level in trading_loop.py"
    return ast.get_source_segment(_loop_src, _FUNCS[name])


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, *a, **k): pass

    def warning(self, *a, **k): self.warnings.append(str(a[0]) if a else "")

    def error(self, *a, **k): pass


class _FakeMemory:
    def __init__(self):
        self.tracked = []
        self.updated_positions = []

    def get_day_start_ts(self):
        return 0  # skip the contributions fetch entirely (sod_ts_ms <= 0)

    def track_daily_pnl(self, eq, contrib):
        self.tracked.append(eq)

    def update_open_positions(self, pos):
        self.updated_positions.append(pos)

    def flush(self):
        pass


def _run_sync(monkey_state, held_coins, logger, mem):
    """Exec the REAL _sync_account_state source with stubbed globals."""
    ns = {
        "logger": logger,
        "memory": mem,
        "resolve_user_address": lambda: "0xUSER",
        "fetch_account_state": lambda user, include_hip3=False: dict(monkey_state),
        "fetch_aggregate_contributions_since": lambda user, ms: 0.0,
        "active_position_coins": lambda: set(held_coins),
    }
    exec(compile(_extract("_sync_account_state"), "<loop>", "exec"), ns)
    return ns["_sync_account_state"]()


def test_failed_dex_with_no_position_is_degraded():
    """THE incident replay: HIP-3 dex query failed, NO open position on it
    (idle funds only). Old guard: blind (held_dexes empty) -> memory poisoned.
    New guard: failed_dexes non-empty -> degraded path (equity 0, no memory
    update)."""
    logger, mem = _Logger(), _FakeMemory()
    equity, positions, available, spot, queried, state = _run_sync(
        {"equity": 6.19, "available": 5.0, "spot_usdc": 954.87,
         "asset_positions": [], "queried_dexes": {""}, "failed_dexes": {"xyz"}},
        held_coins=set(), logger=logger, mem=mem)
    assert equity == 0.0 and positions == [] and queried == set()
    assert mem.tracked == [], "degraded read must NEVER reach memory"
    assert any("partial-dex degraded read" in w for w in logger.warnings)


def test_held_dex_missing_still_degraded():
    """Existing behavior preserved: a dex backing an open tracker missing
    from queried_dexes degrades the read even with failed_dexes absent."""
    logger, mem = _Logger(), _FakeMemory()
    equity, positions, available, spot, queried, state = _run_sync(
        {"equity": 56.65, "available": 50.0, "spot_usdc": 0.0,
         "asset_positions": [], "queried_dexes": {""}},
        held_coins={"xyz:MU"}, logger=logger, mem=mem)
    assert equity == 0.0 and queried == set()
    assert mem.tracked == []
    assert any("partial-dex degraded read" in w for w in logger.warnings)


def test_all_dexes_responded_is_clean():
    """Clean fan-out (failed_dexes empty, held dexes queried) -> normal path:
    memory updated, state returned intact."""
    logger, mem = _Logger(), _FakeMemory()
    equity, positions, available, spot, queried, state = _run_sync(
        {"equity": 954.46, "available": 947.89, "spot_usdc": 954.87,
         "asset_positions": [{"position": {"coin": "MON", "szi": "100"}}],
         "queried_dexes": {"", "xyz"}, "failed_dexes": set()},
        held_coins={"xyz:MON"}, logger=logger, mem=mem)
    assert equity == 954.46
    assert mem.tracked == [954.46]
    assert queried == {"", "xyz"}


def test_missing_failed_dexes_key_is_tolerated():
    """A state dict without the key (older/other producers) must not raise —
    guard degrades to the held-dex check only."""
    logger, mem = _Logger(), _FakeMemory()
    equity, positions, available, spot, queried, state = _run_sync(
        {"equity": 954.46, "available": 947.89, "spot_usdc": 0.0,
         "asset_positions": [], "queried_dexes": {"", "xyz"}},
        held_coins=set(), logger=logger, mem=mem)
    assert equity == 954.46 and mem.tracked == [954.46]
