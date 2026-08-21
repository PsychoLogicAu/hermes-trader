"""Per-coin log prefix for the parallel research phase (research_max_workers>1).

Proves CoinLogFormatter prepends "[COIN] " to a record only while log_coin is
set on the CURRENT thread, that the contextvar propagates into
ThreadPoolExecutor worker tasks (so each worker tags its OWN coin), and that it
does NOT leak to the main thread after the pool drains (the sequential
workers==1 path must not stamp cycle-level lines with a stale coin).

NO network, NO exchange calls. Runs standalone (no pytest):
    python3 tests/test_log_coin_prefix.py
pytest-compatible too (CI runs the tests/ suite).
"""
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

# ── Test isolation (mirror tests/conftest.py) — MUST run before hermes imports
_tmp = tempfile.mkdtemp(prefix="hermes-test-logcoin-")
os.environ["HERMES_AGENT_MEMORY_FILE"] = os.path.join(_tmp, ".agent-memory.json")
os.environ["HERMES_AGENT_CONFIG_FILE"] = os.path.join(_tmp, ".agent-config.json")
os.environ["HERMES_DSL_STATE_FILE"] = os.path.join(_tmp, ".dsl-state.json")
os.environ["SESSION_LOG_PATH"] = os.path.join(_tmp, "session.jsonl")
os.environ.pop("HYPERLIQUID_PRIVATE_KEY", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_trader.log_context import (  # noqa: E402
    CoinLogFormatter,
    log_coin,
    set_coin_context,
)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


class _Capture(logging.Handler):
    """Records the formatted line of each record into self.lines."""

    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(self.format(record))


def _setup_logger():
    lg = logging.getLogger("hermes_trader.test_logcoin")
    cap = _Capture()
    lg.handlers = [cap]
    lg.setLevel(logging.INFO)
    lg.propagate = False
    cap.setFormatter(CoinLogFormatter("%(message)s"))
    return lg, cap


def test_plain_line_untagged():
    lg, cap = _setup_logger()
    lg.info("hello bare")
    check("no coin -> line unchanged", cap.lines[-1] == "hello bare",
          repr(cap.lines[-1]))


def test_coin_prefix_and_restore():
    lg, cap = _setup_logger()
    with set_coin_context("ETH"):
        lg.info("verdict long")
    check("coin set -> [ETH] prefix", cap.lines[-1] == "[ETH] verdict long",
          repr(cap.lines[-1]))
    check("context restored on exit", log_coin.get() == "",
          repr(log_coin.get()))


def test_pool_propagation_and_no_leak():
    lg, cap = _setup_logger()
    coins = ["ETH", "SOL", "AVAX", "DOGE"]

    def worker(c):
        with set_coin_context(c):
            lg.info("verdict for %s", c)

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="research") as p:
        for f in [p.submit(worker, c) for c in coins]:
            f.result()

    tagged = [l for l in cap.lines if l.startswith("[")]
    check("4 worker lines emitted", len(tagged) == 4, repr(cap.lines))
    expected = {f"[{c}]" for c in coins}
    got = {l.split(" ", 1)[0] for l in tagged}
    check("each worker tagged its own coin", got == expected, repr(got))
    check("no main-thread leak after pool", log_coin.get() == "",
          repr(log_coin.get()))
    lg.info("cycle level line")
    check("main line after pool is bare", cap.lines[-1] == "cycle level line",
          repr(cap.lines[-1]))


if __name__ == "__main__":
    test_plain_line_untagged()
    test_coin_prefix_and_restore()
    test_pool_propagation_and_no_leak()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)