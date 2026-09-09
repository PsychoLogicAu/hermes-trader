"""T4.4 — tradable_universe allowlist toggle (upstream 16a2a0a98f9d, ported
default-OFF).

`hermes_trader.client.universe.filter_universe` is the pure helper: an empty or
absent allowlist returns the universe UNCHANGED (byte-identical current
behavior); a non-empty allowlist keeps only coins whose bare ticker (the part
after ':' for HIP-3 dex names) is in the list, case-insensitively.

scripts/trading_loop.py is NOT importable (module-level `while True` would
start trading), so the two call sites are pinned by source-text assertion —
the same pattern as tests/test_cooldown_research_skip.py. No network, no live
config (tests/conftest.py redirects HERMES_AGENT_CONFIG_FILE to a temp dir).
"""
import os

from hermes_trader.client.universe import filter_universe

_LOOP_SRC = open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "trading_loop.py",
)).read()


def _rows(*coins):
    return [{"coin": c, "type": "perp", "dex": c.split(":")[0] if ":" in c else None}
            for c in coins]


# ── Pure helper ─────────────────────────────────────────────────────────────

def test_empty_allowlist_unchanged():
    uni = _rows("BTC", "ETH", "xyz:GOLD", "PEPE")
    out = filter_universe(uni, [])
    assert out is uni  # same objects, order, length — no copy, no sort


def test_none_allowlist_unchanged():
    uni = _rows("BTC", "ETH", "xyz:GOLD", "PEPE")
    out = filter_universe(uni, None)
    assert out is uni


def test_filters_main_dex_exact_ticker_case_insensitive():
    uni = _rows("BTC", "ETH", "pepe", "SOL")
    out = filter_universe(uni, ["btc", "SOL"])
    assert [r["coin"] for r in out] == ["BTC", "SOL"]
    assert out[0] is uni[0]  # survivors are the SAME row objects, original order


def test_hip3_prefix_matches_bare_ticker_only():
    uni = _rows("MU", "xyz:MU", "km:MU", "xyz:OTHER")
    out = filter_universe(uni, ["MU"])
    # "MU" covers the bare listing AND every venue's dex-prefixed name;
    # a different ticker is excluded.
    assert [r["coin"] for r in out] == ["MU", "xyz:MU", "km:MU"]


def test_whitespace_only_allowlist_treated_as_empty():
    uni = _rows("BTC", "PEPE")
    assert filter_universe(uni, ["", "   "]) is uni
    assert filter_universe(uni, ["", None, "  "]) is uni  # no crash on junk


def test_strip_and_case_fold_of_entries():
    uni = _rows("btc", "XYZ:GOLD")
    out = filter_universe(uni, ["  Btc ", "gold"])
    assert [r["coin"] for r in out] == ["btc", "XYZ:GOLD"]


def test_no_match_keeps_nothing():
    assert filter_universe(_rows("PEPE", "BONK"), ["BTC"]) == []


# ── Wiring: both construction sites apply the toggle (source pin) ───────────

def test_loop_wiring():
    assert "from hermes_trader.client.universe import filter_universe, get_universe" in _LOOP_SRC
    # Startup site (module level): hot-read via read_agent_config().
    assert 'universe = filter_universe(universe, (read_agent_config() or {}).get("tradable_universe") or [])' in _LOOP_SRC
    # Periodic refresh site (inside the TTL block): reuses the tick's _cfg.
    assert 'universe = filter_universe(universe, _cfg.get("tradable_universe") or [])' in _LOOP_SRC
    # Each filter sits IMMEDIATELY after its get_universe fetch, before the
    # snapshot is used (startup: before the "Universe loaded" log; refresh:
    # before _last_universe_refresh is stamped).
    i_fetch, i_filter, i_log = (
        _LOOP_SRC.index("universe = get_universe(include_hip3=_enable_hip3)"),
        _LOOP_SRC.index('(read_agent_config() or {}).get("tradable_universe")'),
        _LOOP_SRC.index('f"Universe loaded: {len(universe)} markets"'),
    )
    assert i_fetch < i_filter < i_log
    # Anchor the STAMP on the INDENTED in-block assignment, not the module-level
    # `_last_universe_refresh = time.time()` init (which str.index would find first
    # and which sits before the refetch — an invalid order anchor).
    i_refetch, i_refilter, i_stamp = (
        _LOOP_SRC.index("universe = get_universe(force_refresh=True, include_hip3=_enable_hip3)"),
        _LOOP_SRC.index('_cfg.get("tradable_universe")'),
        _LOOP_SRC.index("                _last_universe_refresh = time.time()"),
    )
    assert i_refetch < i_refilter < i_stamp
