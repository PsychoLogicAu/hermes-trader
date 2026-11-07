"""Tests for the log-only OI accrual (hermes-trader kanban t_9f660914).

Covers the four things that matter about a shadow accrual: it is INERT when switched off, it
never raises into the scan, it writes an append-only series on the redirected path, and — the
part that keeps it a shadow — no decision surface reads it.
"""

import inspect
import json
import logging

import hermes_trader.agents.oi_accrual as oa
from hermes_trader.agents import perception


def _payload(rows):
    """rows: [(coin, oi_coins, px)] in metaAndAssetCtxs shape (index-aligned meta/ctx)."""
    meta = {"universe": [{"name": c} for c, _, _ in rows]}
    ctxs = [{"openInterest": oi, "midPx": px} for _, oi, px in rows]
    return [meta, ctxs]


def _enable(monkeypatch, tmp_path, **over):
    monkeypatch.setenv("HERMES_OI_SERIES_FILE", str(tmp_path / "oi-series.jsonl"))
    oa.reset_history()
    cfg = {"oi_accrual": {"enabled": True, "windows_min": [15, 60, 240],
                          "min_oi_usd": 5e6, "min_doi_pct": 3.0,
                          "max_px_move_pct": 4.0, "min_interval_s": 60.0, **over}}
    return cfg


def _series_lines(tmp_path):
    path = tmp_path / "oi-series.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ── inert / dormancy ─────────────────────────────────────────────────────────

def test_absent_block_is_inert_and_makes_no_fetch(monkeypatch):
    calls = []
    monkeypatch.setattr(oa, "_fetch_ctx_rows", lambda: calls.append(1) or _payload([("BTC", 10, 100)]))
    oa.reset_history()
    assert oa.accrual_enabled({}) is False
    assert oa.accrual_enabled({"oi_accrual": {"enabled": False}}) is False
    assert oa.snapshot_and_log({}) == []
    assert oa.snapshot_and_log({"oi_accrual": {"enabled": False}}) == []
    assert calls == []                       # zero HL weight spent while off


def test_malformed_block_is_inert(monkeypatch):
    calls = []
    monkeypatch.setattr(oa, "_fetch_ctx_rows", lambda: calls.append(1) or None)
    assert oa.accrual_enabled({"oi_accrual": True}) is False      # not a dict
    assert oa.accrual_enabled({"oi_accrual": {}}) is False        # no enabled key
    assert oa.snapshot_and_log({"oi_accrual": "yes"}) == []
    assert calls == []


# ── append-only series ───────────────────────────────────────────────────────

def test_series_appends_one_line_per_snapshot(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path)
    monkeypatch.setattr(oa, "_fetch_ctx_rows",
                        lambda: _payload([("BTC", 100.0, 50_000.0), ("SOL", 900.0, 200.0)]))
    oa.snapshot_and_log(cfg, now=1_800_000_000.0, force=True)
    oa.snapshot_and_log(cfg, now=1_800_000_200.0, force=True)
    lines = _series_lines(tmp_path)
    assert len(lines) == 2                                   # append-only, never rewrite
    assert lines[0]["ts"] == 1_800_000_000 and lines[1]["ts"] == 1_800_000_200
    assert lines[0]["m"]["BTC"] == [100.0, 50000.0]
    assert lines[0]["m"]["SOL"][0] == 900.0


def test_throttle_bounds_fetches_per_scan(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path, min_interval_s=60.0)
    calls = []

    def fake():
        calls.append(1)
        return _payload([("BTC", 100.0, 50_000.0)])

    monkeypatch.setattr(oa, "_fetch_ctx_rows", fake)
    oa.snapshot_and_log(cfg, now=1_800_000_000.0)
    oa.snapshot_and_log(cfg, now=1_800_000_010.0)            # 10 s later: skipped
    oa.snapshot_and_log(cfg, now=1_800_000_120.0)            # 2 min later: fetched
    assert len(calls) == 2


# ── flag semantics ───────────────────────────────────────────────────────────

def test_quiet_oi_build_flags_with_window_values(monkeypatch, tmp_path, caplog):
    cfg = _enable(monkeypatch, tmp_path)
    t0, t1 = 1_800_000_000.0, 1_800_014_400.0               # exactly 4 h apart
    monkeypatch.setattr(oa, "_fetch_ctx_rows",
                        lambda: _payload([("BTC", 110.0, 50_250.0)]))  # +10% OI, +0.5% px
    oa._HISTORY.append((t0, {"BTC": (100.0, 50_000.0)}))
    with caplog.at_level(logging.INFO, logger="hermes_trader.agents.oi_accrual"):
        flags = oa.snapshot_and_log(cfg, now=t1, force=True)
    assert len(flags) == 1
    f = flags[0]
    assert f["coin"] == "BTC"
    # One 4 h-old snapshot: every lookback resolves to it, and the FIRST window that
    # clears the bar is the reported hit (15 m), so the line orders windows predictably.
    assert f["doi_pct"] == [10.0, 10.0, 10.0]
    assert f["px_move_pct"] == [0.5, 0.5, 0.5]
    assert f["hit_window_min"] == 15 and f["hit_doi_pct"] == 10.0
    lines = [r.getMessage() for r in caplog.records]
    shadow = [l for l in lines if l.startswith("[oi-shadow] BTC")]
    assert shadow and "dOI(15m,60m,240m)=+10.0,+10.0,+10.0%" in shadow[0]
    assert "px(+0.5,+0.5,+0.5)%" in shadow[0] and "hit=15m" in shadow[0]
    assert any(l.startswith("[oi-accrual] snap ") for l in lines)   # liveness heartbeat


def test_price_already_ran_suppresses_the_flag(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path)
    monkeypatch.setattr(oa, "_fetch_ctx_rows",
                        lambda: _payload([("BTC", 110.0, 53_000.0)]))  # +9% OI but +6% px
    oa._HISTORY.append((1_800_000_000.0, {"BTC": (100.0, 50_000.0)}))
    assert oa.snapshot_and_log(cfg, now=1_800_008_640.0, force=True) == []


def test_small_notional_is_skipped(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path)
    monkeypatch.setattr(oa, "_fetch_ctx_rows",
                        lambda: _payload([("MEME", 100.0, 0.01)]))     # $1 OI notional
    oa._HISTORY.append((1_800_000_000.0, {"MEME": (10.0, 0.01)}))      # +900% OI
    assert oa.snapshot_and_log(cfg, now=1_800_008_640.0, force=True) == []


def test_oi_growth_is_coin_units_not_usd_notional(monkeypatch, tmp_path):
    """A doubling price on flat coin-OI must not read as position building."""
    cfg = _enable(monkeypatch, tmp_path)
    monkeypatch.setattr(oa, "_fetch_ctx_rows",
                        lambda: _payload([("BTC", 100.0, 100_000.0)]))   # px 2x, OI flat
    oa._HISTORY.append((1_800_000_000.0, {"BTC": (100.0, 50_000.0)}))
    flags = oa.snapshot_and_log(cfg, now=1_800_008_640.0, force=True)
    assert flags == []


def test_cold_start_seeds_windows_from_the_file(monkeypatch, tmp_path):
    """A recreate must not lose the lookbacks: the series tail rebuilds the history."""
    cfg = _enable(monkeypatch, tmp_path)                       # reset_history() inside
    (tmp_path / "oi-series.jsonl").write_text(json.dumps(
        {"ts": 1_800_000_000, "m": {"BTC": [100.0, 50_000.0]}}) + "\n")
    monkeypatch.setattr(oa, "_fetch_ctx_rows",
                        lambda: _payload([("BTC", 108.0, 50_100.0)]))    # +8% OI, +0.2% px
    flags = oa.snapshot_and_log(cfg, now=1_800_014_400.0, force=True)
    assert len(flags) == 1 and flags[0]["doi_pct"] == [8.0, 8.0, 8.0]


# ── degradation ──────────────────────────────────────────────────────────────

def test_rate_budget_skip_costs_nothing_and_never_raises(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path)
    monkeypatch.setattr(oa, "_fetch_ctx_rows", lambda: None)   # _http_post's budget-exhausted path
    assert oa.snapshot_and_log(cfg, now=1_800_000_000.0, force=True) == []
    assert _series_lines(tmp_path) == []


def test_unexpected_error_never_reaches_the_scan(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path)

    def boom():
        raise RuntimeError("socket exploded")

    monkeypatch.setattr(oa, "_fetch_ctx_rows", boom)
    assert oa.snapshot_and_log(cfg, now=1_800_000_000.0, force=True) == []


def test_malformed_payload_rows_are_skipped(monkeypatch, tmp_path):
    cfg = _enable(monkeypatch, tmp_path)
    monkeypatch.setattr(oa, "_fetch_ctx_rows", lambda: [{"universe": [{"name": "BTC"}, {}]},
                                                        [{"openInterest": None}, {"midPx": 0}]])
    assert oa.snapshot_and_log(cfg, now=1_800_000_000.0, force=True) == []
    assert _series_lines(tmp_path) == []
    assert oa.parse_ctx_rows("garbage") == {}


# ── the shadow guarantee ─────────────────────────────────────────────────────

def test_only_perception_wires_it_and_no_decision_surface_reads_it():
    """The accrual must stay log-only: the scan is its single consumer.

    A future seat/gate wiring has to be a deliberate act that breaks this test, goes through
    its own gauntlet, and gets its own board task (owner rule: accrue log-only first).
    """
    import hermes_trader.agents.decision_voice as dv
    import hermes_trader.agents.executor as ex
    import hermes_trader.agents.research as rs
    import hermes_trader.agents.risk_gates as rg

    assert "oi_accrual" in inspect.getsource(perception)          # wiring exists
    for mod in (dv, ex, rs, rg):
        assert "oi_accrual" not in inspect.getsource(mod), mod.__name__
