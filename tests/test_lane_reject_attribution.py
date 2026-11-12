"""t_c777983d — the LIVE lane's `reason=<X>` must name the rejecting gate.

In the 11-01..06 aligned-short-lane live experiment 9/10 admits logged
`maybe_execute -> executed=False reason=-`: maybe_execute's gate-block result
carries `blocked_by` (list of per-gate reason strings), not `reason`, and the
log line printed only `reason`. The ad-hoc decode in trading_loop.py (56dec6a)
was never pinned by any test — this file pins `risk_gates.lane_reject_reason`,
the decode the live lane line now calls, for EVERY result shape:

1. pre-gate refusals: `reason` passes through (mode_off, cooldown, sizing...).
2. gate blocks: every `blocked_by` string lands in the output (ctx values
   embedded — "confidence 0.42 < 0.65"), never '-'.
3. a blocked gate with a null reason still names the failing gate via
   `gate_results` keys.
4. an unexpected result shape self-reports as `unattributed(keys:...)` so
   shape drift can't silently regress to '-'.
5. success lines keep the meaningless '-'.

Plus the end-to-end pin the board asks for: a REAL `maybe_execute` gate-block
result (hermetic `_exec_baseline` stubs + a blocking eval_all_gates) must be
attributable — the exact object the trading_loop lane line decodes.
"""
from __future__ import annotations

from test_cleanup import _analysis, _exec_baseline

from hermes_trader.agents.risk_gates import lane_reject_reason


# ── 1. pre-gate refusals: reason rides through ──────────────────────────────
def test_pre_gate_reason_passes_through():
    assert lane_reject_reason(
        {"executed": False, "mode": "LIVE", "reason": "mode_off"}) == "mode_off"


# ── 2. gate blocks: blocked_by strings (with ctx values) are the reason ────
def test_gate_block_joins_blocked_by_reasons():
    res = {
        "executed": False, "mode": "LIVE", "analysis_id": "a1",
        "blocked_by": [
            "confidence 0.42 < 0.65",
            "short on thin market: 24h vol $3.1M below floor $20.0M",
        ],
        "gate_results": {
            "confidence": {"pass": False, "reason": "confidence 0.42 < 0.65"},
            "thin_short": {"pass": False,
                           "reason": "short on thin market: 24h vol $3.1M below floor $20.0M"},
        },
    }
    out = lane_reject_reason(res)
    assert out != "-"
    assert "confidence 0.42 < 0.65" in out
    assert "below floor $20.0M" in out


# ── 3. null reason strings never erase attribution: gate keys carry it ─────
def test_null_reason_entries_fall_back_to_failing_gate_keys():
    # eval_all_gates appends result.get("reason", key) — a gate returning
    # {"pass": False, "reason": None} puts a bare None into blocked_by.
    res = {
        "executed": False, "mode": "LIVE",
        "blocked_by": [None],
        "gate_results": {
            "confidence": {"pass": True},
            "quiet_tape": {"pass": False, "reason": None},
        },
    }
    out = lane_reject_reason(res)
    assert out == "gates:quiet_tape"
    assert out != "-"


def test_mixed_null_and_real_reasons_name_every_failing_gate():
    # Independent-review nit: with SOME entries None, the naive blocked_by
    # filter would silently drop the null-reason gate's identity. The decode
    # must name every failing gate — real reason where present, key otherwise.
    res = {
        "executed": False, "mode": "LIVE",
        "blocked_by": ["confidence 0.42 < 0.65", None],
        "gate_results": {
            "confidence": {"pass": False, "reason": "confidence 0.42 < 0.65"},
            "quiet_tape": {"pass": False, "reason": None},
        },
    }
    out = lane_reject_reason(res)
    assert "confidence 0.42 < 0.65" in out
    assert "gates:quiet_tape" in out


# ── 4. shape drift self-reports instead of silently printing '-' ───────────
def test_unexpected_reject_shape_self_reports():
    out = lane_reject_reason({"executed": False, "mode": "LIVE", "analysis_id": "a1"})
    assert out == "unattributed(keys:analysis_id,executed,mode)"


# ── 5. executed=True keeps the '-' (reason slot meaningless on success) ────
def test_success_line_keeps_dash():
    assert lane_reject_reason({"executed": True, "mode": "LIVE",
                               "order_id": "OID1"}) == "-"


# ── end-to-end: a REAL maybe_execute gate block is attributable ────────────
def test_maybe_execute_gate_block_line_is_attributable(monkeypatch):
    """The exact object trading_loop logs for the LIVE lane: gate-block shape
    (no reason key, blocked_by list) must decode to the blocking gate's text —
    the 9x `reason=-` blind spot pinned shut."""
    ex, _captured, _cfg = _exec_baseline(monkeypatch)
    monkeypatch.setattr(ex, "_get_market_volume_24h", lambda c: 5e7)

    def _blocked_gates(ctx, config, last_trade_time=None):
        return {
            "results": {
                "confidence": {"pass": False, "reason": "confidence 0.42 < 0.65"},
            },
            "blocked": True,
            "block_reasons": ["confidence 0.42 < 0.65"],
        }

    monkeypatch.setattr(
        "hermes_trader.agents.executor.eval_all_gates", _blocked_gates)

    r = ex.maybe_execute(_analysis(coin="MELANIA", verdict="SHORT",
                                   side="short", confidence=0.42))
    assert r["executed"] is False, r
    # The gate-block shape: NO reason key, blocked_by present.
    assert "reason" not in r and r["blocked_by"] == ["confidence 0.42 < 0.65"]
    # What the LIVE lane line now prints:
    assert lane_reject_reason(r) == "confidence 0.42 < 0.65"
