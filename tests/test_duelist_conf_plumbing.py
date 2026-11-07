"""t_5baddef5 — the duelist's OWN confidence reaches the risk gates.

`GateContext.duelist_confidence` (risk_gates.py) is plumbing only: it carries
the second voice's confidence in its own answer, from research.py's
`duelist_at_entry` snapshot, to the gate pipeline. The intended consumer is
`abstention_veto_gate`, whose key IS that confidence (CEILING-keyed, measured
2026-11-06/07); the older `duelist_veto_gate` never reads it. The value must
arrive WITHOUT changing any outcome until a config block arms the new gate.

The regression this pins is the 2026-08 pitfall: `duelist_at_entry` lost its
fields in research.py's analysis whitelist and the executor silently saw None.
A dropped whitelist key is indistinguishable from a data gap in production, so
the pin lives on the executor side:

1. `maybe_execute` puts `duelist_at_entry["confidence"]` on the GateContext
   (alongside the verdict and the answering seat's model).
2. No duel row / a row without the confidence key => None, i.e. the gate's
   fail-open input, and the entry still executes.
3. With NO `abstention_veto_gate` block in the config — the live posture —
   a carried confidence changes no gate outcome: the entry executes and
   `gate_results["abstention_veto"]` is a bare pass.

Hermetic: the shared `test_cleanup._exec_baseline` executor I/O stubs.
"""
from __future__ import annotations

from test_cleanup import _analysis, _exec_baseline


def _harness(monkeypatch):
    """_exec_baseline + a hermetic 24h-volume stub (no universe/disk read)."""
    ex, captured, _cfg = _exec_baseline(monkeypatch)
    monkeypatch.setattr(ex, "_get_market_volume_24h", lambda c: 5e7)
    return ex, captured


def _capture_ctx(monkeypatch):
    """Swap eval_all_gates for a ctx recorder that blocks nothing."""
    seen: dict = {}

    def _fake_gates(ctx, config, last_trade_time=None):
        seen["duelist_verdict"] = ctx.duelist_verdict
        seen["duelist_confidence"] = ctx.duelist_confidence
        seen["duelist_model"] = ctx.duelist_model
        return {"results": {}, "blocked": False, "block_reasons": []}

    monkeypatch.setattr("hermes_trader.agents.executor.eval_all_gates", _fake_gates)
    return seen


# ── 1. the confidence rides the whitelist into the GateContext ─────────────
def test_executor_passes_duelist_confidence_into_gate_context(monkeypatch):
    ex, captured = _harness(monkeypatch)
    seen = _capture_ctx(monkeypatch)
    a = _analysis(
        coin="BTC", confidence=0.70, composite_score=40,
        duelist_at_entry={"model": "qwen3.8-flash-next-iq3_s",
                          "verdict": "PASS", "confidence": 0.45, "side": None},
    )
    r = ex.maybe_execute(a)
    assert r["executed"] is True, r
    assert captured.get("coin") == "BTC"
    assert seen["duelist_confidence"] == 0.45
    # The other two fields the abstention gate needs: fail-open on a missing
    # verdict, and the per-seat ceiling lookup on the ANSWERING model.
    assert seen["duelist_verdict"] == "PASS"
    assert seen["duelist_model"] == "qwen3.8-flash-next-iq3_s"


# ── 2. data gaps stay None: absence is never an objection ─────────────────
def test_no_duel_row_leaves_duelist_confidence_none(monkeypatch):
    ex, _ = _harness(monkeypatch)
    seen = _capture_ctx(monkeypatch)
    r = ex.maybe_execute(_analysis(coin="BTC", confidence=0.70,
                                   composite_score=40, duelist_at_entry=None))
    assert r["executed"] is True, r
    assert seen["duelist_confidence"] is None
    assert seen["duelist_verdict"] is None
    assert seen["duelist_model"] is None


def test_duel_row_without_confidence_key_stays_none(monkeypatch):
    # A duel row written before the confidence field existed (or a seat whose
    # answer carried none): no measurable conviction is NOT an abstention
    # veto — abstention_veto_gate fails open on exactly this None.
    ex, _ = _harness(monkeypatch)
    seen = _capture_ctx(monkeypatch)
    r = ex.maybe_execute(_analysis(
        coin="BTC", confidence=0.70, composite_score=40,
        duelist_at_entry={"model": "Spark-X2.6-4B-GGUF-Q4_K_M",
                          "verdict": "PASS", "side": None}))
    assert r["executed"] is True, r
    assert seen["duelist_confidence"] is None


# ── 3. plumbing alone moves no gate (live config has no such block) ───────
def test_carried_confidence_changes_no_gate_outcome(monkeypatch):
    # Real eval_all_gates (NOT stubbed): the harness config has no
    # `abstention_veto_gate` block, so the gate is disabled and returns a bare
    # pass even for a conf (0.45) that WOULD sit under the fitted 0.55 ceiling.
    ex, captured = _harness(monkeypatch)
    a = _analysis(
        coin="BTC", confidence=0.70, composite_score=40,
        duelist_at_entry={"model": "qwen3.8-flash-next-iq3_s",
                          "verdict": "PASS", "confidence": 0.45, "side": None},
    )
    r = ex.maybe_execute(a)
    assert r["executed"] is True, r
    assert captured.get("coin") == "BTC"
    assert r["gate_results"].get("abstention_veto") == {"pass": True}
    assert not any("abstention" in str(x).lower() for x in r.get("blocked_by", []))
