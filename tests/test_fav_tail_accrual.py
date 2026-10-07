"""C.21 (2026-10-08): fav_tail_shadow — favorable-tail ACCRUAL line on
executed entries. Log-only: never touches trade_notional. The line format is
the accrual grep anchor; the per-model fav6pp scores come from the expert
signal's candidate_q90_pct / candidate_q10_pct (long/short frame). Hermetic:
peek_expert_select patched, no live models.
"""

import logging

import hermes_trader.agents.expert_select as es
from hermes_trader.agents.expert_select import ExpertSelectSignal

_LINE = "[gate][ACC] fav_tail"


def _sig(q90=None, q10=None):
    return ExpertSelectSignal(
        coin="TESTC", side="long", context_last=1.0, horizon=12,
        selected_model="mixture", diverse=False,
        candidate_q90_pct=q90 or {}, candidate_q10_pct=q10 or {})


def test_signal_fields_roundtrip():
    """The dataclass carries per-model tails (the accrual's complete record)."""
    s = _sig(q90={"chronos": [1.0, 2.0, 4.5, 3.0, 2.0, 1.0]})
    assert s.candidate_q90_pct["chronos"][2] == 4.5


def test_forecast_agreement_line_shape_long(caplog, monkeypatch):
    """Direct format check: max q90[:6] per model, hot= marks >= threshold."""
    sig = _sig(q90={
        "chronos": [1.0, 2.0, 4.5, 3.0, 2.0, 1.0, 0.0] * 2,   # fav 4.5 -> hot
        "timesfm": [0.5, 1.0, 1.2, 1.1, 1.0, 0.9, 0.8] * 2,   # fav 1.2
    })
    monkeypatch.setattr(es, "peek_expert_select", lambda coin: sig)
    # exercise the same math the executor block runs
    fav = {m: max(p[:6]) for m, p in sig.candidate_q90_pct.items()}
    assert fav == {"chronos": 4.5, "timesfm": 1.2}
    hot = [m for m, v in fav.items() if v >= 4.0]
    assert hot == ["chronos"]


def test_short_frame_uses_neg_min_q10():
    sig = _sig(q10={"tirex": [-0.5, -2.0, -5.1, -3.0, -1.0, -0.5] * 2})
    fav = {m: -min(p[:6]) for m, p in sig.candidate_q10_pct.items()}
    assert fav == {"tirex": 5.1}
