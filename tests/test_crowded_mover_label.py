"""t_1999904f (2026-10-17): crowdedMover label on the seat state.

The d2 seat reads a stretched CROWDED mover as pure strength: every named
token in its state is positive (trend aligned, triggers firing, catalyst)
and the conviction ladder's top rung ("trend + structure + catalyst all
agree") has no freshness concept — executed mover longs conf 0.71-0.88,
n=29, -$2.64, 4 max_loss; blocked mover class n=158, -$2.08/trade. Fix:
surface the pre-computed crowdedness flag (the dailyMover trigger hit —
>=bar 24h move AND >=$3M vol, the same source the gates veto on) as a named
`crowdedMover: true` field behind the per-profile `crowded_mover_label`
knob. Default off = byte-identical state (breakout_side_label pattern);
emitted ONLY when true so the ~96% non-mover states never change.
"""

from hermes_trader.agents import decision_voice as dv


def _tf(bull: bool = True):
    return {
        "ema8": 101.0 if bull else 99.0,
        "ema21": 100.0,
        "rsi14": 61.0 if bull else 39.0,
        "atr14": 1.2, "adx14": 25.0,
        "last_close": 100.8, "last_time": 1_700_000_000_000,
    }


def _perception(mover: bool, extra=None):
    triggers = [
        {"name": "dailyMover", "score": 10 if mover else 0,
         "reason": "+12.4% 24h mover, vol $40.00M" if mover
                   else "+12.4% 24h / vol $0.50M",
         "fired": mover},
    ]
    if extra:
        triggers.extend(extra)
    return {
        "id": "pid-cm-1", "coin": "BTC", "mid": 50_000.0,
        "composite_score": 62.0, "daily_move_pct": 12.4,
        "triggers": triggers,
    }


def _build(p, **knobs):
    return dv.build_state("BTC", p, _tf(), _tf(), _tf(), "0.0100%/hr", "n",
                          [], win_rate=0.5, n_closes=10, **knobs)


def test_default_off_is_byte_identical():
    p = _perception(mover=True)
    assert _build(p) == _build(p, crowded_mover_label=False)
    assert "crowdedMover" not in _build(p)


def test_label_present_when_mover_fired():
    state = _build(_perception(mover=True), crowded_mover_label=True)
    assert state.get("crowdedMover") is True


def test_label_absent_when_mover_not_fired():
    # thin >10% mover (flag not fired) = the best cell (n=32, 91% WR) —
    # must stay byte-identical, no caution token emitted.
    state = _build(_perception(mover=False), crowded_mover_label=True)
    assert "crowdedMover" not in state


def test_label_absent_without_triggers():
    p = _perception(mover=False)
    p["triggers"] = []
    state = _build(p, crowded_mover_label=True)
    assert "crowdedMover" not in state


def test_label_does_not_disturb_other_fields():
    p = _perception(mover=True, extra=[
        {"name": "volumeSpike", "score": 8, "reason": "rvol 3.1",
         "fired": True},
    ])
    off = _build(p)
    on = _build(p, crowded_mover_label=True)
    assert on == {**off, "crowdedMover": True}
    # trigger name rendering untouched (dailyMover stays in triggers_fired)
    assert "dailyMover" in on["triggers_fired"]
