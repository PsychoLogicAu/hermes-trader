"""t_8577433e (2026-11-26): short_chase_risk — extension-evidence label on
the seat state (ext_fields knob, plan 2026-11-26_102317).

The gauntlet (2,954 rows, three arms, replica seat): the label arm is net
+106pp under BN2 5%/6h bar-gated pricing, both time-halves positive, and on
the REALIZED last-week book vetoed 25/63 entry calls for +$12.69 net. The
numeric-only arm was rejected (raised off-target conf). So exactly ONE gated
state key rides the prompt, behind a per-profile knob:
  short_chase_risk = "price <off>% above 24h low, <d>% <above|below> fast 1h
                      EMA — short is entering the bounce, not the flush"
emitted iff off_24h_low_pct >= 1.5 OR ema_dist_pct > 0, computed from
CONFIRMED 1h bars (bar.t + 1h <= call ts) off the candlestore.

The arithmetic MUST stay lockstep with the gauntlet's shared module
(scratch/extf_fields.fields_at) or live prompts stop matching the priced
arm — test_lockstep_with_gauntlet_module pins that directly.
"""

import os, sys
import pytest

from hermes_trader.agents import decision_voice as dv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scratch"))
try:
    import extf_fields  # gauntlet module (untracked scratch/) — pins below skip without it
    HAVE_EXT = True
except Exception:  # noqa: BLE001
    HAVE_EXT = False
from hermes_trader.client import candlestore as cs  # noqa: E402

H = 3_600_000


def _bars(n=120, t_end=1_795_600_000_000):
    """Ascending 1h bars with a sawtooth so min-low and the EMA are
    non-degenerate. Last bar CLOSES exactly at t_end + H."""
    t0 = t_end - (n - 1) * H
    return [{"t": t0 + k * H,
             "l": 100.0 + (k % 7) - 3.0,
             "c": 100.0 + (k % 5),
             "h": 100.0 + (k % 7) + 4.0}
            for k in range(n)]


def _tf():
    return {"ema8": 101.0, "ema21": 100.0, "rsi14": 61.0,
            "atr14": 1.2, "adx14": 25.0,
            "last_close": 100.8, "last_time": 1_795_600_000_000}


def _perception(mid=110.0):
    return {"id": "pid-ext-1", "coin": "BTC", "mid": mid,
            "composite_score": 40.0, "daily_move_pct": 3.1, "triggers": []}


def _build(p, **knobs):
    return dv.build_state("BTC", p, _tf(), _tf(), _tf(), "0.0100%/hr", "n",
                          [], win_rate=0.5, n_closes=10, **knobs)


# ---- knob + rendering ----------------------------------------------------

def test_default_off_is_byte_identical(monkeypatch):
    monkeypatch.setattr(dv, "_ext_position_fields",
                        lambda coin, mid, now_ms: (9.9, 5.0))
    p = _perception()
    assert _build(p) == _build(p, ext_fields=False)
    assert "short_chase_risk" not in _build(p)


@pytest.mark.skipif(not HAVE_EXT, reason="gauntlet module (scratch/extf_fields.py) absent")
def test_label_exact_gauntlet_string(monkeypatch):
    # byte-identical to extf_fields.apply_ext_fields(mode="label") output
    monkeypatch.setattr(dv, "_ext_position_fields",
                        lambda coin, mid, now_ms: (2.4, 1.9))
    state = _build(_perception(), ext_fields=True)
    assert state["short_chase_risk"] == (
        "price 2.4% above 24h low, 1.9% above fast 1h EMA"
        " \u2014 short is entering the bounce, not the flush")
    ref = {}
    extf_fields.apply_ext_fields(ref, "label", 2.4, 1.9)
    assert state["short_chase_risk"] == ref["short_chase_risk"]


def test_below_ema_wording(monkeypatch):
    monkeypatch.setattr(dv, "_ext_position_fields",
                        lambda coin, mid, now_ms: (3.0, -1.2))
    state = _build(_perception(), ext_fields=True)
    assert state["short_chase_risk"] == (
        "price 3.0% above 24h low, 1.2% below fast 1h EMA"
        " \u2014 short is entering the bounce, not the flush")


def test_emit_gate_boundaries(monkeypatch):
    def _fake(v):
        monkeypatch.setattr(dv, "_ext_position_fields",
                            lambda coin, mid, now_ms: v)
        return "short_chase_risk" in _build(_perception(), ext_fields=True)

    assert _fake((1.49, -0.01)) is False   # under low-off bar, below EMA
    assert _fake((1.50, -0.01)) is True    # off bar inclusive
    assert _fake((0.0, 0.01)) is True      # above EMA fires alone
    assert _fake((0.0, 0.0)) is False      # neither
    assert _fake((1.4, 0.0)) is False


def test_key_rides_last(monkeypatch):
    # the priced arm APPENDED the key to the finished state; JSON position
    # is part of byte-identity, so it must be the final key.
    monkeypatch.setattr(dv, "_ext_position_fields",
                        lambda coin, mid, now_ms: (2.0, 1.0))
    state = _build(_perception(), ext_fields=True)
    assert list(state.keys())[-1] == "short_chase_risk"


def test_store_fault_is_no_field(monkeypatch):
    # best-effort: a candlestore explosion never breaks the scan and never
    # fakes a value.
    def _boom(coin, mid, now_ms):
        raise RuntimeError("store down")
    monkeypatch.setattr(dv, "_ext_position_fields", _boom)
    assert "short_chase_risk" not in _build(_perception(), ext_fields=True)


# ---- arithmetic (confirmed bars, windows, guards) -------------------------

def test_confirmed_bar_cutoff():
    bars = _bars()
    ts_mid_bar = bars[-1]["t"] + H // 2  # last bar NOT confirmed yet
    off, dist = dv._ext_fields_from_rows(bars, ts_mid_bar, 110.0)
    # without the last bar's low (100+((n-1)%7)-3): manual 24-low over bars[:-1]
    lows = [b["l"] for b in bars[:-1]]
    assert off == max(0.0, (110.0 - min(lows[-24:])) / min(lows[-24:]) * 100)
    off_full, _ = dv._ext_fields_from_rows(bars, bars[-1]["t"] + H, 110.0)
    lows_all = [b["l"] for b in bars]
    assert off_full == (110.0 - min(lows_all[-24:])) / min(lows_all[-24:]) * 100
    assert dist != 0.0


def test_guards_none():
    bars = _bars()
    assert dv._ext_fields_from_rows(bars[:29], bars[-1]["t"] + H, 110.0) is None
    broken = [dict(b) for b in bars]
    broken[-3]["l"] = None
    assert dv._ext_fields_from_rows(broken, bars[-1]["t"] + H, 110.0) is None
    broken2 = [dict(b) for b in bars]
    broken2[-5]["c"] = None
    assert dv._ext_fields_from_rows(broken2, bars[-1]["t"] + H, 110.0) is None
    assert dv._ext_fields_from_rows([], 1_795_600_000_000, 110.0) is None


def test_off_clamped_nonnegative():
    bars = _bars()
    # mid under the 24h window min: gauntlet clamps off to 0 (dist may go
    # negative); a negative off would break the label's "above" wording.
    low_mid = min(b["l"] for b in bars[-24:]) - 5.0
    off, _ = dv._ext_fields_from_rows(bars, bars[-1]["t"] + H, low_mid)
    assert off == 0.0


@pytest.mark.skipif(not HAVE_EXT, reason="gauntlet module (scratch/extf_fields.py) absent")
def test_lockstep_with_gauntlet_module(monkeypatch):
    # THE drift pin: live _ext_position_fields and the gauntlet's fields_at
    # run over identical store rows and must agree to the bit — otherwise
    # live prompts no longer match the arm the +106pp was priced with.
    bars = _bars()
    ts = bars[-1]["t"] + H // 2  # mid-bar: exercises the confirm rule
    monkeypatch.setattr(cs, "series", lambda coin, tf: [dict(b) for b in bars])
    monkeypatch.setattr(dv.time, "time", lambda: ts / 1000.0)  # frozen clock:
    # build_state's now_ms then equals the ref ts — same confirmed-bar cut.
    extf_fields._cache.clear()
    ref = extf_fields.fields_at("BTC", ts, 110.0)
    got = dv._ext_position_fields("BTC", 110.0, ts)
    assert ref is not None and got == ref
    # and the live emission decision equals the gauntlet arm's on the same
    # numbers:
    state = {}
    added = extf_fields.apply_ext_fields(state, "label", *ref)
    if added:
        assert dv.build_state(
            "BTC", _perception(), _tf(), _tf(), _tf(), "0.0100%/hr", "n", [],
            win_rate=0.5, n_closes=10, ext_fields=True
        )["short_chase_risk"] == state["short_chase_risk"]


# ---- facts-only variants (t_ce765165, owner ruling 2026-11-27: give the
# model the data, let it decide). 'facts' drops only the directive clause;
# 'factsx' additionally renames the key to the neutral price_extension —
# the key name is itself interpretation. Knob off + default style stay
# byte-identical (pins above); these arms are measured before any arming.

def test_style_knob_parse():
    assert dv._ext_style_of(False) == "directive"
    assert dv._ext_style_of("false") == "directive"
    assert dv._ext_style_of(True) == "directive"
    assert dv._ext_style_of("true") == "directive"
    assert dv._ext_style_of("facts") == "facts"
    assert dv._ext_style_of("FACTSX") == "factsx"


def test_facts_variant_exact_and_lockstep(monkeypatch):
    monkeypatch.setattr(dv, "_ext_position_fields",
                        lambda coin, mid, now_ms: (2.4, 1.9))
    state = _build(_perception(), ext_fields=True, ext_style="facts")
    assert state["short_chase_risk"] == (
        "price 2.4% above 24h low, 1.9% above fast 1h EMA")
    assert " — short is" not in state["short_chase_risk"]
    if HAVE_EXT:
        ref = {}
        extf_fields.apply_ext_fields(ref, "facts", 2.4, 1.9)
        assert state["short_chase_risk"] == ref["short_chase_risk"]


def test_factsx_neutral_key(monkeypatch):
    monkeypatch.setattr(dv, "_ext_position_fields",
                        lambda coin, mid, now_ms: (2.4, -1.9))
    state = _build(_perception(), ext_fields=True, ext_style="factsx")
    assert "short_chase_risk" not in state
    assert state["price_extension"] == (
        "price 2.4% above 24h low, 1.9% below fast 1h EMA")
    if HAVE_EXT:
        ref = {}
        added = extf_fields.apply_ext_fields(ref, "factsx", 2.4, -1.9)
        assert added == ["price_extension"]
        assert state["price_extension"] == ref["price_extension"]


def test_directive_is_facts_plus_clause(monkeypatch):
    # the rolled-back directive arm stays byte-compatible: directive text ==
    # facts text + the interpretation clause, same key and last position.
    monkeypatch.setattr(dv, "_ext_position_fields",
                        lambda coin, mid, now_ms: (9.9, 5.0))
    p = _perception()
    d = _build(p, ext_fields=True)                       # default style
    f = _build(p, ext_fields=True, ext_style="facts")
    fx = _build(p, ext_fields=True, ext_style="factsx")
    assert d["short_chase_risk"] == (f["short_chase_risk"]
        + " — short is entering the bounce, not the flush")
    assert fx["price_extension"] == f["short_chase_risk"]
    assert list(f)[-1] == "short_chase_risk"
    assert list(fx)[-1] == "price_extension"


def test_facts_variants_share_emit_gate(monkeypatch):
    if not HAVE_EXT:
        pytest.skip("extf_fields module absent")
    for off, dist in ((1.5, 0.0), (1.49, 0.0), (1.4, 0.01), (1.4, 0.0)):
        monkeypatch.setattr(dv, "_ext_position_fields",
                            lambda coin, mid, now_ms, o=off, d=dist: (o, d))
        for style in ("facts", "factsx"):
            state = _build(_perception(), ext_fields=True, ext_style=style)
            keys = {"facts": "short_chase_risk",
                    "factsx": "price_extension"}[style]
            assert (keys in state) == extf_fields.emit_gate(off, dist)
