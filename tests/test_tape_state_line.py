"""[tape-state] shadow accrual line — t_12707dad (2026-11-09).

Covers the pure helper (_tape_state_line) and the research() wiring via AST.
Log-only by design: nothing may reach the prompt — pinned here and proven
post-deploy by a prompt-log grep.
"""
from __future__ import annotations

import ast
import inspect

from hermes_trader.agents import research as research_mod
from hermes_trader.agents.research import (
    TAPE_BOX_ATR_MAX,
    TAPE_MIN_BARS,
    _tape_state_line,
)
from hermes_trader.models.types import Candle


def _candles(seq):
    """seq: list of (o, h, l, c) -> Candle list with fixed spacing/volume."""
    return [
        Candle(t=1_700_000_000_000 + i * 3_600_000, o=o, h=h, l=l, c=c, v=100.0)
        for i, (o, h, l, c) in enumerate(seq)
    ]


def _moderate(n, base=100.0, step=2.5):
    """Trending bars: per-bar body ~step, so ATR ~ step."""
    out = []
    px = base
    for _ in range(n):
        out.append((px, px + step, px - step * 0.4, px + step * 0.8))
        px += step * 0.6
    return out


def _quiet(n, base=110.0, amp=0.15):
    out = []
    for k in range(n):
        c = base + (amp if k % 2 == 0 else -amp)
        out.append((base, base + amp * 1.4, base - amp * 1.4, c))
    return out


def test_short_series_returns_none():
    seq = _moderate(TAPE_MIN_BARS - 1)
    assert _tape_state_line("BTC", _candles(seq)) is None


def test_quiet_box_and_low_percentile():
    # 45 volatile/trending bars then 22 quiet bars -> current ATR% sits in the
    # bottom of its own 80-bar window and a quiet 12-bar box is live.
    seq = _moderate(45)
    seq += _quiet(22, base=seq[-1][3])
    line = _tape_state_line("NEAR", _candles(seq))
    assert line is not None
    assert line.startswith("[tape-state] NEAR ")
    fields = dict(p.split("=", 1) for p in line.split()[2:] if "=" in p)
    assert fields["box"] == "QUIET"
    assert int(fields["age"]) < 12  # box formed recently (age in bars)
    pctr = int(fields["pctrank"])
    assert 0 <= pctr <= 40, f"expected low vol percentile, got {pctr}"
    # inside the box: both edge distances positive (close below top, above floor)
    assert float(fields["up"].rstrip("%")) > 0
    assert float(fields["dn"].rstrip("%")) > 0
    assert float(fields["atrpct"]) > 0


def test_broke_up_flag_and_negative_edge():
    # quiet box forms, then the break bar closes above the window's highest
    # high (the window still contains the box at j=i, so edges >= old top).
    seq = _moderate(40)
    seq += _quiet(16, base=seq[-1][3])  # continue from last close (no gap TR)
    top = max(x[1] for x in seq[-12:])
    c_prev = seq[-1][3]
    # wide break bar: its own range blows the quiet test at j=i (edges then
    # come from j=i-1's box, and the close clears that box's top)
    seq.append((c_prev, top + 5.3, c_prev - 0.1, top + 5.0))
    line = _tape_state_line("SOL", _candles(seq))
    assert line is not None
    fields = dict(p.split("=", 1) for p in line.split()[2:] if "=" in p)
    assert fields["box"] == "BROKE_UP"
    assert int(fields["age"]) >= 1
    # broken above: distance-to-top is negative
    assert float(fields["up"].rstrip("%")) < 0


def test_trend_no_box():
    # a steady trend keeps the 12-bar range far above 2.5 x ATR -> box=NONE,
    # and no age/up/dn fields are emitted.
    seq = _moderate(90, step=6.0)
    line = _tape_state_line("BTC", _candles(seq))
    assert line is not None
    fields = dict(p.split("=", 1) for p in line.split()[2:] if "=" in p)
    assert fields["box"] == "NONE"
    assert "age" not in fields and "up" not in fields and "dn" not in fields
    # 2.5x-ATR box test math must be consistent with NONE here
    assert TAPE_BOX_ATR_MAX == 2.5


def test_helper_pure_no_mutation():
    seq = _moderate(45)
    seq += _quiet(22, base=seq[-1][3])
    cands = _candles(seq)
    before = [(c.t, c.o, c.h, c.l, c.c, c.v) for c in cands]
    _tape_state_line("ETH", cands)
    after = [(c.t, c.o, c.h, c.l, c.c, c.v) for c in cands]
    assert before == after
    assert len(cands) == len(seq)


# ── wiring pins (AST; the executor-pinning precedent) ─────────────────────

def test_exactly_one_emit_owner_and_one_call_site():
    src = inspect.getsource(research_mod)
    assert src.count("_tape_state_line") == 2  # def + the single call
    assert src.count("logger.info(_ts_line)") == 1
    # the accrual line format is owned by the helper alone (one f-string)
    assert src.count('f"[tape-state] {coin}"') == 1


def test_emit_inside_research_fail_soft():
    tree = ast.parse(inspect.getsource(research_mod.research))

    # map every node to its enclosing Try chain
    tried: set[int] = set()
    for tnode in ast.walk(tree):
        if isinstance(tnode, ast.Try):
            for inner in ast.walk(tnode):
                tried.add(id(inner))

    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_tape_state_line"
    ]
    assert len(calls) == 1
    assert id(calls[0]) in tried, "call must sit inside try/except (fail-soft)"


def test_no_tape_state_leak_into_prompt_builders():
    # log-only: the user-message builder must never carry the tape-state
    # primitives. ("tape" alone is legitimate prompt prose about the market.)
    src = inspect.getsource(research_mod._build_user_message)
    for token in ("tape-state", "tape_state", "atrpct", "pctrank", "quietbox"):
        assert token not in src
