"""T2.3 — sub-floor entries round UP to the HL minimum instead of dropping.

Hermetic (in-memory harness, no network). Pins the bump branch of
maybe_execute: when a sized entry falls SHORT of the HL minimum notional it
is rounded UP to the floor (within min_order_bump_max_mult — code default
2.0, hot-read per call; 0 = the legacy skip-only behavior) instead of being
dropped as below_min_order_notional. The added risk is bounded by the floor
itself. Propagation: trade_notional is the single sizing variable, so the
rounded-up notional flows into entry_size_for_notional, the ledger size_usd,
and the DSL position_notional unchanged.
"""
import pytest

from test_cleanup import _analysis, _exec_baseline

# Harness floor: test_cleanup._exec_baseline pins
# min_entry_notional_usd -> 10.5 and get_hl_price -> 100.0.
FLOOR = 10.5
MID = 100.0


def _run(monkeypatch, equity, frac, lev, bump, cap=100000.0):
    """Drive maybe_execute through the legacy sizing path (conviction off).

    trade_notional = equity * frac * lev, clamped to `cap` BEFORE the floor
    check (the production ordering). Returns (result, notionals, captured):
    notionals records what the floor check actually sent to
    entry_size_for_notional (empty when the trade skipped before sizing).
    """
    ex, captured, _ = _exec_baseline(
        monkeypatch,
        cfg_overrides={
            "leverage": lev,
            "equity_fraction_per_trade": frac,
            "conviction_sizing": False,
            "max_trade_notional_usd": cap,
            "min_order_bump_max_mult": bump,
        },
        state_overrides={"equity": equity},
    )
    notionals = []
    monkeypatch.setattr(
        ex, "entry_size_for_notional",
        lambda c, n, mid: notionals.append(n) or n / mid)
    r = ex.maybe_execute(_analysis())
    return r, notionals, captured


# ── THE FIX: a sizing intent just short of the floor executes AT the floor ──
def test_sub_floor_rounds_up_to_floor(monkeypatch):
    # The real xyz:CXMT case: sized $6.76 vs a ~$10.50 floor, 2x ceiling.
    # 6.76 * 2.0 = 13.52 >= 10.50 -> bump; previously dropped entirely.
    r, notionals, captured = _run(monkeypatch, 1000.0, 0.00676, 1, 2.0)
    assert r["executed"] is True, r
    assert notionals == [pytest.approx(FLOOR)], notionals
    # size_in_coin computed from the FLOOR, not the original intent.
    assert captured["size"] == pytest.approx(FLOOR / MID)
    assert r["order_id"] == "OID1"


def test_bump_never_exceeds_floor(monkeypatch):
    # The bump lands exactly on the floor — never above it (bounded risk).
    r, notionals, _ = _run(monkeypatch, 1000.0, 0.009, 1, 2.0)  # sized $9.00
    assert r["executed"] is True, r
    assert notionals == [pytest.approx(FLOOR)]


# ── BELOW THE CEILING: a genuinely tiny intent is still skipped ────────────
def test_far_below_ceiling_still_skips(monkeypatch):
    # sized $1.00: 1.00 * 2.0 = 2.00 < 10.50 -> intent too small to express.
    r, notionals, _ = _run(monkeypatch, 1000.0, 0.001, 1, 2.0)
    assert r["executed"] is False
    assert r["reason"].startswith("below_min_order_notional"), r
    assert notionals == []  # never reached sizing


# ── bump_max = 0: the kill switch restores the legacy skip-only behavior ───
@pytest.mark.parametrize("bump", [0.0, None])
def test_bump_disabled_restores_old_skip(monkeypatch, bump):
    # The very $6.76 case that now bumps: with the bump off it drops, as before.
    r, notionals, _ = _run(monkeypatch, 1000.0, 0.00676, 1, bump)
    assert r["executed"] is False
    assert r["reason"].startswith("below_min_order_notional"), r
    assert notionals == []


# ── AT/ABOVE the floor: untouched (no bump, no behavior change) ────────────
@pytest.mark.parametrize("sized,frac", [
    (10.50, 0.0105),  # exactly AT the floor — the < check is strict
    (20.00, 0.02),    # comfortably above
])
def test_at_or_above_floor_untouched(monkeypatch, sized, frac):
    r, notionals, captured = _run(monkeypatch, 1000.0, frac, 1, 2.0)
    assert r["executed"] is True, r
    assert notionals == [pytest.approx(sized)]
    assert captured["size"] == pytest.approx(sized / MID)


# ── CAP ORDERING: the notional cap always binds before the floor check ─────
def test_trade_at_cap_is_not_bumped(monkeypatch):
    # Sized exactly at the cap (cap >= floor, the realistic case): the cap
    # clamp runs first, so the trade sits ABOVE the floor and the bump branch
    # never fires — it executes at the cap, untouched.
    r, notionals, captured = _run(monkeypatch, 1000.0, 0.30, 1, 2.0, cap=300.0)
    assert r["executed"] is True, r
    assert notionals == [pytest.approx(300.0)]
    assert captured["size"] == pytest.approx(300.0 / MID)


def test_bump_above_cap_is_blocked_by_notional_cap_gate(monkeypatch):
    # Degenerate config (cap below the exchange floor): the cap clamps the
    # intent to $5 first, the bump then rounds to the $10.50 floor — ABOVE the
    # cap — but the per-trade notional gate blocks execution. So even in this
    # corner a bumped trade can never be sent above the cap.
    r, notionals, _ = _run(monkeypatch, 1000.0, 0.10, 1, 3.0, cap=5.0)
    assert r["executed"] is False, r
    assert notionals == [pytest.approx(FLOOR)]  # the bump fired (5*3 >= 10.5)
    assert any("exceeds cap" in b for b in r.get("blocked_by", [])), r
