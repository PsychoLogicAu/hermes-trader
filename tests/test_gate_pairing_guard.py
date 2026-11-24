"""t_5f7a5b5b — gate-pairing config guard tests.

Pins scripts/gate_pairing_check.py: the pairing rule from config-repo commit
5ea1d4c ("classic min_ai_confidence AND runner min_confidence move together so
one gate can't re-block the other's admits") enforced per referenced seat,
mode-aware across the deployed gate code and the conf-floor-precedence code
(cc86fe00, branch fix/conf-floor-precedence).

NOTE on the card's original acceptance ("profile entry_min_short_confidence
0.60 + runner_entry_gate.min_confidence 0.65 fails; 0.60/0.60 passes"): that
predicate was written before t_c002f54c established that under the DEPLOYED
runner, min_confidence=0.65 vs short bar 0.60 is the CORRECT live shape (the
side-aware seat-bar branch admits shorts between the bars by design) — the
parent handoff repointed the guard. The failing fixture here is therefore the
actual one-sided failure mode measured by t_552a0b2c: the CLASSIC floor
(min_ai_confidence, profile gate_bars overlay included) above the seat short
bar. The 0.60/0.60 pairing passes under --mode precedence (the cc86fe00
end-state) and correctly fails LONG-ENFORCEMENT under deployed semantics.
"""

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "scripts", "gate_pairing_check.py")
CONFIG_REPO = os.path.expanduser("~/src/hermes-trader-config")


def _fixture(tmp_path, *, min_ai_confidence, gen, sb_short=None, sb_long=None,
             gate_bars_extra=None, cohort_admissions=None, extra_profiles=None):
    """Write a .agent-config.json + config-models/ tree; seat 'seatx' is the
    referenced primary."""
    cfg = {
        "mode": "LIVE",
        "min_ai_confidence": min_ai_confidence,
        "llm": {"primary_engine": {"engine": "systemone", "model_profile": "seatx"}},
        "runner_entry_gate": {"enabled": True, "allow_shorts": True,
                              "min_confidence": gen},
    }
    if cohort_admissions is not None:
        cfg["runner_entry_gate"]["cohort_admissions"] = cohort_admissions
    prof = {"model": "seatx", "base_url": "http://x"}
    bars = {}
    if sb_short is not None:
        prof["entry_min_short_confidence"] = sb_short
    if sb_long is not None:
        prof["entry_min_confidence"] = sb_long
    gate_bars = {"min_confidence": gen}
    if gate_bars_extra:
        gate_bars.update(gate_bars_extra)
    bars["runner_entry_gate"] = gate_bars
    if min_ai_confidence is not None:
        bars["min_ai_confidence"] = min_ai_confidence
    prof["gate_bars"] = bars
    models = tmp_path / "config-models"
    models.mkdir()
    (models / "seatx.json").write_text(json.dumps(prof))
    for name, body in (extra_profiles or {}).items():
        (models / f"{name}.json").write_text(json.dumps(body))
    (tmp_path / ".agent-config.json").write_text(json.dumps(cfg))
    return tmp_path


def _run(tmp_path, *args):
    return subprocess.run(
        [sys.executable, SCRIPT, "--config", str(tmp_path / ".agent-config.json"),
         "--models", str(tmp_path / "config-models"), *args],
        capture_output=True, text=True,
    )


def test_live_paired_shape_passes_deployed(tmp_path):
    # The live shape applied by t_c002f54c: classic 0.60, runner general 0.65,
    # seat bars long 0.65 / short 0.60, mechanical cohort key present.
    _fixture(tmp_path, min_ai_confidence=0.60, gen=0.65, sb_short=0.60, sb_long=0.65,
             cohort_admissions={"mechanical/aligned_short_live": 0.0})
    r = _run(tmp_path, "--mode", "deployed")
    assert r.returncode == 0, r.stderr


def test_one_sided_short_edit_fails(tmp_path):
    # The pre-t_c002f54c state (t_552a0b2c): profile short bar 0.60 while the
    # classic gate (top-level AND profile gate_bars) sits at 0.65 — the classic
    # gate re-vetoes every 0.60-0.64 SHORT. This is the card's failing case,
    # restated against the deployed gate algebra (parent repoint).
    _fixture(tmp_path, min_ai_confidence=0.65, gen=0.65, sb_short=0.60, sb_long=0.65)
    r = _run(tmp_path, "--mode", "deployed")
    assert r.returncode == 1
    err = r.stderr
    assert "SHORT-PAIRING" in err
    # error names the pairing rule and both offending values + key names
    assert "min_ai_confidence=0.65" in err
    assert "entry_min_short_confidence=0.60" in err
    assert "5ea1d4c" in err


def test_paired_065_065_passes(tmp_path):
    # Owner-flip-A shape (t_40bcf04d option A): everything at 0.65.
    _fixture(tmp_path, min_ai_confidence=0.65, gen=0.65, sb_short=0.65, sb_long=0.65)
    r = _run(tmp_path, "--mode", "deployed")
    assert r.returncode == 0, r.stderr


def test_paired_060_060_passes_only_under_precedence(tmp_path):
    # The audit's end-state (0.60 general bars) is the correct shape ONCE
    # cc86fe00 ships — and genuinely unsafe under deployed code (opens
    # 0.60-0.64 LONGs through both gates), so the guard must fail it there.
    _fixture(tmp_path, min_ai_confidence=0.60, gen=0.60, sb_short=0.60, sb_long=0.65)
    r = _run(tmp_path, "--mode", "precedence")
    assert r.returncode == 0, r.stderr
    r = _run(tmp_path, "--mode", "deployed")
    assert r.returncode == 1 and "LONG-ENFORCEMENT" in r.stderr


def test_dangerous_long_edit_fails_deployed(tmp_path):
    # Lowering ONLY the runner general bar below the long seat bar while the
    # classic floor sits below it too: the seat-bar branch goes inert and
    # sub-floor LONGs pass both gates (the deviation t_c002f54c refused).
    _fixture(tmp_path, min_ai_confidence=0.60, gen=0.55, sb_short=0.60, sb_long=0.65)
    r = _run(tmp_path, "--mode", "deployed")
    assert r.returncode == 1 and "LONG-ENFORCEMENT" in r.stderr
    assert "entry_min_confidence=0.65" in r.stderr
    assert "min_confidence=0.55" in r.stderr


def test_long_classic_veto_fails(tmp_path):
    # Mirror one-sided LONG edit: classic floor above the long seat bar
    # (short bar lifted to match, isolating the LONG-PAIRING rule).
    _fixture(tmp_path, min_ai_confidence=0.70, gen=0.65, sb_short=0.70, sb_long=0.65)
    r = _run(tmp_path, "--mode", "deployed")
    assert r.returncode == 1 and "LONG-PAIRING" in r.stderr


def test_cohort_override_waives_veto(tmp_path):
    # Explicit per-seat cohort_admissions => body's escape hatch: veto rules
    # waived for THAT seat.
    _fixture(tmp_path, min_ai_confidence=0.65, gen=0.65, sb_short=0.60, sb_long=0.65,
             cohort_admissions={"seatx/some_lane": 0.0})
    r = _run(tmp_path, "--mode", "deployed")
    assert r.returncode == 0, r.stderr
    assert "WAIVED" in r.stdout


def test_foreign_cohort_key_does_not_waive(tmp_path):
    # Waiver is model-scoped: the live 'mechanical/...' key must not blank the
    # guard for LLM seats.
    _fixture(tmp_path, min_ai_confidence=0.65, gen=0.65, sb_short=0.60, sb_long=0.65,
             cohort_admissions={"mechanical/aligned_short_live": 0.0})
    r = _run(tmp_path, "--mode", "deployed")
    assert r.returncode == 1 and "SHORT-PAIRING" in r.stderr


def test_bar_sanity_malformed_side_bar(tmp_path):
    # A malformed seat bar is silently fail-opened by the resolvers — the
    # guard must name it.
    _fixture(tmp_path, min_ai_confidence=0.60, gen=0.65, sb_long=0.65)
    prof_path = tmp_path / "config-models" / "seatx.json"
    prof = json.loads(prof_path.read_text())
    prof["entry_min_short_confidence"] = "0.6o"
    prof_path.write_text(json.dumps(prof))
    r = _run(tmp_path, "--mode", "deployed")
    assert r.returncode == 1 and "BAR-SANITY" in r.stderr


def test_unreferenced_profile_not_checked_until_all_profiles(tmp_path):
    # Retired seats (e.g. d3-27b-artemis, long bar 0.50) keep legacy bars and
    # must not block deploys while unreferenced; --all-profiles surfaces them.
    legacy = {"model": "old-seat", "entry_min_confidence": 0.50}
    _fixture(tmp_path, min_ai_confidence=0.60, gen=0.65, sb_short=0.60, sb_long=0.65,
             extra_profiles={"old-seat": legacy})
    assert _run(tmp_path, "--mode", "deployed").returncode == 0
    r = _run(tmp_path, "--mode", "deployed", "--all-profiles")
    assert r.returncode == 1 and "old-seat" in r.stderr


@pytest.mark.skipif(not os.path.isfile(os.path.join(CONFIG_REPO, ".agent-config.json")),
                    reason="config repo not present (CI)")
def test_live_config_repo_passes():
    # The live paired config (63b6944) must pass — pins guard-vs-live green
    # and pins the overlay algebra against the real files.
    r = subprocess.run(
        [sys.executable, SCRIPT, "--config", os.path.join(CONFIG_REPO, ".agent-config.json"),
         "--models", os.path.join(CONFIG_REPO, "config-models"), "--mode", "deployed",
         "--quiet"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
