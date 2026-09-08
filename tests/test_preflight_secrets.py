"""Tests for scripts/preflight_secrets.py — hermetic.

The root conftest.py loads the REAL .env.local into os.environ before every
pytest run, so these tests never touch os.environ's credential values: every
test builds a SYNTHETIC env dict and (for file/git checks) tmp files/repos.
The real .env.local is never read, opened, or leaked.
"""

import os
import stat
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import preflight_secrets  # noqa: E402
from preflight_secrets import _redact, run_preflight  # noqa: E402

# Synthetic values only — none of these are real credentials.
PK = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
WALLET = "0xaaaa000000000000000000000000000000000001"
MASTER = "0xbbbb000000000000000000000000000000000001"
LLM_KEY = "sk-synthetic-llm-key-0123456789"
ACCOUNT = "synthacct"
HF = "hf_synthetic_0123456789"


def good_env() -> dict:
    return {
        "HYPERLIQUID_PRIVATE_KEY": PK,
        "HYPERLIQUID_WALLET_ADDRESS": WALLET,
        "LLM_API_KEY": LLM_KEY,
        "HL_ACCOUNT_NAME": ACCOUNT,
        "HF_TOKEN": HF,
    }


def blocking(findings):
    return [f for f in findings if preflight_secrets.is_blocking(f)]


def warns(findings):
    return [f for f in findings if not preflight_secrets.is_blocking(f)]


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
    )


# ── 1. all required present + distinct → no blocking findings ───────────────

def test_all_present_distinct_passes(tmp_path):
    # Fully clean setup: 0600 env file present, gitless dir (checks 4-5 skip).
    p = tmp_path / ".env.local"
    p.write_text("X=1\n")
    os.chmod(p, 0o600)
    assert run_preflight(good_env(), env_path=str(p)) == []


def test_env_file_absent_is_warn_not_block(tmp_path):
    findings = run_preflight(good_env(), env_path=str(tmp_path / "nope.env"))
    assert blocking(findings) == []
    assert any("not found" in f for f in warns(findings))


# ── 2. each required secret missing/empty → a blocking finding ──────────────

@pytest.mark.parametrize("name", preflight_secrets.REQUIRED_SECRETS)
def test_each_required_missing_blocks(name, tmp_path):
    env = good_env()
    env.pop(name)
    findings = run_preflight(env, env_path=str(tmp_path / "nope.env"))
    assert any(name in f for f in blocking(findings)), name

@pytest.mark.parametrize("name", preflight_secrets.REQUIRED_SECRETS)
def test_each_required_empty_blocks(name, tmp_path):
    env = good_env()
    env[name] = "   "
    findings = run_preflight(env, env_path=str(tmp_path / "nope.env"))
    assert any(name in f for f in blocking(findings)), name

def test_hf_token_missing_is_warn_not_block(tmp_path):
    env = good_env()
    env.pop("HF_TOKEN")
    findings = run_preflight(env, env_path=str(tmp_path / "nope.env"))
    assert blocking(findings) == []
    assert any("HF_TOKEN" in f for f in warns(findings))


# ── 3. containment: WALLET == MASTER → block; distinct / only one → pass ────

def test_wallet_equals_master_blocks(tmp_path):
    env = good_env()
    env["HYPERLIQUID_MASTER_ADDRESS"] = WALLET.upper()  # case-insensitive
    findings = run_preflight(env, env_path=str(tmp_path / "nope.env"))
    assert any("HYPERLIQUID_MASTER_ADDRESS" in f for f in blocking(findings))

def test_wallet_distinct_from_master_passes(tmp_path):
    env = good_env()
    env["HYPERLIQUID_MASTER_ADDRESS"] = MASTER
    findings = run_preflight(env, env_path=str(tmp_path / "nope.env"))
    assert blocking(findings) == []

def test_only_wallet_present_passes(tmp_path):
    # Our single-agent setup: no master address at all.
    assert blocking(run_preflight(good_env(),
                                  env_path=str(tmp_path / "nope.env"))) == []


# ── 4. .env.local mode: 0644 → warn; 0600 → none; absent → warn ─────────────

def test_env_file_0644_warns(tmp_path):
    p = tmp_path / ".env.local"
    p.write_text("X=1\n")
    os.chmod(p, 0o644)
    findings = run_preflight(good_env(), env_path=str(p))
    assert blocking(findings) == []
    assert any("0644" in f or "0o777" in f or "readable" in f for f in warns(findings))

def test_env_file_0600_no_finding(tmp_path):
    p = tmp_path / ".env.local"
    p.write_text("X=1\n")
    os.chmod(p, 0o600)
    findings = run_preflight(good_env(), env_path=str(p))
    assert findings == []


# ── 5. git-tracked .env.local → block; not a repo / allow_no_repo → skip ────

def _init_repo(path: str) -> str:
    _git(path, "init", "-q")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "init")
    return path

def test_env_file_git_tracked_blocks(tmp_path):
    p = tmp_path / ".env.local"
    p.write_text("X=1\n")
    _init_repo(str(tmp_path))
    findings = run_preflight(good_env(), env_path=str(p))
    assert any("git-tracked" in f for f in blocking(findings))

def test_not_a_repo_skips_git_checks(tmp_path):
    p = tmp_path / ".env.local"
    p.write_text("X=1\n")
    os.chmod(p, 0o600)
    # tmp_path is not a git repo (allow_no_repo default) → clean.
    assert run_preflight(good_env(), env_path=str(p)) == []

def test_not_a_repo_allow_no_repo_false_warns(tmp_path):
    p = tmp_path / ".env.local"
    p.write_text("X=1\n")
    os.chmod(p, 0o600)
    findings = run_preflight(good_env(), env_path=str(p), allow_no_repo=False)
    assert blocking(findings) == []
    assert any("not a git repository" in f for f in warns(findings))


# ── 6. secret value in the git index → block; absent → pass ─────────────────

def _value_in_index(repo: str, value: str) -> bool:
    return _git(repo, "grep", "--cached", "-F", value).returncode == 0

def test_secret_value_in_git_index_blocks(tmp_path):
    # Commit a .env file whose content includes the synthetic LLM key value.
    (tmp_path / "leaked.env").write_text(f"LLM_API_KEY={LLM_KEY}\n")
    _init_repo(str(tmp_path))
    assert _value_in_index(str(tmp_path), LLM_KEY)
    findings = run_preflight(good_env(),
                             env_path=str(tmp_path / ".env.local"))
    blocks = blocking(findings)
    assert any("LLM_API_KEY" in f and "git index" in f for f in blocks)
    # Redaction: the VALUE itself never appears in any emitted finding.
    assert all(LLM_KEY not in f for f in findings)

def test_secret_value_not_in_index_passes(tmp_path):
    (tmp_path / "clean.txt").write_text("nothing secret here\n")
    _init_repo(str(tmp_path))
    assert not _value_in_index(str(tmp_path), LLM_KEY)
    findings = run_preflight(good_env(),
                             env_path=str(tmp_path / ".env.local"))
    assert blocking(findings) == []

def test_display_tag_in_docs_is_not_a_leak(tmp_path):
    # Regression (real-env finding 2026-09-08): HL_ACCOUNT_NAME is a human-
    # readable display tag (the repo's own name in production), and its value
    # legitimately appears in README/DEPLOY.md/.env.example. It is 13 chars —
    # long enough to pass MIN_SECRET_SCAN_LEN — so excluding it must be by
    # class (GIT_INDEX_SCAN_SECRETS), not by length.
    (tmp_path / "README.md").write_text(
        f"# hermes-trader\nSet HL_ACCOUNT_NAME={ACCOUNT} before deploy.\n")
    _init_repo(str(tmp_path))
    assert _value_in_index(str(tmp_path), ACCOUNT)  # it IS in tracked docs
    findings = run_preflight(good_env(),
                             env_path=str(tmp_path / ".env.local"))
    assert blocking(findings) == []  # ...and it must NOT block startup
    assert "HL_ACCOUNT_NAME" not in "".join(findings)

def test_wallet_value_in_git_index_blocks(tmp_path):
    # Control: a genuine credential (the wallet address) in the index is a
    # real leak and must still block — only the display tag is exempted.
    (tmp_path / "leaked.env").write_text(f"WALLET={WALLET}\n")
    _init_repo(str(tmp_path))
    assert _value_in_index(str(tmp_path), WALLET)
    findings = run_preflight(good_env(),
                             env_path=str(tmp_path / ".env.local"))
    blocks = blocking(findings)
    assert any("HYPERLIQUID_WALLET_ADDRESS" in f and "git index" in f
               for f in blocks)
    assert all(WALLET not in f for f in findings)  # redaction holds

def test_no_secret_value_in_any_finding(tmp_path):
    # Deliberately plant a finding that could leak: run with the value in the
    # index and assert the emitted text is clean for every secret value.
    (tmp_path / "leaked.env").write_text(f"LLM_API_KEY={LLM_KEY}\n")
    _init_repo(str(tmp_path))
    findings = run_preflight(good_env(),
                             env_path=str(tmp_path / ".env.local"))
    for value in (PK, WALLET, LLM_KEY, ACCOUNT, HF):
        assert all(value not in f for f in findings)


# ── 7. _redact never emits a secret value ────────────────────────────────────

def test_redact_replaces_every_value():
    env = good_env()
    nasty = f"{PK} {WALLET} {LLM_KEY} {HF} {ACCOUNT}"
    out = _redact(nasty, env)
    assert out == "*** *** *** *** ***"
    for value in env.values():
        assert value not in out

def test_redact_noop_without_secrets():
    assert _redact("hello *** world", {}) == "hello *** world"
