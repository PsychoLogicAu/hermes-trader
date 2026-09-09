#!/usr/bin/env python3
"""Boot preflight — refuse to start on an unsafe credential setup.

Gate between "misconfigured" and "trading": today the loop starts up and
only fails mid-trade (a missing signing key raises at order time, an empty
wallet address breaks fetch_account_state). This runs BEFORE the first
credential use and BEFORE any network I/O, and fails CLOSED — a bad or
unsafe setup blocks startup with a clear message naming what and where.

Checks (findings name the secret/file, NEVER its value; _redact() is applied
to every emitted line as defense in depth):
  1. required secrets present + non-empty: HYPERLIQUID_PRIVATE_KEY,
     HYPERLIQUID_WALLET_ADDRESS, LLM_API_KEY, HL_ACCOUNT_NAME — a BLOCK
     finding each. HF_TOKEN is OPTIONAL: a WARN finding, not blocking.
  2. containment boundary: if HYPERLIQUID_WALLET_ADDRESS and
     HYPERLIQUID_MASTER_ADDRESS are BOTH present and EQUAL (case-
     insensitive), the signer IS the master wallet (can withdraw) — BLOCK.
     Distinct → pass. Only one present → pass (single-agent setup).
  3. .env.local mode: any group/other bit set (mode & 0o077) → WARN
     (defense in depth; the container may get creds another way).
     File absent → WARN (not blocking).
  4. .env.local not git-tracked (git ls-files --error-unmatch) → BLOCK if
     tracked. Not a git repo (and allow_no_repo) → skip, no finding.
  5. no CREDENTIAL value in the git index (git grep --cached -F, pattern in a
     0600 temp file so the value never reaches argv / a process listing) →
     BLOCK if found. Scans GIT_INDEX_SCAN_SECRETS only (the credential-class
     secrets); the human-readable HL_ACCOUNT_NAME display tag is NOT a
     credential and is expected to appear in .env.example/docs, so it is
     excluded here (still presence-checked in step 1) — otherwise the repo's
     display name in README/DEPLOY.md would false-positive. Values shorter
     than MIN_SECRET_SCAN_LEN are skipped (a short placeholder would
     false-positive on source). Not a git repo (and allow_no_repo) → skip.

Checks 4-5 are safe in a gitless Docker container: they skip via
allow_no_repo. run_preflight() takes an EXPLICIT env mapping (NOT
os.environ) so tests can run on synthetic values — the root conftest.py
loads the real .env.local into os.environ before every pytest run, and this
module must never read or leak the real secrets.

main() prints every finding (redacted) and exits 1 on any BLOCK finding,
0 otherwise (WARNs do not block).
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from typing import List, Mapping

# Missing/empty => startup refused (fail closed).
REQUIRED_SECRETS = (
    "HYPERLIQUID_PRIVATE_KEY",    # hex signing key (hermes_trader/client/exchange.py)
    "HYPERLIQUID_WALLET_ADDRESS", # signer identity (client/hl_client.resolve_user_address)
    "LLM_API_KEY",                # LLM verdicts (agents/research.py)
    "HL_ACCOUNT_NAME",            # HL account tag
)
# Missing => WARN only, never blocks.
OPTIONAL_SECRETS = ("HF_TOKEN",)  # model/HF fetch

# The git-index value scan (check 5) hunts for CREDENTIALS — the secrets whose
# presence in tracked files means a real leak (a private key / wallet address /
# LLM key committed to the repo). HL_ACCOUNT_NAME is a human-readable display
# tag, not a credential: it is EXPECTED to appear in .env.example and docs (and
# here is the repo's own name), so scanning its value would false-positive on
# ordinary tracked documentation. It is still presence-checked (check 1).
GIT_INDEX_SCAN_SECRETS = (
    "HYPERLIQUID_PRIVATE_KEY",
    "HYPERLIQUID_WALLET_ADDRESS",
    "LLM_API_KEY",
)

BLOCKING = "BLOCK"
WARN = "WARN"

# Values shorter than this are almost always short tags (account names), and a
# git-index scan for them would false-positive on ordinary source text.
MIN_SECRET_SCAN_LEN = 8


def _redact(text: str, env: Mapping[str, str]) -> str:
    """Defense in depth: replace every secret VALUE in `text` with *** so a
    buggy finding message can never leak a value. Short values (len < 4,
    e.g. locale flags when the caller passes os.environ) are skipped —
    redacting a single character would mangle every message."""
    for value in env.values():
        if value and len(value) >= 4:
            text = text.replace(value, "***")
    return text


def is_blocking(finding: str) -> bool:
    return finding.startswith(BLOCKING + ": ")


def _git(repo_dir: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True
    )


def _is_git_repo(repo_dir: str) -> bool:
    return _git(repo_dir, "rev-parse", "--is-inside-work-tree").returncode == 0


def run_preflight(env: Mapping[str, str], *, env_path: str = ".env.local",
                  allow_no_repo: bool = True) -> List[str]:
    """Run all credential checks against `env`. Returns finding strings;
    EMPTY == clean. BLOCK: -prefixed findings block startup, WARN: ones do
    not. Findings name the secret/path only, never its value."""
    findings: List[str] = []

    # 1. required secrets present + non-empty (BLOCK class); optionals WARN.
    for name in REQUIRED_SECRETS:
        if not (env.get(name) or "").strip():
            findings.append(
                f"{BLOCKING}: {name} missing or empty — required secret; "
                f"refusing to start without it")
    for name in OPTIONAL_SECRETS:
        if not (env.get(name) or "").strip():
            findings.append(
                f"{WARN}: {name} missing or empty — model/HF fetch "
                f"unavailable (not blocking)")

    # 2. containment boundary: the agent wallet must not BE the master wallet.
    wallet = (env.get("HYPERLIQUID_WALLET_ADDRESS") or "").strip()
    master = (env.get("HYPERLIQUID_MASTER_ADDRESS") or "").strip()
    if wallet and master and wallet.lower() == master.lower():
        findings.append(
            f"{BLOCKING}: HYPERLIQUID_WALLET_ADDRESS equals "
            f"HYPERLIQUID_MASTER_ADDRESS — the signer IS the master wallet "
            f"(can withdraw), so there is no containment boundary; use a "
            f"distinct agent wallet")

    # 3. .env.local file hygiene (defense in depth; WARN class).
    if os.path.exists(env_path):
        mode = os.stat(env_path).st_mode & 0o777
        if mode & 0o077:
            findings.append(
                f"{WARN}: {env_path} is group/other-readable "
                f"(mode {oct(mode & 0o777)}) — expected 0600")
    else:
        findings.append(
            f"{WARN}: {env_path} not found — credentials must come from the "
            f"process environment (not blocking)")

    # 4+5. git containment (defense in depth). Safe in a gitless container:
    # skip silently via allow_no_repo.
    repo_dir = os.path.dirname(os.path.abspath(env_path))
    if _is_git_repo(repo_dir):
        env_rel = os.path.basename(env_path)
        # 4. the env file must not be git-tracked.
        if _git(repo_dir, "ls-files", "--error-unmatch", env_rel).returncode == 0:
            findings.append(
                f"{BLOCKING}: {env_rel} is git-tracked — a secret file would "
                f"be committed; untrack it and rotate the credentials")
        # 5. no credential value in the git index. The pattern goes in a 0600
        #    temp file (git grep -f) so the value never appears on argv / in a
        #    process listing. Credential-class secrets only (see
        #    GIT_INDEX_SCAN_SECRETS) — the HL_ACCOUNT_NAME display tag is
        #    expected in docs/.env.example and would false-positive.
        for name in GIT_INDEX_SCAN_SECRETS:
            value = (env.get(name) or "").strip()
            if len(value) < MIN_SECRET_SCAN_LEN:
                continue
            fd, pattern_path = tempfile.mkstemp(prefix="preflight-secret-")
            try:
                os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)  # 0600
                with os.fdopen(fd, "w") as f:
                    f.write(value)
                r = _git(repo_dir, "grep", "--cached", "-F", "-f", pattern_path)
                if r.returncode == 0:
                    findings.append(
                        f"{BLOCKING}: {name} value found in the git index — "
                        f"untrack the file containing it and rotate the "
                        f"credential")
            finally:
                try:
                    os.unlink(pattern_path)
                except OSError:
                    pass
    elif not allow_no_repo:
        findings.append(
            f"{WARN}: {repo_dir} is not a git repository — skipped the git "
            f"containment checks")

    return findings


def main() -> int:
    env = os.environ
    findings = run_preflight(env)
    for line in findings:
        print(_redact(line, env))
    return 1 if any(is_blocking(f) for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
