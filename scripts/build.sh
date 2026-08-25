#!/usr/bin/env bash
# Build the hermes-trader image with the container user matched to this
# host's uid:gid, so the bind-mounted volume dirs (trader-logs, hf-cache,
# agent-state) stay writable from the container.
#
# Usage:
#   scripts/build.sh                 # build with $(id -u):$(id -g)
#   scripts/build.sh --no-cache
#
# Override the detected ids (e.g. building for a different user's host):
#   USER_ID=1000 GROUP_ID=1000 scripts/build.sh
#
# Manual equivalent:
#   docker compose build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) hermes-trader

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

USER_ID="${USER_ID:-$(id -u)}"
GROUP_ID="${GROUP_ID:-$(id -g)}"

echo "[build] container user will be ${USER_ID}:${GROUP_ID} (host user: $(id -un) = $(id -u):$(id -g))"

# --- Pre-flight: volume dirs must exist and be owned by the build user ---
# If a dir is missing, Docker auto-creates it as root:root on `up` and the
# container crash-loops with PermissionError. Fix that up front.
BAD=0
for dir in trader-logs hf-cache agent-state duel-logs; do
  if [ ! -d "$dir" ]; then
    echo "[build] creating $dir"
    mkdir -p "$dir"
  fi
  owner="$(stat -c '%u:%g' "$dir" 2>/dev/null || stat -f '%u:%g' "$dir" 2>/dev/null || echo '?')"
  if [ "$owner" != "${USER_ID}:${GROUP_ID}" ]; then
    echo "[build] ! ${dir} is owned by ${owner}, expected ${USER_ID}:${GROUP_ID}"
    BAD=1
  fi
done
if [ "$BAD" -ne 0 ]; then
  echo "[build] fix with: sudo chown -R ${USER_ID}:${GROUP_ID} trader-logs hf-cache agent-state duel-logs"
  exit 1
fi

# Pre-create the duel log file so the container's append writes never race a
# fresh-inode creation (same rationale as the directory pre-flight above).
touch duel-logs/hermes-trader-duel.jsonl

docker compose build "$@" --build-arg USER_ID="$USER_ID" --build-arg GROUP_ID="$GROUP_ID" hermes-trader