# Hermes Trader — Production Dockerfile (python:3.13-slim)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# 1) System deps
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# 2) Copy requirements + install deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip uninstall -y hyperliquid || true && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip uninstall -y hyperliquid || true

# 3) Setup user, directories, and permissions FIRST (before COPY).
# uid/gid are build args (default 1000:1000 — the conventional first-user id,
# matching Fly.io and CI builders). Build with your host user's id so the
# bind-mounted host dirs (trader-logs, hf-cache, agent-state) stay writable:
#   scripts/build.sh                      # detects $(id -u):$(id -g)
#   docker compose build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
# Deliberately NOT a Debian "system" user: high uids (e.g. 3000 > SYS_UID_MAX 999)
# only warn with -r; a plain useradd with an explicit --uid avoids the warning.
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN addgroup --gid "${GROUP_ID}" trader && \
    useradd --uid "${USER_ID}" --gid "${GROUP_ID}" --home-dir /home/trader --shell /bin/false trader && \
    mkdir -p /home/trader/.hermes/universe_cache && \
    chown -R trader:trader /home/trader && \
    chmod -R 755 /home/trader/.hermes

# 4) Setup directories in app
RUN mkdir -p /app/log && chmod 755 /app/log

# 5) Copy source
COPY . .

# 6) Set ownership to trader user (after all files are copied)
RUN chown -R trader:trader /app

USER trader

CMD ["python", "-m", "hermes_trader", "start"]
