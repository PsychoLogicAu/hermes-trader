# Hermes Trader — Production Dockerfile (python:3.13-slim)
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# 1) System deps (no python3-pip, which doesn't exist in 3.13-slim)
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*

# 2) Copy requirements + install deps (cacheable layer)
# First uninstall the CCXT fork hyperliquid (0.4.x) which shadows the SDK's modules
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip uninstall -y hyperliquid || true && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    # Force remove CCXT fork if it snuck in as a transitive dep
    python -m pip uninstall -y hyperliquid || true

# 3) Copy source + editable install (Docker layer caching handles speed)
COPY . .
RUN python -m pip install -e . && python -m pip uninstall -y hyperliquid || true

# 4) Non-root user
RUN useradd --create-home --shell /bin/bash trader && chown -R trader:trader /app
USER trader

CMD ["python", "-m", "hermes_trader", "start"]
