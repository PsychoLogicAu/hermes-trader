# Hermes-Trader

> Autonomous crypto trading agent on Hyperliquid (perpetuals), operated as a Docker Compose stack. Scans markets, runs cheap TA filters, calls a self-hosted LLM only on confirmed setups, enforces 11+ risk gates, and manages dynamic exits via DSL — all without human intervention.

Originally forked from [Julian-dev28/hermes-trader](https://github.com/Julian-dev28/hermes-trader); this repo has since diverged significantly with new LLM wiring, Docker deployment, advanced risk gates, shadow-mode signals, and trailing TP.

**What it does:**
Scans 500+ Hyperliquid markets, fires statistical triggers on price/volume/breakout signals, runs a zero-cost multi-timeframe TA filter (`analyze_perception`), and only calls the LLM on CONFIRMED setups (or momentum bursts). The LLM acts as an analyst — not an oracle. An 11-gate risk framework enforces discipline, and a DSL exit engine manages trailing stops, profit locking, and timeouts.

---

## Quick Start

All commands from the repo root: `/home/oknight/src/hermes-trader`.

```bash
# Start (builds if needed)
docker compose up -d

# Restart (apply config changes that need rebuild, or if container got stale)
docker compose down && docker compose up -d

# Stop
docker compose down

# Tail logs live
docker compose logs -f hermes-trader

# Check container health
docker compose ps
```

Logs: `trader-logs/trader.log` (daily rotation, date-stamped backups).

---

## The problem it solves

Trading signals appear constantly — 5-minute spikes, hourly trends, daily breakouts. Most systems call expensive AI on every signal, burning tokens on noise. Hermes-Trader solves this by separating cheap statistical analysis from expensive AI reasoning:

1. **Scan** — parallel batch scanning with volume pre-filtering, within HL's 1200 weight/min rate limit
2. **TA Filter** — multi-timeframe indicators (EMA, RSI, ATR, ADX, volume) — zero AI cost
3. **AI Research** — only on CONFIRMED signals or fired momentum bursts; full reasoning + structured JSON verdict
4. **Execution** — ATR equal-risk sizing, Hyperliquid order normalization, DSL dynamic exits (loss protection → profit locking)
5. **Discovery** — built-in Hyperfeed Discovery replicates Smart Money leaderboards and whale signals

The LLM is never called on a raw trigger. Cheap math gates the expensive model every time.

---

## Architecture

```
+---------------------------------------------------------------+
|          hermes-trader — autonomous trading pipeline          |
|                                                               |
|  Scan → TA Filter → AI Research → Risk Gates → Execute → DSL Monitor ──▶ Auto-Close
|        (cheap)         (expensive)     (11+ gates)           (per-tick, trailing TP)
|               |
|               └── Hyperfeed Discovery
|                   Leaderboard • Smart Money • OI Anomaly • Whale Tracking
+---------------------------------------------------------------+
```

### Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────┐
│  Perception │───>│  TA Filter   │───>│   AI Research   │───>│  Risk Gates  │───>│   Executor  │
│   Scanner   │    │  (TA Filter) │    │ (local LLM API) │    │  (11+ gates) │    │ HL + DSL TP │
│ 5m/1h/4h    │    │  EMA/RSI/ATR│    │ Verdict + Price │    │  gates       │    │ SL / Trailing│
│ Volume-N    │    └──────────────┘    └─────────────────┘    └──────────────┘    └──────────────┘
└─────────────┘
     │
     ├── Hyperfeed Discovery (leaderboard, whale index, OI anomaly)
     │     ↳ smart_money_concentration(), oi_funding_anomaly()
     │     ↳ discovery_get_top_traders(), leaderboard_get_trader_positions()
     └── Rate-Limit Pipeline (1200 weight/min — batch + cache)
```

Key differences from the original repo:
- **LLM:** Self-hosted via `llama-net` (Qwen-based), not OpenRouter.
- **Deployment:** Docker Compose container (`hermes-trader`), not direct Python/`restart.sh`.
- **Signals:** Shadow-mode signal suite (GEX, whale flow, FINRA short volume, news catalyst) wired into research and enforcement.
- **Risk gates:** Enhanced with high-confidence bypasses (`bypass_low_volume`, `bypass_late_trend_chase`), funding-regime overlays, and trailing TP.

---

## Key Features

### Dockerized Deployment

- **Container**: `docker-compose.yml` runs the entire trading loop as `hermes-trader`.
- **Networks**: Connects to external `llama-net` for self-hosted LLM access.
- **Volume mounts**:
  - `.agent-config.json` → `/app/.agent-config.json` (config)
  - `.env.local` → `/app/.env.local` (credentials, LLM endpoints)
  - `trader-logs/` → `/app/log` (daily-rotated logs)
- **Persistence**: DSL state (`.dsl-state.json`), agent memory (`.agent-memory.json`), and session logs persist across restarts.

### Local LLM Integration

- Calls a self-hosted Qwen-based model via `llama-net`, configured via:
  - `.env.local`: `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL` (default: local endpoint).
- Resilient to credit/API outages: no OpenRouter dependency.
- On a 402 with affordability hint, retries ONCE with a smaller `max_tokens` budget so the bot degrades rather than going blind.

### Shadow-Mode Signal Suite

Free positioning signals gathered per candidate, logged without blocking execution:

- **GEX (Dealer Gamma Exposure)**: `options_gex.py` — parses CBOE delayed options to compute net dealer gamma, call/put walls, gamma flip, and max pain. Used to surface structural context to the AI and to veto/boost forced-override entries.
- **Crypto Whale Flow**: `crypto_whale.py` — Binance aggTrades rolling window detecting whale buying/selling bias.
- **FINRA Short Volume**: `short_volume.py` — identifies crowded short squeezes for equity perps (HIP-3).
- **News Catalyst**: `news_catalyst.py` — GDELT-based breaking-news / surge detection.
- **Chronos-2 Forecast** (`chronos_signal.py`): Amazon Chronos-2 zero-shot time-series forecasting on price history. Runs async as a daemon thread (CPU), TTL-cached per coin. Currently SHADOW-ONLY: logs forecast direction vs LLM verdict (`ALIGN`/`MISMATCH`) for forward-validation; NOT fed to the LLM prompt and NOT used as a gate. Configuration via `.agent-config.json` under `chronos_signal`.

These signals are:
- Fed into the AI research prompt so the LLM's verdict reflects real positioning.
- Used by `signal_enforcement` for Veto/Boost on the forced-override path.
- Run async (`run_shadow_async`) off the hot execute path so they never amplify latency.

### Enhanced Risk Gates

Beyond the original 11 gates:

- **High-confidence liquidity bypass**: `bypass_low_volume` — allows a trade below the standard liquidity floor if AI confidence ≥ `bypass_low_volume_min_conf` (default 0.85).
- **High-confidence trend-chase bypass**: `bypass_late_trend_chase` — allows a late-chase trade if AI confidence ≥ `bypass_late_trend_chase_min_conf` (default 0.80).
- **Funding-regime overlay**: `market_regime_gate` now enforces symmetric counter-funding discipline:
  - Trading against the crowd requires elevated confidence (`max(counter_regime_min_conf, 0.85)`).
  - Trading with the crowd faces a "squeeze risk" check (`crowded_with_min_conf`).
- **Give-back breaker**: `daily_giveback_gate` — once daily PnL peaks ≥ `daily_giveback_min_peak_usd`, halts NEW entries if it retraces > `daily_giveback_halt_pct` from peak (protects green days from round-tripping).
- **GEX Veto**: For equity perps (HIP-3), a forced-override LONG near a gamma pin wall can be suppressed (`gex_override_caution`) so the override path doesn't force entries into mean-revert chop.
- **Whale Veto / Boost**: Large whale sell flows veto long forced-entries; large whale buys can boost (lower) the override bar.

### Dynamic Trailing TP (via DSL Heartbeat)

- Every heartbeat (~60s), the DSL engine adjusts server-side TP orders upward when trailing conditions are met.
- Configured via `.agent-config.json` under `dsl_exit.trailing_tp`:
  - `trail_pct_from_peak`: distance from peak (default 3%)
  - `max_tp_pct_from_entry`: maximum target cap
- Only adjusts if trailing level > current TP + minimum move, so it doesn't ping-pong.
- Full reconciliation with live exchange positions each cycle; persists across restarts via `.dsl-state.json`.

---

## Configuration

Two config files:

| File | Where it goes | You… | When it's read |
|------|---------------|------|----------------|
| **`.agent-config.json`** | Repo root (tracked) | **Edit** (don't recreate) | At container restart (volume-mounted into container) |
| **`.env.local`** | Repo root (gitignored) | **Create** from `.env.example` | At container build/startup |

### `.env.local`

Copy `.env.example` → `.env.local` and set your values:

```bash
# ── LLM (local via llama-net) ─────────────────────────────────────
LLM_MODEL=qwen3.6-27b-architect-polaris2-fable-b-nvfp4-mtp
LLM_API_KEY=your-local-api-key-if-needed
LLM_BASE_URL=http://your-llm-endpoint/api/v1   # defaults to local llama-net

# ── Hyperliquid ────────────────────────────────────────────────────
HYPERLIQUID_WALLET_ADDRESS=0x...          # required
HYPERLIQUID_PRIVATE_KEY=0x...             # required

# ── News (optional — enables news catalyst in research + gates) ─────
BRAVE_API_KEY=BSA...                      # optional; without it news_context = "no news"

# ── Scan tuning (optional — defaults shown) ────────────────────────
HERMES_SCAN_INTERVAL=60
HERMES_MAX_MARKETS=60
HERMES_BATCH_SIZE=20
HERMES_BATCH_SLEEP=0.3
HERMES_WATCHDOG_TIMEOUT_S=600
```

### `.agent-config.json` — live risk settings

All trading behaviour and risk limits live here. Edited on the host; the container sees it via volume mount. **Restart required** after changes:

```bash
cd /home/oknight/src/hermes-trader && docker compose restart hermes-trader
```

Current live shape (abridged — full file has all DSL, shadow signal, and gate settings):

```json
{
  "mode": "LIVE",
  "enable_crypto": true,
  "enable_hip3": false,
  "equity_fraction_per_trade": 0.08,
  "leverage": 5,
  "max_trade_notional_usd": 35,
  "tp_scale_fraction": 0.5,
  "max_concurrent": 5,
  "max_total_notional_pct": 2.0,
  "max_daily_loss_usd": -20,
  "daily_giveback_halt_pct": 0.5,
  "daily_giveback_min_peak_usd": 25.0,
  "min_ai_confidence": 0.7,
  "counter_regime_min_conf": 0.8,
  "min_market_volume_usd": 3500000,
  "min_short_volume_usd": 50000000,
  "cooldown_min": 30,
  "held_research_interval_min": 3,
  "research_cooldown_min": 3,
  "coin_blocklist": ["TON", "TRX"],
  "max_crypto_long_correlated": 2,
  "dsl_exit": {
    "max_loss_pct": 5.0,
    "max_loss_roe_pct": 30.0,
    "protect_pct": 1.0,
    "trailing_tp": {
      "enabled": true,
      "trail_pct_from_peak": 0.03,
      "max_tp_pct_from_entry": 40.0
    },
    "phase2_tiers": [
      {"pct_above_entry": 8.0, "retrace_threshold": 0.35},
      {"pct_above_entry": 15.0, "retrace_threshold": 0.4}
    ]
  },
  "runner_entry_gate": {
    "enabled": true,
    "bypass_late_trend_chase": true,
    "bypass_late_trend_chase_min_conf": 0.8,
    "bypass_low_volume": true,
    "bypass_low_volume_min_conf": 0.85
  },
  "atr_risk_sizing": {
    "enabled": true,
    "risk_per_trade_pct": 0.02,
    "sizing_basis": "primary_stop"
  },
  "shadow_signals": {
    "enabled": true,
    "gex": true,
    "short_volume": true,
    "crypto_whale": true,
    "news": true,
    "whale_window_min": 15
  },
  "chronos_signal": {
    "enabled": true,
    "debug": false,
    "model_id": "amazon/chronos-2",
    "device": "cpu",
    "context_length": 100,
    "forecast_horizon": 48,
    "cache_ttl_seconds": 300,
    "timeout_seconds": 30,
    "quantile_levels": [0.1, 0.5, 0.9],
    "num_samples": 50
  }
}
```

Key parameters:

| Key | Meaning |
|-----|---------|
| `mode` | `LIVE` = real money; `OFF` = analyse-only (exits still monitored) |
| `max_trade_notional_usd` | Hard ceiling per trade notional |
| `max_concurrent` | Max open positions |
| `max_total_notional_pct` | Max portfolio notional as multiple of equity |
| `max_daily_loss_usd` | Hard daily loss floor (triggers flatten-all killswitch) |
| `daily_giveback_*` | Once day peaks ≥ `min_peak`, halts NEW entries if retraced > `halt_pct` |
| `min_ai_confidence` | Minimum LLM confidence to execute |
| `leverage` | Leverage ceiling per trade |
| `cooldown_min` | Min minutes between trades on the same coin |
| `min_market_volume_usd` | Liquidity floor (with high-conf bypass option) |
| `min_short_volume_usd` | Deeper liquidity floor for shorts |
|| `coin_blocklist` | Never trade these symbols |
|| `runner_entry_gate.*` | Runner surface gate thresholds (late-chase / low-vol bypass) |
|| `atr_risk_sizing.*` | ATR-based equal-risk sizing parameters |
|| `dsl_exit.*` | DSL trailing stop + trailing TP configuration |
|| `shadow_signals.*` | Toggle individual free signals in the shadow suite |
|| `chronos_signal.*` | Chronos-2 forecasting shadow signal (shadow-only logging; `enabled` toggles it) |

---

## Logs and Monitoring

All logs go to `trader-logs/trader.log` (inside the container: `/app/log/trader.log`).

```bash
# Tail live
docker compose logs -f hermes-trader

# Check last executed trades with reasons
grep "executed.*True" trader-logs/trader.log | tail -10

# View trades blocked by gates
grep "runner_gate_blocked" trader-logs/trader.log | tail -20

# Verify a config value is live in the container
docker compose exec hermes-trader sh -c 'cat /app/.agent-config.json | python -c "import json,sys; c=json.load(sys.stdin); print(c.get(\"max_trade_notional_usd\"))"'
```

---

## Troubleshooting

- **Container not starting**: Check `docker compose logs hermes-trader` for LLM connection errors or config syntax issues.
- **No trades executing**: Check `runner_gate_blocked` reasons in logs — likely liquidity, late-chase gate, confidence threshold, or daily-loss killswitch.
- **Config not applying**: Restart the container (`docker compose restart hermes-trader` or full down/up).
- **LLM unreachable**: Verify `llama-net` exists (`docker network ls`) and the container is connected.

---

## Project Structure

```
hermes-trader/
├── hermes_trader/                  # Core agent package
│   ├── agents/                    # Agent logic
│   │   ├── perception.py          # Volume-filtered parallel scanner
│   │   ├── ta_filter.py           # Pre-AI multi-TF TA filter
│   │   ├── research.py            # AI research pipeline (local LLM)
│   │   ├── executor.py            # Trade execution + DSL registration
│   │   ├── risk_gates.py          # 11+ risk gates
│   │   ├── dsl_exit.py            # Two-phase trailing stop + trailing TP
│   │   ├── hyperfeed.py           # Discovery API (leaderboard, whale index)
│   │   ├── market_regime.py       # Regime detection + funding overlays
│   │   ├── options_gex.py         # GEX / max-pain / gamma walls from CBOE
│   │   ├── crypto_whale.py        # Whale order-flow (Binance aggTrades)
│   │   ├── short_volume.py        # FINRA short volume signals
│   │   ├── news_catalyst.py       # GDELT breaking-news / surge detection
│   │   ├── shadow_signals.py      # Shadow-mode signal suite + enforcement
│   │   ├── chronos_signal.py      # Chronos-2 forecasting shadow signal (shadow logging only)
│   │   └── memory.py              # File-backed state
│   ├── client/                    # External API clients
│   │   ├── hl_client.py           # Hyperliquid REST + WebSocket
│   │   ├── exchange.py            # Order placement, leverage, TP adjustments
│   │   ├── ws_client.py           # Persistent WebSocket for real-time mids
│   │   └── universe.py            # Volume-ranked market loader
│   └── indicators/                # TA math
│       ├── math.py                # EMA, SMA, ATR, RSI, ADX
│       └── triggers.py            # Trigger detection + composite scoring
├── scripts/
│   ├── trading_loop.py            # Continuous trading loop
│   └── hermes-mcp-server.py       # MCP server (stdio, 100 tools) — legacy integration
├── tests/                         # pytest suite — offline / online / live e2e
├── docker-compose.yml             # Docker Compose stack (production)
├── .agent-config.json             # Live risk settings (volume-mounted into container)
├── .env.local                     # Credentials, LLM endpoints (gitignored)
├── .agent-memory.json             # Persistent memory (perceptions, trades)
├── .dsl-state.json                # DSL tracker state
└── trader-logs/                   # Daily-rotated logs (gitignored)
```

---

## Rate Limit Math

| Operation | Weight | Notes |
|-----------|--------|-------|
| `allMids` | 2 | Real-time prices |
| `metaAndAssetCtxs` | 20 | Universe + volume + OI (perp) |
| `candleSnapshot` (per coin) | 20 | Plus per-item weight |
| **Total per scan cycle** | ~1,200 | Top 60 markets, one candle fetch each |

With `HERMES_MAX_MARKETS=60` and a 50s candle-cache TTL, each 60s scan fetches fresh candles (~1,200 weight).

---

## Built With

- Hyperliquid Python SDK — perpetual futures DEX
- Docker Compose — containerized deployment with external `llama-net`
- Self-hosted LLM via llama-net — no OpenRouter dependency
- Brave Search API (optional) — news headlines
- Prometheus (`prometheus-client`) — `/metrics` instrumentation + observability

**Note:** Project trunk is `main` (Python). The legacy TypeScript/Next.js implementation lives on archived branches.
