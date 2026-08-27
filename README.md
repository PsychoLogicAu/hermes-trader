# Hermes-Trader

> Autonomous crypto trading agent on Hyperliquid (perpetuals), operated as a Docker Compose stack. Scans markets, runs cheap TA filters, calls a self-hosted LLM only on confirmed setups, enforces 14 risk gates, and manages dynamic exits via DSL — all without human intervention.

Originally forked from [Julian-dev28/hermes-trader](https://github.com/Julian-dev28/hermes-trader); this repo has since diverged significantly with new LLM wiring, Docker deployment, advanced risk gates, shadow-mode signals, and trailing TP.

**What it does:**
Scans 500+ Hyperliquid markets, fires statistical triggers on price/volume/breakout signals, runs a zero-cost multi-timeframe TA filter (`analyze_perception`), and only calls the LLM on CONFIRMED setups (or momentum bursts). The LLM acts as an analyst — not an oracle. A 14-gate risk framework enforces discipline, and a DSL exit engine manages trailing stops, profit locking, and timeouts.

---

## Quick Start

### Prerequisites

- Docker + Docker Compose v2
- An **LLM endpoint** the bot can reach — see [External LLM](#external-llm).
  The research step calls any OpenAI-compatible `/chat/completions` server.
  This deployment runs one on the shared `llama-net` Docker network; if yours
  lives on the host or another machine, skip the network step below.
- A Hyperliquid account: main wallet address + an agent/API wallet private key

### First run

```bash
# 1. Create the Docker network the stack attaches to
docker network create llama-net

# 2. Configure credentials + LLM endpoint (edit the real values)
cp .env.example .env.local
$EDITOR .env.local        # HYPERLIQUID_*, LLM_BASE_URL, LLM_MODEL

# 3. Configure trading risk params
cp .agent-config.example.json .agent-config.json
$EDITOR .agent-config.json   # template ships with mode: OFF (analyse-only)

# 4. Start (builds the image on first run)
docker compose up -d

# 5. Tail logs — expect the startup grace delay and the scan interval line
docker compose logs -f hermes-trader
```

The bot runs in **`OFF` mode by default**: it scans, researches, and manages
any exits, but opens no new positions. Verify one or two cycles look sane
(logs, `docker compose ps`), then flip `mode` to `LIVE` in
`.agent-config.json` when you're ready (hot-reloaded — see below).

### Everyday commands

```bash
# Restart container (after rename-based config edits — see inode trap below)
docker compose restart hermes-trader

# Stop
docker compose down

# Rebuild + re-run (after code changes) — build.sh matches the container
# user to this host's uid/gid so the bind-mounted volume dirs stay writable
scripts/build.sh && docker compose up -d --force-recreate hermes-trader

# Tail logs live
docker compose logs -f hermes-trader

# Check container health
docker compose ps
```

Logs: `trader-logs/trader.log` (daily rotation, date-stamped backups).

### Config hot-reload (and the inode trap)

`.agent-config.json` is volume-mounted and **re-read every cycle** (~60s), so
edits take effect within a cycle — *if* the container's bind mount still points
at the same inode.

- **Safe (in-place) edits**: Python `open(path, 'w')` overwrites, `echo >>`
  appends, the `hermes config` CLI. No restart needed.
- **Inode trap**: editors that write a temp file and rename over the original
  (`sed -i`, many editor save flows) create a *new* inode; the container keeps
  serving the old file and your change silently does nothing. Fix:
  `docker compose restart hermes-trader` (~15s).
- **Verify what the container sees**:
  `docker exec hermes-trader grep max_trade_notional_usd /app/.agent-config.json`

`.env.local` (credentials, LLM endpoint) is different — it is loaded at process
start, so changes need `docker compose up -d --force-recreate hermes-trader`.

Full operating procedures (deploying code, changing the LLM, testing, the 14
risk gates, invariants, troubleshooting, ledger queries) are in
[`docs/OPERATIONS.md`](docs/OPERATIONS.md).

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
|        (cheap)         (expensive)     (14 gates)          (per-tick, trailing TP)
|               |
|               └── Hyperfeed Discovery
|                   Leaderboard • Smart Money • OI Anomaly • Whale Tracking
+---------------------------------------------------------------+
```

### Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────────┐
│  Perception │───>│  TA Filter   │───>│   AI Research   │───>│  Risk Gates  │───>│   Executor  │
│   Scanner   │    │  (TA Filter) │    │ (local LLM API) │    │  (14 gates) │    │ HL + DSL TP │
│ 5m/1h/4h    │    │  EMA/RSI/ATR│    │ Verdict + Price │    │ (14 gates)   │    │ SL / Trailing│
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
- **Risk gates:** 14 gates, enhanced with high-confidence bypasses (`bypass_low_volume`, `bypass_late_trend_chase`), funding-regime overlays, and trailing TP.

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

- Calls a self-hosted Qwen-based model via `llama-net`, configured via
  `.env.local`: `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`.
- Resilient to credit/API outages: no OpenRouter dependency.
- On a 402 with affordability hint, retries ONCE with a smaller `max_tokens`
  budget so the bot degrades rather than going blind.
- **Output budget**: 8192 completion tokens for both the primary research
  call and the duelist (`LLM_DUEL_MAX_TOKENS` inherits `LLM_MAX_TOKENS` when
  unset; both are env-tunable via `LLM_MAX_TOKENS`). They cap completion
  tokens only — the prompt/context window is the model server's `ctx_size`,
  not this setting. (Raised to 32768 on 2026-08-26 during the duelist
  runaway-generation forensics; reverted 2026-08-27 — the budget does not
  cause runaways, it only caps their cost, and 8192 is 12x the healthy
  duelist output p99 of 487 tokens.)

### External LLM

The research step needs **any OpenAI-compatible `/chat/completions` endpoint**
reachable from the `hermes-trader` container — its own model server, a hosted
API, or a local inference server on the shared `llama-net` Docker network.
It is an external project to this repo; hermes-trader only consumes the URL.

This deployment runs [lemonade](https://github.com/lemonade-sdk/lemonade)
(`lemond`) on `llama-net`, serving a GGUF model via llama.cpp — a small
Q4_K_M model (e.g. ~9B, thinking disabled) fits an 8GB GPU with a 32k
context, which is plenty for the analyst prompt. A reference
`docker-compose.yml` for the lemonade side, plus the matching `.env.local`
values (`LLM_BASE_URL=http://lemond:13305/api/v1`), live in
[`docs/OPERATIONS.md`](docs/OPERATIONS.md#external-llm-reference).

### Model Duel (A/B LLM Evaluation)

A second LLM can be run against the **exact same research prompt** as a pure
observer — A/B evaluation of model performance. The duelist never executes and
never gates: it runs after the primary verdict, so its slowness or outages can
neither delay nor break trading, and a failed call simply logs no observation.

- **Enable**: set `LLM_DUEL_MODEL` in `.env.local` (one line; base URL and API
  key inherit the primary `LLM_*` values, so dueling a different model on the
  same server needs nothing else). Feature is dormant unless the model is set.
- **Record**: every paired call appends to `~/.hermes-trader-duel.jsonl`
  (both verdicts, confidences, reasoning excerpts, agree/split) and emits a
  `duel` event to the session log (`✓ agree` / `≈ split`).
- **Score**: the duelist's verdict is snapshotted into the trade's entry
  context, so when a position closes the close row carries it. The report then
  computes the duelist's *hypothetical* P&L on the same trades: concurs with
  the live side → same P&L; opposes → mirrored; PASS/CLOSE → 0 (a win only if
  the live side lost).
- **Read**: `hermes duel` prints paired-call agreement plus both models'
  realized stats (trades, win rate, P&L, avg win/loss). The primary is
  unaffected and always scores every close; the duelist scores only the closes
  it has a verdict for.

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
| **`.agent-config.json`** | Repo root (gitignored) | **Create** from `.agent-config.example.json`, then **edit** | Fresh every cycle (hot-reloaded — see [Quick Start](#config-hot-reload-and-the-inode-trap)) |
| **`.env.local`** | Repo root (gitignored) | **Create** from `.env.example` | At process start — force-recreate after changes |

### `.env.local`

Copy `.env.example` → `.env.local` and fill in your real values:

```bash
# ── Hyperliquid ──────────────────────────────────────────────────────────────
HYPERLIQUID_WALLET_ADDRESS=0x...          # main account address
HYPERLIQUID_PRIVATE_KEY=0x...             # agent/API wallet signing key

# ── LLM (OpenAI-compatible endpoint) ────────────────────────────────────────
# A local inference server on llama-net (see External LLM below) or any hosted
# /chat/completions API.
LLM_BASE_URL=http://lemond:13305/api/v1
LLM_API_KEY=local
LLM_MODEL=<your-model>

# ── News (optional — enables news catalyst in research + gates) ─────────────
# BRAVE_API_KEY=...

# ── Scan tuning (optional — defaults shown) ──────────────────────────────────
# HERMES_SCAN_INTERVAL=60
# HERMES_MAX_MARKETS=60
# HERMES_BATCH_SIZE=20
# HERMES_BATCH_SLEEP=0.3
```

### `.agent-config.json` — live risk settings

All trading behaviour and risk limits live here. Create it from the template
(`cp .agent-config.example.json .agent-config.json`) — the template ships with
`mode: OFF` (analyse-only) — then tune. It is hot-reloaded every cycle; the
inode trap in the [Quick Start](#config-hot-reload-and-the-inode-trap) decides
whether you also need a restart. Every key (defaults, ranges, what each gate
reads) is documented in [`docs/CONFIG.md`](docs/CONFIG.md); operating
procedures are in [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

A minimal starting shape:

```json
{
  "mode": "OFF",
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
  "min_market_volume_usd": 1000000,
  "min_short_volume_usd": 50000000,
  "cooldown_min": 30,
  "held_research_interval_min": 3,
  "research_cooldown_min": 3,
  "coin_blocklist": ["TON", "TRX"],
  "max_crypto_long_correlated": 2
}
```

The full template (`.agent-config.example.json`) additionally carries
`dsl_exit.*` (stops, trailing TP, phase tiers), `runner_entry_gate.*`,
`atr_risk_sizing.*`, `shadow_signals.*`, and `chronos_signal.*`.

Key parameters:

| Key | Meaning |
|-----|---------|
| `mode` | `LIVE` = real money; `OFF` = analyse-only (exits still monitored); `SHADOW` = full pipeline, no orders |
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
| `coin_blocklist` | Never trade these symbols |
| `runner_entry_gate.*` | Runner surface gate thresholds (late-chase / low-vol bypass) |
| `atr_risk_sizing.*` | ATR-based equal-risk sizing parameters |
| `dsl_exit.*` | DSL trailing stop + trailing TP configuration |
| `shadow_signals.*` | Toggle individual free signals in the shadow suite |
| `chronos_signal.*` | Chronos-2 forecasting shadow signal (shadow-only logging; `enabled` toggles it) |

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
- **Config not applying**: First check the inode trap (host file vs container view) — `docker exec hermes-trader grep <key> /app/.agent-config.json` — then `docker compose restart hermes-trader`. `.env.local` changes need a full `--force-recreate`.
- **LLM unreachable**: Verify `llama-net` exists (`docker network ls`) and the LLM endpoint itself answers (see [External LLM](#external-llm)).

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
│   │   ├── risk_gates.py          # 14 risk gates
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
├── .env.example                   # Committed placeholder template for .env.local
├── .env.local                     # Credentials, LLM endpoint (gitignored — copy from .env.example)
├── .agent-config.example.json     # Committed safe template (mode OFF) for .agent-config.json
├── .agent-config.json             # Live risk settings (gitignored — copy from .agent-config.example.json)
├── .agent-memory.json             # Persistent memory (perceptions, trades) (gitignored)
├── .dsl-state.json                # DSL tracker state (gitignored)
├── docs/                          # CONFIG.md (key reference), OPERATIONS.md (day-to-day), ARCHITECTURE.md
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
