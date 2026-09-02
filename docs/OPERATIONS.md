# Hermes Trader — Operations Manual

Day-to-day operating procedures for anyone running hermes-trader: start/stop,
config editing (hot-reload + the inode trap), deploying code, changing the LLM,
testing, and the invariants the code depends on.

> This is the shared operations manual. Operator-specific state (live mode,
> current sizing, account notes, change log) belongs in your own local notes,
> not in this file.

## Where state lives

| File | Purpose | Committed? |
|---|---|---|
| `.agent-config.json` | Live risk params, gates, sizing. Hot-reloaded each cycle. | No (gitignored) |
| `.agent-config.example.json` | Safe template (`mode: OFF`) for the above. | Yes |
| `.env.local` | Credentials + LLM endpoint + scan tuning. | No (gitignored) |
| `.env.example` | Placeholder template for the above. | Yes |
| `trader-logs/trader.log` | Main log (daily rotation, UTC). | No |
| `trader-logs/trades.jsonl` | Append-only trade ledger (epoch-ms). | No |
| `.agent-memory.json` | Perceptions/analyses/trades across restarts. | No |
| `.dsl-state.json` | Exit-engine trackers (persists across restarts). | No |
| `hf-cache/` | HuggingFace model cache (Chronos-2) mounted into the container. | No |

## Start / stop / restart

```bash
# Start (builds if needed)
docker compose up -d

# Restart (after rename-based config edits — see inode trap)
docker compose restart hermes-trader

# Stop
docker compose down

# Tail logs
docker compose logs -f hermes-trader
```

## Editing config (hot-reload)

`.agent-config.json` is volume-mounted and **re-read at the start of every
cycle** (~60s), so changes take effect within a cycle — **if the container's
bind mount still points at the same inode**.

- **Safe (in-place) edits** — the container sees them on the next cycle, no
  restart:
  - `python3 -c` / any Python `open(path, 'w')` overwrite of the same file
  - `echo '...' >> .agent-config.json` (append)
  - `hermes_trader.agents.config_store.write_agent_config` (writes a tmp file
    then `os.replace`s — the mount follows the path, so this is safe)
- **Inode trap** — editors that write a temp file and rename over the original
  (`patch`, `sed -i`, most editor "write + rename" flows) create a **new
  inode**; the container keeps serving the old file and your change silently
  does nothing. Fix: `docker compose restart hermes-trader` (~15s).

Always verify what the container actually sees:

```bash
docker exec hermes-trader grep max_trade_notional_usd /app/.agent-config.json
```

Env vars (`.env.local`: LLM endpoint, credentials) are different — they are
loaded at process start, so any change to `.env.local` needs a
**force-recreate**:

```bash
docker compose up -d --force-recreate hermes-trader
```

## Deploying code changes

Code is **baked into the image** (not bind-mounted). Procedure:

1. Commit the change.
2. `scripts/build.sh` (builds with this host's uid/gid — the image bakes in
   the container user at build time, so a plain `docker compose build` would
   default to 1000:1000)
3. `docker compose up -d --force-recreate hermes-trader`
4. Verify inside the container that the new code is present:
   `docker exec hermes-trader grep <new_symbol> /app/hermes_trader/...`
5. Watch `docker compose logs -f hermes-trader` for ~30s — expect the startup
   grace delay, the scan interval line, and zero errors.

**No clean-book requirement:** open positions survive a force-recreate —
`rehydrate_from_exchange` re-synthesizes DSL trackers from the live exchange on
the first cycle after restart. Do not close positions just to deploy.

A plain `restart` will NOT pick up code changes.

## Changing the LLM

Edit `.env.local` (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`), then force-
recreate (a plain restart does not reload env vars). The LLM is an external
project to this repo — see [External LLM (reference)](#external-llm-reference)
for the lemonade setup this deployment uses.

## Testing

- Offline suite (no network, no credentials, no orders):
  `pytest -x -q` — markers `online`/`live` are deselected by default.
- Online (read-only public HL API, no money): `pytest -m online`
- Real-money e2e (double-gated, spends fees + a billable LLM call):
  `HERMES_E2E=1 pytest -m live` — only when you mean it.
- `tests/conftest.py` redirects all state files to a throwaway temp dir before
  any module import, so a pytest run can never touch live state.

## Risk gates

15 gates, evaluated pre-execution, **all of them every cycle** (no
short-circuit — results are collected for telemetry). Any `pass: False`
blocks the trade; the `reason` strings land in the log's `blocked_by`.
(14 can block; `chronos_mismatch` is shadow-only by default — it passes and
logs would-blocks until `chronos_mismatch_gate.shadow_mode` is set false.)

1. **Confidence** — LLM confidence ≥ `min_ai_confidence` (with a regime-aligned
   floor override via `aligned_min_conf`)
2. **Max concurrent** — open positions < `max_concurrent`
3. **Notional cap** — trade notional ≤ `max_trade_notional_usd`
4. **Daily loss** — day PnL below −T (`daily_kill_pct_of_equity` × equity)
   blocks new entries + arms the halt timer; below −1.25·T the heartbeat
   hard-killswitch flattens all positions
5. **Daily giveback** — halts new entries if daily PnL retraces
   > `daily_giveback_halt_pct` from a peak ≥ `daily_giveback_min_peak_usd`
6. **Liquidity** — 24h volume ≥ `min_market_volume_usd` (HIP-3:
   `min_hip3_volume_usd`), with a high-confidence bypass
7. **Short liquidity** — 24h volume ≥ `min_short_volume_usd` for shorts
8. **Coin filter** — blocklist/allowlist
9. **No pyramid** — no re-entry (same or opposite side) on a coin you hold
10. **Cooldown** — `cooldown_min` since the most recent real trade on the coin
11. **Correlation cap** — ≤ `max_crypto_long_correlated` major-crypto longs
12. **Equity risk** — total open notional ≤ `max_total_notional_pct` × equity
13. **Market regime** — counter-trend blocked unless confidence ≥
    `counter_regime_min_conf` (with crowd/squeeze overlays)
14. **News** — binary news risk (Fed/CPI/earnings) stands the bot down
15. **Chronos mismatch** — entry side contradicts the cached Chronos-2
    forecast (long vs negative median / short vs positive, beyond
    `min_abs_median_pct`) unless confidence ≥ `chronos_mismatch_gate.min_conf`
    (0.90) or composite ≥ `min_composite` (60). **SHADOW by default**:
    `shadow_mode: true` makes it structurally pass and log would-blocks —
    it never blocks. Flip `shadow_mode` to `false` to go live.

## Key rules

- **One knob at a time** — never change two config params at once on a live bot.
- **Gate-first** — structural gates discipline the LLM; the LLM justifies
  entries, the gates enforce limits.
- **Start OFF** — a missing/empty config resolves to `{"mode": "OFF"}`
  (analyse-only). Flip to `LIVE` deliberately. `SHADOW` runs the full pipeline
  and logs intended trades without sending orders.
- **Equity accounting** — on a Hyperliquid *unified* account, equity = spot
  USDC + perp uPnL (held perp margin is reserved spot, not extra equity). If
  reported equity runs ~$30 above the real portfolio with a full book, check
  the equity formula in `client/hl_client.py`.
- **UTC everywhere** — logs, ledger, and exit windows use UTC. Naive
  `datetime.fromisoformat()` on a stored timestamp shifts windows by your
  UTC offset.

## Critical invariants (do not break)

1. **Gate result shape**: every gate returns `{"pass": bool, "reason": str}` —
   never `None`, never a bare bool.
2. **No short-circuit**: `eval_all_gates` calls all 14 gates every cycle.
3. **Ledger append-only**: `trades.jsonl` is never truncated or rewritten.
4. **Epoch-ms timestamps**: ledger uses `int(time.time()*1000)`; convert with
   `datetime.fromtimestamp(ts/1000, tz=timezone.utc)`.
5. **Config read once per cycle**: the loop reads `.agent-config.json` at the
   start of each cycle; changes land within one cycle if the mount sees the
   new content (inode trap above).
6. **No hardcoded credentials**: the private key comes from
   `HYPERLIQUID_PRIVATE_KEY` only. `.env.local` is never committed.

## Troubleshooting

1. **Container not starting** — `docker compose logs hermes-trader`; check
   `.env.local` exists, `.agent-config.json` is valid JSON, and `llama-net`
   exists.
2. **No trades executing** — grep `runner_gate_blocked` / `blocked_by` in
   `trader-logs/trader.log` — likely liquidity, late-chase, confidence, or the
   daily-loss kill switch.
3. **Config not applying** — inode mismatch (host file vs container view):
   verify with `docker exec hermes-trader grep <key> /app/.agent-config.json`,
   then restart if stale.
4. **Code change not applying** — code is in the image: rebuild +
   force-recreate, not restart.
5. **LLM unreachable** — `docker network ls` to confirm `llama-net`, and that
   the LLM endpoint itself answers (see the reference below).

## Ledger quick queries

```bash
# Recent closes with P/L
grep '"CLOSE"' trader-logs/trades.jsonl | jq -r '.coin + " | " + (.realized_pnl_pct|tostring) + "% | " + (.hold_minutes|tostring) + "m"'

# Winning trades
grep '"CLOSE"' trader-logs/trades.jsonl | jq -r 'select(.realized_pnl_usd > 0) | .coin + " +\(round(.realized_pnl_usd * 100)/100)usd"'

# Open positions (entries not yet closed)
grep '"OPEN"' trader-logs/trades.jsonl | jq -r '.coin + " | " + .side + " @ " + (.entry_px|tostring)'
```

## External LLM (reference)

The research step calls any **OpenAI-compatible** `/chat/completions` endpoint
(`LLM_BASE_URL` + `LLM_MODEL` + `LLM_API_KEY` in `.env.local`). It is a
separate project from this repo; this repo only needs it to be reachable from
the Docker network.

This deployment runs [lemonade](https://github.com/lemonade-sdk/lemonade)
(`lemond`) on the same `llama-net` network, aliased as `lemond`, serving a
GGUF model via llama.cpp. Reference `docker-compose.yml`:

```yaml
services:
  lemond:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        # Limit parallel jobs to avoid swap storms on 32GB systems
        # Adjust based on your CPU core count (e.g., 8 for 8-core, 12 for 16-core)
        BUILD_JOBS: 8
    deploy:
      resources:
        limits:
          memory: 24G
        reservations:
          memory: 16G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    container_name: lemonade-server
    ports:
      - "13305:13305"
    environment:
      - LEMONADE_API_KEY=${LEMONADE_API_KEY:-}
      - LEMONADE_ALLOWED_ORIGINS=${LEMONADE_ALLOWED_ORIGINS:-}
    volumes:
      - lemonade-cache:/opt/lemonade/.cache
    networks:
      llama-net:
        aliases:
          - lemond

volumes:
  lemonade-cache:

networks:
  llama-net:
    external: true
```

A small Q4_K_M GGUF model (e.g. a ~9B instruct model with thinking disabled via
llama.cpp `--chat-template-kwargs '{"enable_thinking": false}'`) runs in an
8GB GPU with a 32k context — plenty for the analyst prompt, which is
structured and short. Point hermes-trader at it with:

```bash
LLM_BASE_URL=http://lemond:13305/api/v1
LLM_MODEL=<the model name lemonade serves>
LLM_API_KEY=local
```

If your LLM runs elsewhere (host process, another machine), just put its
reachable URL in `LLM_BASE_URL`.