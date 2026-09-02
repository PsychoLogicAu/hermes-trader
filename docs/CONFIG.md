# `.agent-config.json` — Reference

Every key the bot reads at trade-time. **Hot-reloaded** on each trade (no
restart for most changes). Exceptions:

- **`enable_hip3`** — universe is fetched once at startup; flipping it
  mid-run requires a loop restart.
- **`.env.local` (separate file, not this one)** — loaded at process start;
  changes require a force-recreate.
- **The container inode trap** (Docker path only) — a rename-based host edit
  creates a new inode the container won't follow; see the README Quick Start.

For one-shot tuning by account size, use:

```bash
scripts/config_preset.py list                          # show available presets
scripts/config_preset.py apply small_aggressive        # apply (with diff preview)
scripts/config_preset.py apply --account-size 250      # auto-pick by equity
```

---

## Mode + asset class

### `mode` (string: `"OFF"` | `"LIVE"` | `"SHADOW"`)
- `OFF` — bot scans + researches but executes nothing. Exits on positions
  already open are still managed (DSL + stop orders).
- `LIVE` — orders go to real money.
- `SHADOW` — full pipeline runs; intended trades are logged but never sent.

### `enable_crypto` (bool, default `true`)
Scan native HL perps (BTC, ETH, SOL, etc.). Hot-reloaded — takes effect on the
next cycle (filter is applied per-scan, not at startup).

### `enable_hip3` (bool, default `false`)
Scan HIP-3 tokenized-equity / commodity perps (`xyz:NVDA`, `km:USOIL`, etc.). Adds ~8 HTTP POSTs per scan. **Requires loop restart to take effect**.

---

## Sizing

### `equity_fraction_per_trade` (float, 0–1, default `0.01`)
Fraction of perp equity committed as MARGIN per trade. With leverage, notional = `equity × fraction × leverage`.

| Account size | Recommended fraction |
|---|---|
| < $500 | 0.05–0.10 (sized for $25-50 margin per trade) |
| $500–$2000 | 0.03–0.05 |
| $2000+ | 0.01–0.03 |

**Math check**: if `equity_fraction × max_concurrent > 1.0`, you'll fully deploy and start tripping `max_total_notional_pct`. For $250 with 0.10 fraction × 18 concurrent = 180% over-deployment, but with 10–40x leverage the per-trade *margin* commitment stays modest. Watch `available` (free margin) on the dashboard; if it drops below 10% of equity, the executor refuses new trades.

### `leverage` (int, default `5`)
Max leverage to request per trade. Actual = `min(this, per-coin HL max)`. BTC: 40x, ETH: 25x, mid-cap alts: 5-20x, HIP-3 equity: 5-20x.

| Account size | Recommended |
|---|---|
| < $500 | 20–40x (need leverage to size meaningfully) |
| $500–$2000 | 10–20x |
| $2000+ | 5–10x |

### `max_trade_notional_usd` (int, default `100000`)
Per-trade hard cap regardless of formula above. Keep well above intended deployment or trades get blocked.

### `max_concurrent` (int, default `10`)
Max simultaneous open positions. With 60s scan + 180min hold, on a busy day you can fill 18-20 slots quickly. Trades over this cap are deferred.

### `max_total_notional_pct` (float, default `1.0`)
Combined open notional cap as multiple of equity. `40.0` = max 40× equity in total notional. Bounds total deployment even when individual trades are within `max_trade_notional_usd`.

### `min_available_margin_pct` (float, default `0.10`)
Refuse new trade if `available / equity < this`. Default 10% leaves headroom for maintenance margin + slippage. Lower = more aggressive deployment, higher risk of HL "insufficient margin" rejections.

### `conviction_sizing` (bool, default `true`)
Scale position size by AI confidence: a high-conviction setup bets a larger fraction of equity. Set `false` for flat sizing across all trades.

### `conviction_tiers` (list of `[min_confidence, multiplier]`, optional)
Overrides the built-in confidence tiers used by `conviction_sizing`. Each pair is `[threshold, size_multiplier]`; the highest threshold the AI confidence clears wins, and the multiplier scales `equity_fraction_per_trade` for that trade. Hot-reloaded — no restart needed.

Default (when unset) reproduces the prior hardcoded behavior:
```json
"conviction_tiers": [[0.80, 1.5], [0.65, 1.0], [0.0, 0.7]]
```

Example — bet more aggressively on strong setups and smaller on weak ones:
```json
"conviction_tiers": [[0.85, 2.0], [0.70, 1.2], [0.0, 0.5]]
```
So at `equity_fraction_per_trade: 0.10`, a 0.90-confidence trade sizes at an effective 0.20 fraction (2.0×), while a 0.55-confidence trade sizes at 0.05 (0.5×). Malformed entries are ignored and fall back to the default tiers. The whale-signal boost (`whale_size_multiplier`) still multiplies on top, clamped at 2× base.

---

## Risk safety

### `daily_kill_pct_of_equity` (float 0-1, default `0.10`)
Daily-loss killswitch, equity-relative (2026-09). Threshold T =
`clamp(pct × equity, daily_kill_min_usd, daily_kill_cap_usd)`. Two tiers,
both off the same T:

1. **Halt / entry gate** at **−T**: blocks NEW trades and arms a halt
   *timer* (`daily_loss_halt.halt_min`, default 360 min) that blocks
   re-entry until it expires or the day's PnL recovers into the release
   band — not a UTC-midnight lock.
2. **Hard flatten** at **−1.25·T** (override with
   `daily_kill_flatten_mult`, `1.0` = old flat behaviour): the heartbeat
   closes ALL open positions. It is deliberately higher than the halt so
   the open book has a grace band (−T … −1.25·T) to claw the day back
   and clear the halt early — daily PnL is equity-based and stays pinned
   red once the book is empty, so a flat flatten at T would make that
   early release unreachable.

`0` disables all three (gate, halt, flatten).

| Account size | Suggested pct (cap $) |
|---|---|
| < $500 | 0.10–0.15 (cap $25–50) |
| $500–$2000 | 0.15–0.20 (cap $150) |
| $2000+ | 0.25 (cap $500) |

`daily_kill_cap_usd` (default `100`) is the absolute ceiling on T — "never
lose more than $X in one day" regardless of equity growth (the flatten
then tops out at 1.25·cap). `daily_kill_min_usd` (default `8`) is a
small-account noise guard.

Too loose = catastrophic days possible. Too tight = locks you out on a normal variance day.

### `daily_giveback_halt_pct` (float 0-1, default `0` = off)
**Daily give-back breaker** (2026-06-06). Once the day's PnL has peaked at `>= daily_giveback_min_peak_usd`, block NEW entries if it then retraces more than this fraction from that peak. Existing positions keep riding their own stops; resets at the UTC roll. Locks in green days so a won day can't fully round-trip (e.g. `0.35` = halt after giving back 35% from peak). Measures TRUE account PnL (aggregate equity, not main-dex-only). `0` disables.

### `daily_giveback_min_peak_usd` (float, default `20`)
Arm threshold for the give-back breaker — it stays disarmed until the day's peak PnL reaches this, so a tiny `+$2` peak can't trip a halt. Scale to account size.

### `cooldown_min` (int, default `60`)
Minimum minutes between trades on the same coin. Prevents over-trading a single market. 30-60 reasonable for active strategies; 120+ for slower. Also skips the paid AI research call for a non-held coin still inside this window (a re-entry would be gate-blocked anyway).

### `held_research_interval_min` (int, default `10`)
How often a coin you ALREADY HOLD is re-researched for a possible AI `CLOSE`. Without this, a held position that keeps triggering pays for a "hold" PASS on every ~60s scan. The DSL exit engine still handles fast/loss exits in real time every scan regardless — this only paces the slower "thesis broke → close" judgment. Lower = more responsive AI closes but more token spend; higher = leans more on DSL for exits. Hot-reloaded.

### `min_ai_confidence` (float 0-1, default `0.35`)
Floor for AI-verdict confidence to execute. Raise to filter out borderline trades; lower to accept more setups. Current default 0.30 with conviction_sizing reducing those bets to 0.7×.

### `counter_regime_min_conf` (float 0-1, default `0.7`)
For trades against the BTC/SP500 regime trend, AI confidence must clear this OR `composite_score ≥ 50` OR `momentumBurst` fired OR a slow-burn trigger fired. Loosened bypass paths added 2026-05-28. The `composite ≥ 50` path is NOT disabled by `block_counter_trend_bypass` (only the binary-trigger bypass is).

### `crowded_with_min_conf` (float 0-1, default `0` = off)
**SHORT_CROWDED squeeze caution** (2026-06-06). A trend-aligned trade that is ALSO *with the crowd* (a short into `SHORT_CROWDED` funding, or a long into `LONG_CROWDED`) normally gets a free "aligned" pass — but those are exactly what gets squeezed on a reversal. When set, such a trade must clear this confidence bar or it's blocked `via:crowded_squeeze`. Filters squeeze-prone weak entries while letting strong setups through. `0.80` is a moderate filter; too high neuters the down-short edge (SHORT_CROWDED is common in downtrends). `0` disables.

### `tp_scale_fraction` (float 0-1, default `0.5`)
Fraction of a position auto-banked at the take-profit target via a server-side reduce-only TP trigger placed at entry (`1 ATR` past entry). Banks e.g. half at target while the rest rides the DSL trail — captures profit instead of round-tripping into the trailing stop. `0` = no TP scale-out (trail only).

### `aligned_min_conf` (float 0-1, optional, default unset)
Confidence bar for a trade WITH the regime (trend-aligned), typically lower than `counter_regime_min_conf`. Lets aligned shorts in a selloff / aligned longs in a rally clear at a lower bar than counter-trend trades. Unset = use `min_ai_confidence`.

### `block_counter_trend_bypass` (bool, default `false`)
When `true`, the binary-trigger bypass (momentum_burst / slow_burn / whale) can NO LONGER push a counter-trend trade through the regime gate — it must clear real conviction (conf or `composite ≥ 50`). Stops low-conviction longs being forced into a downtrend. Does NOT touch aligned/neutral trades or the composite≥50 path.

### `whale_scan_bypass` (bool, default `false`)
Let whale-accumulation signals (oi_funding_anomaly) surface a coin for research even when it scores below the composite scan gate (whale loads on FLAT price, which scores low on momentum triggers).

### `max_crypto_long_correlated` (int, default `2`)
Cap on simultaneous correlated crypto longs. Prevents stacking 5 alt longs that all dump together. HIP-3 equity/commodity longs don't count against this.

## Signal surfacing (gated)

These surface extra candidates for AI research beyond the weighted composite gate; the AI + risk gates still adjudicate. All default OFF unless noted.

### `momentum_continuation` (nested, `enabled` default `false`)
`{enabled, min_trend_pct (8), max_pullback_pct (6), weight (0.4), log_near_miss}`. Surfaces a coin in a sustained ORDERLY uptrend now consolidating (already-extended movers the spike/breakout triggers miss) and adds its weight to the composite — so a strong momentum long can clear the regime gate's `composite ≥ 50` path even counter-trend. Enable when you want to ride extended momentum; the counter-trend gate + caps back it up.

### `candlestick_patterns` (nested, `enabled` default `false`)
`{enabled, wick_body_ratio (2.0), context_lookback (6), context_pct (1.5)}`. Reversal candles at exhaustion — shooting-star/bearish-engulfing (→ SHORT) and hammer/bullish-engulfing (→ LONG), each requiring a preceding move so they fire at tops/bottoms, not every bar. Weight-0 surfacing signal; the research prompt also gets the last 12 raw 1h OHLC bars so the LLM reads price action directly.

### `force_execute_composite` (float, default `40`)
If AI says PASS but trigger composite hits this AND `force_execute_slow_burn_count` slow-burn triggers fire, the executor upgrades to LONG conf 0.70. The structure overrides the AI's hedge. Set to 999 to disable.

### `force_execute_slow_burn_count` (int, default `2`)
Min slow-burn triggers (volumeBuildup1h / trendFlip1h / higherLows1h) required for the structural override. Combined with `force_execute_composite`.

### Whale-signal priority (oi_funding_anomaly accumulation flag)
Three independent knobs, all default-on, to capitalize on smart-money accumulation (deeply-negative funding + flat price + high OI):

- `whale_regime_bypass` (bool, default `true`) — a whale signal lets a trade bypass the counter-regime gate (even against trend).
- `whale_force_execute` (bool, default `true`) — a whale signal alone upgrades an AI PASS to LONG conf 0.70 (the structural override).
- `whale_size_multiplier` (float, default `1.3`) — whale-backed trades multiply their conviction sizing by this, clamped at 2× base. Set `1.0` to keep override/bypass but no size boost.

Set all three off (`false`/`false`/`1.0`) to treat whale signals as informational only (still shown in the AI prompt, no gate/sizing effect).

---

## Liquidity (volume floors)

### `min_market_volume_usd` (int, default `5000000`)
Crypto perps below this 24h volume are blocked. Default $5M screens illiquid microcaps.

### `min_hip3_volume_usd` (int, default `500000`)
HIP-3 perps below this 24h volume are blocked. Lower because HIP-3 markets carry less volume than crypto majors (xyz:CRCL at $4M is well-tradable).

### `min_short_volume_usd` (int, default `0` = off)
A SEPARATE, deeper 24h-volume floor for SHORTS only — thin markets squeeze, so a short needs more liquidity than a long in the same name. `0` disables (shorts use the general floor).

---

## Filters

### `coin_allowlist` (list, default `[]` empty = allow all)
If non-empty, ONLY these coins are tradable. Useful for whitelisting a focused basket.

### `coin_blocklist` (list, default `[]`)
Coins always blocked regardless of allowlist. Use for known-bad markets.

---

## DSL exit (nested object)

### `dsl_exit.max_loss_pct` (float, default `2.5`)
Max adverse SPOT% move before forced exit. Combined with the ROE cap below — whichever fires first.

### `dsl_exit.max_loss_roe_pct` (float, default `50.0`)
Max ROE% loss (margin %). At 40x leverage, 40% ROE = 1% spot. The min of (max_loss_pct, max_loss_roe_pct/leverage) is the effective stop. Tighter cap = smaller losses per trade but more stop-outs on noise.

### `dsl_exit.protect_pct` (float, default `1.5`)
Spot% move required to engage phase-2 trailing. Lower = trailing locks profit earlier; higher = lets winners run further before tightening.

### `dsl_exit.retrace_threshold` (float 0-1, default `0.30`)
Phase-2 floor gives back this fraction of peak gains. `0.30` locks 70% of peak profit; `0.20` locks 80%; `0.50` lets price retrace half before exit.

### `dsl_exit.hard_timeout_minutes` (float, default `180`)
Max time a position stays open before forced close. 90 = tight (frequent timeouts on slow movers); 180 = balanced; 360 = lets multi-hour setups breathe.

---

## Account-size presets

The `scripts/config_preset.py` tool ships with these presets:

| Preset | For | Style |
|---|---|---|
| `small_aggressive` | $100-500 | Max conviction, high leverage, tight daily loss cap |
| `small_conservative` | $100-500 | Lower leverage, looser stops, longer holds |
| `medium_balanced` | $500-2000 | Default-ish, balanced risk |
| `large_steady` | $2000+ | Low leverage, tight per-trade size, looser caps |
| `hip3_only` | any | Disables crypto, focuses on tokenized equity |
| `crypto_only` | any | Disables HIP-3 |

Run `scripts/config_preset.py show small_aggressive` to see the full values without applying.

---

## Exit, sizing & signal blocks (nested)

All hot-read. README "Configuration" has the concise version.

### `dsl_exit` — trailing-stop engine
- `max_loss_pct` (3.5) + `max_loss_roe_pct` (18) — hard stop, whichever binds first (ROE cap = `pct / leverage` in spot terms; at 10x, 18% ROE = 1.8% spot).
- `protect_pct` (1.5) + `retrace_threshold` (0.30) — trail tightness. **Low = scalp (bank fast); high = trend-ride (let it run).** `phase2_tiers` = profit-scaled retrace ladder.
- `stale_flat_timeout_minutes` — flatten a position that never reaches `protect_pct` within this window.
- `regime_aware {enabled, trend_ride{…}}` — when `detect_regime()=='up'`, swap to looser trend-ride params (scalp chop / ride trends). Default OFF.

### `atr_risk_sizing` `{enabled, risk_per_trade_pct}`
Equal-risk (Turtle-N): notional = `risk_per_trade_pct × equity / stop_width`. Overrides flat `equity_fraction` — volatile coins get smaller size, tight-stop coins bigger (capped by `max_trade_notional_usd`).

### `signal_enforcement` `{enabled, veto, boost, gex_veto, boost_bar_delta, whale_*}`
Free signal suite acting on the **forced-override path only**. VETO blocks chop-traps (GEX pin-trap) / whales dumping; BOOST lowers the override bar on a catalyst. Cache-only.

### `shadow_signals` `{enabled, gex, short_volume, crypto_whale, news}`
Logs the free signals per candidate **without affecting trades** — forward validation.

### `gex_signal` / `momentum_reentry`
Gated experiments (see commit history). `momentum_reentry` backtested net-negative → OFF.

### `chronos_mismatch_gate` `{enabled, shadow_mode, min_conf, min_composite, min_abs_median_pct}`
Direction-mismatch conviction gate (the 15th gate). When the entry side contradicts the coin's cached Chronos-2 forecast (long vs negative median / short vs positive, beyond `min_abs_median_pct`, default 0.5%), the trade must clear elevated conviction: `confidence >= min_conf` (0.90 — same bar as the late-trend-chase bypass) OR `composite >= min_composite` (60). No usable forecast (cold cache) or a directionally-neutral forecast = no opinion = pass. **SHADOW by default**: `shadow_mode: true` makes it structurally pass and the executor logs `[gate][SHADOW] chronos_mismatch WOULD HAVE BLOCKED …` at WARNING — it cannot alter live execution. Flip `shadow_mode` to `false` (one knob, hot-read per cycle) to make it block.

### `band_counter_breach_gate` `{enabled, shadow_mode, min_conf, min_breach_pct, drift_ref_span}`
The GRASS-shape deterministic gate (LIVE by default, `shadow_mode: false`): band TRENDING + price ≥ `min_breach_pct` (1.0) past the drift's OPPOSITE-side edge + entry on the bounce/dip side → require `confidence >= min_conf` (0.90; no composite bypass). `drift_ref_span` (bars, **default 32 live**) sets the drift-reference LAG for the verdict only: the band edges stay the trigger's `band_span` MA, but "trending?" samples the SAME edge `drift_ref_span` bars back instead of `band_span` bars back. The 2026-08-31 single-window trigger rework collapsed this lag onto `band_span` (16) — correct for the trigger's chop gate, but a 16-bar EMA hugs price so tightly its own-window drift is ~4-5x smaller (the GRASS 2026-08-26 tick read 6.4% drift over the old 48-bar ref and only 1.1% over 16 — the live gate slept). Absent = `band_span` (byte-identical to the reworked default). Sweep evidence: drift saturates past ~30 bars (48 buys nothing over 32); over the 2026-08-27→31 executed window ref 32 caught 3/18 stale-flat traps at P/L-neutral cost. The trigger (`band_snapback`) never sees this key.

### `duelist_veto_gate` `{enabled, shadow_mode, min_conf, min_composite}`
Duelist-veto conviction gate (the 19th gate). Keys off the A/B duelist — the second model (`.env.local` `LLM_DUEL_MODEL`, resolved via `duel_store.duelist_config()`) that re-answers the same research prompt and logs its verdict to `duel-logs/hermes-trader-duel.jsonl`. When the primary issues a directional entry (LONG/SHORT) and the duelist abstains with **PASS** (or takes the **opposite side**), the entry must clear elevated conviction: `confidence >= min_conf` (0.90) OR `composite >= min_composite` (60) — otherwise it is vetoed. The duelist's verdict rides into the executor via the analysis dict's `duelist_at_entry` snapshot (written by research.py at entry), so the gate is a pure context read — no extra LLM call, no new I/O. Fail-safes (no-opinion pass): disabled; no duelist verdict (duelist disabled or the second call failed); duelist agrees with the primary's side; unrecognised verdict. A data gap can never block a trade. **SHADOW by default**: `shadow_mode: true` (and how it ships) makes it structurally pass and the executor logs `[gate][SHADOW] duelist_veto WOULD HAVE BLOCKED {coin} {side} (duelist {model} said {verdict}; …)` at WARNING; the `shadow_would_block` marker also lands in the `Trade result` line / `execute` event's `gate_results`, joinable to the ledger (coin/side/ts) and to the duel JSONL for free. Flip `shadow_mode` to `false` (one knob, hot-read per cycle) to make it block. Rationale: the 2026-08-31 48h replay (41 executed trades) found the strict veto removed 12 entries — 5 losers (+$4.58, mostly the 240-min stale-flat-timeout trap) and 7 winners (+$2.75) — net +$1.83 vs the −$2.72 baseline; it is a conviction veto (kills low-conviction stall-prone entries), not a directionally-better model.

### Structural-override gates (LONG-only — upgrade an AI PASS on strong TA/whale)
`force_execute_composite` (bar), `composite_force_execute` (forcing AI-rejects = adverse selection → OFF), `breakout_force_execute`, `whale_force_execute`, `force_execute_slow_burn_count`.

---

## What to actually tune day-to-day

Most of these knobs you set once and leave. The three you'd realistically touch:

1. **`mode`**: flip to `OFF` when you want the bot to stop trading (it keeps scanning, just doesn't execute)
2. **`daily_kill_pct_of_equity`**: raise if you want a looser circuit breaker for the day; pair with `daily_kill_cap_usd` for the absolute ceiling
3. **`min_ai_confidence`**: raise to filter trades when the AI is being too loose; lower to accept more

Everything else is structural — change it deliberately, not reactively.
