"""CHASE-SHADOW — log-only chase-crash discriminator at short-candidate time.

Board card t_c95ea397. The 105d short-cohort backtest (scratch/_ext_bt2.py,
2026-11-26) found exactly ONE entry feature that was OOS-protective: entering
SHORT while the coin already sits <= 8% under its trailing-24h high
("chasing the crash"). Blocked cohort avg -$0.254/trade vs retained +$0.101,
negative in all four months. Post-hoc from the same sweep — per doctrine this
gets NO live veto: it accrues forward evidence as a log line and promotion is
decided by the card's gate (>=30 forward decisions, blocked-net negative
across >=2 regimes; kill if the forward blocked cohort turns positive).

This module is PURE OBSERVATION:
  * it never returns a verdict to any caller (signature returns None; the one
    call site in executor.py discards it — pinned by tests/test_chase_shadow.py),
  * it never raises (fail-open; a broken store or missing history only
    changes the log line),
  * it costs ZERO rate weight: reads the persistent candle store
      (client/candlestore.py), which accumulates confirmed bars from the
      decision path's own fetches (harvest-on-pass). No new API calls.

Field math mirrors the backtest harness (scratch/_ext_bt.py feats() +
btc_mom4h + scratch/_ext_bt2.py btc_drift24) EXACTLY, on CONFIRMED 1h bars
only (t + 1h <= tdec; the store already stores confirmed-only, the filter is
belt-and-braces for a store written mid-hour):
  ema4p      EMA over the last <=300 confirmed 1h closes, alpha 0.4, seed =
             first close in the window; (last/e - 1) * 100
  off24lo    (last / min(last 24 closes) - 1) * 100
  dd24h      (last / max(last 24 closes) - 1) * 100   <- the discriminator
  tape_age_h (tdec - t of the 24h-low CLOSE bar) / 3_600_000
  btc_drift24 (BTC conf[-1].c / conf[-25].c - 1) * 100   (>=26 confirmed bars)
  btc_mom4h   (BTC conf[-1].c / conf[-5].c - 1) * 100    (>=6 confirmed bars)

Config (hot, optional): {"chase_shadow": {"enabled": true,
"dd24h_bar_pct": -8.0}} — enabled defaults True so accrual needs no config
deploy; flip false in the config repo to silence without a rebuild.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

H_MS = 3_600_000
_DEFAULT_BAR_PCT = -8.0
_ERR_STATE = {"count": 0, "last_msg": None}


def _fmt(v: Optional[float], spec: str = "+.2f") -> str:
    return "NA" if v is None else format(v, spec)


def coin_features(bars: List[Dict[str, Any]], tdec_ms: int) -> Optional[Dict[str, float]]:
    """Per-coin extension fields, mirroring the backtest harness. None = no
    opinion (fewer than 30 confirmed 1h bars — same threshold the backtest
    cohort used; thin history must not fake an opinion)."""
    conf = [b for b in bars if b["t"] + H_MS <= tdec_ms]
    if len(conf) < 30:
        return None
    cl = [float(b["c"]) for b in conf]
    win = cl[-300:]
    e = win[0]
    for x in win:
        e = 0.4 * x + 0.6 * e
    last = cl[-1]
    w24 = conf[-24:]
    w24c = [float(b["c"]) for b in w24]
    lo = min(w24c)
    lo_t = int(w24[w24c.index(lo)]["t"])
    return {
        "ema4p": (last / e - 1.0) * 100.0,
        "off24lo": (last / lo - 1.0) * 100.0,
        "dd24h": (last / max(w24c) - 1.0) * 100.0,
        "tape_age_h": (tdec_ms - lo_t) / 3_600_000.0,
        "mark": last,
    }


def btc_features(bars: List[Dict[str, Any]], tdec_ms: int) -> Dict[str, Optional[float]]:
    conf = [b for b in bars if b["t"] + H_MS <= tdec_ms]
    cl = [float(b["c"]) for b in conf]
    drift = (cl[-1] / cl[-25] - 1.0) * 100.0 if len(cl) >= 26 and cl[-25] else None
    mom4 = (cl[-1] / cl[-5] - 1.0) * 100.0 if len(cl) >= 6 and cl[-5] else None
    return {"btc_drift24": drift, "btc_mom4h": mom4}


def log_chase_shadow(coin: str, trade_side: Any, config: Any,
                     now_ms: Optional[int] = None) -> None:
    """Emit ONE `[gate][CHASE-SHADOW]` line per SHORT entry evaluation.

    (Any-typed on purpose: hot config values can be any JSON shape and this
    instrument tolerates all of them — pins pass deliberately-garbage types.)

    Call-and-discard: the return value is None BY CONTRACT (pinned test) so no
    decision path can consume it. Fail-open: any error is swallowed to a
    once-warned line — this instrument must never affect a live decision.
    EVERY guard runs INSIDE the try (independent audit 2026-11-26: guards
    outside it let a hot-config shorthand like {"chase_shadow": true} raise
    AttributeError into the executor call path — the one defect class this
    instrument must never have).
    """
    try:
        if str(trade_side).lower() != "short":
            return
        cfg = config.get("chase_shadow") if isinstance(config, dict) else None
        if not isinstance(cfg, dict):
            cfg = {}  # shorthand/typo (true, list, missing) -> defaults
        en = cfg.get("enabled", True)
        if isinstance(en, str):  # "false" quoted in config must silence
            en = en.strip().lower() not in ("false", "0", "off", "no")
        if not en:
            return
        import time
        import hermes_trader.client.candlestore as cs

        if now_ms is None:
            now_ms = int(time.time() * 1000)
        bar_pct = float(cfg.get("dd24h_bar_pct", _DEFAULT_BAR_PCT))

        coin_bars = cs.series(coin, "1h")
        btc_bars = cs.series("BTC", "1h")
        cf = coin_features(coin_bars, now_ms) if coin_bars else None
        bf = btc_features(btc_bars, now_ms)

        if cf is None:
            # Data gap = no opinion (fail-open), but keep the denominator so
            # accrual coverage is auditable from the log itself.
            logger.warning(
                f"[gate][CHASE-SHADOW] {coin} nodata "
                f"coin_bars={len(coin_bars or [])} btc_bars={len(btc_bars or [])} "
                f"would_block=N bar={bar_pct:+.1f}")
            return
        would = cf["dd24h"] <= bar_pct
        logger.warning(
            f"[gate][CHASE-SHADOW] {coin} dd24h={_fmt(cf['dd24h'])} "
            f"would_block={'Y' if would else 'N'} bar={bar_pct:+.1f} "
            f"mark={_fmt(cf['mark'], '.6g')} ema4p={_fmt(cf['ema4p'])} "
            f"off24lo={_fmt(cf['off24lo'])} tape_age_h={_fmt(cf['tape_age_h'], '.1f')} "
            f"btc_drift24={_fmt(bf['btc_drift24'])} btc_mom4h={_fmt(bf['btc_mom4h'])}")
    except Exception as e:  # never touches the live decision
        _ERR_STATE["count"] += 1
        msg = str(e)
        if _ERR_STATE["last_msg"] != msg:
            logger.warning(f"[gate][CHASE-SHADOW] fail-open (no decision impact): {e}")
            _ERR_STATE["last_msg"] = msg
