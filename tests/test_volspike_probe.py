"""Shadow probe for the volume-spike leg (t_c3e7ab77, 2026-11-28).

Pins the probe's three arms against the live volume_spike rule:
  - live: z of candles[-1] (the FORMING bar, per HL candleSnapshot)
  - arm 1: closed-bar lookback (last 3 closed bars, each vs own 20-bar window)
  - arm 2: time-scaled forming bar (v_partial / elapsed_fraction), guarded
    to fraction >= 0.4, bucket derived from the `interval` param (prod Candle
    objects have no 'T').

The probe is log-only: it must NEVER look like a trigger hit, and the live
rule's behavior must not move. These pins are the contract for that.

NOTE (independent audit 2026-11-28): the LIVE effective sigmaThreshold is
2.0 (TRIGGER_CONFIG default, config.py:47 — no config key overrides it).
These pins pass 3 explicitly for readable arithmetic; production parity is
on the formula, not the sigma value. Tests pass `now_ms` (probe test seam)
so fraction math is deterministic — wall-clock gaps between fixture build
and probe call must never flip a pin (audit flake finding).
"""
import time

from hermes_trader.indicators.triggers import volume_spike, volume_spike_probe


def _series(vols, bucket_ms=300_000, forming_age_ms=None, now=None, with_T=True):
    """Candles ending in a FORMING bar: last bar's t is now-age_ms, T=next bucket.
    Closed bars get closed t/T pairs. The same `now` must be passed to the
    probe as now_ms for deterministic fractions. with_T=False yields Candle-style
    dicts WITHOUT 'T' (the production shape) to exercise the bucket param."""
    now = now if now is not None else time.time() * 1000
    out = []
    for i, v in enumerate(vols):
        if i == len(vols) - 1:
            age = forming_age_ms if forming_age_ms is not None else 150_000
            t0 = now - age
            # HL forming candle: T is the bucket end = t0 + bucket, exactly
            t_end = int(t0 + bucket_ms)
        else:
            t0 = now - (len(vols) - i) * bucket_ms
            t_end = int(t0 + bucket_ms - 1)
        row = {"t": int(t0), "o": 1, "h": 1, "l": 1, "c": 1, "v": v}
        if with_T:
            row["T"] = t_end
        out.append(row)
    return out


def _base():
    # 24 quiet bars with small alternation so std is nonzero but ~4
    return [10 + (4 if i % 2 else -4) for i in range(24)]


def test_closed_bar_spike_missed_by_live_detected_by_probe():
    """Surge bar closed one bar ago: live rule (forming bar flat) stays flat,
    the closed-bar-lookback arm fires at age=0."""
    vols = _base() + [200, 10]  # surge at [-2] (last CLOSED), forming bar flat
    live = volume_spike(_series(vols), 3)
    probe = volume_spike_probe(_series(vols), 3)
    assert not live["fired"], "live rule must stay flat on a flat forming bar"
    assert probe["would_fire"] and probe["age_bars"] == 0
    assert probe["z3max"] >= 3

def test_time_scaled_forming_bar_fires_when_live_is_diluted():
    """Owner-proposal arm: a real burst mid-bar (60% elapsed) is diluted in the
    live z but crosses the sigma bar once scaled by elapsed fraction."""
    now = time.time() * 1000
    vols = _base() + [20]  # forming bar at frac=0.6: live z=2.5 (<3), scaled 33
    s = _series(vols, forming_age_ms=180_000, now=now)
    live = volume_spike(s, 3)
    probe = volume_spike_probe(s, 3, now_ms=now)
    assert not live["fired"], "live rule must be diluted below the bar"
    assert probe["frac"] == 0.6  # deterministic via now_ms seam
    assert probe["would_fire_scaled"]
    assert probe["z_forming_scaled"] is not None and probe["z_forming_scaled"] >= 3

def test_early_forming_bar_not_extrapolated():
    """Fraction guard: at 0.2 elapsed the scaled arm stays dark (noise guard)."""
    now = time.time() * 1000
    vols = _base() + [30]
    s = _series(vols, forming_age_ms=60_000, now=now)
    probe = volume_spike_probe(s, 3, now_ms=now)
    assert probe["frac"] is None and probe["z_forming_scaled"] is None
    assert not probe["would_fire_scaled"]

def test_bucket_follows_interval_param():
    """Audit finding: Candle has no 'T' — the bucket MUST come from interval,
    not a hardcoded 5m. Same bar 200s old, no T: 15m bucket -> frac=0.13
    guard-dark, 5m bucket -> frac=0.67 lit."""
    now = time.time() * 1000
    vols = _base() + [20]
    s = _series(vols, forming_age_ms=200_000, now=now, with_T=False)
    probe15 = volume_spike_probe(s, 3, interval="15m", now_ms=now)
    assert probe15["frac"] is None and not probe15["would_fire_scaled"]
    probe5 = volume_spike_probe(s, 3, now_ms=now)
    assert probe5["frac"] == 0.67 and probe5["would_fire_scaled"]

def test_quiet_series_no_fire_and_shape_stable():
    probe = volume_spike_probe(_series(_base() + [10, 10]), 3)
    assert not probe["would_fire"] and not probe["would_fire_scaled"]
    for k in ("would_fire", "would_fire_scaled", "z_forming", "z_closed1",
              "z3max", "age_bars", "z_forming_scaled", "frac"):
        assert k in probe
    # probe rows must not masquerade as trigger hits for any()/score consumers
    assert "name" not in probe and "fired" not in probe and "score" not in probe

def test_short_series_is_safe():
    probe = volume_spike_probe(_series([10] * 5), 3)
    assert probe["z3max"] is None and probe["z_forming"] is None
    assert not probe["would_fire"]

def test_malformed_interval_falls_back_to_5m_bucket():
    now = time.time() * 1000
    vols = _base() + [20]
    s = _series(vols, forming_age_ms=180_000, now=now)
    probe = volume_spike_probe(s, 3, interval="weird", now_ms=now)
    assert probe["frac"] == 0.6  # 5m fallback, no crash
