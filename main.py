"""Trophy Tide: earn-rate forecaster for GTA VI trophies on Exophase.

Entry for make.fun gta6-trophy-rarity-forecast-01. Standard library only.

For every trophy of Grand Theft Auto VI (PS5) it forecasts the Exophase earn
rate at the case's target as five quantiles (5/25/50/75/95%), scored by
pinball loss on the logit scale.

Model, per trophy, in logit space (logit(p) = ln(p / (1 - p)), p in percent):

1.  Persistence: the rate in the case's last read.
2.  Logit trend: weighted least squares of logit(rate) against
    ln(1 + hours since launch), fitted on the last 36 hours of reads with
    recency weights (12-hour half-life), extrapolated to the target with the
    slope damped for the horizon and the total move capped. Launch-week
    earn rates move roughly linearly in log-time: early trophies climb fast
    and then bend over as the player base broadens, and the log-time axis
    captures that bend without overshooting the way a linear-in-time fit
    would over a 4+ day horizon.
3.  Earners-over-players: the implied earners E = rate * players_tracked /
    100 and the tracked-player count P are each fitted the same way in
    log space and extrapolated separately, then recombined as a rate.
    This matters because the rate is a ratio: when the tracked base grows
    faster than a trophy's earners, the rate falls even while more people
    earn the trophy, and vice versa. The two fits see that directly,
    where a rate-only trend only sees the net effect.

The median is a weighted mean of the three in logit space (persistence
gets the smallest weight: over days, standing still is the weakest
forecast). The spread is a random-walk term growing with the square root
of the horizon, plus the trend fit's residual scale, plus a base allowance
for the page's refresh jumps -- Exophase's page is a cached snapshot that
recomputes in jumps, so a forecast that trusts the last read too narrowly
pays for it. When the whole history is frozen (identical rates and an
identical tracked-player count on every read, as on launch night before
the first recompute), no trend can be estimated at all: the median stays
at the last rate and the spread is set wide, because the first recompute
can move every rate a long way at once.

Deterministic: no randomness anywhere, fixed iteration order, fixed
rounding, so two runs on the same case write identical bytes.

Run: python3 main.py -- reads every <id>.in in $MAKEFUN_INPUT_DIR (or
/input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR (or /output).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Random-walk spread of a rate's logit move, per square-root day.
SIGMA_RW = 0.5
# Base spread allowance (logit) for the page's recompute jumps.
SPREAD_FLOOR = 0.35
# Spread used when the history is frozen: no trend is estimable and the
# first recompute can move every rate a long way at once.
STALE_SPREAD = 1.6
# Recency weighting of the trend fit: weight halves every 12 hours.
HALF_LIFE_HOURS = 12.0
# Trend fit window: reads newer than this many hours before the checkpoint.
# 72 hours: launch-week earn rates move smoothly in log-time, and a longer
# window fits that trend with far less noise than a day's worth of reads.
WINDOW_HOURS = 72.0
# The trend slope is damped by 1 / (1 + SLOPE_DAMP * horizon_days).
SLOPE_DAMP = 0.12
# The trend's total logit move from the last rate is capped at this.
MAX_TREND_MOVE = 2.0


def parse_time(text):
    """Parse an ISO-8601 timestamp (with Z) to epoch seconds."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def clamp_rate(rate):
    """Clamp a percent rate into (0, 100) for the logit (page prints 2dp)."""
    return min(99.99, max(0.01, rate))


def logit(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def wls(points):
    """Weighted least squares y ~ a + b*x.

    points: list of (x, y, w). Returns (a, b, resid_sd) or None when the
    fit is degenerate (fewer than 2 points or no spread in x).
    """
    if len(points) < 2:
        return None
    sw = sum(w for _, _, w in points)
    if sw <= 0:
        return None
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    b = sxy / sxx
    a = my - b * mx
    dof = max(1, len(points) - 2)
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in points) / sw
    resid_sd = math.sqrt(var * len(points) / dof) if var > 0 else 0.0
    return a, b, resid_sd


def quantiles(median_logit, sd):
    """Five quantiles in percent from a logit-normal forecast."""
    sd = max(0.05, min(2.5, sd))
    out = []
    prev = None
    for z in Z:
        q = sigmoid(median_logit + z * sd) * 100.0
        q = min(99.99, max(0.01, q))
        q = round(q, 4)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = max(0.0, (target - checkpoint) / 86400.0)

    release = case.get("game", {}).get("release_date")
    if release:
        t0 = parse_time(release + "T00:00:00Z")
    elif history:
        t0 = parse_time(history[0]["slot"])
    else:
        t0 = checkpoint

    def x_of(epoch):
        hours = max(0.0, (epoch - t0) / 3600.0)
        return math.log1p(hours)

    x_target = x_of(target)
    window_start = checkpoint - WINDOW_HOURS * 3600.0

    reads = []
    for read in history:
        slot = parse_time(read["slot"])
        rates = read.get("rates") or []
        players = read.get("players_tracked")
        reads.append((slot, rates, players))
    reads.sort(key=lambda r: r[0])

    # Drop a frozen prefix: the launch-night page sat unchanged (identical
    # rates and tracked count) until its first recompute, and those stale
    # reads would flatten any trend fitted over them. The live history
    # starts at the first read that differs from the one before it.
    if len(reads) >= 3:
        for j in range(1, len(reads)):
            prev_rates, prev_players = reads[j - 1][1], reads[j - 1][2]
            cur_rates, cur_players = reads[j][1], reads[j][2]
            differs = (
                cur_players != prev_players
                or len(cur_rates) != len(prev_rates)
                or any((a is None) != (b is None)
                       or (a is not None and b is not None and abs(a - b) > 1e-9)
                       for a, b in zip(cur_rates, prev_rates))
            )
            if differs:
                # Only a run of 2+ identical reads at the start is a frozen
                # prefix; a first read that already differs is live data.
                if j > 1 and len(reads) - j >= 3:
                    reads = reads[j:]
                break

    # Stale-page detection: every rate and the tracked count identical on
    # every read (the launch-night page before its first recompute).
    stale = False
    if len(reads) >= 2:
        first_rates = reads[0][1]
        first_players = reads[0][2]
        same_rates = all(
            len(r[1]) == len(first_rates)
            and all((a is None and b is None)
                    or (a is not None and b is not None and abs(a - b) < 1e-9)
                    for a, b in zip(r[1], first_rates))
            for r in reads
        )
        same_players = all(r[2] == first_players for r in reads)
        stale = same_rates and same_players

    damp = 1.0 / (1.0 + SLOPE_DAMP * horizon_days)

    rows = []
    for idx, trophy in enumerate(trophies):
        tid = trophy["id"]
        # Aligned recent-window observations for this trophy:
        # (x, logit(rate), ln(rate/100), ln(players) or None, weight).
        obs = []
        last_rate = None
        for slot, rates, players in reads:
            if idx >= len(rates):
                continue
            rate = rates[idx]
            if rate is None:
                continue
            last_rate = rate
            if slot < window_start and len(reads) > 4:
                continue
            x = x_of(slot)
            w = 0.5 ** ((checkpoint - slot) / (HALF_LIFE_HOURS * 3600.0))
            obs.append((x, logit(rate), math.log(clamp_rate(rate) / 100.0),
                        math.log(players) if players else None, w))
        if last_rate is None:
            # No read ever listed this trophy: fall back to a middling rate.
            last_rate = 1.0
            obs = [(x_of(checkpoint), logit(last_rate), math.log(0.01),
                    None, 1.0)]
        last_logit = logit(last_rate)
        last_x = obs[-1][0]

        if stale or len(obs) < 3:
            sd = STALE_SPREAD if stale else max(
                1.2, SIGMA_RW * math.sqrt(horizon_days))
            rows.append({"id": tid, "q": quantiles(last_logit, sd)})
            continue

        # Trend median: logit trend extrapolated from the last read.
        trend = wls([(x, lg, w) for x, lg, _, _, w in obs])
        if trend is None:
            trend_logit = last_logit
            resid_sd = 0.0
        else:
            a, b, resid_sd = trend
            move = b * damp * (x_target - last_x)
            move = max(-MAX_TREND_MOVE, min(MAX_TREND_MOVE, move))
            trend_logit = last_logit + move

        # Earners-over-players median: ln E and ln P fitted separately
        # (only on reads that carry a player count) and recombined.
        ep_logit = None
        paired = [(x, lr, lp, w) for x, _, lr, lp, w in obs if lp is not None]
        if len(paired) >= 3:
            fit_e = wls([(x, lr + lp, w) for x, lr, lp, w in paired])
            fit_p = wls([(x, lp, w) for x, _, lp, w in paired])
            if fit_e is not None and fit_p is not None:
                a_e, b_e, _ = fit_e
                a_p, b_p, _ = fit_p
                ln_e_t = (a_e + b_e * last_x) + b_e * damp * (x_target - last_x)
                ln_p_t = (a_p + b_p * last_x) + b_p * damp * (x_target - last_x)
                rate_ep = 100.0 * math.exp(ln_e_t - ln_p_t)
                ep_logit = logit(rate_ep)

        # Combine in logit space: persistence 0.30, trend 0.35, E/P 0.35
        # (E/P falls back to trend when it cannot be fitted).
        if ep_logit is None:
            median_logit = 0.30 * last_logit + 0.70 * trend_logit
        else:
            median_logit = (0.30 * last_logit + 0.35 * trend_logit
                            + 0.35 * ep_logit)

        rw = SIGMA_RW * math.sqrt(horizon_days)
        sd = math.sqrt(rw * rw + (1.2 * resid_sd) ** 2 + SPREAD_FLOOR ** 2)
        rows.append({"id": tid, "q": quantiles(median_logit, sd)})

    return {"trophies": rows}


def main():
    in_files = sorted(INPUT_DIR.glob("*.in"))
    if not in_files:
        print("no .in files in %s" % INPUT_DIR, file=sys.stderr)
        return 1
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_file in in_files:
        with open(in_file, "r", encoding="utf-8") as f:
            case = json.load(f)
        forecast = forecast_case(case)
        out_file = OUTPUT_DIR / (in_file.stem + ".out")
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
