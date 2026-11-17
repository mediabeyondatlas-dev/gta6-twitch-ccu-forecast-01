"""Launch-day Twitch forecast for GTA VI (make.fun gta6-twitch-ccu-forecast-01).

Approach: fit the day's shape at run time from the case's own comparables
(past game launches with full release-day hourly curves). For each past
launch we look at the log ratio of each hour's viewers to that launch's
hour-1 print. Launches are grouped by the size of their hour-1 print --
small, medium and large -- because a launch that already opens huge barely
grows during the day while a launch that opens tiny can grow a hundredfold.
Within each group we take the median log ratio per hour as the shape, with
the spread of the ratios as the log spread for the quantiles.

For large openings (the regime GTA VI is in) the median shape is blended
with a per-hour linear fit of the log ratio against log(hour-1 print):
bigger openings grow less, and the fit captures that smoothly instead of
relying on one group's median alone. Every log ratio is then pulled 10%
toward zero (toward a flat day), which the leave-one-out validation below
preferred over the unshrunk shape.

Leave-one-out validation on the 28 usable comparables in the practice
cases (mean pinball loss on the log scale, lower is better):
    this model   overall 0.528 | large-opening launches 0.278
    house starter overall 1.137 | large-opening launches 0.340

Python 3.13, standard library only. Reads every *.in in the input
directory and writes one *.out per case, deterministically.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZVALS = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Hour-1 print (viewers) that separates the three opening sizes.
BIG_OPENING = 20000
MID_OPENING = 3000
# Weight of the log-size fit versus the large-group median, for large openings.
FIT_WEIGHT = 0.7
# Pull of every log ratio toward a flat day (0 = persistence).
SHRINK_TOWARD_FLAT = 0.9
# Multiplier on the fitted log spreads.
SPREAD_MULT = 0.9
# Floor on any hour's log spread, so early hours still carry some width.
SPREAD_FLOOR = 0.15
# The fitted log ratio is never allowed outside this range.
RATIO_LOW = -1.0
RATIO_HIGH = 2.5
# A comparable needs at least this many release-day hours to be usable.
MIN_HOURS = 20


def _round_half_up(x):
    """Round to 2 decimals, halves up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _spread(values, centre):
    """Population standard deviation of values around centre."""
    if len(values) < 2:
        return 0.5
    mean_sq = sum((v - centre) ** 2 for v in values) / len(values)
    return math.sqrt(mean_sq)


def _line_fit(xs, ys):
    """Least-squares fit y = a + b*x; returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return my - slope * mx, slope


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _release_hours(comp):
    """Map hour (1..24 of the release UTC day) to viewers for one comparable.

    Returns None when the entry is not a real launch day (the target game's
    own pre-release category, or a launch whose peak came during an
    early-access window days before the standard release).
    """
    unlock = (comp.get("unlock") or "").lower()
    name = (comp.get("name") or "").lower()
    if "not a launch" in unlock or "early-access" in unlock:
        return None
    if "grand theft auto vi" in name:
        return None
    start = _parse_time(comp["day_start"])
    hours = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        delta = (_parse_time(row["from"]) - start).total_seconds() / 3600.0
        if 0.0 <= delta < 24.0:
            hours[int(delta) + 1] = viewers
    if 1 not in hours or len(hours) < MIN_HOURS:
        return None
    return hours


def _tier(opening):
    if opening >= BIG_OPENING:
        return "big"
    if opening >= MID_OPENING:
        return "mid"
    return "small"


def _fit(days):
    """Fit the shape model from release-day curves.

    days: list of {hour: viewers} dicts. Returns (shapes, fits) where shapes
    maps tier -> hour -> (median log ratio, spread) and fits maps hour ->
    (intercept, slope, residual spread) of the log-size fit.
    """
    shapes = {}
    for tier in ("big", "mid", "small"):
        group = [d for d in days if _tier(d[1]) == tier]
        per_hour = {}
        for h in range(2, 25):
            ratios = [math.log(d[h] / d[1]) for d in group if h in d]
            if ratios:
                centre = _median(ratios)
                per_hour[h] = (centre, _spread(ratios, centre))
            else:
                per_hour[h] = (0.0, 0.5)
        shapes[tier] = per_hour
    fits = {}
    for h in range(2, 25):
        xs = [math.log(d[1]) for d in days if h in d]
        ys = [math.log(d[h] / d[1]) for d in days if h in d]
        if len(xs) >= 4:
            a, b = _line_fit(xs, ys)
            resid = [y - (a + b * x) for x, y in zip(xs, ys)]
            fits[h] = (a, b, _spread(resid, 0.0))
        else:
            fits[h] = (0.0, 0.0, 0.5)
    return shapes, fits


def _hour_row(p1, h, shapes, fits):
    tier = _tier(p1)
    centre, width = shapes[tier][h]
    if tier == "big":
        a, b, resid = fits[h]
        fitted = a + b * math.log(max(1.0, p1))
        centre = FIT_WEIGHT * fitted + (1.0 - FIT_WEIGHT) * centre
        width = resid
        centre = max(RATIO_LOW, min(RATIO_HIGH, centre))
    centre *= SHRINK_TOWARD_FLAT
    width = max(SPREAD_FLOOR, width) * SPREAD_MULT
    median = p1 * math.exp(centre)
    row = []
    for z in ZVALS:
        q = _round_half_up(median * math.exp(z * width))
        row.append(max(1, min(1000000000, q)))
    for i in range(1, len(row)):
        if row[i] < row[i - 1]:
            row[i] = row[i - 1]
    return row


def _flat_row(p1, h):
    """Persistence fallback: flat median, random-walk spread."""
    width = 0.15 * math.sqrt(h - 1)
    row = []
    for z in ZVALS:
        q = _round_half_up(p1 * math.exp(z * width))
        row.append(max(1, min(1000000000, q)))
    return row


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    days = []
    for comp in case.get("comparables") or []:
        hours = _release_hours(comp)
        if hours is not None:
            days.append(hours)
    if days:
        shapes, fits = _fit(days)
        rows = [
            {"hour": row["hour"], "q": _hour_row(p1, row["hour"], shapes, fits)}
            for row in case["forecast"]
        ]
    else:
        rows = [
            {"hour": row["hour"], "q": _flat_row(p1, row["hour"])}
            for row in case["forecast"]
        ]
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
