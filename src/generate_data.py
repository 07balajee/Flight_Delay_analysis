"""Synthetic flight-level data generator for the "Why are flights delayed?" case.

The PDF supplies only the schema, so we simulate one year of operations for a
domestic airline and then deliberately inject realistic data-quality problems
(duplicates, inconsistent codes, missing values, typos) so that the cleaning
and validation step is meaningful.

Output schema (exactly as specified in the brief):
    Flight_ID, Airport, Route, Aircraft, Scheduled_Departure, Actual_Departure,
    Weather, Previous_Flight, Turnaround_Time

Delay mechanics baked into the simulation (the analysis should rediscover them):
    * airport congestion (BOM, DEL highest)
    * time-of-day (evening bank) and day-of-week (Fri/Sun/Mon) peaks
    * seasonal weather: winter fog in the north, monsoon storms on the west coast
    * reactionary / knock-on delay from the previous leg of the same aircraft
    * turnarounds scheduled below the aircraft type's minimum ground time
    * a handful of structurally delayed routes and two unreliable tails
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 22
YEAR = 2025
N_AIRCRAFT = 44

# code: (lat, lon, congestion factor, region)
AIRPORTS = {
    "DEL": (28.56, 77.10, 1.55, "north"),
    "BOM": (19.09, 72.87, 1.70, "west"),
    "BLR": (13.20, 77.71, 1.20, "south"),
    "HYD": (17.24, 78.43, 0.90, "south"),
    "MAA": (12.99, 80.17, 1.05, "southeast"),
    "CCU": (22.65, 88.45, 1.15, "east"),
    "GOI": (15.38, 73.83, 0.85, "west"),
    "PNQ": (18.58, 73.92, 0.95, "west"),
    "AMD": (23.07, 72.63, 0.85, "north"),
    "COK": (10.15, 76.40, 0.90, "west"),
    "JAI": (26.82, 75.81, 0.75, "north"),
    "LKO": (26.76, 80.88, 0.80, "north"),
}
HUBS = ["DEL", "BOM", "BLR", "HYD", "MAA", "CCU"]
SPOKES = {
    "GOI": ["BOM", "DEL", "BLR"],
    "PNQ": ["DEL", "BLR", "HYD"],
    "AMD": ["DEL", "BOM", "BLR"],
    "COK": ["BOM", "BLR", "MAA"],
    "JAI": ["DEL", "BOM", "HYD"],
    "LKO": ["DEL", "BOM", "BLR"],
}
# Structurally problematic routes: mean extra minutes of delay.
ROUTE_PENALTY = {
    "BOM-DEL": 14, "DEL-BOM": 12, "BOM-GOI": 10, "CCU-DEL": 11,
    "BLR-BOM": 8, "DEL-LKO": 7,
}
# type: (min turnaround minutes, share of fleet)
AIRCRAFT_TYPES = {"A320": (35, 0.40), "A321": (45, 0.20), "B737": (40, 0.25), "ATR72": (25, 0.15)}
PROBLEM_TAILS = 2

HOUR_FACTOR = {**{h: 0.75 for h in range(0, 9)}, **{h: 1.0 for h in range(9, 13)},
               **{h: 1.1 for h in range(13, 17)}, **{h: 1.55 for h in range(17, 22)},
               **{h: 1.25 for h in range(22, 24)}}
DOW_FACTOR = {0: 1.15, 1: 0.88, 2: 0.90, 3: 1.00, 4: 1.30, 5: 0.95, 6: 1.25}
MONTH_FACTOR = {1: 1.15, 2: 1.0, 3: 0.95, 4: 1.0, 5: 1.15, 6: 1.05, 7: 1.0, 8: 1.0,
                9: 0.95, 10: 1.1, 11: 1.15, 12: 1.25}
WEATHER_EXTRA = {"Clear": 0, "Cloudy": 2, "Haze": 6, "Rain": 12, "Thunderstorm": 48, "Fog": 62}


def _haversine(a: str, b: str) -> float:
    la1, lo1, *_ = AIRPORTS[a]
    la2, lo2, *_ = AIRPORTS[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def _network() -> dict[str, list[str]]:
    net: dict[str, list[str]] = {a: [] for a in AIRPORTS}
    for i, a in enumerate(HUBS):
        for b in HUBS[i + 1:]:
            net[a].append(b)
            net[b].append(a)
    for spoke, hubs in SPOKES.items():
        for h in hubs:
            net[spoke].append(h)
            net[h].append(spoke)
    return net


def _weather_regimes(rng: np.random.Generator, dates: pd.DatetimeIndex) -> dict:
    """Daily weather regime per airport: fog / storm / rain / haze / fair."""
    regimes = {}
    for ap, (_, _, _, region) in AIRPORTS.items():
        for d in dates:
            m = d.month
            p_fog = 0.0
            p_storm = 0.02
            p_rain = 0.06
            p_haze = 0.05
            if region == "north" and m in (12, 1, 2):
                p_fog = {12: 0.35, 1: 0.45, 2: 0.15}[m]
                p_haze = 0.25
            if region in ("west", "east") and m in (6, 7, 8, 9):
                p_storm, p_rain = 0.28, 0.35
            if region == "south" and m in (6, 7, 8, 9, 10):
                p_storm, p_rain = 0.12, 0.25
            if region == "southeast" and m in (10, 11, 12):
                p_storm, p_rain = 0.25, 0.30
            if region == "north" and m in (7, 8):
                p_storm, p_rain = 0.12, 0.25
            r = rng.random()
            if r < p_fog:
                reg = "fog"
            elif r < p_fog + p_storm:
                reg = "storm"
            elif r < p_fog + p_storm + p_rain:
                reg = "rain"
            elif r < p_fog + p_storm + p_rain + p_haze:
                reg = "haze"
            else:
                reg = "fair"
            regimes[(ap, d.date())] = reg
    return regimes


def _flight_weather(rng: np.random.Generator, regime: str, hour: int) -> str:
    if regime == "fog":
        return "Fog" if hour < 12 else ("Haze" if rng.random() < 0.6 else "Clear")
    if regime == "storm":
        if 13 <= hour <= 21:
            return "Thunderstorm" if rng.random() < 0.7 else "Rain"
        return "Rain" if rng.random() < 0.5 else "Cloudy"
    if regime == "rain":
        return "Rain" if rng.random() < 0.7 else "Cloudy"
    if regime == "haze":
        return "Haze" if rng.random() < 0.7 else "Clear"
    return "Clear" if rng.random() < 0.75 else "Cloudy"


def simulate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    net = _network()
    dates = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="D")
    regimes = _weather_regimes(rng, dates)

    # Fleet
    types = list(AIRCRAFT_TYPES)
    shares = np.array([AIRCRAFT_TYPES[t][1] for t in types])
    fleet = []
    counters = {t: 100 for t in types}
    for _ in range(N_AIRCRAFT):
        t = rng.choice(types, p=shares / shares.sum())
        counters[t] += 1
        fleet.append(f"{t}-{counters[t]}")
    problem_tails = set(rng.choice(fleet, size=PROBLEM_TAILS, replace=False))
    tail_effect = {a: (14 if a in problem_tails else rng.normal(0, 1.2)) for a in fleet}

    rows = []
    fid = 10000
    hub_weight = {a: (3.0 if a in HUBS else 1.0) for a in AIRPORTS}
    for tail in fleet:
        atype = tail.split("-")[0]
        min_turn = AIRCRAFT_TYPES[atype][0]
        location = rng.choice(HUBS)
        prev = None  # dict with id, sched_arr, act_arr
        for d in dates:
            sched = d + pd.Timedelta(minutes=int(rng.integers(5 * 60 + 30, 7 * 60 + 45) // 5 * 5))
            while sched.hour < 23 and sched.date() == d.date():
                choices = net[location]
                w = np.array([hub_weight[c] for c in choices])
                dest = rng.choice(choices, p=w / w.sum())
                route = f"{location}-{dest}"
                block = int(round((35 + _haversine(location, dest) / 8.5) / 5) * 5)

                if prev is None:
                    turnaround = np.nan
                    propagated = 0.0
                    prev_id = ""
                else:
                    turnaround = (sched - prev["sched_arr"]).total_seconds() / 60
                    slack = max(0.0, turnaround - min_turn)
                    propagated = max(0.0, prev["arr_delay"] - slack)
                    prev_id = prev["id"]

                hour, dow, month = sched.hour, sched.dayofweek, sched.month
                weather = _flight_weather(rng, regimes[(location, d.date())], hour)
                congestion = AIRPORTS[location][2]
                scale = 3.6 * congestion * HOUR_FACTOR[hour] * DOW_FACTOR[dow] * MONTH_FACTOR[month]

                own = rng.exponential(scale)
                wx = rng.exponential(WEATHER_EXTRA[weather]) if WEATHER_EXTRA[weather] else 0.0
                if weather in ("Fog", "Thunderstorm") and rng.random() < 0.25:
                    wx *= 0.1  # not every flight in bad weather is hit hard
                ground = 0.0
                if not np.isnan(turnaround) and turnaround < min_turn:
                    short = min_turn - turnaround
                    ground = short * rng.uniform(0.6, 1.3) + (rng.exponential(10) if rng.random() < 0.7 else 0)
                route_pen = rng.exponential(ROUTE_PENALTY[route]) if route in ROUTE_PENALTY else 0.0
                tail_pen = max(0.0, rng.normal(tail_effect[tail], 4))

                delay = 0.85 * propagated + own + wx + ground + route_pen + tail_pen - rng.uniform(0, 12)
                delay = int(round(max(delay, -10)))

                actual = sched + pd.Timedelta(minutes=delay)
                arr_delay = delay + rng.normal(-7, 6)  # some block-time recovery en route
                rows.append({
                    "Flight_ID": f"FL{fid}",
                    "Airport": location,
                    "Route": route,
                    "Aircraft": tail,
                    "Scheduled_Departure": sched,
                    "Actual_Departure": actual,
                    "Weather": weather,
                    "Previous_Flight": prev_id,
                    "Turnaround_Time": turnaround,
                })
                prev = {"id": f"FL{fid}", "sched_arr": sched + pd.Timedelta(minutes=block),
                        "arr_delay": max(0.0, arr_delay)}
                fid += 1
                location = dest

                # plan next leg's ground time: airline often schedules tight turns
                offset = rng.choice([-12, -8, -5, 0, 5, 10, 15, 20, 30, 45, 60],
                                    p=[.03, .04, .06, .12, .14, .15, .13, .11, .10, .07, .05])
                next_turn = max(15, min_turn + offset)
                sched = prev["sched_arr"] + pd.Timedelta(minutes=int(next_turn // 5 * 5))

    df = pd.DataFrame(rows).sort_values("Scheduled_Departure").reset_index(drop=True)
    return df


def inject_dirt(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Add realistic data-quality issues so cleaning has something to fix."""
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    n = len(df)

    def pick(frac):
        return rng.choice(n, size=int(n * frac), replace=False)

    fmt = "%Y-%m-%d %H:%M"
    df["Scheduled_Departure"] = df["Scheduled_Departure"].dt.strftime(fmt)
    df["Actual_Departure"] = df["Actual_Departure"].dt.strftime(fmt)

    # mixed date format (DD/MM/YYYY) for a slice of rows
    idx = pick(0.01)
    for col in ("Scheduled_Departure", "Actual_Departure"):
        df.loc[idx, col] = pd.to_datetime(df.loc[idx, col]).dt.strftime("%d/%m/%Y %H:%M")

    # inconsistent airport / route / weather text
    idx = pick(0.02)
    df.loc[idx, "Airport"] = df.loc[idx, "Airport"].str.lower()
    idx = pick(0.01)
    df.loc[idx, "Airport"] = " " + df.loc[idx, "Airport"] + " "
    idx = pick(0.01)
    df.loc[idx, "Route"] = df.loc[idx, "Route"].str.replace("-", " - ", regex=False)
    idx = pick(0.005)
    df.loc[idx, "Route"] = df.loc[idx, "Route"].str.lower()
    idx = pick(0.02)
    df.loc[idx, "Weather"] = df.loc[idx, "Weather"].str.upper()
    idx = pick(0.01)
    df.loc[idx, "Weather"] = df.loc[idx, "Weather"].str.lower() + " "

    # airport field not matching route origin
    idx = pick(0.003)
    codes = np.array(list(AIRPORTS))
    df.loc[idx, "Airport"] = rng.choice(codes, size=len(idx))

    # missing values
    df.loc[pick(0.03), "Weather"] = np.nan
    df.loc[pick(0.015), "Turnaround_Time"] = np.nan
    df.loc[pick(0.008), "Actual_Departure"] = np.nan  # cancelled / not recorded

    # invalid numerics
    idx = pick(0.003)
    df.loc[idx, "Turnaround_Time"] = -df.loc[idx, "Turnaround_Time"]
    idx = pick(0.002)
    df.loc[idx, "Turnaround_Time"] = 9999

    # date-entry typos in actual departure (wrong day)
    idx = pick(0.002)
    shifted = pd.to_datetime(df.loc[idx, "Actual_Departure"], format=fmt, errors="coerce") - pd.Timedelta(days=1)
    df.loc[idx, "Actual_Departure"] = shifted.dt.strftime(fmt)

    # exact duplicate records
    dups = df.iloc[pick(0.015)]
    df = pd.concat([df, dups], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main(out: Path | None = None) -> Path:
    out = out or Path(__file__).resolve().parents[1] / "data" / "raw_flights.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = simulate()
    dirty = inject_dirt(clean)
    dirty.to_csv(out, index=False)
    print(f"Generated {len(clean):,} flights -> {len(dirty):,} raw rows written to {out}")
    return out


if __name__ == "__main__":
    main()
