"""Cleaning, validation and feature engineering for the flight-delay dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

DELAY_THRESHOLD = 15  # minutes; industry-standard on-time definition
MIN_TURN = {"A320": 35, "A321": 45, "B737": 40, "ATR72": 25}
ADVERSE_WEATHER = {"Fog", "Thunderstorm", "Rain"}
VALID_WEATHER = {"Clear", "Cloudy", "Haze", "Rain", "Thunderstorm", "Fog"}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TURN_BINS = [-np.inf, 0, 10, 20, 40, 240, np.inf]
TURN_LABELS = ["Below minimum", "0-10 min slack", "10-20 min slack", "20-40 min slack",
               "40+ min slack", "Overnight"]


def _parse_dt(s: pd.Series) -> pd.Series:
    """Parse ISO timestamps, falling back to DD/MM/YYYY for the rows that use it."""
    out = pd.to_datetime(s, format="%Y-%m-%d %H:%M", errors="coerce")
    alt = pd.to_datetime(s[out.isna()], format="%d/%m/%Y %H:%M", errors="coerce")
    out.loc[alt.index] = alt
    return out


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean dataframe, cleaning log)."""
    log: list[dict] = []
    df = raw.copy()

    def note(step, rows, action):
        log.append({"Step": step, "Rows affected": int(rows), "Action": action})

    note("Raw rows loaded", len(df), "-")

    n = df.duplicated().sum()
    df = df.drop_duplicates()
    note("Exact duplicate rows", n, "Dropped")

    n = df["Flight_ID"].duplicated().sum()
    df = df.drop_duplicates("Flight_ID", keep="first")
    note("Duplicate Flight_ID (non-identical)", n, "Kept first occurrence")

    # Text normalisation
    for col in ("Airport", "Route", "Aircraft", "Weather"):
        before = df[col].copy()
        df[col] = df[col].astype("string").str.strip()
        if col in ("Airport", "Route"):
            df[col] = df[col].str.upper().str.replace(r"\s*-\s*", "-", regex=True)
        elif col == "Weather":
            df[col] = df[col].str.title()
        changed = (before.astype("string") != df[col]).fillna(False).sum()
        note(f"Inconsistent text in {col}", changed, "Trimmed / case-normalised")

    # Route validity and Airport consistency
    parts = df["Route"].str.split("-", expand=True)
    df["Origin"], df["Destination"] = parts[0], parts[1]
    mismatch = df["Airport"] != df["Origin"]
    note("Airport does not match Route origin", mismatch.sum(), "Airport set to Route origin")
    df.loc[mismatch, "Airport"] = df.loc[mismatch, "Origin"]

    bad_wx = df["Weather"].notna() & ~df["Weather"].isin(VALID_WEATHER)
    df.loc[bad_wx, "Weather"] = pd.NA
    note("Unrecognised Weather label", bad_wx.sum(), "Set to missing")
    n = df["Weather"].isna().sum()
    df["Weather"] = df["Weather"].fillna("Unknown")
    note("Missing Weather", n, "Labelled 'Unknown' (kept as its own category)")

    # Timestamps
    df["Scheduled_Departure"] = _parse_dt(df["Scheduled_Departure"])
    df["Actual_Departure"] = _parse_dt(df["Actual_Departure"])
    n = df["Scheduled_Departure"].isna().sum()
    df = df[df["Scheduled_Departure"].notna()]
    note("Unparseable Scheduled_Departure", n, "Dropped")

    df["Delay_Min"] = (df["Actual_Departure"] - df["Scheduled_Departure"]).dt.total_seconds() / 60
    df["Cancelled_or_Missing"] = df["Actual_Departure"].isna()
    note("Missing Actual_Departure (cancelled / not recorded)", df["Cancelled_or_Missing"].sum(),
         "Flagged; excluded from delay metrics")

    invalid = df["Delay_Min"].notna() & ((df["Delay_Min"] < -60) | (df["Delay_Min"] > 24 * 60))
    fixable = invalid & ((df["Delay_Min"] + 1440).between(-60, 24 * 60))
    df.loc[fixable, "Actual_Departure"] += pd.Timedelta(days=1)
    df.loc[fixable, "Delay_Min"] += 1440
    note("Actual_Departure logged on wrong date (-1 day typo)", fixable.sum(), "Corrected by +1 day")
    still_bad = df["Delay_Min"].notna() & ((df["Delay_Min"] < -60) | (df["Delay_Min"] > 24 * 60))
    df.loc[still_bad, ["Actual_Departure", "Delay_Min"]] = np.nan
    note("Implausible delay (< -60 min or > 24 h)", still_bad.sum(), "Set to missing")

    # Aircraft & turnaround
    df["Aircraft_Type"] = df["Aircraft"].str.split("-").str[0]
    df["Min_Turn"] = df["Aircraft_Type"].map(MIN_TURN)
    neg = df["Turnaround_Time"] < 0
    df.loc[neg, "Turnaround_Time"] = df.loc[neg, "Turnaround_Time"].abs()
    note("Negative Turnaround_Time (sign error)", neg.sum(), "Absolute value taken")
    absurd = df["Turnaround_Time"] > 1440
    df.loc[absurd, "Turnaround_Time"] = np.nan
    note("Turnaround_Time > 24 h", absurd.sum(), "Set to missing")
    missing_turn = df["Turnaround_Time"].isna() & df["Previous_Flight"].notna()
    df["Turnaround_Imputed"] = missing_turn
    # A previous leg scheduled on an earlier date means this is an overnight turn
    prev_sched = df["Previous_Flight"].map(df.set_index("Flight_ID")["Scheduled_Departure"])
    overnight = prev_sched.dt.date != df["Scheduled_Departure"].dt.date
    same_day_med = df.groupby(["Aircraft_Type", "Airport"])["Turnaround_Time"].transform(
        lambda s: s[s < 240].median())
    overnight_med = df.loc[df["Turnaround_Time"] >= 240, "Turnaround_Time"].median()
    df.loc[missing_turn & ~overnight, "Turnaround_Time"] = same_day_med[missing_turn & ~overnight]
    df.loc[missing_turn & overnight, "Turnaround_Time"] = overnight_med
    note("Missing Turnaround_Time", missing_turn.sum(),
         "Imputed: aircraft-type x airport median (same-day) or overnight median")
    first_leg = df["Previous_Flight"].isna()
    note("First recorded flight of an aircraft (no previous flight)", first_leg.sum(),
         "Kept; treated as overnight start")

    # Derived features
    df["Is_Delayed"] = (df["Delay_Min"] > DELAY_THRESHOLD).astype("Int64")
    df.loc[df["Delay_Min"].isna(), "Is_Delayed"] = pd.NA
    df["Date"] = df["Scheduled_Departure"].dt.normalize()
    df["Dep_Hour"] = df["Scheduled_Departure"].dt.hour
    df["Day_of_Week"] = pd.Categorical(df["Scheduled_Departure"].dt.dayofweek.map(dict(enumerate(DAY_NAMES))),
                                       categories=DAY_NAMES, ordered=True)
    df["Month"] = df["Scheduled_Departure"].dt.month
    df["Month_Name"] = df["Scheduled_Departure"].dt.strftime("%b")

    prev_delay = df.set_index("Flight_ID")["Delay_Min"]
    df["Prev_Delay_Min"] = df["Previous_Flight"].map(prev_delay).clip(lower=0)
    df["Prev_Delay_Known"] = df["Prev_Delay_Min"].notna()
    df["Prev_Delay_Min"] = df["Prev_Delay_Min"].fillna(0)

    df["Turn_Slack"] = df["Turnaround_Time"] - df["Min_Turn"]
    df.loc[first_leg, "Turn_Slack"] = 600
    df["Turnaround_Time"] = df["Turnaround_Time"].fillna(600)
    df["Turn_Bucket"] = pd.cut(df["Turn_Slack"], TURN_BINS, labels=TURN_LABELS, right=False)
    df["Short_Turnaround"] = df["Turn_Slack"] < 0
    df["Adverse_Weather"] = df["Weather"].isin(ADVERSE_WEATHER)

    # Knock-on delay the previous leg pushes into this one after absorbing the ground slack
    df["Propagated_Min"] = (df["Prev_Delay_Min"] - df["Turn_Slack"].clip(lower=0)).clip(lower=0)

    df["Delay_Cause"] = attribute_cause(df)
    df["Cause_Group"] = np.where(df["Delay_Cause"] == "Weather", "Weather",
                                 np.where(df["Delay_Cause"] == "On time", "On time", "Operational"))
    df = df.sort_values("Scheduled_Departure").reset_index(drop=True)
    note("Clean rows", len(df), f"{df['Delay_Min'].notna().sum():,} with a valid delay value")
    return df, pd.DataFrame(log)


def attribute_cause(df: pd.DataFrame) -> pd.Series:
    """Rule-based primary cause for each delayed departure (>15 min).

    Priority: Weather (Fog/Thunderstorm/Rain at origin) -> Reactionary (previous
    leg's delay not absorbed by ground slack explains > half the delay) ->
    Short turnaround (ground time scheduled below type minimum) ->
    Airport / other operational (congestion, ATC, handling, crew, technical).
    """
    delayed = df["Delay_Min"] > DELAY_THRESHOLD
    reactionary = df["Propagated_Min"] >= 0.5 * df["Delay_Min"]
    conds = [
        ~delayed,
        df["Adverse_Weather"],
        reactionary,
        df["Short_Turnaround"],
    ]
    choices = ["On time", "Weather", "Reactionary (knock-on)", "Short turnaround"]
    out = pd.Series(np.select(conds, choices, default="Airport / other operational"), index=df.index)
    out[df["Delay_Min"].isna()] = "Unknown"
    return out
