"""Descriptive analysis: delay rates by airport, time, turnaround, route, cause."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def valid(df: pd.DataFrame) -> pd.DataFrame:
    """Flights with a usable delay value (cancelled / invalid rows excluded)."""
    return df[df["Delay_Min"].notna()].copy()


def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def kpis(df: pd.DataFrame) -> dict:
    v = valid(df)
    delayed = v[v["Is_Delayed"] == 1]
    return {
        "flights": len(df),
        "valid": len(v),
        "delay_rate": v["Is_Delayed"].mean(),
        "avg_delay": v["Delay_Min"].clip(lower=0).mean(),
        "median_delay_delayed": delayed["Delay_Min"].median(),
        "delay_minutes": delayed["Delay_Min"].sum(),
        "otp": 1 - v["Is_Delayed"].mean(),
    }


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    v = valid(df)
    cols = ["Delay_Min", "Turnaround_Time", "Turn_Slack", "Prev_Delay_Min", "Dep_Hour"]
    d = v[cols].describe(percentiles=[.25, .5, .75, .9, .95]).T
    d["skew"] = v[cols].skew()
    return d.round(2)


def by_airport(df: pd.DataFrame) -> pd.DataFrame:
    v = valid(df)
    g = v.groupby("Airport").agg(
        Flights=("Flight_ID", "size"),
        Delayed=("Is_Delayed", "sum"),
        Avg_Delay=("Delay_Min", lambda s: s.clip(lower=0).mean()),
        P90_Delay=("Delay_Min", lambda s: s.quantile(.9)),
        Weather_Share=("Cause_Group", lambda s: (s == "Weather").sum()),
    )
    g["Delay_Rate"] = g["Delayed"] / g["Flights"]
    g["CI_Low"], g["CI_High"] = wilson_ci(g["Delayed"].to_numpy(float), g["Flights"].to_numpy(float))
    g["Weather_Share"] = g["Weather_Share"] / g["Delayed"]
    return g.sort_values("Delay_Rate", ascending=False).reset_index()


def hour_day_matrix(df: pd.DataFrame, value: str = "rate") -> pd.DataFrame:
    v = valid(df)
    col = "Is_Delayed" if value == "rate" else "Delay_Min"
    m = v.pivot_table(index="Day_of_Week", columns="Dep_Hour", values=col, aggfunc="mean", observed=False)
    return m.astype(float)


def by_hour(df: pd.DataFrame) -> pd.DataFrame:
    v = valid(df)
    return v.groupby("Dep_Hour").agg(Flights=("Flight_ID", "size"), Delay_Rate=("Is_Delayed", "mean"),
                                     Avg_Delay=("Delay_Min", "mean")).reset_index()


def by_day(df: pd.DataFrame) -> pd.DataFrame:
    v = valid(df)
    return v.groupby("Day_of_Week", observed=False).agg(
        Flights=("Flight_ID", "size"), Delay_Rate=("Is_Delayed", "mean"),
        Avg_Delay=("Delay_Min", "mean")).reset_index()


def by_month(df: pd.DataFrame) -> pd.DataFrame:
    v = valid(df)
    g = v.groupby(["Month", "Month_Name"]).agg(
        Flights=("Flight_ID", "size"), Delay_Rate=("Is_Delayed", "mean"),
        Weather_Delays=("Cause_Group", lambda s: (s == "Weather").sum()),
        Operational_Delays=("Cause_Group", lambda s: (s == "Operational").sum()),
    ).reset_index().sort_values("Month")
    return g


def turnaround_effect(df: pd.DataFrame) -> pd.DataFrame:
    v = valid(df)
    g = v.groupby("Turn_Bucket", observed=False).agg(
        Flights=("Flight_ID", "size"), Delay_Rate=("Is_Delayed", "mean"),
        Avg_Delay=("Delay_Min", "mean"), Avg_Prev_Delay=("Prev_Delay_Min", "mean")).reset_index()
    return g


def turnaround_x_prev_delay(df: pd.DataFrame) -> pd.DataFrame:
    v = valid(df)
    v = v[v["Turn_Bucket"] != "Overnight"]
    v["Inbound"] = pd.cut(v["Prev_Delay_Min"], [-1, 15, 30, 60, np.inf],
                          labels=["Inbound on time", "Inbound 15-30 late", "Inbound 30-60 late",
                                  "Inbound 60+ late"])
    return v.pivot_table(index="Inbound", columns="Turn_Bucket", values="Is_Delayed",
                         aggfunc="mean", observed=True).astype(float)


def turnaround_stats(df: pd.DataFrame) -> dict:
    v = valid(df)
    same_day = v[v["Turn_Bucket"] != "Overnight"]
    rho, p = stats.spearmanr(same_day["Turn_Slack"], same_day["Delay_Min"])
    short = same_day[same_day["Short_Turnaround"]]["Is_Delayed"]
    ok = same_day[~same_day["Short_Turnaround"]]["Is_Delayed"]
    table = np.array([[short.sum(), len(short) - short.sum()], [ok.sum(), len(ok) - ok.sum()]], dtype=float)
    chi2, p_chi, *_ = stats.chi2_contingency(table)
    return {"spearman_rho": rho, "spearman_p": p, "short_rate": short.mean(), "ok_rate": ok.mean(),
            "short_share": len(short) / len(same_day), "chi2": chi2, "chi2_p": p_chi,
            "prop_corr": v[["Prev_Delay_Min", "Delay_Min"]].corr().iloc[0, 1]}


def route_consistency(df: pd.DataFrame, min_flights: int = 150) -> pd.DataFrame:
    """A route is 'consistently delayed' if it beats the network delay rate in
    at least 10 of 12 months AND is significantly worse overall (binomial test)."""
    v = valid(df)
    network = v["Is_Delayed"].mean()
    monthly_net = v.groupby("Month")["Is_Delayed"].mean()
    rm = v.groupby(["Route", "Month"])["Is_Delayed"].mean().unstack()
    months_above = (rm.gt(monthly_net, axis=1)).sum(axis=1)
    g = v.groupby("Route").agg(Flights=("Flight_ID", "size"), Delayed=("Is_Delayed", "sum"),
                               Avg_Delay=("Delay_Min", "mean"))
    g["Delay_Rate"] = g["Delayed"] / g["Flights"]
    g["Months_Above_Network"] = months_above
    g["Monthly_Std"] = rm.std(axis=1)
    g["p_value"] = [stats.binomtest(int(k), int(n), network, alternative="greater").pvalue
                    for k, n in zip(g["Delayed"], g["Flights"])]
    g = g[g["Flights"] >= min_flights]
    g["Consistently_Delayed"] = (g["Months_Above_Network"] >= 10) & (g["p_value"] < 0.01)
    g["Lift_vs_Network"] = g["Delay_Rate"] / network
    # Lift vs the origin airport separates route-specific problems from airport-wide congestion
    origin_rate = v.groupby("Airport")["Is_Delayed"].mean()
    g["Origin_Rate"] = g.index.str.split("-").str[0].map(origin_rate)
    g["Lift_vs_Origin"] = g["Delay_Rate"] / g["Origin_Rate"]
    g["Route_Specific"] = g["Consistently_Delayed"] & (g["Lift_vs_Origin"] >= 1.15)
    return g.sort_values("Delay_Rate", ascending=False).reset_index()


def route_month_matrix(df: pd.DataFrame, routes: list[str]) -> pd.DataFrame:
    v = valid(df)
    v = v[v["Route"].isin(routes)]
    m = v.pivot_table(index="Route", columns="Month_Name", values="Is_Delayed", aggfunc="mean")
    order = [c for c in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
             if c in m.columns]
    return m[order].reindex(routes)


def cause_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    v = valid(df)
    d = v[v["Is_Delayed"] == 1]
    g = d.groupby(["Cause_Group", "Delay_Cause"]).agg(Flights=("Flight_ID", "size"),
                                                      Delay_Minutes=("Delay_Min", "sum"),
                                                      Avg_Delay=("Delay_Min", "mean")).reset_index()
    g["Share_Flights"] = g["Flights"] / g["Flights"].sum()
    g["Share_Minutes"] = g["Delay_Minutes"] / g["Delay_Minutes"].sum()
    return g.sort_values("Delay_Minutes", ascending=False)


def by_weather(df: pd.DataFrame) -> pd.DataFrame:
    v = valid(df)
    g = v.groupby("Weather").agg(Flights=("Flight_ID", "size"), Delay_Rate=("Is_Delayed", "mean"),
                                 Avg_Delay=("Delay_Min", "mean"),
                                 P90_Delay=("Delay_Min", lambda s: s.quantile(.9))).reset_index()
    clear = g.loc[g["Weather"] == "Clear", "Avg_Delay"].iloc[0] if (g["Weather"] == "Clear").any() else 0
    g["Excess_vs_Clear"] = g["Avg_Delay"] - clear
    return g.sort_values("Avg_Delay", ascending=False)


def weather_test(df: pd.DataFrame) -> dict:
    v = valid(df)
    adv = v[v["Adverse_Weather"]]["Delay_Min"]
    fair = v[v["Weather"].isin(["Clear", "Cloudy"])]["Delay_Min"]
    u, p = stats.mannwhitneyu(adv, fair, alternative="greater")
    return {"adverse_median": adv.median(), "fair_median": fair.median(), "adverse_mean": adv.mean(),
            "fair_mean": fair.mean(), "mw_p": p, "adverse_share": len(adv) / len(v)}


def by_aircraft(df: pd.DataFrame) -> pd.DataFrame:
    v = valid(df)
    g = v.groupby(["Aircraft", "Aircraft_Type"]).agg(
        Flights=("Flight_ID", "size"), Delay_Rate=("Is_Delayed", "mean"),
        Own_Delay=("Delay_Cause", lambda s: (s == "Airport / other operational").mean()),
        Avg_Delay=("Delay_Min", "mean")).reset_index()
    fleet = v["Is_Delayed"].mean()
    g["Z"] = (g["Delay_Rate"] - fleet) / np.sqrt(fleet * (1 - fleet) / g["Flights"])
    return g.sort_values("Delay_Rate", ascending=False)
