"""Key insights and operational action plan, computed from the full-year data."""
from __future__ import annotations

import pandas as pd

from . import analysis as A
from .charts import FEATURE_LABELS


def facts(df: pd.DataFrame, metrics: dict) -> dict:
    K = A.kpis(df)
    air = A.by_airport(df)
    days = A.by_day(df)
    routes = A.route_consistency(df)
    causes = A.cause_breakdown(df)
    tails = A.by_aircraft(df)
    group_minutes = causes.groupby("Cause_Group")["Delay_Minutes"].sum()
    top2 = air.head(2)
    v = A.valid(df)
    return {
        "K": K, "air": air, "top2": top2,
        "top2_share": top2["Delayed"].sum() / air["Delayed"].sum(),
        "top2_flight_share": top2["Flights"].sum() / air["Flights"].sum(),
        "eve": v.loc[v["Dep_Hour"].between(17, 22), "Is_Delayed"].mean(),
        "morn": v.loc[v["Dep_Hour"].between(5, 7), "Is_Delayed"].mean(),
        "worst_days": days.sort_values("Delay_Rate", ascending=False).head(2),
        "best_day": days.sort_values("Delay_Rate").iloc[0],
        "tstats": A.turnaround_stats(df),
        "turn_ix": A.turnaround_effect(df).set_index("Turn_Bucket"),
        "cause_min": causes.set_index("Delay_Cause")["Share_Minutes"],
        "weather_min_share": group_minutes.get("Weather", 0) / group_minutes.sum(),
        "consistent": routes[routes["Consistently_Delayed"]],
        "route_specific": routes[routes["Route_Specific"]],
        "wx_ix": A.by_weather(df).set_index("Weather"),
        "problem_tails": tails[(tails["Z"] > 4) & (tails["Delay_Rate"] >= 1.5 * K["delay_rate"])],
        "top_feature": FEATURE_LABELS[max(metrics["perm_importance"], key=metrics["perm_importance"].get)],
        "roc_auc": metrics["roc_auc"],
    }


def insights(f: dict) -> list[tuple[str, str]]:
    K, top2, air, wx, tst, wd, turn = f["K"], f["top2"], f["air"], f["wx_ix"], f["tstats"], f["worst_days"], f["turn_ix"]
    pt = f["problem_tails"]
    return [
        ("Delays concentrate at two hubs",
         f"Network on-time performance is **{K['otp']:.1%}** ({K['delay_rate']:.1%} of departures delayed >15 min). "
         f"**{top2.iloc[0]['Airport']}** ({top2.iloc[0]['Delay_Rate']:.0%}) and **{top2.iloc[1]['Airport']}** "
         f"({top2.iloc[1]['Delay_Rate']:.0%}) run the worst delay rates and together generate "
         f"**{f['top2_share']:.0%}** of all delayed departures from {f['top2_flight_share']:.0%} of flights; "
         f"{air.iloc[-1]['Airport']} is best at {air.iloc[-1]['Delay_Rate']:.0%}."),
        ("Delays build up through the day",
         f"Departures between 17:00 and 22:59 are delayed **{f['eve']:.0%}** of the time vs **{f['morn']:.0%}** for "
         f"the first wave (05:00–07:59) — a **{f['eve'] / f['morn']:.1f}×** gap. {wd.iloc[0]['Day_of_Week']} "
         f"({wd.iloc[0]['Delay_Rate']:.0%}) and {wd.iloc[1]['Day_of_Week']} ({wd.iloc[1]['Delay_Rate']:.0%}) are "
         f"the worst days; {f['best_day']['Day_of_Week']} is best ({f['best_day']['Delay_Rate']:.0%})."),
        ("Tight turnarounds are the single most controllable cause",
         f"**{tst['short_share']:.0%}** of same-day turns are scheduled *below* the aircraft type's minimum ground "
         f"time; these depart late **{tst['short_rate']:.0%}** of the time vs **{tst['ok_rate']:.0%}** for the rest "
         f"(χ² p < 0.001). Delay rate falls steadily with slack — from "
         f"{turn.loc['0-10 min slack', 'Delay_Rate']:.0%} (0–10 min) to "
         f"{turn.loc['40+ min slack', 'Delay_Rate']:.0%} (40+ min)."),
        ("Knock-on delay is the largest operational contributor",
         f"Reactionary delay — a late inbound aircraft that the ground time cannot absorb — explains "
         f"**{f['cause_min'].get('Reactionary (knock-on)', 0):.0%}** of delay minutes. Inbound and outbound delays "
         f"correlate at r = {tst['prop_corr']:.2f}, and the Random Forest ranks *{f['top_feature']}* as its most "
         f"important predictor."),
        ("A few routes are late every single month",
         f"**{len(f['consistent'])}** routes beat the network delay rate in ≥10 of 12 months (binomial p < 0.01). "
         f"**{len(f['route_specific'])}** are worse than their own departure airport by ≥15% — a route-specific "
         f"problem, not just hub congestion: "
         + ", ".join(f"{r.Route} ({r.Delay_Rate:.0%})" for r in f["route_specific"].head(6).itertuples()) + "."),
        ("Weather matters, but operations matter more",
         f"Weather-attributed delays are **{f['weather_min_share']:.0%}** of delay minutes; "
         f"**{1 - f['weather_min_share']:.0%}** are operational. Fog and thunderstorms add "
         f"**+{wx.loc['Fog', 'Excess_vs_Clear']:.0f}** and **+{wx.loc['Thunderstorm', 'Excess_vs_Clear']:.0f} min** "
         f"average delay vs clear skies, but affect only "
         f"{(wx.loc['Fog', 'Flights'] + wx.loc['Thunderstorm', 'Flights']) / K['valid']:.1%} of departures — "
         f"concentrated in the monsoon (Jun–Sep) and winter fog (Dec–Jan) months."),
        (f"{len(pt)} aircraft are chronic under-performers",
         (f"Tails **{', '.join(pt['Aircraft'])}** run delay rates of "
          f"{', '.join(f'{x:.0%}' for x in pt['Delay_Rate'])} vs a fleet average of {K['delay_rate']:.0%} "
          f"(≥1.5× fleet, z > 4) — a reliability / maintenance signal rather than a scheduling one.")
         if len(pt) else "No individual aircraft deviates significantly from the fleet average."),
    ]


def action_plan(f: dict, buffer10: dict) -> pd.DataFrame:
    tst, wx, wd, top2 = f["tstats"], f["wx_ix"], f["worst_days"], f["top2"]
    return pd.DataFrame([
        {"#": 1, "Action": "Re-time schedules so no turnaround is below the aircraft-type minimum; add a 10-min "
                           "buffer on hub turns after 15:00",
         "Evidence": f"{tst['short_share']:.0%} of turns below minimum → {tst['short_rate']:.0%} delayed vs "
                     f"{tst['ok_rate']:.0%}",
         "KPI / target": f"Below-minimum turns → 0%; OTP +{buffer10['avoided'] / f['K']['valid']:.1%} (model estimate)",
         "Owner": "Network planning", "Horizon": "Next schedule season"},
        {"#": 2, "Action": "Contain knock-on delay: hold spare aircraft/crew at BOM and DEL in the evening bank; "
                           "swap tails when inbound delay exceeds available slack",
         "Evidence": f"Reactionary = {f['cause_min'].get('Reactionary (knock-on)', 0):.0%} of delay minutes; "
                     f"'{f['top_feature']}' is the top model feature",
         "KPI / target": "Reactionary delay minutes −25%", "Owner": "Ops control centre (OCC)",
         "Horizon": "0–3 months"},
        {"#": 3, "Action": f"Hub recovery plan at {top2.iloc[0]['Airport']} and {top2.iloc[1]['Airport']}: dedicated "
                           "turn coordinators, priority ground-handling for departures, slot review with ATC",
         "Evidence": f"{f['top2_share']:.0%} of delayed departures from {f['top2_flight_share']:.0%} of flights",
         "KPI / target": "Hub delay rate to network average", "Owner": "Airport ops / ground handling",
         "Horizon": "0–6 months"},
        {"#": 4, "Action": "Fix the route-specific problem routes: add block/turn time, move to less congested slots, "
                           "or reassign reliable aircraft: " + ", ".join(f["route_specific"]["Route"].head(6)),
         "Evidence": f"{len(f['route_specific'])} routes worse than network in ≥10/12 months and ≥15% worse than "
                     "their origin airport",
         "KPI / target": "Each route within 1.15× origin delay rate", "Owner": "Network planning",
         "Horizon": "Next schedule season"},
        {"#": 5, "Action": "Protect the evening bank (17:00–22:59) and Fri/Sun peaks: spread departures, add "
                           "standby staff, avoid scheduling tight turns late in the aircraft's day",
         "Evidence": f"Evening delay rate {f['eve']:.0%} vs {f['morn']:.0%} first wave; "
                     f"{wd.iloc[0]['Day_of_Week']}/{wd.iloc[1]['Day_of_Week']} worst days",
         "KPI / target": "Evening delay rate −10 pts", "Owner": "Network planning / HR rostering",
         "Horizon": "3–6 months"},
        {"#": 6, "Action": "Seasonal weather playbooks: pre-emptive re-timing and CAT-III-capable crew/aircraft for "
                           "north-India fog (Dec–Jan); monsoon buffers at BOM/GOI/COK (Jun–Sep)",
         "Evidence": f"Weather = {f['weather_min_share']:.0%} of delay minutes; fog "
                     f"+{wx.loc['Fog', 'Excess_vs_Clear']:.0f} min, thunderstorm "
                     f"+{wx.loc['Thunderstorm', 'Excess_vs_Clear']:.0f} min vs clear",
         "KPI / target": "Weather delay minutes −15% in season", "Owner": "OCC / flight ops",
         "Horizon": "Before next season"},
        {"#": 7, "Action": "Reliability review of chronic tails " + ", ".join(f["problem_tails"]["Aircraft"]) +
                           " (deferred defects, maintenance findings); deploy the RF risk score daily so the OCC "
                           "sees high-risk departures before they happen",
         "Evidence": f"Tail delay rate z > 4 vs fleet; RF ROC-AUC {f['roc_auc']:.2f}",
         "KPI / target": "Tail delay rates within fleet ±5 pts", "Owner": "Engineering & maintenance / OCC",
         "Horizon": "0–3 months"},
    ])
