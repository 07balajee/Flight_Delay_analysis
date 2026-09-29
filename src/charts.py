"""Plotly figure builders shared by the Streamlit dashboard and the notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import analysis as A
from .cleaning import DELAY_THRESHOLD

# Fixed-order categorical palette (validated for CVD on adjacent pairs) + one-hue sequential ramp
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8985"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAUSE_COLORS = {"Reactionary (knock-on)": BLUE, "Weather": ORANGE, "Short turnaround": AQUA,
                "Airport / other operational": YELLOW}
ADVERSE = ("Fog", "Thunderstorm", "Rain")
PCT = ".0%"
FEATURE_LABELS = {"Prev_Delay_Min": "Inbound (previous-leg) delay", "Turn_Slack": "Turnaround slack",
                  "Turnaround_Time": "Turnaround time", "Dep_Hour": "Departure hour",
                  "Weather": "Weather", "Airport": "Departure airport", "Destination": "Destination",
                  "Month": "Month", "Day_Num": "Day of week", "Aircraft_Type": "Aircraft type"}


def style(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(namelength=-1),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def _line(x, y, hover: str) -> go.Scatter:
    return go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=8),
                      hovertemplate=hover)


# ------------------------------------------------------------------ EDA
def delay_histogram(d: pd.DataFrame) -> go.Figure:
    v = A.valid(d)
    h = v[v["Delay_Min"].between(-15, 240)]
    fig = px.histogram(h, x="Delay_Min", nbins=85, color_discrete_sequence=[BLUE])
    fig.add_vline(x=DELAY_THRESHOLD, line_dash="dot", line_color=ORANGE,
                  annotation_text=f"{DELAY_THRESHOLD}-min threshold")
    fig.update_traces(marker_line_width=1, marker_line_color="rgba(255,255,255,0.8)")
    fig.update_xaxes(title="Departure delay (min, clipped to -15…240)")
    fig.update_yaxes(title="Flights")
    return style(fig, title="Distribution of departure delay (right-skewed)")


def monthly_rate(d: pd.DataFrame) -> go.Figure:
    mo = A.by_month(d)
    fig = go.Figure(_line(mo["Month_Name"], mo["Delay_Rate"], "%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=PCT, title="Delay rate", rangemode="tozero")
    return style(fig, title="Monthly delay rate")


# ------------------------------------------------------------------ airports, routes, tails
def airport_rates(d: pd.DataFrame) -> go.Figure:
    a = A.by_airport(d).sort_values("Delay_Rate")
    net = A.valid(d)["Is_Delayed"].mean()
    fig = go.Figure(go.Bar(
        x=a["Delay_Rate"], y=a["Airport"], orientation="h", marker_color=BLUE, marker_cornerradius=4,
        error_x=dict(type="data", symmetric=False, array=a["CI_High"] - a["Delay_Rate"],
                     arrayminus=a["Delay_Rate"] - a["CI_Low"], color=GRAY, thickness=1.2),
        text=a["Delay_Rate"].map(lambda x: f"{x:.0%}"), textposition="outside", cliponaxis=False,
        customdata=np.c_[a["Flights"], a["Avg_Delay"], a["P90_Delay"]],
        hovertemplate="<b>%{y}</b><br>Delay rate %{x:.1%}<br>Flights %{customdata[0]:,}"
                      "<br>Avg delay %{customdata[1]:.1f} min<br>P90 delay %{customdata[2]:.0f} min<extra></extra>"))
    fig.add_vline(x=net, line_dash="dot", line_color=GRAY,
                  annotation_text=f"Network {net:.0%}", annotation_position="top")
    fig.update_xaxes(tickformat=PCT, title="Share of departures delayed >15 min (95% CI)")
    return style(fig, 420, title="Delay rate by departure airport")


def top_routes(d: pd.DataFrame, rt: pd.DataFrame, n: int = 15) -> go.Figure:
    top = rt.head(n)
    fig = go.Figure(go.Bar(
        x=top["Route"], y=top["Delay_Rate"], marker_cornerradius=4,
        marker_color=[ORANGE if s else (BLUE if c else GRAY)
                      for s, c in zip(top["Route_Specific"], top["Consistently_Delayed"])],
        customdata=np.c_[top["Flights"], top["Months_Above_Network"], top["Lift_vs_Origin"]],
        hovertemplate="<b>%{x}</b><br>Delay rate %{y:.1%}<br>Flights %{customdata[0]:,}<br>"
                      "Months above network %{customdata[1]}/12<br>Lift vs origin %{customdata[2]:.2f}×"
                      "<extra></extra>"))
    fig.add_hline(y=A.valid(d)["Is_Delayed"].mean(), line_dash="dot", line_color=GRAY,
                  annotation_text="Network rate")
    fig.update_yaxes(tickformat=PCT, title="Delay rate")
    return style(fig, 380, title=f"{n} most-delayed routes — orange: route-specific, blue: consistently "
                                 "delayed (hub-driven), gray: not consistent")


def route_month_heatmap(d: pd.DataFrame, routes: list[str]) -> go.Figure:
    mm = A.route_month_matrix(d, routes)
    fig = go.Figure(go.Heatmap(z=mm.values, x=mm.columns, y=mm.index, colorscale=BLUES, xgap=2, ygap=2,
                               colorbar=dict(tickformat=PCT),
                               hovertemplate="%{y} · %{x}: %{z:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return style(fig, 420, title="Month-by-month delay rate of the worst routes (consistency check)")


def tail_rates(d: pd.DataFrame, flagged: set[str]) -> go.Figure:
    ta = A.by_aircraft(d)
    fig = go.Figure(go.Bar(x=ta["Aircraft"], y=ta["Delay_Rate"], marker_cornerradius=3,
                           marker_color=[ORANGE if a in flagged else BLUE for a in ta["Aircraft"]],
                           customdata=np.c_[ta["Flights"], ta["Z"]],
                           hovertemplate="<b>%{x}</b><br>Delay rate %{y:.1%}<br>Flights %{customdata[0]:,}"
                                         "<br>z-score %{customdata[1]:.1f}<extra></extra>"))
    fig.update_yaxes(tickformat=PCT, title="Delay rate")
    return style(fig, 340, title="Delay rate by aircraft tail (orange: ≥1.5× fleet rate and z > 4)")


# ------------------------------------------------------------------ hour & day
def hour_day_heatmap(d: pd.DataFrame, value: str = "rate") -> go.Figure:
    m = A.hour_day_matrix(d, value)
    is_rate = value == "rate"
    fig = go.Figure(go.Heatmap(
        z=m.values, x=[f"{h:02d}:00" for h in m.columns], y=list(m.index), colorscale=BLUES, xgap=2, ygap=2,
        colorbar=dict(title="Delay rate" if is_rate else "Avg min", tickformat=PCT if is_rate else ""),
        hovertemplate="%{y} %{x}<br>" + ("Delay rate %{z:.1%}" if is_rate else "Avg delay %{z:.1f} min")
                      + "<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return style(fig, 380, title="Delay rate by scheduled hour and day of week" if is_rate
                 else "Average delay (min) by scheduled hour and day of week")


def hour_rates(d: pd.DataFrame) -> go.Figure:
    bh = A.by_hour(d)
    fig = go.Figure(go.Bar(x=bh["Dep_Hour"], y=bh["Delay_Rate"], marker_color=BLUE, marker_cornerradius=4,
                           customdata=bh["Flights"],
                           hovertemplate="%{x}:00 · %{y:.1%} delayed · %{customdata:,} flights<extra></extra>"))
    fig.update_yaxes(tickformat=PCT, title="Delay rate")
    fig.update_xaxes(title="Scheduled departure hour", dtick=1)
    return style(fig, title="Delay rate by hour")


def day_rates(d: pd.DataFrame) -> go.Figure:
    bd = A.by_day(d)
    worst = bd["Delay_Rate"].nlargest(2).min()
    fig = go.Figure(go.Bar(x=bd["Day_of_Week"].astype(str), y=bd["Delay_Rate"], marker_cornerradius=4,
                           marker_color=[ORANGE if r >= worst else BLUE for r in bd["Delay_Rate"]],
                           text=bd["Delay_Rate"].map(lambda x: f"{x:.0%}"), textposition="outside",
                           cliponaxis=False, hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=PCT, title="Delay rate")
    return style(fig, title="Delay rate by day of week (orange: two worst days)")


# ------------------------------------------------------------------ turnaround & knock-on
def turnaround_rates(d: pd.DataFrame) -> go.Figure:
    t = A.turnaround_effect(d)
    t = t[t["Flights"] > 0]
    fig = go.Figure(go.Bar(
        x=t["Turn_Bucket"].astype(str), y=t["Delay_Rate"], marker_cornerradius=4,
        marker_color=[ORANGE if b == "Below minimum" else BLUE for b in t["Turn_Bucket"]],
        text=t["Delay_Rate"].map(lambda x: f"{x:.0%}"), textposition="outside", cliponaxis=False,
        customdata=np.c_[t["Flights"], t["Avg_Delay"]],
        hovertemplate="<b>%{x}</b><br>Delay rate %{y:.1%}<br>Flights %{customdata[0]:,}"
                      "<br>Avg delay %{customdata[1]:.1f} min<extra></extra>"))
    fig.update_yaxes(tickformat=PCT, title="Delay rate")
    fig.update_xaxes(title="Scheduled ground time vs aircraft-type minimum")
    return style(fig, 400, title="Delay rate by turnaround slack (orange = scheduled below minimum)")


def inbound_x_slack_heatmap(d: pd.DataFrame) -> go.Figure:
    tx = A.turnaround_x_prev_delay(d)
    fig = go.Figure(go.Heatmap(z=tx.values, x=[str(c) for c in tx.columns], y=[str(i) for i in tx.index],
                               colorscale=BLUES, xgap=2, ygap=2, colorbar=dict(tickformat=PCT),
                               texttemplate="%{z:.0%}",
                               hovertemplate="%{y} · %{x}<br>Delay rate %{z:.1%}<extra></extra>"))
    fig.update_yaxes(showgrid=False)
    return style(fig, 400, title="Late inbound × turnaround slack → delay rate")


def propagation_scatter(d: pd.DataFrame, n: int = 6000) -> go.Figure:
    v = A.valid(d)
    same = v[v["Turn_Bucket"] != "Overnight"]
    same = same.sample(min(n, len(same)), random_state=1)
    fig = px.scatter(same, x="Prev_Delay_Min", y="Delay_Min", opacity=0.25, color_discrete_sequence=[BLUE],
                     labels={"Prev_Delay_Min": "Inbound (previous-leg) delay, min",
                             "Delay_Min": "This departure's delay, min"})
    fig.update_traces(marker=dict(size=5))
    fig.add_trace(go.Scatter(x=[0, 240], y=[0, 240], mode="lines", line=dict(color=GRAY, dash="dot"),
                             name="y = x (delay fully passed on)", hoverinfo="skip"))
    fig.update_xaxes(range=[0, 240])
    fig.update_yaxes(range=[-15, 260])
    return style(fig, 420, title="Delay propagation: inbound vs outbound delay (same-day turns, sample)")


def leg_accumulation(d: pd.DataFrame) -> go.Figure:
    v = A.valid(d)
    agg = v.groupby(v.groupby(["Aircraft", "Date"]).cumcount() + 1).agg(
        Flights=("Flight_ID", "size"), Delay_Rate=("Is_Delayed", "mean"))
    agg = agg[agg["Flights"] > 200]
    fig = go.Figure(_line(agg.index, agg["Delay_Rate"], "Leg %{x} of the day: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=PCT, title="Delay rate", rangemode="tozero")
    fig.update_xaxes(title="Leg number of the aircraft's day", dtick=1)
    return style(fig, 320, title="Delay accumulates with each successive leg of an aircraft's day")


# ------------------------------------------------------------------ weather vs operational
def cause_hours(d: pd.DataFrame) -> go.Figure:
    c = A.cause_breakdown(d).sort_values("Delay_Minutes")
    fig = go.Figure()
    for r in c.itertuples():
        fig.add_bar(x=[r.Delay_Minutes / 60], y=[r.Cause_Group], name=r.Delay_Cause, orientation="h",
                    marker_color=CAUSE_COLORS[r.Delay_Cause], marker_line_width=2,
                    marker_line_color="rgba(255,255,255,0.9)",
                    text=f"{r.Share_Minutes:.0%}", textposition="inside",
                    hovertemplate=f"<b>{r.Delay_Cause}</b><br>{r.Flights:,} delayed flights<br>"
                                  f"{r.Delay_Minutes / 60:,.0f} delay hours ({r.Share_Minutes:.1%})<br>"
                                  f"Avg {r.Avg_Delay:.0f} min<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Delay hours (departures delayed >15 min)")
    return style(fig, 320, title="Delay hours by primary cause: weather vs operational")


def weather_delay(d: pd.DataFrame) -> go.Figure:
    w = A.by_weather(d).sort_values("Avg_Delay")
    fig = go.Figure(go.Bar(x=w["Avg_Delay"], y=w["Weather"], orientation="h", marker_cornerradius=4,
                           marker_color=[ORANGE if x in ADVERSE else BLUE for x in w["Weather"]],
                           customdata=np.c_[w["Flights"], w["Delay_Rate"], w["Excess_vs_Clear"]],
                           hovertemplate="<b>%{y}</b><br>Avg delay %{x:.1f} min<br>Delay rate %{customdata[1]:.1%}"
                                         "<br>+%{customdata[2]:.1f} min vs Clear<br>Flights %{customdata[0]:,}"
                                         "<extra></extra>"))
    fig.update_xaxes(title="Average departure delay (min)")
    return style(fig, title="Average delay by weather (orange: adverse)")


def monthly_cause_split(d: pd.DataFrame) -> go.Figure:
    mo = A.by_month(d)
    fig = go.Figure()
    fig.add_bar(x=mo["Month_Name"], y=mo["Operational_Delays"], name="Operational", marker_color=BLUE,
                hovertemplate="%{x}: %{y:,} operational delays<extra></extra>")
    fig.add_bar(x=mo["Month_Name"], y=mo["Weather_Delays"], name="Weather", marker_color=ORANGE,
                hovertemplate="%{x}: %{y:,} weather delays<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_traces(marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)")
    fig.update_yaxes(title="Delayed departures")
    return style(fig, title="Weather vs operational delays by month")


# ------------------------------------------------------------------ model
def confusion(metrics: dict) -> go.Figure:
    cm = np.array(metrics["confusion_matrix"])
    labels = ["On time", "Delayed"]
    fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=BLUES, showscale=False, xgap=2, ygap=2,
                               texttemplate="%{z:,}",
                               hovertemplate="Actual %{y} · Predicted %{x}: %{z:,}<extra></extra>"))
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual", autorange="reversed", showgrid=False)
    return style(fig, 340, title="Confusion matrix (test set)")


def roc(metrics: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(x=metrics["roc"]["fpr"], y=metrics["roc"]["tpr"], mode="lines",
                               line=dict(color=BLUE, width=2), name="Random Forest",
                               hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRAY, dash="dot"),
                             name="Chance", hoverinfo="skip"))
    fig.update_xaxes(title="False positive rate")
    fig.update_yaxes(title="True positive rate")
    return style(fig, 340, title=f"ROC curve (AUC {metrics['roc_auc']:.3f})")


def precision_recall(metrics: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(x=metrics["pr"]["recall"], y=metrics["pr"]["precision"], mode="lines",
                               line=dict(color=BLUE, width=2), name="Random Forest",
                               hovertemplate="Recall %{x:.2f} · Precision %{y:.2f}<extra></extra>"))
    fig.add_hline(y=metrics["positive_rate"], line_dash="dot", line_color=GRAY, annotation_text="Base rate")
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return style(fig, 340, title=f"Precision-recall (AP {metrics['pr_auc']:.3f})")


def permutation_importance(metrics: dict) -> go.Figure:
    pi = pd.DataFrame({"f": list(metrics["perm_importance"]), "v": list(metrics["perm_importance"].values()),
                       "s": list(metrics["perm_importance_std"].values())}).sort_values("v")
    fig = go.Figure(go.Bar(x=pi["v"], y=pi["f"].map(FEATURE_LABELS), orientation="h", marker_color=BLUE,
                           marker_cornerradius=4,
                           error_x=dict(type="data", array=pi["s"], color=GRAY, thickness=1.2),
                           hovertemplate="%{y}: ROC-AUC drop %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="Mean drop in test ROC-AUC when feature is shuffled")
    return style(fig, 400, title="Permutation importance (test set)")


def impurity_importance(metrics: dict) -> go.Figure:
    ii = pd.Series(metrics["importance"]).sort_values()
    fig = go.Figure(go.Bar(x=ii.values, y=ii.index.map(FEATURE_LABELS), orientation="h", marker_color=BLUE,
                           marker_cornerradius=4, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Mean decrease in impurity (Gini)")
    return style(fig, 400, title="Impurity-based importance")


def buffer_curve(scenarios: pd.DataFrame) -> go.Figure:
    fig = go.Figure(_line(scenarios["Min_Slack"], scenarios["Reduction"],
                          "Buffer %{x} min → %{y:.1%} fewer delayed departures<extra></extra>"))
    fig.update_yaxes(tickformat=PCT, title="Predicted reduction in delayed departures", rangemode="tozero")
    fig.update_xaxes(title="Minimum slack enforced above the type minimum (min)", dtick=5)
    return style(fig, 340, title="What-if: enforcing a turnaround buffer (Random Forest estimate)")
