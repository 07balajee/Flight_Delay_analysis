"""Streamlit dashboard: Aviation — why are flights delayed?

Run:  streamlit run app.py
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analysis as A
from src import charts as C
from src import report as R
from src.cleaning import DAY_NAMES, DELAY_THRESHOLD, MIN_TURN
from src.model import CAT_FEATURES, FEATURES
from src.model import buffer_scenario as _buffer_scenario
from src.pipeline import CLEAN, LOG, MODELS, RAW, run

st.set_page_config(page_title="Why Are Flights Delayed?", page_icon="✈️", layout="wide")

_chart_ids = iter(range(10_000))


def plot(where, fig: go.Figure) -> None:
    """plotly_chart with a unique key, so the same figure can appear in several tabs."""
    where.plotly_chart(fig, key=f"chart_{next(_chart_ids)}", width="stretch")


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner="Generating data, cleaning and training the Random Forest (first run ~1 min)...")
def ensure_artifacts() -> bool:
    if not (CLEAN.exists() and (MODELS / "rf_model.joblib").exists() and (MODELS / "metrics.json").exists()):
        run(regenerate=not RAW.exists())
    return True


@st.cache_data
def load_data():
    df = pd.read_parquet(CLEAN)
    log = pd.read_csv(LOG)
    raw = pd.read_csv(RAW, nrows=200)
    metrics = json.loads((MODELS / "metrics.json").read_text())
    return df, log, raw, metrics


@st.cache_resource
def load_model():
    return joblib.load(MODELS / "rf_model.joblib")


@st.cache_data(show_spinner="Scoring scenarios with the Random Forest...")
def buffer_scenario(min_slack: int) -> dict:
    return _buffer_scenario(load_model(), load_data()[0], min_slack)


ensure_artifacts()
df_all, clean_log, raw_sample, metrics = load_data()
model = load_model()

# ------------------------------------------------------------------ sidebar filters
st.sidebar.title("✈️ Filters")
st.sidebar.caption("Filters apply to the deep-dive tabs. Executive summary, model and action plan "
                   "always use the full year.")
airports = sorted(df_all["Airport"].unique())
sel_airports = st.sidebar.multiselect("Departure airport", airports, default=airports)
months = st.sidebar.slider("Month", 1, 12, (1, 12))
types = sorted(df_all["Aircraft_Type"].unique())
sel_types = st.sidebar.multiselect("Aircraft type", types, default=types)
weathers = sorted(df_all["Weather"].unique())
sel_weather = st.sidebar.multiselect("Weather", weathers, default=weathers)
df = df_all[df_all["Airport"].isin(sel_airports) & df_all["Month"].between(*months)
            & df_all["Aircraft_Type"].isin(sel_types) & df_all["Weather"].isin(sel_weather)]
st.sidebar.metric("Flights in selection", f"{len(df):,}")
st.sidebar.markdown(f"**Delayed** = departed more than **{DELAY_THRESHOLD} min** after schedule.")

if A.valid(df).empty:
    st.warning("No flights match the current filters.")
    st.stop()

F = R.facts(df_all, metrics)
K = F["K"]
flagged_tails = set(F["problem_tails"]["Aircraft"])

# ------------------------------------------------------------------ layout
st.title("✈️ Why are flights delayed?")
st.caption("Aviation case study · synthetic one-year flight-level dataset for a domestic airline · "
           "Random Forest delay model · operational recommendations")

tabs = st.tabs(["📌 Executive summary", "🧹 Data quality & EDA", "🛫 Airports & routes", "🕒 Hour & day",
                "🔁 Turnaround & knock-on", "🌦️ Weather vs operational", "🌲 Random Forest model",
                "🛠️ Action plan"])

# ---- Executive summary
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Flights analysed", f"{K['valid']:,}")
    c[1].metric("On-time performance", f"{K['otp']:.1%}")
    c[2].metric("Avg departure delay", f"{K['avg_delay']:.1f} min")
    c[3].metric("Delay hours (>15 min)", f"{K['delay_minutes'] / 60:,.0f}")
    c[4].metric("RF model ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.subheader("Key insights")
    for i, (title, body) in enumerate(R.insights(F), 1):
        st.markdown(f"**{i}. {title}.** {body}")

    st.subheader("Four key visualisations")
    l, r = st.columns(2)
    plot(l, C.airport_rates(df_all))
    plot(r, C.hour_day_heatmap(df_all))
    l, r = st.columns(2)
    plot(l, C.turnaround_rates(df_all))
    plot(r, C.cause_hours(df_all))

# ---- Data quality & EDA
with tabs[1]:
    st.subheader("Dataset")
    st.markdown(
        "The brief supplies only the schema, so a **realistic synthetic dataset** was generated "
        f"(`src/generate_data.py`): {df_all['Aircraft'].nunique()} aircraft flying "
        f"{df_all['Route'].nunique()} routes between {df_all['Airport'].nunique()} Indian airports for one year "
        f"({len(df_all):,} flights). Seasonal weather, hub congestion, peak-hour effects, tight turnarounds, "
        "knock-on delay, problem routes and unreliable tails were simulated, and realistic data-quality issues "
        "were injected so cleaning is meaningful.")
    with st.expander("Raw data sample (as delivered, before cleaning)"):
        st.dataframe(raw_sample, width="stretch", hide_index=True)

    st.subheader("Cleaning & validation log")
    st.dataframe(clean_log, width="stretch", hide_index=True)
    st.caption("Engineered fields: Delay_Min, Is_Delayed (>15 min), Dep_Hour, Day_of_Week, Month, Origin/"
               "Destination, Aircraft_Type, Min_Turn (type minimum ground time), Turn_Slack, Prev_Delay_Min "
               "(joined via Previous_Flight), Propagated_Min, Adverse_Weather, Delay_Cause.")
    st.markdown("Minimum ground time assumed per type: " +
                ", ".join(f"**{k}** {v} min" for k, v in MIN_TURN.items()))

    st.subheader("Descriptive statistics (current filter)")
    st.dataframe(A.descriptive_stats(df), width="stretch")
    l, r = st.columns(2)
    plot(l, C.delay_histogram(df))
    plot(r, C.monthly_rate(df))

# ---- Airports & routes
with tabs[2]:
    plot(st, C.airport_rates(df))
    st.dataframe(A.by_airport(df).style.format({"Delay_Rate": "{:.1%}", "CI_Low": "{:.1%}", "CI_High": "{:.1%}",
                                                "Avg_Delay": "{:.1f}", "P90_Delay": "{:.0f}",
                                                "Weather_Share": "{:.1%}"}),
                 width="stretch", hide_index=True)

    st.subheader("Do certain routes consistently experience delays?")
    st.markdown("A route is **consistently delayed** if its delay rate exceeds the network rate in **≥10 of 12 "
                "months** *and* a one-sided binomial test vs the network rate gives **p < 0.01**. "
                "**Route-specific** routes are additionally ≥15% worse than their own departure airport, "
                "so the problem is the route rather than hub congestion.")
    rt = A.route_consistency(df, min_flights=50)
    if rt.empty:
        st.info("Not enough flights per route in the current selection.")
    else:
        plot(st, C.top_routes(df, rt))
        plot(st, C.route_month_heatmap(df, rt["Route"].head(12).tolist()))
        st.dataframe(rt[["Route", "Flights", "Delay_Rate", "Avg_Delay", "Months_Above_Network", "Monthly_Std",
                         "p_value", "Lift_vs_Network", "Lift_vs_Origin", "Consistently_Delayed", "Route_Specific"]]
                     .style.format({"Delay_Rate": "{:.1%}", "Avg_Delay": "{:.1f}", "Monthly_Std": "{:.3f}",
                                    "p_value": "{:.1e}", "Lift_vs_Network": "{:.2f}", "Lift_vs_Origin": "{:.2f}"}),
                     width="stretch", hide_index=True)

    st.subheader("Aircraft (tail) reliability")
    plot(st, C.tail_rates(df, flagged_tails))

# ---- Hour & day
with tabs[3]:
    measure = st.radio("Measure", ["Delay rate", "Average delay (min)"], horizontal=True)
    plot(st, C.hour_day_heatmap(df, "rate" if measure == "Delay rate" else "mean"))
    l, r = st.columns(2)
    plot(l, C.hour_rates(df))
    plot(r, C.day_rates(df))
    st.info("Delay rate climbs steadily after the morning wave: early departures start with fresh aircraft and "
            "overnight slack, while later legs inherit accumulated knock-on delay and meet the evening "
            "congestion bank. Friday and Sunday carry peak leisure/business demand.")

# ---- Turnaround & knock-on
with tabs[4]:
    c = st.columns(4)
    ts = A.turnaround_stats(df)
    c[0].metric("Turns below type minimum", f"{ts['short_share']:.1%}")
    c[1].metric("Delay rate — below minimum", f"{ts['short_rate']:.1%}")
    c[2].metric("Delay rate — adequate turn", f"{ts['ok_rate']:.1%}")
    c[3].metric("Spearman ρ (slack vs delay)", f"{ts['spearman_rho']:.2f}")
    l, r = st.columns(2)
    plot(l, C.turnaround_rates(df))
    plot(r, C.inbound_x_slack_heatmap(df))
    st.markdown("**Reading it:** moving right (more slack) the delay rate falls in every row — ground buffer "
                "absorbs a late inbound aircraft. Moving down (later inbound) it rises sharply unless slack is large. "
                "That is the knock-on (reactionary) mechanism.")
    plot(st, C.propagation_scatter(df))
    plot(st, C.leg_accumulation(df))

# ---- Weather vs operational
with tabs[5]:
    st.markdown(
        "Each delayed departure (>15 min) receives one **primary cause**, applied in priority order: "
        "1) **Weather** — Fog, Thunderstorm or Rain at the origin; 2) **Reactionary** — the inbound delay "
        "left over after ground slack explains ≥50% of the delay; 3) **Short turnaround** — ground time scheduled "
        "below the type minimum; 4) **Airport / other operational** — congestion, ATC, handling, crew, technical. "
        "Causes 2–4 are *operational* (within the airline's or airport's control).")
    plot(st, C.cause_hours(df))
    l, r = st.columns(2)
    plot(l, C.weather_delay(df))
    plot(r, C.monthly_cause_split(df))
    wt = A.weather_test(df)
    st.markdown(f"**Statistical check:** median delay in adverse weather is **{wt['adverse_median']:.0f} min** vs "
                f"**{wt['fair_median']:.0f} min** in clear/cloudy conditions (one-sided Mann-Whitney U, "
                f"p = {wt['mw_p']:.1e}). Adverse weather affects {wt['adverse_share']:.1%} of departures.")
    st.dataframe(A.cause_breakdown(df).style.format({"Delay_Minutes": "{:,.0f}", "Avg_Delay": "{:.1f}",
                                                     "Share_Flights": "{:.1%}", "Share_Minutes": "{:.1%}"}),
                 width="stretch", hide_index=True)

# ---- Random Forest model
with tabs[6]:
    st.markdown(
        "**Task:** classify whether a departure will be delayed >15 min, using only information known before "
        "push-back — " + ", ".join(f"`{f}`" for f in FEATURES) + ". Categorical features are ordinal-encoded; "
        "`class_weight='balanced_subsample'` handles the ~27% positive class. Hyper-parameters chosen by 3-fold "
        "stratified CV (ROC-AUC) on an 80% training split; metrics below are on the untouched 20% test split.")
    c = st.columns(6)
    c[0].metric("Accuracy", f"{metrics['accuracy']:.3f}",
                f"{metrics['accuracy'] - metrics['baseline_accuracy']:+.3f} vs majority baseline")
    c[1].metric("Precision", f"{metrics['precision']:.3f}")
    c[2].metric("Recall", f"{metrics['recall']:.3f}")
    c[3].metric("F1", f"{metrics['f1']:.3f}")
    c[4].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c[5].metric("PR-AUC", f"{metrics['pr_auc']:.3f}", f"base rate {metrics['positive_rate']:.2f}",
                delta_color="off")
    st.caption(f"Train {metrics['n_train']:,} · test {metrics['n_test']:,} · best params {metrics['best_params']} · "
               f"CV ROC-AUC {metrics['cv_auc']:.3f}")
    l, m, r = st.columns(3)
    plot(l, C.confusion(metrics))
    plot(m, C.roc(metrics))
    plot(r, C.precision_recall(metrics))

    st.subheader("Most significant delay contributors (feature importance)")
    l, r = st.columns(2)
    plot(l, C.permutation_importance(metrics))
    plot(r, C.impurity_importance(metrics))
    st.caption("Permutation importance is the more reliable of the two: impurity importance is biased toward "
               "high-cardinality numeric features such as turnaround time.")
    with st.expander("Cross-validation results"):
        st.dataframe(pd.DataFrame(metrics["cv_results"]), width="stretch", hide_index=True)

    st.subheader("Predict a single departure")
    with st.form("predict"):
        c = st.columns(5)
        f_air = c[0].selectbox("Departure airport", airports, index=airports.index("BOM"))
        dests = sorted(df_all.loc[df_all["Airport"] == f_air, "Destination"].unique())
        f_dest = c[1].selectbox("Destination", dests)
        f_type = c[2].selectbox("Aircraft type", types)
        f_wx = c[3].selectbox("Weather", [w for w in weathers if w != "Unknown"])
        f_hour = c[4].slider("Scheduled hour", 5, 22, 18)
        c = st.columns(4)
        f_day = c[0].selectbox("Day of week", DAY_NAMES, index=4)
        f_month = c[1].slider("Month", 1, 12, 7)
        f_turn = c[2].number_input("Planned turnaround (min)", 15, 900, MIN_TURN.get(f_type, 35) + 5)
        f_prev = c[3].number_input("Inbound delay (min)", 0, 600, 20)
        submitted = st.form_submit_button("Predict")
    if submitted:
        x = pd.DataFrame([{"Airport": f_air, "Destination": f_dest, "Aircraft_Type": f_type, "Weather": f_wx,
                           "Dep_Hour": f_hour, "Day_Num": DAY_NAMES.index(f_day), "Month": f_month,
                           "Turnaround_Time": f_turn, "Turn_Slack": f_turn - MIN_TURN[f_type],
                           "Prev_Delay_Min": f_prev}])[FEATURES]
        for col in CAT_FEATURES:
            x[col] = x[col].astype(str)
        p = model.predict_proba(x)[0, 1]
        st.metric("Probability of >15 min delay", f"{p:.0%}", "HIGH RISK" if p >= 0.5 else "low risk",
                  delta_color="inverse" if p >= 0.5 else "normal")
        if f_turn < MIN_TURN[f_type]:
            st.warning(f"Planned turnaround is {MIN_TURN[f_type] - f_turn} min below the {f_type} minimum.")

# ---- Action plan
with tabs[7]:
    st.subheader("Operational action plan")
    st.caption("Priorities ordered by expected impact on delay minutes and by how controllable the lever is.")

    st.markdown("#### What-if: enforce a minimum turnaround buffer")
    slack = st.slider("Minimum ground-time slack above the type minimum (min)", 0, 30, 10, step=5)
    sc = buffer_scenario(slack)
    c = st.columns(4)
    c[0].metric("Turns re-timed", f"{sc['flights_changed']:,}")
    c[1].metric("Ground time added", f"{sc['minutes_added'] / 60:,.0f} h / year")
    c[2].metric("Predicted delayed departures", f"{sc['new']:,.0f}",
                f"{-sc['avoided']:,.0f} ({-sc['rel']:.1%})", delta_color="inverse")
    c[3].metric("Predicted OTP change", f"{sc['avoided'] / K['valid']:+.1%}")
    st.caption("Estimated by re-scoring every flight with the Random Forest after raising its turnaround slack to "
               "the chosen buffer (inbound delays held fixed, so second-order knock-on benefits are not counted "
               "and the estimate is conservative). A model-based estimate, not a causal guarantee — validate with "
               "a pilot on the worst routes first.")

    plan = R.action_plan(F, buffer_scenario(10))
    st.dataframe(plan, width="stretch", hide_index=True,
                 column_config={"Action": st.column_config.TextColumn(width="large"),
                                "Evidence": st.column_config.TextColumn(width="medium")})
    st.download_button("Download action plan (CSV)", plan.to_csv(index=False), "action_plan.csv", "text/csv")
    st.download_button("Download cleaned dataset (CSV)", df_all.to_csv(index=False), "clean_flights.csv", "text/csv")
