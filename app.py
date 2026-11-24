"""Streamlit app: ER wait time analysis and prediction.

Run locally:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src.analysis import (
    DAY_ORDER,
    SEASON_ORDER,
    TARGET,
    TIME_OF_DAY_ORDER,
    URGENCY_ORDER,
    load_visits,
    outcome_by_urgency,
    plot_arrivals_by_hour,
    plot_outcome_by_urgency,
    plot_stage_breakdown,
    plot_wait_by_staffing,
    plot_wait_by_time_of_day,
    plot_wait_by_urgency,
    plot_wait_distribution,
    quality_checks,
    stage_breakdown,
    staffing_within_urgency,
    wait_summary,
)
from src.model import (
    FEATURES,
    compare_models,
    feature_importance,
    leakage_demo,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison,
    test_predictions,
    train_final_model,
)

st.set_page_config(page_title="ER Wait Time Analysis and Prediction", layout="wide")


@st.cache_data
def get_data():
    return load_visits()


@st.cache_resource
def get_model(_df):
    return train_final_model(_df)


@st.cache_data
def get_evaluation(_df):
    return compare_models(_df), leakage_demo(_df), feature_importance(_df), test_predictions(_df)


df = get_data()
model = get_model(df)

st.title("ER Wait Time Analysis and Prediction")
st.write(
    "Analysis of 5,000 emergency room visits across five hospitals in 2024, and a model that predicts "
    "a patient's total wait using only what is known when they arrive."
)

tab_predict, tab_overview, tab_drivers, tab_model, tab_quality = st.tabs(
    ["Predict a wait", "Overview", "What drives the wait", "Model", "Data quality"]
)

# ---------------------------------------------------------------- prediction
with tab_predict:
    st.subheader("Estimate the wait for a new arrival")
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        urgency = c1.selectbox("Urgency level", URGENCY_ORDER, index=2)
        time_of_day = c1.selectbox("Time of day", TIME_OF_DAY_ORDER, index=3)
        hour = c1.slider("Hour of arrival", 0, 23, 18)
        day = c2.selectbox("Day of week", DAY_ORDER, index=2)
        season = c2.selectbox("Season", SEASON_ORDER)
        hospital = c2.selectbox("Hospital", sorted(df["Hospital Name"].unique()))
        nurse_ratio = c3.select_slider("Nurse-to-patient ratio (patients per nurse)", options=[1, 2, 3, 4, 5], value=3)
        specialists = c3.slider("Specialists available", 0, int(df["Specialist Availability"].max()), 3)
        beds = c3.slider("Facility size (beds)", int(df["Facility Size (Beds)"].min()),
                         int(df["Facility Size (Beds)"].max()), 80)
        submitted = st.form_submit_button("Predict wait time")

    region = df.loc[df["Hospital Name"] == hospital, "Region"].mode().iloc[0]
    row = pd.DataFrame([{
        "Urgency Level": urgency, "Time of Day": time_of_day, "Day of Week": day, "Season": season,
        "Region": region, "Hospital Name": hospital, "Nurse-to-Patient Ratio": nurse_ratio,
        "Specialist Availability": specialists, "Facility Size (Beds)": beds, "Hour": hour,
    }])[FEATURES]
    prediction = float(model.predict(row)[0])

    similar = df[(df["Urgency Level"] == urgency) & (df["Time of Day"] == time_of_day)][TARGET]
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted total wait", f"{prediction:.0f} min")
    m2.metric("Typical range", f"{max(prediction - 12, 0):.0f} to {prediction + 12:.0f} min")
    m3.metric(f"Median for {urgency.lower()} urgency, {time_of_day.lower()}", f"{similar.median():.0f} min")
    st.caption(
        "The range uses the model's average error on unseen visits (about 12 minutes). "
        "In this data, staffing levels are tied to urgency: critical patients were seen at ratios 1 to 2 and "
        "low-urgency patients at 4 to 5, so other combinations are outside what the model has seen."
    )

# ---------------------------------------------------------------- overview
with tab_overview:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Visits", f"{len(df):,}")
    k2.metric("Average wait", f"{df[TARGET].mean():.0f} min")
    k3.metric("Median wait", f"{df[TARGET].median():.0f} min")
    lwbs = (df["Patient Outcome"] == "Left Without Being Seen").mean() * 100
    k4.metric("Left without being seen", f"{lwbs:.1f}%")
    left, right = st.columns(2)
    left.pyplot(plot_wait_distribution(df))
    right.pyplot(plot_arrivals_by_hour(df))
    st.dataframe(wait_summary(df, "Hospital Name"), hide_index=True, width="stretch")

# ---------------------------------------------------------------- drivers
with tab_drivers:
    left, right = st.columns(2)
    left.pyplot(plot_wait_by_urgency(df))
    right.pyplot(plot_stage_breakdown(df))
    left, right = st.columns(2)
    left.pyplot(plot_wait_by_time_of_day(df))
    right.pyplot(plot_outcome_by_urgency(df))
    st.subheader("Staffing")
    left, right = st.columns(2)
    left.pyplot(plot_wait_by_staffing(df))
    right.markdown("**Average wait by staffing level within each urgency level (minutes)**")
    right.dataframe(staffing_within_urgency(df), width="stretch")
    right.caption(
        "Staffing ratio follows urgency in this data, so the chart on the left mostly shows the urgency effect. "
        "Within the same urgency level, one more patient per nurse adds about 5 to 16 minutes."
    )
    left, right = st.columns(2)
    left.dataframe(wait_summary(df, "Season"), hide_index=True, width="stretch")
    right.dataframe(wait_summary(df, "Day of Week"), hide_index=True, width="stretch")

# ---------------------------------------------------------------- model
with tab_model:
    with st.spinner("Evaluating models..."):
        comparison, leakage, importance, preds = get_evaluation(df)
    st.pyplot(plot_model_comparison(comparison))
    st.dataframe(comparison, hide_index=True, width="stretch")
    left, right = st.columns(2)
    left.pyplot(plot_actual_vs_predicted(preds))
    right.pyplot(plot_feature_importance(importance))
    st.subheader("Why the inputs are limited to arrival information")
    st.write(
        "Total wait is exactly the sum of time to registration, time to triage and time to a clinician. "
        "Patient outcome and satisfaction are only recorded after the visit. A model given these columns "
        "scores almost perfectly, but it could never be used for a patient who has just walked in."
    )
    st.dataframe(leakage, hide_index=True, width="stretch")

# ---------------------------------------------------------------- data quality
with tab_quality:
    st.dataframe(quality_checks(df).astype(str).rename("value"), width="stretch")
    st.markdown(
        "- No missing values or duplicate visits.\n"
        "- The `Time of Day` label agrees with the clock time in `Visit Date` for only about 18% of visits, "
        "so it is used as given and the hour is kept as a separate input.\n"
        "- Wait times are right-skewed (median 60, mean 82 minutes), so medians and the 90th percentile are "
        "reported next to averages.\n"
        "- Source: [ER Wait Time dataset on Kaggle](https://www.kaggle.com/datasets/rivalytics/er-wait-time)."
    )
    st.dataframe(outcome_by_urgency(df), width="stretch")
    st.download_button("Download the dataset", df.drop(columns=["Hour", "Month"]).to_csv(index=False),
                       file_name="er_visits.csv")
