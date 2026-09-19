"""Streamlit dashboard for employee attrition analysis and model review."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import OUTPUT_DIR, TARGET_COLUMN
from src.dashboard_data import load_dashboard_data


st.set_page_config(page_title="Employee Attrition Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def get_dashboard_data(version: int) -> dict:
    """Cache saved results; file modification time invalidates the cache."""
    return load_dashboard_data(OUTPUT_DIR / "dashboard_data.json")


def apply_filters(frame: pd.DataFrame, departments: list[str], genders: list[str], ages: list[str]) -> pd.DataFrame:
    """Apply sidebar filters while leaving the original data unchanged."""
    filtered = frame.copy()
    if departments:
        filtered = filtered[filtered["department"].isin(departments)]
    if genders:
        filtered = filtered[filtered["gender"].isin(genders)]
    if ages:
        filtered = filtered[filtered["age_group"].isin(ages)]
    return filtered


FILTER_KEYS = ("department_filter", "gender_filter", "age_filter")


def clear_filters() -> None:
    """Reset sidebar filters through their stable Streamlit state keys."""
    for key in FILTER_KEYS:
        st.session_state[key] = []


try:
    results_path = OUTPUT_DIR / "dashboard_data.json"
    results = get_dashboard_data(results_path.stat().st_mtime_ns)
except (OSError, ValueError, KeyError, TypeError):
    st.error("Dashboard results are missing or invalid. Run python run_pipeline.py and deploy outputs/dashboard_data.json with the app.")
    st.stop()

frame = results["frame"]
quality_report = results["quality_report"]
best_model_name = results["best_model_name"]
risk_review = results["risk_review"]
test_record_count = len(risk_review)

st.title("Employee Attrition Dashboard")
st.caption("Decision-support prototype using the Saudi Employee Attrition Dataset. It does not make automated HR decisions.")

with st.sidebar:
    st.header("Filters")
    st.button(
        "Clear all filters",
        on_click=clear_filters,
        disabled=not any(st.session_state.get(key) for key in FILTER_KEYS),
        width="stretch",
    )
    department_filter = st.multiselect(
        "Department", sorted(frame["department"].unique()), key="department_filter"
    )
    gender_filter = st.multiselect("Gender", sorted(frame["gender"].unique()), key="gender_filter")
    age_filter = st.multiselect("Age group", sorted(frame["age_group"].unique()), key="age_filter")
    risk_threshold = st.slider(
        "High-risk threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Records with a predicted risk at or above this value are labelled High risk.",
    )

filtered = apply_filters(frame, department_filter, gender_filter, age_filter)
total_employees = len(filtered)
left_employees = int(filtered[TARGET_COLUMN].sum())
attrition_rate = left_employees / total_employees if total_employees else 0
st.subheader("Overview")
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Employees reviewed", f"{total_employees:,}")
metric_2.metric("Employees who left", f"{left_employees:,}")
metric_3.metric("Attrition rate", f"{attrition_rate:.1%}")
metric_4.metric("Employees who stayed", f"{total_employees - left_employees:,}")

if filtered.empty:
    st.warning("No records match the selected filters. Change or clear a filter.")
    st.stop()

overview_left, overview_right = st.columns(2)
with overview_left:
    stay_leave = filtered["attrition_label"].value_counts().rename_axis("status").reset_index(name="employees")
    st.plotly_chart(
        px.pie(stay_leave, names="status", values="employees", hole=0.55, title="Stayed and left"),
        width="stretch",
    )
with overview_right:
    department_rate = (
        filtered.groupby("department", as_index=False)[TARGET_COLUMN]
        .mean()
        .sort_values(TARGET_COLUMN, ascending=False)
    )
    st.plotly_chart(
        px.bar(department_rate, x="department", y=TARGET_COLUMN, title="Attrition rate by department", labels={TARGET_COLUMN: "Attrition rate", "department": "Department"}).update_yaxes(tickformat=".0%"),
        width="stretch",
    )

st.subheader("Attrition factors")
factor_left, factor_right = st.columns(2)
with factor_left:
    overtime_rate = filtered.groupby("overtime", as_index=False)[TARGET_COLUMN].mean()
    st.plotly_chart(
        px.bar(overtime_rate, x="overtime", y=TARGET_COLUMN, title="Attrition rate by overtime", labels={TARGET_COLUMN: "Attrition rate", "overtime": "Overtime"}).update_yaxes(tickformat=".0%"),
        width="stretch",
    )
with factor_right:
    satisfaction_rate = (
        filtered.groupby("job_satisfaction", as_index=False)[TARGET_COLUMN]
        .mean()
        .sort_values(TARGET_COLUMN, ascending=False)
    )
    st.plotly_chart(
        px.bar(satisfaction_rate, x="job_satisfaction", y=TARGET_COLUMN, title="Attrition rate by job satisfaction", labels={TARGET_COLUMN: "Attrition rate", "job_satisfaction": "Job satisfaction"}).update_yaxes(tickformat=".0%"),
        width="stretch",
    )

st.subheader("Prediction model")
st.write(f"Selected model: **{best_model_name}**. The model is selected using hold-out F1 score, which balances missed leavers and false alerts.")
st.dataframe(
    results["metrics"].style.format({column: "{:.3f}" for column in results["metrics"].columns if column != "model"}),
    width="stretch",
    hide_index=True,
)

filtered_risk_review = apply_filters(risk_review, department_filter, gender_filter, age_filter)
filtered_risk_review["risk_category"] = filtered_risk_review["attrition_risk"].ge(risk_threshold).map(
    {True: "High risk", False: "Lower risk"}
)
high_risk_count = int((filtered_risk_review["risk_category"] == "High risk").sum())
risk_left, risk_right = st.columns([1, 3])
with risk_left:
    st.metric("High-risk test records", high_risk_count)
    st.caption(f"Threshold: {risk_threshold:.0%}")
with risk_right:
    # Keep classification probabilities on 0–1; scale only the displayed table.
    displayed_risk_review = filtered_risk_review.sort_values("attrition_risk", ascending=False).copy()
    displayed_risk_review["attrition_risk"] *= 100
    st.dataframe(
        displayed_risk_review,
        column_config={"attrition_risk": st.column_config.ProgressColumn("Predicted risk", min_value=0.0, max_value=100.0, format="%.1f%%")},
        width="stretch",
        hide_index=True,
    )
st.info(
    "**How to interpret the prediction table:** Predicted risk is a model estimate, not confirmation that an "
    "employee will leave. **Left** and **Stayed** are the recorded outcomes in the dataset. Records at or above "
    "the selected threshold are labelled **High risk**. This dashboard supports human review and must not be "
    "used for automated employment decisions."
)
st.caption("Risk estimates are shown only for the held-out test set.")

importance = results["importance"].head(12).sort_values("importance_mean")
st.plotly_chart(
    px.bar(importance, x="importance_mean", y="feature", orientation="h", error_x="importance_std", title="Global feature importance", labels={"importance_mean": "Decrease in F1 when shuffled", "feature": "Feature"}),
    width="stretch",
)
st.caption(f"Feature importance: {best_model_name}, evaluated on all {test_record_count:,} held-out test records, independent of sidebar filters. Permutation importance describes predictive association, not causation.")

st.subheader("Fairness and responsible use")
st.caption(
    f"Fairness tables: all {test_record_count:,} held-out test records, "
    f"using {best_model_name} at the model’s default 50% decision threshold. "
    "Sidebar filters and the high-risk threshold slider do not change these tables."
)
fairness_tabs = st.tabs(["Gender", "Age group", "Data quality"])
with fairness_tabs[0]:
    st.dataframe(results["gender_fairness"].style.format({"predicted_high_risk_rate": "{:.1%}", "recall": "{:.1%}", "false_positive_rate": "{:.1%}"}), width="stretch", hide_index=True)
with fairness_tabs[1]:
    st.dataframe(results["age_fairness"].style.format({"predicted_high_risk_rate": "{:.1%}", "recall": "{:.1%}", "false_positive_rate": "{:.1%}"}), width="stretch", hide_index=True)
with fairness_tabs[2]:
    st.dataframe(quality_report, width="stretch", hide_index=True)

st.warning("Use model outputs to identify areas for human review and support. Do not use them alone for promotion, termination, or other high-impact employment decisions.")
