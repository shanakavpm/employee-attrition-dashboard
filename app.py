"""Streamlit dashboard for employee attrition analysis and model review."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MIN_GROUP_SIZE, OUTPUT_DIR, TARGET_COLUMN
from src.dashboard_data import load_dashboard_data
from src.dashboard_filters import FILTER_DEFINITIONS, FILTER_KEYS, apply_filters, filter_options


st.set_page_config(page_title="Employee Attrition Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def get_dashboard_data(version: int) -> dict:
    """Cache saved results; file modification time invalidates the cache."""
    return load_dashboard_data(OUTPUT_DIR / "dashboard_data.json")


def clear_filters() -> None:
    """Reset sidebar filters through their stable Streamlit state keys."""
    for key in FILTER_KEYS:
        st.session_state[key] = []


try:
    results_path = OUTPUT_DIR / "dashboard_data.json"
    results = get_dashboard_data(results_path.stat().st_mtime_ns)
except (OSError, ValueError, KeyError, TypeError) as error:
    st.error("Dashboard results are missing or invalid. Run python run_pipeline.py and deploy outputs/dashboard_data.json with the app.")
    st.caption(f"Loader detail: {type(error).__name__}: {error}")
    st.stop()

frame = results["frame"]
quality_report = results["quality_report"]
best_model_name = results["best_model_name"]
selection_reason = results["selection_reason"]
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
    selected_filters: dict[str, list[object]] = {}
    for key, label, column in FILTER_DEFINITIONS:
        selected_filters[column] = st.multiselect(
            label,
            filter_options(frame, column),
            key=key,
        )
    risk_threshold = st.slider(
        "High-risk threshold",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help=(
            "The threshold determines which held-out test records are labelled as high risk. "
            "Lowering the threshold identifies more records but may increase false alerts. "
            "Raising it identifies fewer records but may miss some employees who later left. "
            "This model output supports human review and is not an employment decision."
        ),
    )

filtered = apply_filters(frame, selected_filters)
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
if total_employees < MIN_GROUP_SIZE:
    st.warning(
        f"This result is based on {total_employees} records, below the configured minimum of "
        f"{MIN_GROUP_SIZE}, and should be interpreted cautiously."
    )

overview_left, overview_right = st.columns(2)
with overview_left:
    stay_leave = filtered["attrition_label"].value_counts().rename_axis("status").reset_index(name="employees")
    st.plotly_chart(
        px.pie(stay_leave, names="status", values="employees", hole=0.55, title="Stayed and left"),
        use_container_width=True,
    )
with overview_right:
    department_rate = (
        filtered.groupby("department", as_index=False)
        .agg(attrition_rate=(TARGET_COLUMN, "mean"), records=(TARGET_COLUMN, "size"))
        .sort_values("attrition_rate", ascending=False)
    )
    department_rate["sample_note"] = department_rate["records"].map(
        lambda count: "Small group: interpret cautiously" if count < MIN_GROUP_SIZE else ""
    )
    st.plotly_chart(
        px.bar(
            department_rate,
            x="department",
            y="attrition_rate",
            text="records",
            hover_data=["records", "sample_note"],
            title="Attrition rate by department",
            labels={"attrition_rate": "Attrition rate", "department": "Department", "records": "Records"},
        ).update_traces(texttemplate="n=%{text}", textposition="outside").update_yaxes(tickformat=".0%"),
        use_container_width=True,
    )

st.subheader("Attrition factors")
factor_left, factor_right = st.columns(2)
with factor_left:
    overtime_rate = filtered.groupby("overtime", as_index=False).agg(
        attrition_rate=(TARGET_COLUMN, "mean"), records=(TARGET_COLUMN, "size")
    )
    overtime_rate["sample_note"] = overtime_rate["records"].map(
        lambda count: "Small group: interpret cautiously" if count < MIN_GROUP_SIZE else ""
    )
    st.plotly_chart(
        px.bar(
            overtime_rate,
            x="overtime",
            y="attrition_rate",
            text="records",
            hover_data=["records", "sample_note"],
            title="Attrition rate by overtime",
            labels={"attrition_rate": "Attrition rate", "overtime": "Overtime", "records": "Records"},
        ).update_traces(texttemplate="n=%{text}", textposition="outside").update_yaxes(tickformat=".0%"),
        use_container_width=True,
    )
with factor_right:
    satisfaction_rate = (
        filtered.groupby("job_satisfaction", as_index=False)
        .agg(attrition_rate=(TARGET_COLUMN, "mean"), records=(TARGET_COLUMN, "size"))
        .sort_values("attrition_rate", ascending=False)
    )
    satisfaction_rate["sample_note"] = satisfaction_rate["records"].map(
        lambda count: "Small group: interpret cautiously" if count < MIN_GROUP_SIZE else ""
    )
    st.plotly_chart(
        px.bar(
            satisfaction_rate,
            x="job_satisfaction",
            y="attrition_rate",
            text="records",
            hover_data=["records", "sample_note"],
            title="Attrition rate by job satisfaction",
            labels={"attrition_rate": "Attrition rate", "job_satisfaction": "Job satisfaction", "records": "Records"},
        ).update_traces(texttemplate="n=%{text}", textposition="outside").update_yaxes(tickformat=".0%"),
        use_container_width=True,
    )
if any(selected_filters.values()) and any(
    summary["records"].lt(MIN_GROUP_SIZE).any()
    for summary in (department_rate, overtime_rate, satisfaction_rate)
):
    st.warning(
        f"One or more displayed subgroup rates are based on fewer than {MIN_GROUP_SIZE} records and should "
        "be interpreted cautiously. Record counts are shown on the bars and in chart details."
    )

st.subheader("Prediction model")
st.write(f"Selected model: **{best_model_name}**.")
st.caption(
    f"{selection_reason} The independent hold-out set is used only for final evaluation. "
    "These model and calibration results do not change with sidebar filters."
)
metrics_table = results["metrics"][[
    "model",
    "selected",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "cv_evaluations",
    "cv_f1_mean",
    "cv_f1_std",
    "cv_f1_p2_5",
    "cv_f1_p97_5",
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
    "cv_roc_auc_p2_5",
    "cv_roc_auc_p97_5",
]]
metric_formats = {
    column: "{:.3f}"
    for column in metrics_table.select_dtypes(include="number").columns
    if column != "cv_evaluations"
}
st.dataframe(
    metrics_table.style.format(metric_formats),
    column_config={
        "selected": "Selected",
        "cv_evaluations": "CV validation folds",
        "cv_f1_mean": "CV F1 mean",
        "cv_f1_std": "CV F1 SD",
        "cv_f1_p2_5": "CV F1 2.5th pct",
        "cv_f1_p97_5": "CV F1 97.5th pct",
        "cv_roc_auc_mean": "CV ROC-AUC mean",
        "cv_roc_auc_std": "CV ROC-AUC SD",
        "cv_roc_auc_p2_5": "CV ROC-AUC 2.5th pct",
        "cv_roc_auc_p97_5": "CV ROC-AUC 97.5th pct",
    },
    width="stretch",
    hide_index=True,
)

filtered_risk_review = apply_filters(risk_review, selected_filters)
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
    displayed_risk_review = displayed_risk_review[
        ["id", "department", "jobtitle", "gender", "age_group", "attrition_label", "attrition_risk", "risk_category"]
    ]
    if displayed_risk_review.empty:
        st.warning("No held-out test records match the selected filters.")
    else:
        st.dataframe(
            displayed_risk_review,
            column_config={
                "id": "Dataset record ID",
                "attrition_risk": st.column_config.ProgressColumn(
                    "Predicted risk", min_value=0.0, max_value=100.0, format="%.1f%%"
                ),
            },
            width="stretch",
            hide_index=True,
        )
st.info(
    "**How to interpret the prediction table:** Predicted risk is a model estimate, not confirmation that an "
    "employee will leave. **Left** and **Stayed** are the recorded outcomes in the dataset. Records at or above "
    "the selected threshold are labelled **High risk**. Model outputs must not be used alone for promotion, "
    "termination or other high-impact employment decisions."
)
st.caption(
    "Risk estimates are shown only for the held-out test set. Dataset record IDs are reference keys for review, "
    "not evidence for an employment decision."
)

st.subheader("Probability calibration")
calibration_metrics = results["calibration_metrics"].iloc[0]
calibration_left, calibration_right = st.columns([1, 3])
with calibration_left:
    st.metric("Held-out Brier score", f"{calibration_metrics['brier_score']:.3f}")
    st.caption(
        f"Baseline Brier score: {calibration_metrics['baseline_brier_score']:.3f}. "
        "Lower scores are better."
    )
with calibration_right:
    calibration_curve = results["calibration_curve"]
    calibration_figure = px.line(
        calibration_curve,
        x="mean_predicted_risk",
        y="observed_attrition_rate",
        markers=True,
        hover_data=["calibration_bin", "records", "absolute_gap"],
        title=f"Held-out reliability diagram: {best_model_name}",
        labels={
            "mean_predicted_risk": "Mean predicted risk",
            "observed_attrition_rate": "Observed attrition rate",
            "calibration_bin": "Risk bin",
            "records": "Records",
            "absolute_gap": "Absolute gap",
        },
    )
    calibration_figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line={"color": "gray", "dash": "dash"},
    )
    calibration_figure.update_xaxes(tickformat=".0%", range=[0, 1])
    calibration_figure.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(calibration_figure, use_container_width=True)
st.info(str(calibration_metrics["interpretation"]))

importance = results["importance"].head(12).sort_values("importance_mean")
st.plotly_chart(
    px.bar(importance, x="importance_mean", y="feature", orientation="h", error_x="importance_std", title="Global feature importance", labels={"importance_mean": "Decrease in F1 when shuffled", "feature": "Feature"}),
    use_container_width=True,
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
