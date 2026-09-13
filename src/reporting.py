"""Static evidence figures for the written report and GitHub repository."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# The project generates report images in a non-interactive environment.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import MIN_GROUP_SIZE, TARGET_COLUMN
from src.modeling import AnalysisResult, fairness_by_group, global_feature_importance


def _save_figure(path: Path) -> None:
    """Apply consistent export settings and close the active chart."""
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _percentage_axis(axis: plt.Axes) -> None:
    """Display a proportion axis as percentages without an extra dependency."""
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")


def generate_evidence_figures(
    frame: pd.DataFrame, result: AnalysisResult, figure_directory: Path
) -> pd.DataFrame:
    """Generate concise PNG figures used as reproducible assignment evidence."""
    figure_directory.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    index: list[dict[str, str]] = []

    attrition_counts = frame["attrition_label"].value_counts().reindex(["Stayed", "Left"], fill_value=0)
    ax = attrition_counts.plot(kind="bar", color=["#4C78A8", "#E45756"], figsize=(7, 4))
    ax.set(title="Employee attrition distribution", xlabel="Employee status", ylabel="Employees")
    figure_path = figure_directory / "fig_01_attrition_distribution.png"
    _save_figure(figure_path)
    index.append({"file": figure_path.name, "purpose": "Employee stay and leave distribution"})

    department_summary = frame.groupby("department")[TARGET_COLUMN].agg(records="size", attrition_rate="mean")
    department_summary = department_summary[department_summary["records"] >= MIN_GROUP_SIZE].sort_values("attrition_rate")
    department_labels = [f"{name} (n={records})" for name, records in department_summary["records"].items()]
    ax = department_summary["attrition_rate"].set_axis(department_labels).plot(kind="barh", color="#4C78A8", figsize=(8, 5))
    ax.set(title="Attrition rate by department", xlabel="Attrition rate", ylabel="Department")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    figure_path = figure_directory / "fig_02_attrition_by_department.png"
    _save_figure(figure_path)
    index.append({"file": figure_path.name, "purpose": "Department-level attrition comparison"})

    satisfaction_rates = frame.groupby("job_satisfaction")[TARGET_COLUMN].mean().sort_values(ascending=False)
    ax = satisfaction_rates.plot(kind="bar", color="#F58518", figsize=(7, 4))
    ax.set(title="Attrition rate by job satisfaction", xlabel="Job satisfaction", ylabel="Attrition rate")
    _percentage_axis(ax)
    figure_path = figure_directory / "fig_03_attrition_by_job_satisfaction.png"
    _save_figure(figure_path)
    index.append({"file": figure_path.name, "purpose": "Job satisfaction and attrition comparison"})

    model_metrics = result.metrics.set_index("model")[["f1", "roc_auc"]]
    ax = model_metrics.plot(kind="bar", color=["#54A24B", "#B279A2"], figsize=(8, 4))
    ax.set(title="Hold-out model comparison", xlabel="Model", ylabel="Score", ylim=(0, 1))
    ax.legend(["F1 score", "ROC-AUC"], loc="lower right")
    figure_path = figure_directory / "fig_04_model_comparison.png"
    _save_figure(figure_path)
    index.append({"file": figure_path.name, "purpose": "Model selection evidence"})

    importance = global_feature_importance(result).head(12).sort_values("importance_mean")
    ax = importance.plot.barh(x="feature", y="importance_mean", xerr="importance_std", legend=False, color="#72B7B2", figsize=(8, 6))
    ax.set(title="Global feature importance", xlabel="Decrease in held-out F1 when shuffled", ylabel="Feature")
    figure_path = figure_directory / "fig_05_feature_importance.png"
    _save_figure(figure_path)
    index.append({"file": figure_path.name, "purpose": "Model explanation evidence"})

    gender_fairness = fairness_by_group(result, frame, "gender").set_index("group")
    ax = gender_fairness[["predicted_high_risk_rate", "recall", "false_positive_rate"]].plot(kind="bar", figsize=(8, 4))
    ax.set(title="Held-out fairness diagnostic by gender", xlabel="Gender", ylabel="Rate", ylim=(0, 1))
    _percentage_axis(ax)
    ax.legend(["Predicted high-risk rate", "Recall", "False-positive rate"], loc="upper right")
    figure_path = figure_directory / "fig_06_fairness_by_gender.png"
    _save_figure(figure_path)
    index.append({"file": figure_path.name, "purpose": "Gender-level fairness diagnostic"})

    return pd.DataFrame(index)
