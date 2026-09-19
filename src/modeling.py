"""Predictive modelling, evaluation, fairness checks, and explanations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CV_FOLDS, CV_REPEATS, MIN_GROUP_SIZE, RANDOM_STATE, TEST_SIZE
from src.data import model_features

# Streamlit Community Cloud's current Python 3.14 image can fail inside
# scikit-learn's joblib wrapper, even when ``n_jobs=1``. Dashboard evaluation is
# intentionally serial, avoiding that wrapper altogether. The dataset is small,
# so the latency remains suitable for an interactive app.
PERMUTATION_REPEATS = 15


@dataclass
class AnalysisResult:
    """All objects needed to report and visualise a reproducible model run."""

    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    best_model_name: str
    selection_reason: str
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    probabilities: pd.Series
    predictions: pd.Series


def _preprocessor(features: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """Build a preprocessing step that is fitted only within each model pipeline."""
    categorical_columns = features.select_dtypes(include="object").columns.tolist()
    numeric_columns = features.select_dtypes(include="number").columns.tolist()

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def _logistic_pipeline(features: pd.DataFrame, scale_numeric: bool = True) -> Pipeline:
    """Build the transparent classifier used for model selection and ablation."""
    return Pipeline(
        steps=[
            ("preprocessor", _preprocessor(features, scale_numeric=scale_numeric)),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_candidate_models(features: pd.DataFrame) -> dict[str, Pipeline]:
    """Create transparent baseline and non-linear candidate models.

    Class weighting addresses imbalance without synthetic observations. All
    transformations remain inside the pipeline, preventing train-test leakage.
    """
    return {
        "Logistic Regression": _logistic_pipeline(features),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", _preprocessor(features, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def preprocessing_ablation(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Measure the effect of numeric standardisation on the logistic pipeline."""
    features = model_features(frame)
    target = frame[target_column]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str | bool]] = []
    for variant, scale_numeric in (("With numeric standardisation", True), ("Without numeric standardisation", False)):
        pipeline = _logistic_pipeline(x_train, scale_numeric=scale_numeric)
        cv_result = _serial_cross_validation(pipeline, x_train, y_train, cv)
        pipeline.fit(x_train, y_train)
        row = _metric_row(variant, pipeline, x_test, y_test)
        row["numeric_standardisation"] = scale_numeric
        row["cv_f1_mean"] = float(np.mean(cv_result["f1"]))
        row["cv_f1_std"] = float(np.std(cv_result["f1"], ddof=1))
        row["cv_roc_auc_mean"] = float(np.mean(cv_result["roc_auc"]))
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_row(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | str]:
    """Calculate hold-out metrics for one fitted model."""
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def _serial_cross_validation(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    cv: StratifiedKFold | RepeatedStratifiedKFold,
) -> dict[str, list[float]]:
    """Evaluate folds without joblib, which is more reliable in hosted sessions."""
    scores: dict[str, list[float]] = {"f1": [], "roc_auc": []}
    for train_index, validation_index in cv.split(features, target):
        candidate = clone(pipeline)
        x_train = features.iloc[train_index]
        x_validation = features.iloc[validation_index]
        y_train = target.iloc[train_index]
        y_validation = target.iloc[validation_index]
        candidate.fit(x_train, y_train)
        predictions = candidate.predict(x_validation)
        probabilities = candidate.predict_proba(x_validation)[:, 1]
        scores["f1"].append(float(f1_score(y_validation, predictions, zero_division=0)))
        scores["roc_auc"].append(float(roc_auc_score(y_validation, probabilities)))
    return scores


def _repeated_score_summary(scores: dict[str, list[float]]) -> dict[str, float | int]:
    """Summarise repeated-fold scores with an empirical 95% percentile interval."""
    summary: dict[str, float | int] = {"cv_evaluations": len(scores["f1"])}
    for metric in ("f1", "roc_auc"):
        values = np.asarray(scores[metric], dtype=float)
        summary[f"cv_{metric}_mean"] = float(values.mean())
        summary[f"cv_{metric}_std"] = float(values.std(ddof=1))
        summary[f"cv_{metric}_p2_5"] = float(np.percentile(values, 2.5))
        summary[f"cv_{metric}_p97_5"] = float(np.percentile(values, 97.5))
    return summary


def _select_model(metrics: pd.DataFrame) -> tuple[str, str]:
    """Select from training-only CV results with a transparent-model preference."""
    best_mean = float(metrics["cv_f1_mean"].max())
    comparable = metrics[metrics["cv_f1_mean"] >= best_mean - 0.01]
    if "Logistic Regression" in comparable["model"].values:
        selected = "Logistic Regression"
        reason = (
            "Selected from 5-fold, 5-repeat cross-validation on the training data. "
            "Its mean F1 was within 0.01 of the strongest candidate, so the more "
            "interpretable and maintainable model was preferred."
        )
    else:
        ranked = comparable.sort_values(
            ["cv_f1_mean", "cv_f1_std", "cv_roc_auc_mean"],
            ascending=[False, True, False],
        )
        selected = str(ranked.iloc[0]["model"])
        reason = (
            "Selected from 5-fold, 5-repeat cross-validation on the training data "
            "using mean F1, stability and ROC-AUC."
        )
    return selected, reason


def train_and_evaluate(frame: pd.DataFrame, target_column: str) -> AnalysisResult:
    """Select a model by repeated CV, then evaluate it once on the hold-out set."""
    features = model_features(frame)
    target = frame[target_column]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    candidates = build_candidate_models(x_train)
    cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    cv_rows: list[dict[str, float | str | int]] = []

    for name, pipeline in candidates.items():
        cv_result = _serial_cross_validation(pipeline, x_train, y_train, cv)
        row: dict[str, float | str | int] = {"model": name}
        row.update(_repeated_score_summary(cv_result))
        cv_rows.append(row)

    cv_metrics = pd.DataFrame(cv_rows)
    best_model_name, selection_reason = _select_model(cv_metrics)

    holdout_rows: list[dict[str, float | str]] = []
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        holdout_rows.append(_metric_row(name, pipeline, x_test, y_test))
    metrics = pd.DataFrame(holdout_rows).merge(cv_metrics, on="model", validate="one_to_one")
    metrics["selected"] = metrics["model"].eq(best_model_name)
    metrics = metrics.sort_values(
        ["selected", "cv_f1_mean", "cv_f1_std"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_model = candidates[best_model_name]
    probabilities = pd.Series(best_model.predict_proba(x_test)[:, 1], index=x_test.index, name="attrition_risk")
    predictions = pd.Series(best_model.predict(x_test), index=x_test.index, name="predicted_attrition")
    return AnalysisResult(
        models=candidates,
        metrics=metrics,
        best_model_name=best_model_name,
        selection_reason=selection_reason,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        probabilities=probabilities,
        predictions=predictions,
    )


def calibration_evaluation(result: AnalysisResult, bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate uncalibrated hold-out scores without fitting on the hold-out set."""
    actual = result.y_test.astype(int)
    predicted_risk = result.probabilities.astype(float)
    brier = float(brier_score_loss(actual, predicted_risk))
    training_prevalence = float(result.y_train.mean())
    baseline_brier = float(np.mean((actual.to_numpy() - training_prevalence) ** 2))
    brier_skill = 1.0 - (brier / baseline_brier) if baseline_brier else np.nan

    bin_count = min(bins, int(predicted_risk.nunique()))
    bin_ids = pd.qcut(predicted_risk, q=bin_count, labels=False, duplicates="drop")
    curve = (
        pd.DataFrame({"actual": actual, "predicted_risk": predicted_risk, "bin": bin_ids})
        .groupby("bin", as_index=False, observed=True)
        .agg(
            records=("actual", "size"),
            mean_predicted_risk=("predicted_risk", "mean"),
            observed_attrition_rate=("actual", "mean"),
        )
    )
    curve["calibration_bin"] = np.arange(1, len(curve) + 1)
    curve["absolute_gap"] = (
        curve["mean_predicted_risk"] - curve["observed_attrition_rate"]
    ).abs()
    curve = curve[
        [
            "calibration_bin",
            "records",
            "mean_predicted_risk",
            "observed_attrition_rate",
            "absolute_gap",
        ]
    ]
    weighted_gap = float(np.average(curve["absolute_gap"], weights=curve["records"]))
    weak_calibration = bool(brier_skill <= 0 or weighted_gap > 0.10)
    if weak_calibration:
        interpretation = (
            "Calibration is weak on the held-out test set. Treat the scores as relative risk estimates, "
            "not literal probabilities."
        )
    else:
        interpretation = (
            "Calibration is reasonably close on the held-out test set, but each score remains a model "
            "estimate rather than a guaranteed probability."
        )
    metrics = pd.DataFrame(
        [
            {
                "model": result.best_model_name,
                "holdout_records": len(actual),
                "brier_score": brier,
                "baseline_brier_score": baseline_brier,
                "brier_skill_score": brier_skill,
                "mean_absolute_calibration_gap": weighted_gap,
                "calibration_assessment": "Weak" if weak_calibration else "Reasonably close",
                "interpretation": interpretation,
            }
        ]
    )
    return metrics, curve


def fairness_by_group(result: AnalysisResult, source_frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Compare error rates across a demographic group on the held-out data.

    This is a diagnostic, not evidence that the model is fair. Small groups are
    flagged rather than interpreted as reliable evidence.
    """
    if group_column not in source_frame.columns:
        return pd.DataFrame()

    review = pd.DataFrame(
        {
            "group": source_frame.loc[result.x_test.index, group_column],
            "actual": result.y_test,
            "predicted": result.predictions,
        }
    )
    rows: list[dict[str, float | int | str]] = []
    for group, subset in review.groupby("group", dropna=False):
        actual = subset["actual"].to_numpy()
        predicted = subset["predicted"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
        group_size = len(subset)
        rows.append(
            {
                "group": str(group),
                "records": group_size,
                "predicted_high_risk_rate": float(np.mean(predicted)),
                "recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else np.nan,
                "interpretation": "Small group: interpret cautiously" if group_size < MIN_GROUP_SIZE else "Review group differences",
            }
        )
    return pd.DataFrame(rows).sort_values("records", ascending=False).reset_index(drop=True)


def global_feature_importance(result: AnalysisResult) -> pd.DataFrame:
    """Return model-agnostic feature importance from serial held-out permutations."""
    best_model = result.models[result.best_model_name]
    baseline_f1 = f1_score(result.y_test, best_model.predict(result.x_test), zero_division=0)
    random_generator = np.random.default_rng(RANDOM_STATE)
    importance_values: list[np.ndarray] = []
    for feature in result.x_test.columns:
        decreases: list[float] = []
        for _ in range(PERMUTATION_REPEATS):
            shuffled = result.x_test.copy()
            shuffled[feature] = random_generator.permutation(shuffled[feature].to_numpy())
            shuffled_f1 = f1_score(result.y_test, best_model.predict(shuffled), zero_division=0)
            decreases.append(float(baseline_f1 - shuffled_f1))
        importance_values.append(np.asarray(decreases))

    importance_mean = np.asarray([values.mean() for values in importance_values])
    importance_std = np.asarray([values.std() for values in importance_values])
    return (
        pd.DataFrame(
            {
                "feature": result.x_test.columns,
                "importance_mean": importance_mean,
                "importance_std": importance_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
