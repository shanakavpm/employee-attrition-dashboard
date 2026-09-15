"""Predictive modelling, evaluation, fairness checks, and explanations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CV_FOLDS, MIN_GROUP_SIZE, RANDOM_STATE, TEST_SIZE
from src.data import model_features


@dataclass
class AnalysisResult:
    """All objects needed to report and visualise a reproducible model run."""

    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    best_model_name: str
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
                        n_jobs=-1,
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
        cv_result = cross_validate(pipeline, x_train, y_train, cv=cv, scoring={"f1": "f1", "roc_auc": "roc_auc"}, n_jobs=-1)
        pipeline.fit(x_train, y_train)
        row = _metric_row(variant, pipeline, x_test, y_test)
        row["numeric_standardisation"] = scale_numeric
        row["cv_f1_mean"] = float(np.mean(cv_result["test_f1"]))
        row["cv_f1_std"] = float(np.std(cv_result["test_f1"], ddof=1))
        row["cv_roc_auc_mean"] = float(np.mean(cv_result["test_roc_auc"]))
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


def train_and_evaluate(frame: pd.DataFrame, target_column: str) -> AnalysisResult:
    """Train both models and select the best by hold-out F1 score.

    F1 is explicit because missed leavers and unnecessary retention actions both
    matter. Cross-validation means and standard deviations show model stability.
    """
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
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | str]] = []

    for name, pipeline in candidates.items():
        cv_result = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipeline.fit(x_train, y_train)
        row = _metric_row(name, pipeline, x_test, y_test)
        row["cv_f1_mean"] = float(np.mean(cv_result["test_f1"]))
        row["cv_f1_std"] = float(np.std(cv_result["test_f1"], ddof=1))
        row["cv_roc_auc_mean"] = float(np.mean(cv_result["test_roc_auc"]))
        row["cv_roc_auc_std"] = float(np.std(cv_result["test_roc_auc"], ddof=1))
        rows.append(row)

    metrics = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["model"])
    best_model = candidates[best_model_name]
    probabilities = pd.Series(best_model.predict_proba(x_test)[:, 1], index=x_test.index, name="attrition_risk")
    predictions = pd.Series(best_model.predict(x_test), index=x_test.index, name="predicted_attrition")
    return AnalysisResult(
        models=candidates,
        metrics=metrics,
        best_model_name=best_model_name,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        probabilities=probabilities,
        predictions=predictions,
    )


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
    """Return model-agnostic feature importance from held-out permutation tests."""
    best_model = result.models[result.best_model_name]
    importance = permutation_importance(
        best_model,
        result.x_test,
        result.y_test,
        scoring="f1",
        n_repeats=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": result.x_test.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
