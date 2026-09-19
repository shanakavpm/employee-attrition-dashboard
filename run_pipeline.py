"""Create reproducible data and modelling outputs for the Streamlit dashboard."""

from __future__ import annotations

import json

from src.config import FIGURE_DIR, OUTPUT_DIR, PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET_COLUMN
from src.data import create_features, load_and_prepare_data
from src.modeling import (
    calibration_evaluation,
    fairness_by_group,
    global_feature_importance,
    preprocessing_ablation,
    train_and_evaluate,
)
from src.reporting import generate_evidence_figures
from src.dashboard_data import save_dashboard_data


def main() -> None:
    """Run the complete pipeline from original data to saved evidence files."""
    prepared = load_and_prepare_data(RAW_DATA_PATH)
    frame = create_features(prepared.frame)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame.to_csv(PROCESSED_DATA_PATH, index=False)
    prepared.quality_report.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

    result = train_and_evaluate(frame, TARGET_COLUMN)
    result.metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    preprocessing_ablation(frame, TARGET_COLUMN).to_csv(OUTPUT_DIR / "preprocessing_ablation.csv", index=False)
    importance = global_feature_importance(result)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    fairness_by_group(result, frame, "gender").to_csv(OUTPUT_DIR / "fairness_by_gender.csv", index=False)
    fairness_by_group(result, frame, "age_group").to_csv(OUTPUT_DIR / "fairness_by_age_group.csv", index=False)
    calibration_metrics, calibration_curve = calibration_evaluation(result)
    calibration_metrics.to_csv(OUTPUT_DIR / "calibration_metrics.csv", index=False)
    calibration_curve.to_csv(OUTPUT_DIR / "calibration_curve.csv", index=False)
    generate_evidence_figures(frame, result, calibration_curve, FIGURE_DIR).to_csv(
        OUTPUT_DIR / "figure_index.csv", index=False
    )

    with (OUTPUT_DIR / "run_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "best_model": result.best_model_name,
                "selection_method": "5-fold, 5-repeat stratified cross-validation on training data",
                "selection_reason": result.selection_reason,
                "holdout_role": "Final evaluation only",
                "records": int(len(frame)),
                "features": int(result.x_train.shape[1]),
            },
            output_file,
            indent=2,
        )
    risk_review_columns = [
        "id",
        "department",
        "jobtitle",
        "gender",
        "age_group",
        "job_satisfaction",
        "overtime",
        "years_experience",
        "salary_band",
        "attrition_label",
    ]
    risk_review = frame.loc[result.x_test.index, risk_review_columns].copy()
    risk_review["attrition_risk"] = result.probabilities
    save_dashboard_data(
        OUTPUT_DIR / "dashboard_data.json",
        best_model_name=result.best_model_name,
        selection_reason=result.selection_reason,
        frame=frame, quality_report=prepared.quality_report, metrics=result.metrics,
        risk_review=risk_review, importance=importance,
        gender_fairness=fairness_by_group(result, frame, "gender"),
        age_fairness=fairness_by_group(result, frame, "age_group"),
        calibration_metrics=calibration_metrics,
        calibration_curve=calibration_curve,
    )
    print(f"Pipeline complete. Evidence files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
