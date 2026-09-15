"""Create reproducible data and modelling outputs for the Streamlit dashboard."""

from __future__ import annotations

import json

from src.config import FIGURE_DIR, OUTPUT_DIR, PROCESSED_DATA_PATH, RAW_DATA_PATH, TARGET_COLUMN
from src.data import create_features, load_and_prepare_data
from src.modeling import fairness_by_group, global_feature_importance, preprocessing_ablation, train_and_evaluate
from src.reporting import generate_evidence_figures


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
    global_feature_importance(result).to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    fairness_by_group(result, frame, "gender").to_csv(OUTPUT_DIR / "fairness_by_gender.csv", index=False)
    fairness_by_group(result, frame, "age_group").to_csv(OUTPUT_DIR / "fairness_by_age_group.csv", index=False)
    generate_evidence_figures(frame, result, FIGURE_DIR).to_csv(OUTPUT_DIR / "figure_index.csv", index=False)

    with (OUTPUT_DIR / "run_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "best_model": result.best_model_name,
                "selection_metric": "hold-out F1 score",
                "records": int(len(frame)),
                "features": int(result.x_train.shape[1]),
            },
            output_file,
            indent=2,
        )
    print(f"Pipeline complete. Evidence files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
