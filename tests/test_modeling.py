"""Focused checks for repeated validation, selection, and calibration evidence."""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from src.modeling import (
    AnalysisResult,
    _select_model,
    _serial_cross_validation,
    calibration_evaluation,
)


class ModelingTests(unittest.TestCase):
    def test_repeated_cross_validation_is_reproducible(self):
        features = pd.DataFrame(
            {
                "signal": np.linspace(-2, 2, 100),
                "secondary": np.tile([0.0, 1.0], 50),
            }
        )
        target = pd.Series(([0] * 50) + ([1] * 50))

        def evaluate():
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
            return _serial_cross_validation(LogisticRegression(), features, target, cv)

        first = evaluate()
        second = evaluate()
        self.assertEqual(first, second)
        self.assertEqual(len(first["f1"]), 25)
        self.assertEqual(len(first["roc_auc"]), 25)

    def test_selection_uses_cross_validation_not_holdout_f1(self):
        metrics = pd.DataFrame(
            [
                {"model": "Logistic Regression", "f1": 0.70, "cv_f1_mean": 0.80,
                 "cv_f1_std": 0.02, "cv_roc_auc_mean": 0.85},
                {"model": "Random Forest", "f1": 0.99, "cv_f1_mean": 0.75,
                 "cv_f1_std": 0.02, "cv_roc_auc_mean": 0.86},
            ]
        )
        selected, _ = _select_model(metrics)
        self.assertEqual(selected, "Logistic Regression")

    def test_calibration_evidence_is_reproducible(self):
        index = pd.RangeIndex(10)
        result = AnalysisResult(
            models={},
            metrics=pd.DataFrame(),
            best_model_name="Test model",
            selection_reason="Test",
            x_train=pd.DataFrame(index=pd.RangeIndex(20)),
            x_test=pd.DataFrame(index=index),
            y_train=pd.Series(([0] * 12) + ([1] * 8)),
            y_test=pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], index=index),
            probabilities=pd.Series(np.linspace(0.05, 0.95, 10), index=index),
            predictions=pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], index=index),
        )
        first_metrics, first_curve = calibration_evaluation(result, bins=5)
        second_metrics, second_curve = calibration_evaluation(result, bins=5)
        pd.testing.assert_frame_equal(first_metrics, second_metrics)
        pd.testing.assert_frame_equal(first_curve, second_curve)
        self.assertEqual(int(first_curve["records"].sum()), 10)
        self.assertTrue(first_metrics.loc[0, "brier_score"] >= 0)


if __name__ == "__main__":
    unittest.main()
