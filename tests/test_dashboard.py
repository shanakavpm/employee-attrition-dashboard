"""Regression checks for serving saved results without online training."""
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.dashboard_data import load_dashboard_data
from src.dashboard_filters import FILTER_DEFINITIONS, apply_filters, filter_options


class DashboardTests(unittest.TestCase):
    def test_saved_results_match_report(self):
        results = load_dashboard_data(OUTPUT_DIR / "dashboard_data.json")
        for key, filename in (("metrics", "model_metrics"), ("importance", "feature_importance"),
                              ("gender_fairness", "fairness_by_gender"), ("age_fairness", "fairness_by_age_group"),
                              ("calibration_metrics", "calibration_metrics"), ("calibration_curve", "calibration_curve")):
            pd.testing.assert_frame_equal(results[key], pd.read_csv(OUTPUT_DIR / f"{filename}.csv"), check_dtype=False)
        risk = results["risk_review"]
        self.assertEqual(len(risk), 298)
        self.assertTrue(risk["id"].is_unique)
        self.assertTrue(risk["attrition_risk"].between(0, 1).all())
        self.assertEqual(results["best_model_name"], results["metrics"].iloc[0]["model"])
        self.assertEqual(set(results["metrics"]["cv_evaluations"]), {25})
        self.assertEqual(int(results["calibration_curve"]["records"].sum()), 298)

    def test_all_supported_filters_individually_and_in_combination(self):
        frame = load_dashboard_data(OUTPUT_DIR / "dashboard_data.json")["frame"]
        first_record = frame.iloc[0]
        combined: dict[str, list[object]] = {}
        for _, _, column in FILTER_DEFINITIONS:
            value = first_record[column]
            filtered = apply_filters(frame, {column: [value]})
            self.assertFalse(filtered.empty, column)
            self.assertTrue(filtered[column].eq(value).all(), column)
            combined[column] = [value]
        combined_result = apply_filters(frame, combined)
        self.assertIn(first_record["id"], combined_result["id"].values)
        self.assertEqual(filter_options(frame, "years_experience")[0], "Less than 5 years")
        self.assertEqual(filter_options(frame, "salary_band")[0], "Less than 5000 SAR")

    def test_interactions_do_not_train_or_compute_importance(self):
        with patch("src.modeling.train_and_evaluate", side_effect=AssertionError("Online training")), \
             patch("src.modeling.global_feature_importance", side_effect=AssertionError("Online permutation")):
            app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "1,191")
            self.assertEqual(len(app.multiselect), len(FILTER_DEFINITIONS))
            self.assertEqual(len(app.sidebar.slider), 1)
            self.assertIn("Lowering the threshold", app.sidebar.slider[0].help)
            explanation = app.main.info[0].value
            self.assertIn("model estimate, not confirmation", explanation)
            self.assertIn("Left** and **Stayed", explanation)
            self.assertIn("at or above the selected threshold", explanation)
            self.assertIn("must not be used alone", explanation)
            app.multiselect[0].select("Accounting").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "64")
            self.assertEqual(app.metric[2].value, "60.9%")
            app.button[0].click().run()
            for threshold, count in ((0.3, "165"), (0.8, "74")):
                app.slider[0].set_value(threshold).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[4].value, count)
                risk = app.dataframe[1].value
                self.assertTrue((risk.loc[risk.risk_category.eq("High risk"), "attrition_risk"] >= threshold * 100).all())
                self.assertEqual(app.dataframe[2].value.records.sum(), 298)

    def test_empty_and_small_filter_results_are_safe(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
        app.multiselect[0].select("Accounting").run()
        app.multiselect[3].select("Teacher").run()
        self.assertFalse(app.exception)
        self.assertIn("No records match", app.warning[0].value)
        app.button[0].click().run()
        self.assertEqual(app.metric[0].value, "1,191")

        app.multiselect[3].select("Cybersecurity director").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "1")
        self.assertIn("should be interpreted cautiously", app.warning[0].value)


if __name__ == "__main__":
    unittest.main()
