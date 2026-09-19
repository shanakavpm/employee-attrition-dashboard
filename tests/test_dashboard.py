"""Regression checks for serving saved results without online training."""
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.dashboard_data import load_dashboard_data


class DashboardTests(unittest.TestCase):
    def test_saved_results_match_report(self):
        results = load_dashboard_data(OUTPUT_DIR / "dashboard_data.json")
        for key, filename in (("metrics", "model_metrics"), ("importance", "feature_importance"),
                              ("gender_fairness", "fairness_by_gender"), ("age_fairness", "fairness_by_age_group")):
            pd.testing.assert_frame_equal(results[key], pd.read_csv(OUTPUT_DIR / f"{filename}.csv"), check_dtype=False)
        risk = results["risk_review"]
        self.assertEqual(len(risk), 298)
        self.assertTrue(risk["id"].is_unique)
        self.assertTrue(risk["attrition_risk"].between(0, 1).all())
        self.assertEqual(results["best_model_name"], results["metrics"].iloc[0]["model"])

    def test_interactions_do_not_train_or_compute_importance(self):
        with patch("src.modeling.train_and_evaluate", side_effect=AssertionError("Online training")), \
             patch("src.modeling.global_feature_importance", side_effect=AssertionError("Online permutation")):
            app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "1,191")
            self.assertEqual(len(app.sidebar.slider), 1)
            explanation = app.main.info[0].value
            self.assertIn("model estimate, not confirmation", explanation)
            self.assertIn("Left** and **Stayed", explanation)
            self.assertIn("at or above the selected threshold", explanation)
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


if __name__ == "__main__":
    unittest.main()
