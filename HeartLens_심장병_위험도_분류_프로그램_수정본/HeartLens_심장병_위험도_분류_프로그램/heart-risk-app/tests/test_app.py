import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risk_engine import load_model, predict, validate_payload


VALID = {
    "age": 58, "sex": 1, "cp": 4, "trestbps": 142, "chol": 240,
    "fbs": 0, "restecg": 0, "thalach": 132, "exang": 1,
    "oldpeak": 1.8, "slope": 2, "ca": 1, "thal": 7, "acute": {},
}


class RiskEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()

    def test_model_shape(self):
        p = self.model["preprocessor"]
        self.assertEqual(len(p["feature_names"]), len(self.model["coefficients"]))
        self.assertEqual(self.model["training_summary"]["rows"], 920)

    def test_valid_prediction(self):
        result = predict(VALID, self.model)
        self.assertTrue(result["ok"])
        self.assertGreaterEqual(result["probability"], 0)
        self.assertLessEqual(result["probability"], 1)
        self.assertIn(result["tier"], {"low", "attention", "high"})
        self.assertGreaterEqual(len(result["factors"]), 3)

    def test_missing_optional_is_allowed(self):
        payload = dict(VALID, slope=None, ca=None, thal=None)
        result = predict(payload, self.model)
        self.assertTrue(result["ok"])
        self.assertEqual(result["input_quality"]["optional_missing"], 3)

    def test_out_of_range_rejected(self):
        payload = dict(VALID, age=8, trestbps=900)
        result = predict(payload, self.model)
        self.assertFalse(result["ok"])
        self.assertTrue(any("age" in x for x in result["errors"]))
        self.assertTrue(any("trestbps" in x for x in result["errors"]))

    def test_emergency_overrides_display_warning(self):
        payload = dict(VALID, acute={"chest_pain_now": True})
        result = predict(payload, self.model)
        self.assertTrue(result["emergency"])
        self.assertIn("119", result["warnings"][0])

    def test_evaluation_contains_all_sites(self):
        report = json.loads((ROOT / "model" / "evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(report["training_rows"], 920)
        self.assertEqual(len(report["leave_one_site_out"]), 4)
        self.assertGreater(report["oof_metrics_at_0_5"]["auroc"], 0.75)


if __name__ == "__main__":
    unittest.main()
