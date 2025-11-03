from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd

from credit_default_risk_model.data import (
    MODEL_FEATURES,
    SENSITIVE,
    TARGET,
    make_smoke_data,
    validate_schema,
)
from credit_default_risk_model.evaluate import evaluate
from credit_default_risk_model.modeling import (
    MODEL_ARTIFACT_VERSION,
    ModelArtifactError,
    choose_cost_threshold,
)
from credit_default_risk_model.predict import predict
from credit_default_risk_model.train import train


class CreditRiskPipelineTest(unittest.TestCase):
    def test_smoke_data_is_deterministic(self) -> None:
        pd.testing.assert_frame_equal(make_smoke_data(180), make_smoke_data(180))

    def test_schema_rejects_duplicate_id(self) -> None:
        data = make_smoke_data(120)
        data.loc[1, "ID"] = data.loc[0, "ID"]
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_schema(data)

    def test_cost_threshold_uses_exact_observed_score_boundary(self) -> None:
        threshold, mean_cost = choose_cost_threshold(
            pd.Series([1, 0]),
            probability=pd.Series([0.504, 0.503]).to_numpy(),
            false_negative_cost=1.0,
            false_positive_cost=1.0,
        )

        self.assertAlmostEqual(threshold, 0.504)
        self.assertEqual(mean_cost, 0.0)

    def test_predict_rejects_stale_artifact_feature_schema(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.csv"
            artifact_path = root / "stale.joblib"
            make_smoke_data(120).drop(columns=[TARGET]).head(3).to_csv(input_path, index=False)
            joblib.dump(
                {
                    "artifact_version": MODEL_ARTIFACT_VERSION,
                    "model": object(),
                    "model_features": list(reversed(MODEL_FEATURES)),
                    "target": TARGET,
                    "sensitive_attribute": SENSITIVE,
                    "threshold": 0.5,
                },
                artifact_path,
            )

            with self.assertRaisesRegex(ModelArtifactError, "feature schema"):
                predict(input_path, artifact_path)

    def test_train_evaluate_predict_and_diagnostics(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "credit.csv"
            output = root / "artifacts"
            make_smoke_data(240).to_csv(input_path, index=False)

            metrics = train(input_path, output)
            evaluation = evaluate(input_path, output / "model.joblib")
            scored = predict(input_path, output / "model.joblib")

            self.assertTrue((output / "test_predictions.csv").exists())
            self.assertIn("brier_score", metrics["test"])
            self.assertIn("gaps", metrics["fairness_by_sex"])
            self.assertIn("calibration_bins", evaluation)
            self.assertEqual(len(scored), 240)
            self.assertTrue(scored["default_probability"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
