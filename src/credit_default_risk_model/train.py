"""Train, calibrate, select threshold and persist the credit model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import MODEL_FEATURES, SENSITIVE, TARGET, load_data
from .modeling import (
    MODEL_ARTIFACT_VERSION,
    build_model,
    choose_cost_threshold,
    classification_metrics,
    fairness_diagnostics,
)


def train(
    input_path: str | Path,
    output_dir: str | Path,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
    seed: int = 42,
) -> dict[str, object]:
    data = load_data(input_path)
    train_validation, test = train_test_split(
        data, test_size=0.2, stratify=data[TARGET], random_state=seed
    )
    train_frame, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=train_validation[TARGET],
        random_state=seed,
    )
    model = build_model(seed)
    model.fit(train_frame[MODEL_FEATURES], train_frame[TARGET])
    validation_probability = model.predict_proba(validation[MODEL_FEATURES])[:, 1]
    threshold, validation_cost = choose_cost_threshold(
        validation[TARGET], validation_probability, false_negative_cost, false_positive_cost
    )
    test_probability = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    test_metrics = classification_metrics(
        test[TARGET], test_probability, threshold, false_negative_cost, false_positive_cost
    )
    baseline_probability = np.full(len(test), train_frame[TARGET].mean())
    baseline = classification_metrics(
        test[TARGET],
        baseline_probability,
        threshold=0.5,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    fairness = fairness_diagnostics(test[TARGET], test_probability, test[SENSITIVE], threshold)
    metrics: dict[str, object] = {
        "split": {"train": len(train_frame), "validation": len(validation), "test": len(test)},
        "seed": seed,
        "costs": {"false_negative": false_negative_cost, "false_positive": false_positive_cost},
        "validation_selected_threshold": threshold,
        "validation_mean_cost": validation_cost,
        "test": test_metrics,
        "constant_probability_baseline": baseline,
        "fairness_by_sex": fairness,
        "data_note": "Metrics describe the supplied file; smoke data are not real evaluation.",
    }
    artifact = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "model": model,
        "threshold": threshold,
        "model_features": MODEL_FEATURES,
        "target": TARGET,
        "sensitive_attribute": SENSITIVE,
        "false_negative_cost": false_negative_cost,
        "false_positive_cost": false_positive_cost,
        "seed": seed,
    }
    predictions = pd.DataFrame(
        {
            "ID": test["ID"].to_numpy(),
            "target": test[TARGET].to_numpy(),
            "probability": test_probability,
            "prediction": (test_probability >= threshold).astype(int),
            "SEX": test[SENSITIVE].to_numpy(),
        }
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output / "model.joblib")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated credit-default model")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--false-negative-cost", type=float, default=5.0)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = train(
        args.input,
        args.output_dir,
        args.false_negative_cost,
        args.false_positive_cost,
        args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
