"""Evaluate a persisted credit model on separately supplied labeled data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .data import SENSITIVE, TARGET, load_data
from .modeling import classification_metrics, fairness_diagnostics, validate_model_artifact


def evaluate(input_path: str | Path, artifact_path: str | Path) -> dict[str, object]:
    data = load_data(input_path)
    artifact = validate_model_artifact(joblib.load(artifact_path))
    probability = artifact["model"].predict_proba(data[artifact["model_features"]])[:, 1]
    metrics = classification_metrics(
        data[TARGET],
        probability,
        artifact["threshold"],
        artifact["false_negative_cost"],
        artifact["false_positive_cost"],
    )
    metrics["fairness_by_sex"] = fairness_diagnostics(
        data[TARGET], probability, data[SENSITIVE], artifact["threshold"]
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a persisted credit-default model")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input, args.artifact), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
