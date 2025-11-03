"""Batch credit-risk scoring CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import load_data
from .modeling import validate_model_artifact


def predict(input_path: str | Path, artifact_path: str | Path) -> pd.DataFrame:
    data = load_data(input_path, require_target=False)
    artifact = validate_model_artifact(joblib.load(artifact_path))
    probability = artifact["model"].predict_proba(data[artifact["model_features"]])[:, 1]
    return pd.DataFrame(
        {
            "ID": data["ID"].to_numpy(),
            "default_probability": probability,
            "risk_flag": (probability >= artifact["threshold"]).astype(int),
            "threshold": artifact["threshold"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score clients with a persisted credit model")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict(args.input, args.artifact)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Wrote {len(result)} scores to {args.output}")
    else:
        print(result.to_csv(index=False))


if __name__ == "__main__":
    main()
