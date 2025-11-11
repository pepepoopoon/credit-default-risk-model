"""Run a reproducible synthetic credit-risk sensitivity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    SENSITIVE,
    TARGET,
    make_smoke_data,
    validate_schema,
)
from .modeling import (
    build_model,
    choose_cost_threshold,
    classification_metrics,
    fairness_diagnostics,
)


def prepare_stress_frame(
    frame: pd.DataFrame,
    *,
    seed: int,
    missing_rate: float,
    prevalence_multiplier: float,
) -> pd.DataFrame:
    """Apply deterministic missingness and prevalence shifts to synthetic data."""
    if not 0 <= missing_rate < 0.5:
        raise ValueError("missing_rate must be in [0, 0.5)")
    if prevalence_multiplier <= 0:
        raise ValueError("prevalence_multiplier must be positive")
    rng = np.random.default_rng(seed + 50_021)
    result = frame.copy()
    if missing_rate:
        for column in MODEL_FEATURES:
            result.loc[rng.random(len(result)) < missing_rate, column] = np.nan

    current_rate = float(result[TARGET].mean())
    target_rate = float(np.clip(current_rate * prevalence_multiplier, 0.05, 0.75))
    positives = result[result[TARGET] == 1]
    negatives = result[result[TARGET] == 0]
    if target_rate < current_rate:
        keep_positive = min(len(positives), round(len(negatives) * target_rate / (1 - target_rate)))
        positives = positives.sample(n=max(2, keep_positive), random_state=seed)
    elif target_rate > current_rate:
        keep_negative = min(len(negatives), round(len(positives) * (1 - target_rate) / target_rate))
        negatives = negatives.sample(n=max(2, keep_negative), random_state=seed)
    combined = pd.concat([positives, negatives])
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def inject_unseen_categories(
    frame: pd.DataFrame, *, rate: float, seed: int
) -> pd.DataFrame:
    if not 0 <= rate < 0.5:
        raise ValueError("unseen_category_rate must be in [0, 0.5)")
    result = frame.copy()
    rng = np.random.default_rng(seed + 60_013)
    for column in CATEGORICAL_FEATURES:
        result.loc[rng.random(len(result)) < rate, column] = 99
    return result


def run_experiment(
    *,
    rows: int,
    data_seed: int,
    split_seed: int,
    model_seed: int,
    false_negative_cost: float,
    false_positive_cost: float,
    calibration_method: str,
    hypothesis: str,
    missing_rate: float = 0.0,
    unseen_category_rate: float = 0.0,
    prevalence_multiplier: float = 1.0,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    """Train a calibrated model and record validation, test and group diagnostics."""
    if not hypothesis.strip():
        raise ValueError("hypothesis must not be empty")
    frame = validate_schema(
        prepare_stress_frame(
            make_smoke_data(rows=rows, seed=data_seed),
            seed=data_seed,
            missing_rate=missing_rate,
            prevalence_multiplier=prevalence_multiplier,
        )
    )
    train_validation, test = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame[TARGET],
        random_state=split_seed,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        stratify=train_validation[TARGET],
        random_state=split_seed,
    )
    validation = inject_unseen_categories(
        validation,
        rate=unseen_category_rate,
        seed=data_seed + split_seed,
    )
    test = inject_unseen_categories(
        test,
        rate=unseen_category_rate,
        seed=data_seed + split_seed + 1,
    )

    model = build_model(model_seed, calibration_method=calibration_method)
    model.fit(train[MODEL_FEATURES], train[TARGET])
    validation_probability = model.predict_proba(validation[MODEL_FEATURES])[:, 1]
    threshold, validation_cost = choose_cost_threshold(
        validation[TARGET],
        validation_probability,
        false_negative_cost,
        false_positive_cost,
    )
    test_probability = model.predict_proba(test[MODEL_FEATURES])[:, 1]
    test_metrics = classification_metrics(
        test[TARGET],
        test_probability,
        threshold,
        false_negative_cost,
        false_positive_cost,
    )
    constant_probability = np.full(len(test), train[TARGET].mean())
    constant_threshold, _ = choose_cost_threshold(
        test[TARGET],
        constant_probability,
        false_negative_cost,
        false_positive_cost,
    )
    constant_metrics = classification_metrics(
        test[TARGET],
        constant_probability,
        constant_threshold,
        false_negative_cost,
        false_positive_cost,
    )

    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "synthetic_credit_sensitivity",
        "hypothesis": hypothesis.strip(),
        "parameters": {
            "rows": rows,
            "data_seed": data_seed,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost,
            "calibration_method": calibration_method,
            "missing_rate": missing_rate,
            "unseen_category_rate": unseen_category_rate,
            "prevalence_multiplier": prevalence_multiplier,
        },
        "dataset": {
            "mode": "synthetic",
            "rows": len(frame),
            "default_rate": float(frame[TARGET].mean()),
            "missing_feature_values": int(frame[MODEL_FEATURES].isna().sum().sum()),
            "unseen_category_values": int(
                (validation[CATEGORICAL_FEATURES] == 99).sum().sum()
                + (test[CATEGORICAL_FEATURES] == 99).sum().sum()
            ),
            "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        },
        "validation": {
            "selected_threshold": threshold,
            "mean_cost": validation_cost,
        },
        "test": test_metrics,
        "constant_probability_baseline": constant_metrics,
        "group_diagnostics": fairness_diagnostics(
            test[TARGET], test_probability, test[SENSITIVE], threshold
        ),
    }
    if baseline is not None:
        baseline_test = baseline.get("test")
        baseline_validation = baseline.get("validation")
        if not isinstance(baseline_test, dict) or not isinstance(baseline_validation, dict):
            raise ValueError("baseline does not satisfy the experiment schema")
        result["comparison"] = {
            "test_delta": {
                metric: float(test_metrics[metric]) - float(baseline_test[metric])
                for metric in (
                    "roc_auc",
                    "average_precision",
                    "brier_score",
                    "precision",
                    "recall",
                    "f1",
                    "mean_cost",
                )
            },
            "threshold_delta": threshold - float(baseline_validation["selected_threshold"]),
            "cost_advantage_over_constant": (
                float(constant_metrics["mean_cost"]) - float(test_metrics["mean_cost"])
            ),
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--rows", type=int, default=360)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--false-negative-cost", type=float, default=5.0)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--calibration-method", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--missing-rate", type=float, default=0.0)
    parser.add_argument("--unseen-category-rate", type=float, default=0.0)
    parser.add_argument("--prevalence-multiplier", type=float, default=1.0)
    args = parser.parse_args(argv)

    baseline = None
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    result = run_experiment(
        rows=args.rows,
        data_seed=args.data_seed,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        false_negative_cost=args.false_negative_cost,
        false_positive_cost=args.false_positive_cost,
        calibration_method=args.calibration_method,
        hypothesis=args.hypothesis,
        missing_rate=args.missing_rate,
        unseen_category_rate=args.unseen_category_rate,
        prevalence_multiplier=args.prevalence_multiplier,
        baseline=baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"experiment result written to {args.output}")


if __name__ == "__main__":
    main()
