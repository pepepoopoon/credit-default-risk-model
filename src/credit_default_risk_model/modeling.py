"""Credit model, threshold and diagnostic utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, SENSITIVE, TARGET

MODEL_ARTIFACT_VERSION = 1


class ModelArtifactError(ValueError):
    """Raised when a persisted model is incompatible with the current scoring contract."""


def validate_model_artifact(artifact: object) -> dict[str, object]:
    """Validate versioned feature, target, group, and estimator metadata."""
    if not isinstance(artifact, dict):
        raise ModelArtifactError("Model artifact must be a dictionary")
    if artifact.get("artifact_version") != MODEL_ARTIFACT_VERSION:
        raise ModelArtifactError("Model artifact version is incompatible")
    if artifact.get("model_features") != MODEL_FEATURES:
        raise ModelArtifactError("Model artifact feature schema is incompatible")
    if artifact.get("target") != TARGET or artifact.get("sensitive_attribute") != SENSITIVE:
        raise ModelArtifactError("Model artifact target or sensitive attribute is incompatible")
    if not callable(getattr(artifact.get("model"), "predict_proba", None)):
        raise ModelArtifactError("Model artifact must contain an estimator with predict_proba")
    return artifact


def build_model(seed: int = 42, calibration_method: str = "sigmoid") -> CalibratedClassifierCV:
    if calibration_method not in {"sigmoid", "isotonic"}:
        raise ValueError("calibration_method must be sigmoid or isotonic")
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )
    estimator = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed),
            ),
        ]
    )
    return CalibratedClassifierCV(estimator=estimator, method=calibration_method, cv=3)


def choose_cost_threshold(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    false_negative_cost: float,
    false_positive_cost: float,
) -> tuple[float, float]:
    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("Error costs must be positive")
    target = np.asarray(y_true, dtype=int)
    scores = np.asarray(probability, dtype=float)
    if target.ndim != 1 or scores.ndim != 1 or not len(scores):
        raise ValueError("Target and probability must be non-empty one-dimensional arrays")
    if len(target) != len(scores):
        raise ValueError("Target and probability must have equal length")
    if not np.isfinite(scores).all():
        raise ValueError("Probability must contain only finite values")

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_target = target[order]
    group_end = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    reviewed = np.arange(1, len(scores) + 1)[group_end]
    true_positive = np.cumsum(sorted_target)[group_end]
    false_positive = reviewed - true_positive
    false_negative = target.sum() - true_positive
    thresholds = sorted_scores[group_end]
    costs = (false_negative_cost * false_negative + false_positive_cost * false_positive) / len(
        scores
    )

    no_review_threshold = 1.0 if sorted_scores[0] < 1.0 else np.nextafter(sorted_scores[0], np.inf)
    thresholds = np.r_[no_review_threshold, thresholds]
    costs = np.r_[false_negative_cost * target.sum() / len(scores), costs]
    best = min(range(len(costs)), key=lambda index: (costs[index], -thresholds[index]))
    return float(thresholds[best]), float(costs[best])


def calibration_bins(
    y_true: pd.Series | np.ndarray, probability: np.ndarray, bins: int = 10
) -> list[dict[str, float | int]]:
    work = pd.DataFrame({"target": np.asarray(y_true), "probability": probability})
    work["bin"] = pd.cut(
        work["probability"], bins=np.linspace(0, 1, bins + 1), include_lowest=True, labels=False
    )
    result: list[dict[str, float | int]] = []
    for bin_number, group in work.groupby("bin", observed=True):
        result.append(
            {
                "bin": int(bin_number),
                "count": int(len(group)),
                "mean_probability": float(group["probability"].mean()),
                "observed_rate": float(group["target"].mean()),
            }
        )
    return result


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, object]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "mean_cost": float((false_negative_cost * fn + false_positive_cost * fp) / len(prediction)),
        "calibration_bins": calibration_bins(y_true, probability),
    }


def fairness_diagnostics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    groups: pd.Series,
    threshold: float,
) -> dict[str, object]:
    work = pd.DataFrame(
        {
            "target": np.asarray(y_true),
            "prediction": (probability >= threshold).astype(int),
            "group": groups.astype("string").to_numpy(),
        }
    )
    reports: dict[str, dict[str, float | int | None]] = {}
    for group_name, group in work.groupby("group", observed=True):
        tn, fp, fn, tp = confusion_matrix(
            group["target"], group["prediction"], labels=[0, 1]
        ).ravel()
        reports[str(group_name)] = {
            "count": int(len(group)),
            "positive_rate": float(group["prediction"].mean()),
            "tpr": float(tp / (tp + fn)) if tp + fn else None,
            "fpr": float(fp / (fp + tn)) if fp + tn else None,
        }

    gaps: dict[str, float | None] = {}
    for metric in ("positive_rate", "tpr", "fpr"):
        values = [
            float(report[metric]) for report in reports.values() if report[metric] is not None
        ]
        gaps[f"{metric}_max_gap"] = max(values) - min(values) if len(values) >= 2 else None
    return {"groups": reports, "gaps": gaps, "warning": "Diagnostic only; no fairness conclusion."}
