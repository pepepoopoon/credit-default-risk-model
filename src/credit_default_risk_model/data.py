"""Credit dataset schema and deterministic smoke generator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{month}" for month in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{month}" for month in range(1, 7)]
CATEGORICAL_FEATURES = ["EDUCATION", "MARRIAGE"]
NUMERIC_FEATURES = ["LIMIT_BAL", "AGE", *PAY_STATUS, *BILL_COLUMNS, *PAYMENT_COLUMNS]
MODEL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
TARGET = "default"
SENSITIVE = "SEX"
REQUIRED_COLUMNS = {"ID", SENSITIVE, TARGET, *MODEL_FEATURES}
UCI_ALIASES = {
    **{f"X{index}": name for index, name in enumerate(
        [
            "LIMIT_BAL",
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "AGE",
            *PAY_STATUS,
            *BILL_COLUMNS,
            *PAYMENT_COLUMNS,
        ],
        start=1,
    )},
    "Y": TARGET,
    "default payment next month": TARGET,
    "default.payment.next.month": TARGET,
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common UCI-exported names to the documented canonical schema."""
    aliases = {name: UCI_ALIASES.get(str(name).strip(), name) for name in frame.columns}
    return frame.rename(columns=aliases)


def validate_schema(frame: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Validate identifiers, feature types, and optional binary target."""
    clean = normalize_columns(frame.copy())
    required = REQUIRED_COLUMNS if require_target else REQUIRED_COLUMNS.difference({TARGET})
    missing = sorted(required.difference(clean.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if clean.empty:
        raise ValueError("Credit data is empty")
    if clean["ID"].isna().any() or clean["ID"].duplicated().any():
        raise ValueError("ID must be non-null and unique")
    for name in [SENSITIVE, *MODEL_FEATURES]:
        parsed = pd.to_numeric(clean[name], errors="coerce")
        if (clean[name].notna() & parsed.isna()).any():
            raise ValueError(f"{name} contains non-numeric values")
        clean[name] = parsed
    if clean[SENSITIVE].isna().any():
        raise ValueError("SEX must be non-null for group diagnostics")
    if clean[MODEL_FEATURES].isna().all(axis=0).any():
        empty = clean[MODEL_FEATURES].columns[clean[MODEL_FEATURES].isna().all()].tolist()
        raise ValueError(f"Features contain entirely missing columns: {empty}")
    if require_target:
        target = pd.to_numeric(clean[TARGET], errors="coerce")
        if target.isna().any() or not set(target.unique()).issubset({0, 1}):
            raise ValueError("default must contain only binary 0/1 values")
        clean[TARGET] = target.astype(int)
        if clean[TARGET].nunique() != 2:
            raise ValueError("default must contain both classes")
    return clean


def load_data(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    return validate_schema(pd.read_csv(path), require_target=require_target)


def make_smoke_data(rows: int = 360, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic, non-realistic credit-shaped dataset."""
    if rows < 120:
        raise ValueError("At least 120 rows are required for a stable smoke split")
    rng = np.random.default_rng(seed)
    limit_balance = rng.choice([20_000, 50_000, 100_000, 200_000, 400_000], size=rows)
    sex = rng.choice([1, 2], size=rows)
    education = rng.choice([1, 2, 3, 4], p=[0.2, 0.45, 0.3, 0.05], size=rows)
    marriage = rng.choice([1, 2, 3], p=[0.45, 0.5, 0.05], size=rows)
    age = rng.integers(21, 70, size=rows)
    latent = rng.normal(0, 1, size=rows)
    frame = pd.DataFrame(
        {
            "ID": np.arange(1, rows + 1),
            "LIMIT_BAL": limit_balance,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
        }
    )
    for index, name in enumerate(PAY_STATUS):
        score = latent + rng.normal(0, 0.8, size=rows) - index * 0.05
        frame[name] = np.select([score < -0.8, score < 0.5, score < 1.4], [-1, 0, 1], default=2)
    utilization = np.clip(0.35 + 0.22 * latent + rng.normal(0, 0.12, rows), 0.02, 1.3)
    for month, name in enumerate(BILL_COLUMNS):
        drift = np.clip(1 - month * 0.04 + rng.normal(0, 0.05, rows), 0.5, 1.2)
        frame[name] = np.round(limit_balance * utilization * drift, 2)
    for month, name in enumerate(PAYMENT_COLUMNS):
        ratio = np.clip(0.2 - 0.04 * latent + rng.normal(0, 0.04, rows), 0.01, 0.6)
        frame[name] = np.round(frame[BILL_COLUMNS[month]] * ratio, 2)
    logit = -1.7 + 0.9 * latent + 0.55 * (frame["PAY_0"] > 0) + 0.45 * utilization
    probability = 1 / (1 + np.exp(-logit))
    frame[TARGET] = rng.binomial(1, probability)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic credit smoke data")
    parser.add_argument("--output", type=Path, default=Path("data/smoke.csv"))
    parser.add_argument("--rows", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    make_smoke_data(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"Wrote synthetic credit data to {args.output}")


if __name__ == "__main__":
    main()
