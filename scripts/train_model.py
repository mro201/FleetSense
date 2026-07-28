"""
Train the baseline Random Forest model from a dataset CSV.

Usage:
    uv run scripts/train_model.py
    uv run scripts/train_model.py --data data/dataset/vessel_weekly_features.csv --start 2026-06-01 --end 2026-09-01
"""

import argparse
from pathlib import Path
from datetime import date, timezone, datetime
import polars as pl
from sklearn.model_selection import train_test_split
import pandas as pd
from fleetsense.features.data_loader import FEATURES, TARGET_COLUMN, get_dataset
from fleetsense.model.base_model import RANDOM_STATE, evaluate_model, train_baseline
from fleetsense.monitoring.distribution_monitoring import build_baselines, save_baselines
import json

ROOT = Path(__file__).parent.parent
LAST_TRAINING_PATH = ROOT / "fleetsense" / "outputs" / "last_training.json"
DEFAULT_DATA_PATH = ROOT / "data" / "dataset" / "vessel_weekly_features_sample.csv"
DEFAULT_START = date(2025, 6, 1)
DEFAULT_END = date(2025, 9, 1)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FleetSense baseline model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to training data CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--start",
        type=parse_date,
        default=DEFAULT_START,
        help=f"start parse_date for training data (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=DEFAULT_END,
        help=f"end date for training data (default: {DEFAULT_END})",
    )
    return parser.parse_args()


def save_training_metadata(start: date, end: date) -> None:
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_start": start.isoformat(),
        "data_end": end.isoformat(),
    }
    LAST_TRAINING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_TRAINING_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Training metadata saved to {LAST_TRAINING_PATH}")


def train(start: date = DEFAULT_START, end: date = DEFAULT_END, data_path: Path = DEFAULT_DATA_PATH) -> None:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Run scripts/generate_dataset.py and scripts/sample_dataset.py first, "
            "or pass --data with a valid path."
        )

    print(f"Loading dataset from {data_path} ...")
    df = get_dataset(data_path).to_pandas()
    df = get_dataset(data_path).to_pandas()

    date_column = "timestamp"  # adjust to whatever your actual date/timestamp column is called
    df[date_column] = pd.to_datetime(df[date_column]).dt.date
    df = df[(df[date_column] >= start) & (df[date_column] <= end)]

    if df.empty:
        raise ValueError(
            f"No data found in range {start} to {end}. " "Check the date range or the dataset's date coverage."
        )

    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # PSI baseline
    baseline_data = pl.from_pandas(X_train.assign(**{TARGET_COLUMN: y_train}))
    psi_baseline = build_baselines(baseline_data, FEATURES, class_col=None)
    save_baselines(psi_baseline)

    # train model
    print("Training baseline model ...")
    model = train_baseline(X_train, y_train)

    print("Evaluating ...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 (macro): {metrics['f1_macro']:.3f}")
    print(metrics["report"])

    save_training_metadata(start, end)


if __name__ == "__main__":
    args = parse_args()

    train(args.start, args.end, args.data)
