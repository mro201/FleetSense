"""
Train the baseline Random Forest model from a dataset CSV.

Usage:
    uv run scripts/train_model.py
    uv run scripts/train_model.py --data data/dataset/vessel_weekly_features.csv
"""

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from fleetsense.features.data_loader import FEATURES, TARGET_COLUMN, get_dataset
from fleetsense.model.base_model import RANDOM_STATE, evaluate_model, train_baseline

ROOT = Path(__file__).parent.parent

DEFAULT_DATA_PATH = ROOT / "data" / "dataset" / "vessel_weekly_features_sample.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FleetSense baseline model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to training data CSV (default: {DEFAULT_DATA_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Training data not found at {args.data}. "
            "Run scripts/generate_dataset.py and scripts/sample_dataset.py first, "
            "or pass --data with a valid path."
        )

    print(f"Loading dataset from {args.data} ...")
    df = get_dataset(args.data).to_pandas()

    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print("Training baseline model ...")
    model = train_baseline(X_train, y_train)

    print("Evaluating ...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 (macro): {metrics['f1_macro']:.3f}")
    print(metrics["report"])


if __name__ == "__main__":
    main()
