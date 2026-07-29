"""
Check drift on predictions logged since the last monitoring check.

Usage:
    uv run scripts/check_drift.py
"""

import json
from datetime import datetime, timezone, date
from pathlib import Path

import polars as pl
from train_model import train, LAST_TRAINING_PATH
from fleetsense.model.base_model import LOG_PATH
from fleetsense.features.data_loader import FEATURES
from fleetsense.monitoring.distribution_monitoring import (
    PSI_MODERATE,
    check_drift,
    load_baselines,
    monitor_all_features,
)

ROOT = Path(__file__).parent.parent

LAST_MONITORING_PATH = ROOT / "fleetsense" / "outputs" / "last_monitoring.json"


def load_last_checked() -> datetime | None:
    if not LAST_MONITORING_PATH.exists():
        return None
    with open(LAST_MONITORING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(LAST_TRAINING_PATH, "r", encoding="utf-8") as f:
        data_train = json.load(f)

    checked_up_to = datetime.fromisoformat(data["checked_up_to"])
    data_end = datetime.fromisoformat(data_train["data_end"])  # last date the model is trained on
    return max(checked_up_to, data_end)


def save_last_checked(up_to: datetime) -> None:
    metadata = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checked_up_to": up_to.isoformat(),
    }
    LAST_MONITORING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_MONITORING_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_new_predictions(since: datetime | None) -> pl.DataFrame:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"No prediction log found at {LOG_PATH}")

    valid_lines = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError:
                continue  # skip malformed entries

    df = pl.DataFrame([json.loads(line) for line in valid_lines])
    df = df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z"))

    if since is not None:
        df = df.filter(pl.col("timestamp") > since)

    return df


def main() -> bool:
    last_checked = load_last_checked()
    if last_checked is None:
        print("No previous monitoring check found — checking all logged predictions.")
    else:
        print(f"Checking predictions logged after {last_checked.isoformat()} ...")

    new_predictions = load_new_predictions(last_checked)

    if new_predictions.is_empty():
        print("No new predictions since last check. Nothing to do.")
        return False

    print(f"Found {new_predictions.height} new predictions.")

    df = new_predictions.unnest("features")

    df = df.with_columns(pl.lit(datetime.now(timezone.utc).date()).alias("period"))
    print(FEATURES)
    baselines = load_baselines()
    psi_results = monitor_all_features(baselines, df, FEATURES, period_col="period", class_col=None)

    flagged = check_drift(psi_results, threshold=PSI_MODERATE, period_col="period", class_col=None)

    latest_timestamp = new_predictions["timestamp"].max()
    save_last_checked(latest_timestamp)

    drift_flagged = flagged.height > 0
    if not drift_flagged:
        print("No drift detected.")
    else:
        train(end=date.today())

    return drift_flagged


if __name__ == "__main__":
    main()
