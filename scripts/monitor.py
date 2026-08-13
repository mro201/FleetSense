"""
Check drift on predictions logged since the last monitoring check.

Usage:
    uv run scripts/check_drift.py
"""

from datetime import datetime, timezone

import polars as pl
from scripts.train_model import LAST_TRAINING_PATH, load_permutation_importance, train

from fleetsense.features.data_loader import FEATURES
from fleetsense.monitoring.distribution_monitoring import (
    PSI_MODERATE,
    add_weighted_psi,
    check_drift,
    load_baselines,
    monitor_all_features,
    weighted_drift_score,
)
from fleetsense.monitoring.monitoring_state import (
    load_last_checked,
    load_new_predictions,
    save_last_checked,
)
from fleetsense.monitoring.report import generate_drift_report
from scripts.train_model import load_permutation_importance

SCORE_THRESHOLD = 0.1


def main() -> bool:
    last_checked = load_last_checked()
    if last_checked is None:
        print("No previous monitoring check found — checking all logged predictions.")
    else:
        print(f"Checking predictions logged after {last_checked.isoformat()} ...")

    new_predictions = load_new_predictions(last_checked)
    if new_predictions.is_empty():
        print("No new predictions since last check. Nothing to do.")
        # return False

    print(f"Found {new_predictions.height} new predictions.")
    df = new_predictions.unnest("features")
    df = df.with_columns(pl.lit(datetime.now(timezone.utc).date()).alias("period"))

    # Load the baselines and compute PSI for all features
    baselines = load_baselines()
    psi_results = monitor_all_features(baselines, df, FEATURES, period_col="timestamp", class_col=None)

    # Load the permutation importance to weight the features in the drift check
    importance_df = load_permutation_importance()
    weights = importance_df["drift_weight"].to_dict()

    psi_results = add_weighted_psi(psi_results, weights)

    per_feature_flagged = check_drift(psi_results, threshold=PSI_MODERATE, class_col=None)
    period_scores = weighted_drift_score(psi_results, period_col="period")
    mean_flagged = period_scores.filter(pl.col("period_weighted_drift_mean") > SCORE_THRESHOLD)

    for row in mean_flagged.iter_rows(named=True):
        print(f"Period {row['period']} flagged for high mean drift: {row['period_weighted_drift_mean']:.4f}")

    drift_flagged = per_feature_flagged.height > 0 or mean_flagged.height > 0

    if drift_flagged:
        reasons = []
        if per_feature_flagged.height > 0:
            reasons.append(f"{per_feature_flagged.height} individual feature breach(es)")
        if mean_flagged.height > 0:
            reasons.append(f"{mean_flagged.height} period(s) with high mean drift")
        print(f"Retraining triggered: {', '.join(reasons)}")
        train(end=date.today())
    else:
        print("No drift detected on any signal.")
    latest_timestamp = new_predictions["timestamp"].max()
    save_last_checked(latest_timestamp)
    generate_drift_report(
        psi_results,
        per_feature_flagged,
        period_scores,
        score_threshold=SCORE_THRESHOLD,
        class_col=None,
    )

    return drift_flagged


if __name__ == "__main__":
    main()
