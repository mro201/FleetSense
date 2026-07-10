import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from fleetsense.features.data_loader import get_dataset

total_features = [
    "length",
    "width",
    "max_draught",
    "min_draught",
    "draught_variability",
    "mean_moving_speed",
    "max_speed",
    "std_speed",
    "sog_p10",
    "sog_median",
    "sog_p90",
    "frac_time_slow",
    "frac_time_fast",
    "cog_variability",
    "rot_mean_abs",
    "rot_std",
    "heading_cog_diff_mean",
    "n_nav_statuses",
    "fishing_ratio",
    "anchor_ratio",
    "underway_engine_ratio",
    "moored_ratio",
    "lat_std",
    "lon_std",
    "lat_mean",
    "lon_mean",
    "bbox_area",
    "CargoX_ratio",
    "CargoY_ratio",
    "CargoZ_ratio",
    "CargoOS_ratio",
    "CargoReserved_ratio",
    "n_pings",
    "time_span_seconds",
    "length_beam_ratio",
    "draught_length_ratio",
    "sog_bimodality",
    "mean_ping_interval_seconds",
]


def compute_feature_importance():
    pddf = get_dataset().to_pandas()

    X_train, X_test, y_train, y_test = train_test_split(
        pddf[total_features], pddf["ship_type"], test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

    importance_df = pd.DataFrame(
        {
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        },
        index=total_features,
    ).sort_values("importance_mean", ascending=False)

    return importance_df


def select_features_by_importance(threshold=0.001):
    importance_df = compute_feature_importance()

    # remove features that have std over mean or importance mean below threshold
    selected_features = importance_df[
        (importance_df["importance_mean"] > threshold)
        & (importance_df["importance_std"] < importance_df["importance_mean"])
    ].index.tolist()

    return selected_features


if __name__ == "__main__":
    features = select_features_by_importance(threshold=0.001)
    print(f"Final selected features: {features}")
