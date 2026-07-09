"""
Baseline Random Forest classifier for vessel type classification.

Centralizes all model design choices (hyperparameters, feature set, train/test
split strategy) so every notebook/experiment that imports this trains and
evaluates the model identically. This ensures drift experiments are comparable
to each other and to the baseline.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


# ── Design choices ────────────────────────────────────────────────────
RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

FEATURE_COLUMNS = [
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
    "bbox_area",
    "length_beam_ratio",
    "draught_length_ratio",
    "sog_bimodality",
    "mean_ping_interval_seconds",
    # ... adjust to your actual final feature list
]

TARGET_COLUMN = "ship_type"


def train_baseline(X_train, y_train) -> RandomForestClassifier:
    """Train the baseline Random Forest with fixed hyperparameters."""
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """Return standard evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }
