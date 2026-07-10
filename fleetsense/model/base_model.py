"""
Baseline Random Forest classifier for vessel type classification.

Centralizes all model design choices (hyperparameters, feature set, train/test
split strategy) so every notebook/experiment that imports this trains and
evaluates the model identically. This ensures drift experiments are comparable
to each other and to the baseline.
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# ── Design choices ────────────────────────────────────────────────────
RANDOM_STATE = 42

MODEL_PATH = Path(__file__).parent.parent / "outputs" / "baseline_rf.pkl"

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

TARGET_COLUMN = "ship_type"


def train_baseline(X_train, y_train) -> RandomForestClassifier:
    """Train the baseline Random Forest with fixed hyperparameters."""
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def load_baseline_model() -> RandomForestClassifier:
    """Load the trained baseline model, raising a clear error if it hasn't been trained yet."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Baseline model not found at {MODEL_PATH}. Train the model first.")


def predict_baseline(X_test) -> list:
    """Make batch predictions with the trained baseline model."""
    return load_baseline_model().predict(X_test)


def predict_baseline_proba(features) -> dict:
    model = load_baseline_model()
    row = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    predicted_class = model.predict(row)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(row)[0].tolist()))

    return {"vessel_type": predicted_class, "probabilities": probabilities}


def evaluate_model(model, X_test, y_test) -> dict:
    """Return standard evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }
