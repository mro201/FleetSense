import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from fleetsense.features.data_loader import get_dataset, SCHEMA_PATH


def get_numeric_features(pddf: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Automatically detect numeric feature columns from the dataset, rather than
    hand-maintaining a static list.
    """
    exclude = exclude or []
    numeric_cols = pddf.select_dtypes(include="number").columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def compute_feature_importance():
    pddf = get_dataset().to_pandas()
    numerical_features = get_numeric_features(pddf)
    X_train, X_test, y_train, y_test = train_test_split(
        pddf[numerical_features], pddf["ship_type"], test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

    importance_df = pd.DataFrame(
        {
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        },
        index=numerical_features,
    ).sort_values("importance_mean", ascending=False)

    return importance_df


def select_features_by_importance(threshold=0.001):
    pddf = get_dataset().to_pandas()
    importance_df = compute_feature_importance()

    selected_features = importance_df[
        (importance_df["importance_mean"] > threshold)
        & (importance_df["importance_std"] < importance_df["importance_mean"])
    ].index.tolist()

    schema = {
        "columns": selected_features,
        "dtypes": {col: str(pddf[col].dtype) for col in selected_features},
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))

    return selected_features


if __name__ == "__main__":
    print("computing....")
    features = select_features_by_importance(threshold=0.001)
    print(f"Final selected features: {features}")
