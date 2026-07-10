"""
Constructs deliberately biased train/test splits to induce known drift types,
for studying how model performance degrades outside the training distribution.
"""

from datetime import date

import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from fleetsense.config import DATA_DATASET, SHIP_TYPES

FEATURES = [
    "sog_p10",
    "sog_median",
    "frac_time_slow",
    "frac_time_fast",
    "heading_cog_diff_mean",
    "fishing_ratio",
    "anchor_ratio",
    "underway_engine_ratio",
    "moored_ratio",
    "length",
    "width",
    "max_draught",
    "draught_variability",
    "length_beam_ratio",
    "draught_length_ratio",
    "bbox_area",
    "mean_ping_interval_seconds",
]

TARGET_COLUMN = "ship_type"

RANDOM_STATE = 42


def get_dataset() -> pl.DataFrame:
    """Load the preprocessed dataset from disk."""
    return pl.read_csv(DATA_DATASET / "vessel_weekly_features.csv")


def get_features_and_target(df: pl.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the model's fixed feature set and target from a features DataFrame, as pandas."""
    X = df.select(FEATURES).to_pandas()
    y = df[TARGET_COLUMN].to_pandas()
    return X, y


def train_test_split_random(df: pl.DataFrame, test_size=0.2):
    """Standard random split used for the i.i.d. baseline (not used for drift experiments)."""
    X, y = get_features_and_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return x_train, x_test, y_train, y_test


def temporal_split(
    df: pl.DataFrame, train_test_split_date: date
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train on data up to a certain date, test on data from a later date (temporal drift)."""
    df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S.%6f"))

    train_df = df.filter(pl.col("timestamp") < train_test_split_date)
    test_df = df.filter(pl.col("timestamp") >= train_test_split_date)

    x_train, y_train = get_features_and_target(train_df)
    x_test, y_test = get_features_and_target(test_df)
    return x_train, x_test, y_train, y_test


def geographic_split(
    df: pl.DataFrame,
    train_region: tuple[float, float, float, float],
    test_region: tuple[float, float, float, float],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Train on one lat/lon bounding box, test on another (geographic drift).

    train_region and test_region are each (lat_min, lat_max, lon_min, lon_max).
    Bounds are inclusive on both ends. Rows outside both boxes are excluded
    from the split entirely; if the boxes overlap, rows in the overlap are
    included in both train and test.
    """
    lat_min, lat_max, lon_min, lon_max = train_region
    train = df.filter(pl.col("lat_mean").is_between(lat_min, lat_max) & pl.col("lon_mean").is_between(lon_min, lon_max))

    lat_min, lat_max, lon_min, lon_max = test_region
    test = df.filter(pl.col("lat_mean").is_between(lat_min, lat_max) & pl.col("lon_mean").is_between(lon_min, lon_max))

    x_train, y_train = get_features_and_target(train)
    x_test, y_test = get_features_and_target(test)
    return x_train, x_test, y_train, y_test


def compositional_split(
    df: pl.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train on a class-balanced sample, test on the natural class distribution (composition drift)."""
    class_col = "ship_type"
    n_total = len(df)
    n_test_total = int(n_total * test_size)

    # First pass: for each class, pull its proportional test slice and see
    # how many rows are left over for train. min_class_size is computed
    # from these post-test-removal pools (not the raw class counts), so a
    # small class that loses a big chunk to test can't leave train short.
    test_parts = []
    remaining_by_class = {}

    for ship_type in SHIP_TYPES:
        class_df = df.filter(pl.col(class_col) == ship_type).sample(fraction=1.0, shuffle=True, seed=random_state)

        # test slice sized proportionally to this class's natural share of the data
        n_class_test = round(len(class_df) / n_total * n_test_total)
        test_parts.append(class_df.head(n_class_test))
        remaining_by_class[ship_type] = class_df.slice(n_class_test)

    min_class_size = min(len(remaining) for remaining in remaining_by_class.values())

    train_parts = [remaining.head(min_class_size) for remaining in remaining_by_class.values()]

    train = pl.concat(train_parts).sample(fraction=1.0, shuffle=True, seed=random_state)
    test = pl.concat(test_parts).sample(fraction=1.0, shuffle=True, seed=random_state)

    # every class should now have exactly min_class_size rows in train
    train_counts = train.group_by(class_col).len()["len"]
    assert train_counts.n_unique() == 1, "compositional_split produced an unbalanced train set"

    x_train, y_train = get_features_and_target(train)
    x_test, y_test = get_features_and_target(test)

    return x_train, x_test, y_train, y_test
