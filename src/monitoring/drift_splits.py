# src/fleetsense/monitoring/drift_splits.py
"""
Constructs deliberately biased train/test splits to induce known drift types,
for studying how model performance degrades outside the training distribution.
"""

import polars as pl


def temporal_split(df: pl.DataFrame, train_months: list[int], test_month: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train on given months, test on a later month (temporal drift)."""
    train = df.filter(pl.col("week_start").dt.month().is_in(train_months))
    test = df.filter(pl.col("week_start").dt.month() == test_month)
    return train, test


def geographic_split(df: pl.DataFrame, train_region: str, test_region: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train on one sea region, test on another (geographic drift)."""
    train = df.filter(pl.col("region") == train_region)
    test = df.filter(pl.col("region") == test_region)
    return train, test
