"""Unit tests for class proportion splitting used in drift monitoring."""

import polars as pl
import pytest

from fleetsense.monitoring.distribution_monitoring import _class_proportions


def test_class_proportions_known_split():
    df = pl.DataFrame({"ship_type": ["Cargo", "Cargo", "Tanker", "Fishing"]})
    classes = ["Cargo", "Tanker", "Fishing", "Passenger"]

    props = _class_proportions(df, "ship_type", classes)

    assert props.tolist() == pytest.approx([0.5, 0.25, 0.25, 0.0])


def test_class_proportions_empty_df_returns_zeros():
    df = pl.DataFrame({"ship_type": []}, schema={"ship_type": pl.Utf8})
    classes = ["Cargo", "Tanker"]

    props = _class_proportions(df, "ship_type", classes)

    assert props.tolist() == [0.0, 0.0]
