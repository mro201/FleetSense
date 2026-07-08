"""Functions for extracting and comparing feature distributions across time periods,
for the purpose of monitoring temporal drift.

Features are compared using the Population Stability Index (PSI), computed against
a fixed set of reference bin edges (deciles of data pooled from a baseline date range,
e.g. the first couple of months). Drift can optionally be computed separately within
each class (e.g. ship type), so a class's later periods are only ever compared against
that same class's own baseline.

Class balance (the proportion of each class per period) is tracked separately via
monitor_class_balance, since a shifting mix of classes is a different question from
drift within a class's feature distributions.
"""

from datetime import date

import numpy as np
import polars as pl

PSI_STABLE = 0.10
PSI_MODERATE = 0.25


def build_reference_bins(baseline: pl.Series, n_bins: int = 10) -> np.ndarray:
    """Compute bin edges from a baseline (reference) sample of a numeric feature.

    Uses equal-frequency (quantile-based) binning, so each bin holds roughly the
    same share of the baseline data. Returns edges, including -inf/+inf as the
    outermost bounds so later periods can't fall outside the binning.
    """
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]  # interior cut points only
    edges = np.quantile(baseline.drop_nulls().to_numpy(), quantiles)
    edges = np.unique(edges)  # guard against duplicate edges from skewed data
    return np.concatenate([[-np.inf], edges, [np.inf]])


def bin_proportions(sample: pl.Series, edges: np.ndarray) -> np.ndarray:
    """Bin a numeric sample using fixed edges and return proportions per bin."""
    values = sample.drop_nulls().to_numpy()
    bin_indices = np.digitize(values, edges[1:-1], right=False)
    counts = np.bincount(bin_indices, minlength=len(edges) - 1)
    return counts / counts.sum()


def psi(reference_proportions: np.ndarray, comparison_proportions: np.ndarray, epsilon: float = 1e-6) -> float:
    """Population Stability Index between two proportion vectors over the same bins."""
    ref = np.clip(reference_proportions, epsilon, None)
    comp = np.clip(comparison_proportions, epsilon, None)
    return float(np.sum((comp - ref) * np.log(comp / ref)))


def _period_as_date_expr(df: pl.DataFrame, period_col: str, period_format: str | None = None) -> pl.Expr:
    """Build an expression that reads period_col as a Date, regardless of whether the
    underlying column is already a Date/Datetime, or stored as a string.

    period_format can be given as an explicit strptime-style format (e.g. "%Y-%m-%d")
    if automatic parsing fails — this commonly happens when the string includes a time
    component (e.g. "2025-06-02 00:00:00") rather than just a date.

    This only affects filtering against baseline_start/baseline_end — the original
    period_col values (string or date) are still what's returned in the output rows.
    """
    dtype = df.schema[period_col]
    if dtype in (pl.Utf8, pl.String):
        if period_format is not None:
            return pl.col(period_col).str.to_datetime(period_format).dt.date()
        # Fall back to datetime parsing (handles a trailing time component) then
        # drop down to just the date, rather than str.to_date() which expects the
        # string to contain nothing but a date.
        return pl.col(period_col).str.to_datetime(strict=False).dt.date()
    if dtype == pl.Date:
        return pl.col(period_col)
    return pl.col(period_col).cast(pl.Date)


def monitor_numeric_feature(
    df: pl.DataFrame,
    feature: str,
    period_col: str,
    baseline_start: date,
    baseline_end: date,
    class_col: str | None = None,
    n_bins: int = 10,
    period_format: str | None = None,
) -> pl.DataFrame:
    """Compute PSI for one numeric feature across all periods, against a baseline range.

    baseline_start and baseline_end define an inclusive date range (e.g. the first
    two months of data). All rows whose period falls within that range are pooled
    together to form the reference distribution — this avoids needing period values
    to match a fixed set of labels, which matters since periods are weekly and won't
    line up with calendar month boundaries.

    If class_col is given, drift is computed separately within each class: each class
    gets its own baseline (from that class's data within the baseline range) and its
    own bin edges, so a class's later periods are only ever compared against that
    same class's typical distribution — not blended with other classes.

    Returns a table with one row per period (and per class, if class_col is given).
    The baseline periods themselves are included in the output, so you can confirm
    PSI comes out near zero there as a sanity check.
    """
    if class_col is None:
        return _monitor_numeric_within_group(
            df, feature, period_col, baseline_start, baseline_end, n_bins=n_bins, period_format=period_format
        )

    results = []
    for cls in df[class_col].unique().sort().to_list():
        class_df = df.filter(pl.col(class_col) == cls)
        class_result = _monitor_numeric_within_group(
            class_df, feature, period_col, baseline_start, baseline_end, n_bins=n_bins, period_format=period_format
        )
        results.append(class_result.with_columns(pl.lit(cls).alias(class_col)))

    return pl.concat(results)


def _monitor_numeric_within_group(
    df: pl.DataFrame,
    feature: str,
    period_col: str,
    baseline_start: date,
    baseline_end: date,
    n_bins: int = 10,
    period_format: str | None = None,
) -> pl.DataFrame:
    """Core PSI computation for one numeric feature within a single group (e.g. one class)."""
    period_as_date = _period_as_date_expr(df, period_col, period_format=period_format)
    baseline_sample = df.filter(period_as_date.is_between(baseline_start, baseline_end, closed="both"))[feature]
    edges = build_reference_bins(baseline_sample, n_bins=n_bins)
    reference_props = bin_proportions(baseline_sample, edges)

    rows = []
    for period in df[period_col].unique().sort().to_list():
        sample = df.filter(pl.col(period_col) == period)[feature]
        comparison_props = bin_proportions(sample, edges)
        rows.append({"period": period, "feature": feature, "psi": psi(reference_props, comparison_props)})

    return pl.DataFrame(rows)


def monitor_all_features(
    df: pl.DataFrame,
    numeric_features: list[str],
    period_col: str,
    baseline_start: date,
    baseline_end: date,
    class_col: str | None = None,
    n_bins: int = 10,
    period_format: str | None = None,
) -> pl.DataFrame:
    """Run PSI drift monitoring across all specified numeric features and combine into one table.

    baseline_start and baseline_end define an inclusive date range whose data is
    pooled together to form the reference (e.g. the first two months of data).

    If class_col is given (e.g. ship type), drift is computed independently within
    each class, and the output includes a column for class alongside feature and
    period, so you can see whether drift is isolated to specific ship types or is
    happening across the board.
    """
    results = [
        monitor_numeric_feature(
            df,
            feature,
            period_col,
            baseline_start,
            baseline_end,
            class_col=class_col,
            n_bins=n_bins,
            period_format=period_format,
        )
        for feature in numeric_features
    ]

    combined = pl.concat(results)
    sort_cols = ["feature", class_col, "period"] if class_col else ["feature", "period"]
    return combined.sort(sort_cols)


def monitor_class_balance(
    df: pl.DataFrame,
    class_col: str,
    period_col: str,
) -> pl.DataFrame:
    """Track the proportion of each class per period.

    This is a separate question from feature drift within a class: it answers
    whether the mix of classes itself is shifting over time (e.g. a given month
    suddenly containing far more Tanker vessels than usual), rather than whether
    a class's own feature distributions are changing.

    Returns a table with one row per class per period and its share of that period.
    """
    counts = df.group_by([period_col, class_col]).agg(pl.len().alias("n"))
    totals = counts.group_by(period_col).agg(pl.col("n").sum().alias("total"))
    return (
        counts.join(totals, on=period_col)
        .with_columns((pl.col("n") / pl.col("total")).alias("proportion"))
        .sort([class_col, period_col])
        .select([class_col, period_col, "n", "total", "proportion"])
    )
