"""Functions for extracting and comparing feature distributions across time periods,
for the purpose of monitoring temporal drift.

Features are compared using the Population Stability Index (PSI), computed against
a fixed set of reference bin edges (deciles of data pooled from a baseline date range,
e.g. the first couple of months). Drift can optionally be computed separately within
each class (e.g. ship type), so a class's later periods are only ever compared against
that same class's own baseline.

Baselines (bin edges and reference proportions per feature, and optionally a class
proportion baseline for the model's predicted class) are computed once via
build_baselines, then reused across all periods being compared by monitor_all_features
— rather than recomputing the same baseline repeatedly. Baselines can be persisted
with save_baselines and reloaded later with load_baselines, so the same reference
doesn't need to be rebuilt in a future session.

Class balance on true labels (the proportion of each class per period) is tracked
separately via monitor_class_balance, since a shifting mix of true classes is a
different question from drift within a class's feature distributions, or from drift
in the model's predicted classes.
"""

import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from fleetsense.features.data_loader import load_schema

# PSI thresholds commonly used for interpretation
PSI_STABLE = 0.10
PSI_MODERATE = 0.25

# Sentinel used as the dict key for baselines when no class_col is given.
_NO_CLASS = "__all__"

# Reserved key under which the predicted-class balance baseline is stored in the
# baselines dict returned by build_baselines, and the "feature" label used for it
# in monitor_all_features' combined output.
_PREDICTED_CLASS_KEY = "__predicted_class_balance__"


class FeatureBaseline(NamedTuple):
    """A feature's reference bin edges and reference proportions, computed once
    from a baseline period (and optionally, one specific class)."""

    edges: np.ndarray
    reference_props: np.ndarray


class ClassBalanceBaseline(NamedTuple):
    """Reference class proportions for a categorical column (e.g. predicted class),
    computed once from a baseline period. classes gives the fixed class ordering
    that reference_props (and any later comparison proportions) must align to."""

    classes: list[str]
    reference_props: np.ndarray


# --- Reference construction ---------------------------------------------------


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


# --- Distribution representation ----------------------------------------------


def bin_proportions(sample: pl.Series, edges: np.ndarray) -> np.ndarray:
    """Bin a numeric sample using fixed edges and return proportions per bin."""
    values = sample.drop_nulls().to_numpy()
    bin_indices = np.digitize(values, edges[1:-1], right=False)
    counts = np.bincount(bin_indices, minlength=len(edges) - 1)
    return counts / counts.sum()


def _class_proportions(df: pl.DataFrame, class_col: str, classes: list[str]) -> np.ndarray:
    """Compute the proportion of each class in classes, in that fixed order, from df.
    A class with zero occurrences in df still gets an entry (0.0), so the resulting
    vector always has the same length/order as classes and lines up for psi()."""
    counts = df[class_col].value_counts()
    lookup = dict(zip(counts[class_col].to_list(), counts["count"].to_list()))
    raw = np.array([lookup.get(cls, 0) for cls in classes], dtype=float)
    total = raw.sum()
    return raw / total if total > 0 else raw


# --- Comparison measures --------------------------------------------------------


def psi(reference_proportions: np.ndarray, comparison_proportions: np.ndarray, epsilon: float = 1e-6) -> float:
    """Population Stability Index between two proportion vectors over the same bins."""
    ref = np.clip(reference_proportions, epsilon, None)
    comp = np.clip(comparison_proportions, epsilon, None)
    return float(np.sum((comp - ref) * np.log(comp / ref)))


# --- Period column handling --------------------------------------------------


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
            return pl.col(period_col).str.to_datetime(period_format, strict=False).dt.date()
        # Automatic format inference doesn't reliably handle strings like
        # "2025-07-21T00:00:00.000000" (ISO format with a T separator and
        # microseconds), so try that exact format explicitly first.
        return pl.col(period_col).str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).dt.date()
    if dtype == pl.Date:
        return pl.col(period_col)
    return pl.col(period_col).cast(pl.Date)


# --- Baseline construction ---------------------------------------------------


def build_baselines(
    baseline_df: pl.DataFrame,
    numeric_features: list[str],
    class_col: str | None = None,
    predicted_class_col: str | None = None,
    n_bins: int = 10,
) -> dict[str, dict[str, FeatureBaseline] | ClassBalanceBaseline]:
    """Compute bin edges and reference proportions once per feature (and per class,
    if class_col is given), plus optionally a predicted-class balance baseline, from
    data falling within the baseline date range.

    Returns a dict keyed by feature name -> {class_value: FeatureBaseline}, the same
    shape as before. If class_col is not given, each feature maps to a single
    baseline under the internal key _NO_CLASS rather than one baseline per class.

    If predicted_class_col is given, the dict also gets one extra entry under the
    reserved key _PREDICTED_CLASS_KEY, holding a ClassBalanceBaseline built from the
    predicted-class proportions in the baseline window — this is what lets
    monitor_all_features also report predicted-class-mix drift, computed the same
    way as feature PSI (against a fixed baseline) but without any quantile binning,
    since predicted class is categorical rather than continuous.

    Pass the result into monitor_numeric_feature / monitor_all_features so the
    baseline only needs to be computed once, rather than being rebuilt on every call.
    """
    class_values = baseline_df[class_col].unique().sort().to_list() if class_col else [_NO_CLASS]
    numeric_features = load_schema()["columns"]
    baselines: dict[str, dict[str, FeatureBaseline] | ClassBalanceBaseline] = {}
    for feature in numeric_features:
        per_class: dict[str, FeatureBaseline] = {}
        for cls in class_values:
            sample = baseline_df.filter(pl.col(class_col) == cls)[feature] if class_col else baseline_df[feature]
            edges = build_reference_bins(sample, n_bins=n_bins)
            reference_props = bin_proportions(sample, edges)
            per_class[cls] = FeatureBaseline(edges=edges, reference_props=reference_props)
        baselines[feature] = per_class

    if predicted_class_col is not None:
        classes = baseline_df[predicted_class_col].unique().sort().to_list()
        reference_props = _class_proportions(baseline_df, predicted_class_col, classes)
        baselines[_PREDICTED_CLASS_KEY] = ClassBalanceBaseline(classes=classes, reference_props=reference_props)

    return baselines


def save_baselines(baselines: dict, path: Path) -> None:
    """Persist a baselines dict (from build_baselines) to disk, so it can be reused
    in a later session without recomputing it.

    Uses pickle, since the baselines are a dict of namedtuples holding NumPy arrays
    — not something that maps cleanly to a plain text format.
    """
    path = Path(path)
    with path.open("wb") as f:
        pickle.dump(baselines, f)


def load_baselines(path: Path) -> dict:
    """Load a baselines dict previously written by save_baselines."""
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301 -- trusted, project-generated file


# --- Orchestration ---------------------------------------------------------------


def monitor_numeric_feature(
    baselines: dict,
    df: pl.DataFrame,
    feature: str,
    period_col: str,
    class_col: str | None = None,
) -> pl.DataFrame:
    """Compute PSI for one numeric feature across all periods, against precomputed
    baselines (see build_baselines).

    If class_col is given, each class's periods are compared only against that same
    class's own baseline (from build_baselines), not blended with other classes.

    Returns a table with one row per period (and per class, if class_col is given).
    The baseline periods themselves are included in the output, so you can confirm
    PSI comes out near zero there as a sanity check.
    """
    feature_baselines = baselines[feature]

    if class_col is None:
        baseline = feature_baselines[_NO_CLASS]
        return _monitor_numeric_within_group(baseline, df, feature, period_col)

    results = []
    for cls in df[class_col].unique().sort().to_list():
        class_df = df.filter(pl.col(class_col) == cls)
        baseline = feature_baselines[cls]
        class_result = _monitor_numeric_within_group(baseline, class_df, feature, period_col)
        results.append(class_result.with_columns(pl.lit(cls).alias(class_col)))

    return pl.concat(results)


def _monitor_numeric_within_group(
    baseline: FeatureBaseline,
    df: pl.DataFrame,
    feature: str,
    period_col: str,
) -> pl.DataFrame:
    """Core PSI computation for one numeric feature within a single group (e.g. one class),
    against a precomputed baseline."""
    rows = []
    for period in df[period_col].unique().sort().to_list():
        sample = df.filter(pl.col(period_col) == period)[feature]
        comparison_props = bin_proportions(sample, baseline.edges)
        rows.append({"period": period, "feature": feature, "psi": psi(baseline.reference_props, comparison_props)})

    return pl.DataFrame(rows)


def _monitor_predicted_class_balance(
    baseline: ClassBalanceBaseline,
    df: pl.DataFrame,
    predicted_class_col: str,
    period_col: str,
) -> pl.DataFrame:
    """Core PSI computation for the predicted-class balance, against a precomputed
    ClassBalanceBaseline. No quantile binning involved — proportions per class are
    computed directly, per period, and compared straight against the baseline's
    fixed class ordering and reference proportions."""
    rows = []
    for period in df[period_col].unique().sort().to_list():
        period_df = df.filter(pl.col(period_col) == period)
        comparison_props = _class_proportions(period_df, predicted_class_col, baseline.classes)
        rows.append(
            {"period": period, "feature": _PREDICTED_CLASS_KEY, "psi": psi(baseline.reference_props, comparison_props)}
        )

    return pl.DataFrame(rows)


def monitor_all_features(
    baselines: dict,
    df: pl.DataFrame,
    numeric_features: list[str],
    period_col: str,
    class_col: str | None = None,
    predicted_class_col: str | None = None,
) -> pl.DataFrame:
    """Run PSI drift monitoring across all specified numeric features, and optionally
    the model's predicted-class balance, combined into one table.

    baselines comes from build_baselines, computed once per feature (and per class,
    if class_col is given), plus a predicted-class balance baseline if
    predicted_class_col was passed to build_baselines.

    If class_col is given (e.g. ship type), feature drift is computed independently
    within each class, and the output includes a column for class alongside feature
    and period, so you can see whether drift is isolated to specific ship types or
    is happening across the board.

    If predicted_class_col is given, the combined output also includes rows with
    feature set to a reserved marker (accessible as
    fleetsense...distribution_monitoring._PREDICTED_CLASS_KEY) representing PSI on
    the model's predicted-class mix for that period — this row has no meaningful
    class_col value, since it's a single signal per period, not per class.
    """
    results = [
        monitor_numeric_feature(baselines, df, feature, period_col, class_col=class_col) for feature in numeric_features
    ]

    combined = pl.concat(results)

    if predicted_class_col is not None:
        predicted_baseline = baselines[_PREDICTED_CLASS_KEY]
        predicted_result = _monitor_predicted_class_balance(predicted_baseline, df, predicted_class_col, period_col)
        if class_col:
            predicted_result = predicted_result.with_columns(pl.lit(_NO_CLASS).alias(class_col))
        combined = pl.concat([combined, predicted_result], how="diagonal")

    sort_cols = ["feature", class_col, "period"] if class_col else ["feature", "period"]
    return combined.sort(sort_cols)


def monitor_class_balance(
    df: pl.DataFrame,
    class_col: str,
    period_col: str,
) -> pl.DataFrame:
    """Track the proportion of each true class per period.

    This is a separate question from feature drift within a class: it answers
    whether the mix of true classes itself is shifting over time (e.g. a given
    month suddenly containing far more Tanker vessels than usual), rather than
    whether a class's own feature distributions are changing, or whether the
    model's predictions are drifting (see build_baselines' predicted_class_col
    and monitor_all_features for that).

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


# --- Drift alarms ---------------------------------------------------------------


def check_drift(
    psi_results: pl.DataFrame,
    threshold: float = PSI_MODERATE,
    period_col: str = "period",
    feature_col: str = "feature",
    psi_col: str = "psi",
    class_col: str | None = "ship_type",
) -> pl.DataFrame:
    """Flag every row in a PSI results table (from monitor_all_features) whose PSI
    breaches the given threshold, and print each one as a human-readable alarm.

    Rows for the predicted-class balance (feature == _PREDICTED_CLASS_KEY) are
    printed without a class label, since that signal is per-period, not per-class.

    Returns the flagged rows as a table, sorted from most to least severe, so they
    can also be inspected or plotted programmatically rather than only printed.
    """
    flagged = psi_results.filter(pl.col(psi_col) > threshold).sort(psi_col, descending=True)

    period_as_date = _period_as_date_expr(flagged, period_col)
    flagged = flagged.with_columns(period_as_date.alias("_period_label"))

    for row in flagged.iter_rows(named=True):
        if row[feature_col] == _PREDICTED_CLASS_KEY:
            print(f"PREDICTION DRIFT FLAGGED — week_start {row['_period_label']} (PSI {row[psi_col]:.2f})")
            continue
        class_part = f"{row[class_col]}: " if class_col else ""
        print(
            f"DRIFT FLAGGED — week_start {row['_period_label']}, "
            f"{class_part}{row[feature_col]} (PSI {row[psi_col]:.2f})"
        )

    return flagged.sort("_period_label").drop("_period_label")
