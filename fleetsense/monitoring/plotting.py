"""Plotting functions for visualizing PSI-based drift monitoring output
(the tables produced by monitor_all_features / monitor_class_balance).

Two main views are provided:
- A heatmap of PSI across features (rows) and periods (columns), one per class if
  class-level drift was computed — good for spotting which features/classes/periods
  are driving drift at a glance.
- A line plot of PSI over time for a single feature, with the standard PSI
  interpretation thresholds drawn in as reference lines, and one line per class
  if class-level drift was computed — good for a closer look once a heatmap has
  flagged something worth investigating.

A third helper ranks features by how much they've drifted, to help decide what's
worth a closer look in the first place.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

from fleetsense.monitoring.distribution_monitoring import PSI_MODERATE, PSI_STABLE


def _thin_labels(labels: list[str], max_labels: int = 12) -> list[str]:
    """Blank out all but every Nth label, so a dense axis stays readable."""
    if len(labels) <= max_labels:
        return labels
    step = math.ceil(len(labels) / max_labels)
    return [label if i % step == 0 else "" for i, label in enumerate(labels)]


def plot_psi_heatmap(
    drift_df: pl.DataFrame,
    class_col: str | None = None,
    figsize_per_panel: tuple[float, float] = (8, 5),
    max_period_labels: int = 12,
) -> plt.Figure:
    """Heatmap of PSI values, features on the y-axis and periods on the x-axis.

    If class_col is present in drift_df, one heatmap panel is drawn per class,
    side by side, sharing the same color scale so panels are directly comparable.

    Period values are parsed and given clean "YYYY-MM-DD" labels, thinned to at
    most max_period_labels per panel, rather than plotting the raw period values
    directly (which can be unreadably long/dense for weekly data).
    """
    pdf = drift_df.to_pandas()
    pdf["_period_label"] = pd.to_datetime(pdf["period"]).dt.strftime("%Y-%m-%d")

    def _draw(ax: plt.Axes, sub: pd.DataFrame, show_cbar: bool) -> None:
        pivot = sub.pivot(index="feature", columns="_period_label", values="psi")
        pivot = pivot[sorted(pivot.columns, key=lambda s: pd.to_datetime(s))]
        vmax = max(pdf["psi"].quantile(0.98), PSI_MODERATE * 1.5)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="rocket_r",
            vmin=0,
            vmax=vmax,
            cbar=show_cbar,
            linewidths=0.3,
            linecolor="white",
            xticklabels=1,  # force one tick per column, so our own thinning below has a matching count
        )
        ax.set_xticklabels(_thin_labels(list(pivot.columns), max_period_labels), rotation=45, ha="right")

    if class_col and class_col in pdf.columns:
        classes = sorted(pdf[class_col].unique())
        fig, axes = plt.subplots(
            len(classes), 1, figsize=(figsize_per_panel[0], figsize_per_panel[1] * len(classes)), sharex=True
        )
        axes = [axes] if len(classes) == 1 else axes

        for ax, cls in zip(axes, classes):
            _draw(ax, pdf[pdf[class_col] == cls], show_cbar=True)
            ax.set_title(str(cls))
            ax.set_ylabel("Feature")
            ax.set_xlabel("Period")

        fig.suptitle("PSI drift by feature and period", y=1.0, fontsize=13)
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=figsize_per_panel)
    _draw(ax, pdf, show_cbar=True)
    ax.set_title("PSI drift by feature and period")
    ax.set_xlabel("Period")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def plot_psi_timeseries(
    drift_df: pl.DataFrame,
    feature: str,
    class_col: str | None = None,
    figsize: tuple[float, float] = (9, 5),
    max_period_labels: int = 12,
) -> plt.Figure:
    """Line plot of PSI over time for a single feature, with PSI interpretation
    thresholds drawn as reference lines. One line per class if class_col is present.

    Period values are parsed and given clean "YYYY-MM-DD" labels, thinned to at
    most max_period_labels, matching the treatment used in plot_psi_heatmap.
    """
    pdf = drift_df.filter(pl.col("feature") == feature).to_pandas()
    pdf["_period_label"] = pd.to_datetime(pdf["period"]).dt.strftime("%Y-%m-%d")
    all_labels = sorted(pdf["_period_label"].unique(), key=lambda s: pd.to_datetime(s))

    fig, ax = plt.subplots(figsize=figsize)

    if class_col and class_col in pdf.columns:
        for cls, group in pdf.groupby(class_col):
            group = group.sort_values("_period_label", key=lambda s: pd.to_datetime(s))
            ax.plot(group["_period_label"], group["psi"], marker="o", label=str(cls))
        ax.legend(title=class_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        pdf = pdf.sort_values("_period_label", key=lambda s: pd.to_datetime(s))
        ax.plot(pdf["_period_label"], pdf["psi"], marker="o", color="steelblue")

    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(_thin_labels(all_labels, max_period_labels), rotation=45, ha="right")

    ax.axhline(PSI_STABLE, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(PSI_MODERATE, color="firebrick", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(ax.get_xlim()[1], PSI_STABLE, " stable cutoff", va="center", fontsize=8, color="gray")
    ax.text(ax.get_xlim()[1], PSI_MODERATE, " significant cutoff", va="center", fontsize=8, color="firebrick")

    ax.set_title(f"PSI over time — {feature}")
    ax.set_xlabel("Period")
    ax.set_ylabel("PSI")
    fig.tight_layout()
    return fig


def rank_features_by_drift(drift_df: pl.DataFrame, class_col: str | None = None) -> pl.DataFrame:
    """Rank features by their maximum observed PSI across all periods (and classes,
    if class_col is given), as a starting point for deciding what to plot in detail.

    Returns a table sorted from most to least drifted.
    """
    group_cols = ["feature", class_col] if class_col else ["feature"]
    return drift_df.group_by(group_cols).agg(pl.col("psi").max().alias("max_psi")).sort("max_psi", descending=True)


def plot_class_balance(
    balance_df: pl.DataFrame,
    class_col: str,
    period_col: str,
    kind: str = "area",
    figsize: tuple[float, float] = (10, 5.5),
    max_period_labels: int = 12,
) -> plt.Figure:
    """Plot the output of monitor_class_balance: the proportion of each class per period.

    kind="area" (default) draws a 100% stacked area chart, which is generally the
    clearest way to see the class mix shifting over time — each band's thickness is
    that class's share of the period, and the whole stack always sums to 1.

    kind="line" instead draws one line per class showing its proportion over time,
    which is useful when you want to focus closely on a single class's trend rather
    than the overall composition (the bands in a stacked chart can be harder to read
    for classes that aren't at the bottom of the stack).

    Period values are parsed and given clean "YYYY-MM-DD" labels, thinned to at
    most max_period_labels, matching the treatment used in the PSI plots.
    """
    pdf = balance_df.to_pandas()
    pdf["_period_label"] = pd.to_datetime(pdf[period_col]).dt.strftime("%Y-%m-%d")

    pivot = pdf.pivot(index="_period_label", columns=class_col, values="proportion").fillna(0)
    pivot = pivot.loc[sorted(pivot.index, key=lambda s: pd.to_datetime(s))]

    fig, ax = plt.subplots(figsize=figsize)

    if kind == "area":
        ax.stackplot(pivot.index, pivot.T.values, labels=pivot.columns, alpha=0.85)
        ax.set_ylabel("Proportion of vessels")
        ax.set_ylim(0, 1)
    elif kind == "line":
        for cls in pivot.columns:
            ax.plot(pivot.index, pivot[cls], marker="o", markersize=4, linewidth=1.5, label=str(cls))
        ax.set_ylabel("Proportion of vessels")
    else:
        raise ValueError(f"Unknown kind {kind!r}, expected 'area' or 'line'")

    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(_thin_labels(list(pivot.index), max_period_labels), rotation=45, ha="right")
    ax.legend(title=class_col, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    ax.set_title("Class balance over time")
    ax.set_xlabel("Period")
    fig.tight_layout()
    return fig
