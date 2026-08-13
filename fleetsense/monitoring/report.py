"""Generate a self-contained, dated HTML drift report from a monitoring run.

Consumes the outputs of build_baselines -> monitor_all_features -> add_weighted_psi
-> check_drift / weighted_drift_score (see distribution_monitoring.py), and renders
one HTML file per run: a status summary, the ranked breach list (every individual
alarm, not just the aggregate), the per-period raw/weighted drift scores, and the
PSI heatmap embedded directly as an image -- no separate image file to manage.

Deliberately not a live dashboard: one static file per run, saved with its date, so
a history builds up as fleetsense/outputs/reports/drift_report_YYYY-MM-DD.html.
"""

import base64
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display available when this runs from a script/cron
import polars as pl

from fleetsense.monitoring.distribution_monitoring import _PREDICTED_CLASS_KEY
from fleetsense.monitoring.plotting import plot_psi_heatmap

REPORTS_DIR = Path(__file__).parent.parent / "outputs" / "reports"


def _fig_to_base64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG string, for embedding directly
    in an <img> tag rather than saving a separate file alongside the report."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    matplotlib.pyplot.close(fig)
    return encoded


def _breach_rows_html(flagged: pl.DataFrame, class_col: str | None) -> str:
    """Build the <tr> rows for the ranked breach list, one row per flagged
    (period, feature[, class]) combination -- every individual alarm stays
    visible, nothing gets collapsed into just the aggregate."""
    if flagged.is_empty():
        return '<tr><td colspan="5" class="empty">No individual feature breaches.</td></tr>'

    rows = []
    for row in flagged.iter_rows(named=True):
        weighted = row.get("weighted_psi")
        weighted_str = f"{weighted:.3f}" if weighted is not None else "—"
        cls = row.get(class_col) if class_col else None
        cls_str = str(cls) if cls is not None else "—"
        feature_label = "Predicted class balance" if row["feature"] == _PREDICTED_CLASS_KEY else row["feature"]
        rows.append(
            f"<tr><td>{row['period']}</td><td>{feature_label}</td><td>{cls_str}</td>"
            f"<td>{row['psi']:.3f}</td><td>{weighted_str}</td></tr>"
        )
    return "\n".join(rows)


def _period_rows_html(period_scores: pl.DataFrame) -> str:
    """Build the <tr> rows for the per-period raw vs. weighted drift score table."""
    rows = []
    for row in period_scores.sort("period").iter_rows(named=True):
        rows.append(
            f"<tr><td>{row['period']}</td>"
            f"<td>{row['period_raw_drift_mean']:.3f}</td>"
            f"<td>{row['period_weighted_drift_mean']:.3f}</td></tr>"
        )
    return "\n".join(rows)


def generate_drift_report(
    psi_results: pl.DataFrame,
    flagged: pl.DataFrame,
    period_scores: pl.DataFrame,
    score_threshold: float = 0.1,
    class_col: str | None = None,
    report_date: date | None = None,
) -> Path:
    """Render a drift report and save it to
    fleetsense/outputs/reports/drift_report_{date}.html, returning the saved path.

    psi_results: full PSI table from monitor_all_features (+ add_weighted_psi) --
        used for the heatmap.
    flagged: output of check_drift -- the ranked, per-alarm breach list.
    period_scores: output of weighted_drift_score -- per-period raw/weighted
        mean and std drift.
    score_threshold: the same threshold used to decide whether a period's mean
        weighted drift counts as flagged (matches SCORE_THRESHOLD in
        check_drift.py) -- only affects the STABLE/FLAGGED status line, not
        which rows appear in the tables (flagged/period_scores are used as-is).
    class_col: name of the class column in psi_results/flagged, if drift was
        computed per class. Used to label the breach table and to shape the
        heatmap correctly.
    report_date: defaults to today; pass explicitly to backfill or re-render.
    """
    report_date = report_date or date.today()
    generated_at = datetime.now().isoformat(timespec="seconds")

    n_breaches = flagged.height
    n_periods_flagged = period_scores.filter(pl.col("period_weighted_drift_mean") > score_threshold).height
    is_flagged = n_breaches > 0 or n_periods_flagged > 0
    status = "FLAGGED" if is_flagged else "STABLE"

    # Heatmap: plot_psi_heatmap expects "feature"/"psi"/"period" and an optional
    # "class" column specifically -- rename class_col to match, and drop the
    # predicted-class-balance row (it has no per-class meaning and would show
    # up as a spurious "__all__" panel/row otherwise).
    heatmap_df = psi_results.filter(pl.col("feature") != _PREDICTED_CLASS_KEY)
    if class_col and class_col in heatmap_df.columns:
        heatmap_df = heatmap_df.rename({class_col: "class"})
    heatmap_fig = plot_psi_heatmap(heatmap_df, figsize_per_panel=(12, 10))
    heatmap_b64 = _fig_to_base64(heatmap_fig)

    breach_rows = _breach_rows_html(flagged, class_col)
    period_rows = _period_rows_html(period_scores)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>FleetSense Drift Report — {report_date}</title>
<style>
    body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif; background: #EEF2F6;
            color: #001f3d; margin: 0; padding: 2rem; }}
    .container {{ max-width: 1000px; margin: 0 auto; }}
    h1 {{ margin-bottom: 0.2rem; }}
    h2 {{ margin-top: 2rem; }}
    .meta {{ color: #4A5C6E; font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .status {{ display: inline-block; padding: 0.4rem 1rem; border-radius: 8px;
               font-weight: 700; margin-bottom: 1.5rem; }}
    .status.stable {{ background: #DCEEDC; color: #1E7B34; }}
    .status.flagged {{ background: #FBDCD3; color: #C74A08; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem;
             background: #fff; border-radius: 8px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #D8E0E8;
              font-size: 0.9rem; }}
    th {{ background: #13315C; color: #fff; }}
    td.empty {{ color: #4A5C6E; font-style: italic; }}
    img {{ max-width: 100%; border-radius: 8px; border: 1px solid #D8E0E8; }}
</style>
</head>
<body>
<div class="container">
    <h1>⚓ FleetSense Drift Report</h1>
    <div class="meta">Period: {report_date} — Generated {generated_at}</div>
    <div class="status {"flagged" if is_flagged else "stable"}">{status}</div>

    <h2>Ranked breach list</h2>
    <table>
        <tr><th>Period</th><th>Feature</th><th>Class</th><th>PSI</th><th>Weighted PSI</th></tr>
        {breach_rows}
    </table>

    <h2>Per-period drift score</h2>
    <table>
        <tr><th>Period</th><th>Raw drift (mean)</th><th>Weighted drift (mean)</th></tr>
        {period_rows}
    </table>

    <h2>PSI heatmap</h2>
    <img src="data:image/png;base64,{heatmap_b64}" alt="PSI heatmap by feature and period">
</div>
</body>
</html>"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"drift_report_{report_date.isoformat()}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
