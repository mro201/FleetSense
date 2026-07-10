"""
Generate a dataset of features for each vessel based on its AIS data.

important thing to take into account:
- very variable number of pings per vessel, some vessels have very few pings, others have many thousands
- unbalanced distribution of ship types, with some types being much more common than others
- missing values in some features, especially draught and navigational status
- number of pings per vessel can vary significantly over time, with some vessels having long periods of inactivity or
    lower reporting frequency

Features generated include:
- Identity: ship type
- Trajectory: mean moving speed, speed variability, course variability
- Status: number of unique navigational statuses, number of anchor pings
- Draught: max and min draught
- Temporal: time span of the data, number of pings

features to be added:
- draught variability (std)
- anchor ratio (anchor_count / n_pings) - maybe remove number of anchor pings and just keep the ratio

features are computed on a weekly basis for each vessel.


Filters to be added:
- Only include vessels with at least 100 pings in the time span.
- Only include vessels with a time span of at least 1 week.
- remove vessels with null or missing values in critical features (e.g., mean_moving_speed, max_draught).
"""

import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path("..").resolve()))
from fleetsense.config import DATA_DATASET, DATA_VESSEL, TIMESTAMP_FMT

IN_DIR = DATA_VESSEL
OUT_DIR = DATA_DATASET
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP_COL = "timestamp"  # renamed on load — avoids issues with '#' in name


# Main feature function
def compute_features_for_vessel(imo: int) -> pl.DataFrame:
    files = sorted(IN_DIR.glob(f"{imo}_*.parquet"))
    if not files:
        return pl.DataFrame()

    # 1. Load, rename timestamp col, parse to Datetime, sort
    combined = (
        pl.concat([pl.read_parquet(f) for f in files])
        .rename({"# Timestamp": TIMESTAMP_COL})  # remove '#' from name
        .with_columns(pl.col(TIMESTAMP_COL).str.to_datetime(TIMESTAMP_FMT).alias(TIMESTAMP_COL))
        .sort(TIMESTAMP_COL)
        .with_columns(pl.col(TIMESTAMP_COL).dt.truncate("1w").alias("_week_start"))
    )

    # 3. Main weekly aggregation
    features = (
        combined.group_by_dynamic(TIMESTAMP_COL, every="1w")
        .agg(
            [
                # ── Identity ──────────────────────────────────────────────
                pl.col("Ship type").mode().first().alias("ship_type"),
                # ── Physical dimensions ───────────────────────────────────
                pl.col("Length").mode().first().alias("length"),
                pl.col("Width").mode().first().alias("width"),
                pl.col("Draught").max().alias("max_draught"),
                pl.col("Draught").min().alias("min_draught"),
                (pl.col("Draught").max() - pl.col("Draught").min()).alias("draught_variability"),
                # ── Speed profile ─────────────────────────────────────────
                pl.col("SOG").filter(pl.col("SOG") > 0.5).mean().alias("mean_moving_speed"),
                pl.col("SOG").max().alias("max_speed"),
                pl.col("SOG").std().alias("std_speed"),
                pl.col("SOG").quantile(0.10).alias("sog_p10"),
                pl.col("SOG").quantile(0.50).alias("sog_median"),
                pl.col("SOG").quantile(0.90).alias("sog_p90"),
                (pl.col("SOG").filter(pl.col("SOG") < 1.0).len() / pl.col("SOG").len()).alias("frac_time_slow"),
                (pl.col("SOG").filter(pl.col("SOG") > 10.0).len() / pl.col("SOG").len()).alias("frac_time_fast"),
                # ── Course & heading behaviour ────────────────────────────
                pl.col("COG").std().alias("cog_variability"),
                pl.col("ROT").abs().mean().alias("rot_mean_abs"),
                pl.col("ROT").abs().std().alias("rot_std"),
                # Circular difference keeps result in [0°, 180°]
                (((pl.col("Heading") - pl.col("COG") + 180) % 360 - 180).abs().mean()).alias("heading_cog_diff_mean"),
                # ── Navigational status ───────────────────────────────────
                pl.col("Navigational status").n_unique().alias("n_nav_statuses"),
                ((pl.col("Navigational status") == "Engaged in fishing").sum() / pl.col(TIMESTAMP_COL).count()).alias(
                    "fishing_ratio"
                ),
                ((pl.col("Navigational status") == "At anchor").sum() / pl.col(TIMESTAMP_COL).count()).alias(
                    "anchor_ratio"
                ),
                (
                    (pl.col("Navigational status") == "Under way using engine").sum() / pl.col(TIMESTAMP_COL).count()
                ).alias("underway_engine_ratio"),
                ((pl.col("Navigational status") == "Moored").sum() / pl.col(TIMESTAMP_COL).count()).alias(
                    "moored_ratio"
                ),
                # ── Spatial range ─────────────────────────────────────────
                pl.col("Latitude").std().alias("lat_std"),
                pl.col("Longitude").std().alias("lon_std"),
                pl.col("Latitude").mean().alias("lat_mean"),
                pl.col("Longitude").mean().alias("lon_mean"),
                (
                    (pl.col("Latitude").max() - pl.col("Latitude").min())
                    * (pl.col("Longitude").max() - pl.col("Longitude").min())
                ).alias("bbox_area"),
                # ── cargo ───────────────────────────────────────
                ((pl.col("Cargo type") == "Category X").sum() / pl.col(TIMESTAMP_COL).count()).alias("CargoX_ratio"),
                ((pl.col("Cargo type") == "Category Y").sum() / pl.col(TIMESTAMP_COL).count()).alias("CargoY_ratio"),
                ((pl.col("Cargo type") == "Category Z").sum() / pl.col(TIMESTAMP_COL).count()).alias("CargoZ_ratio"),
                ((pl.col("Cargo type") == "Category OS").sum() / pl.col(TIMESTAMP_COL).count()).alias("CargoOS_ratio"),
                ((pl.col("Cargo type") == "Reserved for future use").sum() / pl.col(TIMESTAMP_COL).count()).alias(
                    "CargoReserved_ratio"
                ),
                # ── Temporal ──────────────────────────────────────────────
                pl.col(TIMESTAMP_COL).count().alias("n_pings"),
                (pl.col(TIMESTAMP_COL).max() - pl.col(TIMESTAMP_COL).min())
                .dt.total_seconds()
                .alias("time_span_seconds"),
            ]
        )
        # ── Derived features (post-aggregation) ───────────────────────
        .with_columns(
            [
                (pl.col("length") / (pl.col("width") + 1e-6)).alias("length_beam_ratio"),
                (pl.col("max_draught") / (pl.col("length") + 1e-6)).alias("draught_length_ratio"),
                ((pl.col("sog_p90") - pl.col("sog_p10")) / (pl.col("std_speed") + 1e-6)).alias("sog_bimodality"),
                (pl.col("time_span_seconds") / (pl.col("n_pings") - 1).clip(lower_bound=1)).alias(
                    "mean_ping_interval_seconds"
                ),
                # group_by_dynamic index column is already Datetime — just rename it
                pl.col(TIMESTAMP_COL).alias("week_start"),
            ]
        )
    )

    return features


def generate_dataset():
    imo_set = list(
        set(
            p.stem.split("_")[0]
            for p in IN_DIR.glob("*.parquet")
            if p.stem.split("_")[0].isdigit()  # <-- skip 'Unknown' and any other non-numeric
        )
    )
    print(f"Computing features for {len(imo_set)} vessels...")

    out_path = OUT_DIR / "vessel_weekly_features.csv"
    first = True

    for i, imo in enumerate(imo_set):
        if i % 100 == 0:
            print(f"  {i}/{len(imo_set)} vessels processed...")

        features = compute_features_for_vessel(int(imo))

        features = features.filter((pl.col("n_pings") >= 100))

        if features.is_empty():
            continue

        if first:
            features.write_csv(out_path)
            first = False
        else:
            with open(out_path, "ab") as f:
                features.write_csv(f, include_header=False)

    print(f"Done. Output written to {out_path}")
