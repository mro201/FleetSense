"""
Generate a dataset of features for each vessel based on its AIS data.

important thing to take into account:
- very variable number of pings per vessel, some vessels have very few pings, others have many thousands
- unbalanced distribution of ship types, with some types being much more common than others
- missing values in some features, especially draught and navigational status
- number of pings per vessel can vary significantly over time, with some vessels having long periods of inactivity or lower reporting frequency

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

import random
from datetime import date
from pathlib import Path

import parquet as pq
import polars as pl

IN_DIR = Path("data/per_vessel")
OUT_DIR = Path("data/dataset")


def compute_features_for_vessel(mmsi: int) -> pl.DataFrame:
    """Compute features for a single vessel given its MMSI."""

    files = sorted(IN_DIR.glob(f"{mmsi}_*.parquet"))
    if not files:
        return pl.DataFrame()

    combined = pl.concat([pl.read_parquet(f) for f in files])

    features = (
        combined.with_columns(
            pl.col("# Timestamp")
            .str.to_datetime("%d/%m/%Y %H:%M:%S")
            .alias("# Timestamp")
        )
        .sort("# Timestamp")
        .group_by_dynamic("# Timestamp", every="1w", group_by="MMSI")
        .agg(
            [
                # Identity
                pl.col("Ship type").mode().first().alias("ship_type"),
                
                # Trajectory
                # moving speed not mean speed to filter out anchor pings
                pl.col("SOG").filter(pl.col("SOG") > 0).mean().alias("mean_moving_speed"),
                pl.col("SOG").max().alias("max_speed"),
                pl.col("SOG").std().alias("std_speed"),
                pl.col("COG").std().alias("cog_variability"),
                
                # Status
                (pl.col("Navigational status").n_unique()).alias("n_nav_statuses"),
                ((pl.col("Navigational status") == "At anchor").sum()/pl.col("# Timestamp").count()).alias("anchor_ratio"),
                ((pl.col("Navigational status") == "Engaged in fishing").sum()/pl.col("# Timestamp").count()).alias("fishing_ratio"),
                
                
                ((pl.col("Cargo type") == "Category X").sum()/pl.col("# Timestamp").count()).alias("cargoX_ratio"),
                ((pl.col("Cargo type") == "Category OS").sum()/pl.col("# Timestamp").count()).alias("cargoOS_ratio"),
                ((pl.col("Cargo type") == "Category Y").sum()/pl.col("# Timestamp").count()).alias("cargoY_ratio"),
                ((pl.col("Cargo type") == "Category Z").sum()/pl.col("# Timestamp").count()).alias("cargoZ_ratio"),
                
                # Draught
                (pl.col("Draught").max()-pl.col("Draught").min()).alias("draught_variability"),
                pl.col("Draught").min().alias("min_draught"),
                
                # Temporal
                (pl.col("# Timestamp").max() - pl.col("# Timestamp").min()).dt.total_seconds().alias("time_span_seconds"),
                pl.col("# Timestamp").count().alias("n_pings"),
            ]
        )
    )
    return features


# Find all unique MMSIs
mmsi_set = list(set(p.stem.split("_")[0] for p in IN_DIR.glob("*.parquet")))
random.shuffle(mmsi_set)  # Shuffle to avoid any ordering bias
first = True
print(f"Computing features for {len(mmsi_set)} vessels...")
i=0
for mmsi in mmsi_set[:100]:  # Limit to first 100 vessels for testing
    if i % 100 == 0:
        print(f"{i}/{len(mmsi_set)}vessels processed...")
    i += 1
    features = compute_features_for_vessel(int(mmsi))
    features = features.filter(
        (pl.col("n_pings") >= 100)
        & (pl.col("mean_moving_speed").is_not_null())
        & (pl.col("min_draught").is_not_null())
    )
    if features.is_empty():
        continue
    if first:
        features.write_csv("output.csv")
        first = False
    else:
        with open("output.csv", "ab") as f:
            features.write_csv(f, include_header=False)
