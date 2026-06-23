import polars as pl

df = pl.scan_csv("aisdk-2024-01-01.csv")  # lazy, no data loaded yet

features = (
    df
    .group_by("MMSI")
    .agg([
        # Identity
        pl.col("Ship type").mode().first().alias("ship_type"),
        pl.col("Flag").mode().first().alias("flag"),

        # Trajectory
        pl.col("SOG").mean().alias("mean_speed"),
        pl.col("SOG").std().alias("std_speed"),
        pl.col("COG").std().alias("cog_variability"),

        # Status
        pl.col("Navigational status").n_unique().alias("n_nav_statuses"),
        (pl.col("Navigational status") == "At anchor").sum().alias("anchor_count"),

        # Draught
        pl.col("Draught").max().alias("max_draught"),
        pl.col("Draught").min().alias("min_draught"),

        # Temporal
        pl.col("# Timestamp").count().alias("n_pings"),
        (pl.col("# Timestamp").max() - pl.col("# Timestamp").min()).alias("time_span"),
    ])
    .collect()  # only NOW does it execute
)

features.write_parquet("features_2024-01-01.parquet")