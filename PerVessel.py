"""Reads zip files of daily AIS data, filters for relevant ship types, and writes per-vessel per day parquet files."""
import io
import os
import tempfile
import zipfile
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# --- Config ---
ZIP_DIR = Path("data/raw")
OUT_DIR = Path("data/per_vessel")
OUT_DIR.mkdir(exist_ok=True)

RELEVANT_SHIP_TYPES = ["Cargo", "Tanker", "Fishing", "Tug", "Passenger"]


def read_zip_csv(zip_path: Path) -> pl.LazyFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = [f for f in zf.namelist() if f.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            raw = f.read()
    return pl.read_csv(
        io.BytesIO(raw),
        infer_schema_length=1000,
        null_values=["", "NA", "null", "NULL"],
    ).lazy()


def filter_vessels(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("Ship type").is_in(RELEVANT_SHIP_TYPES))


def merge_vessel_files():
    # Find all unique MMSIs
    mmsi_set = set(p.stem.split("_")[0] for p in OUT_DIR.glob("*.parquet"))

    for mmsi in mmsi_set:
        files = sorted(OUT_DIR.glob(f"{mmsi}_*.parquet"))
        # Read and combine one at a time to keep memory low
        combined = pl.concat([pl.read_parquet(f) for f in files])
        combined.write_parquet(OUT_DIR / f"{mmsi}.parquet", compression="snappy")
        # Delete the daily files
        for f in files:
            f.unlink()


def append_to_vessel_parquet(mmsi: int, table: pa.Table):
    out_path = OUT_DIR / f"{mmsi}.parquet"
    if out_path.exists():
        existing = pq.read_table(out_path)
        combined = pa.concat_tables([existing, table])
        # Write to temp file first to avoid Windows file lock
        tmp_fd, tmp_path = tempfile.mkstemp(dir=OUT_DIR, suffix=".parquet")
        os.close(tmp_fd)
        try:
            pq.write_table(combined, tmp_path, compression="snappy")
            os.replace(tmp_path, out_path)
        except Exception:
            os.remove(tmp_path)
            raise
    else:
        pq.write_table(table, out_path, compression="snappy")


def process_day(zip_path: Path):
    print(f"Processing {zip_path.name}...")
    lf = read_zip_csv(zip_path)
    lf = filter_vessels(lf)
    df = lf.collect()

    if df.is_empty():
        return

    date_str = zip_path.stem.replace("aisdk-", "")  # e.g. "2025-06-01"
    for (mmsi,), group_df in df.group_by("MMSI"):
        out_path = OUT_DIR / f"{mmsi}_{date_str}.parquet"
        group_df.write_parquet(out_path, compression="snappy")


# --- Main loop ---
zip_files = sorted(ZIP_DIR.glob("*.zip"))
for zip_file in zip_files[2:]:
    process_day(zip_file)

print(f"Done. Per-vessel parquet files written to {OUT_DIR}/")
