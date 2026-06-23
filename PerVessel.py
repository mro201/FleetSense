import polars as pl
import zipfile
import io
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import tempfile
import os

# --- Config ---
ZIP_DIR = Path("data/raw")
OUT_DIR = Path("data/per_vessel")
OUT_DIR.mkdir(exist_ok=True)

RELEVANT_SHIP_TYPES = ["Cargo", "Tanker", "Fishing", "Tug", "Passenger"]
MIN_ROWS_PER_VESSEL = 60 * 24


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

    for (mmsi,), group_df in df.group_by("MMSI"):  # unpack tuple key
        table = group_df.to_arrow()
        append_to_vessel_parquet(mmsi, table)


# --- Main loop ---
zip_files = sorted(ZIP_DIR.glob("*.zip"))
for zip_file in zip_files:  # Process from June 17th onward
    process_day(zip_file)

print(f"Done. Per-vessel parquet files written to {OUT_DIR}/")