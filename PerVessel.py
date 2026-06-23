import polars as pl
import zipfile
import io
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa

# --- Config ---
ZIP_DIR = Path("data/raw")          # folder with daily .zip files
OUT_DIR = Path("data/per_vessel")   # one .parquet per MMSI
OUT_DIR.mkdir(exist_ok=True)

# filter criteria
RELEVANT_SHIP_TYPES = ["Cargo", "Tanker", "Fishing", "Tug", "Passenger"] #we are analysing only these ship types
MIN_ROWS_PER_VESSEL = 60*24  # minimum number of rows per vessel to keep


def read_zip_csv(zip_path: Path) -> pl.LazyFrame:
    """Read the CSV inside a zip into a Polars LazyFrame."""
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
    """filter unnecesssary vessels based on ship type and number of rows per vessel."""
    return (
        lf
        .filter(pl.col("Ship type").is_in(RELEVANT_SHIP_TYPES))
    )

def append_to_vessel_parquet(mmsi: int, table: pa.Table):
    """Append a PyArrow table to a per-vessel parquet file."""
    out_path = OUT_DIR / f"{mmsi}.parquet"
    if out_path.exists():
        existing = pq.read_table(out_path)
        combined = pa.concat_tables([existing, table])
        pq.write_table(combined, out_path, compression="snappy")
    else:
        pq.write_table(table, out_path, compression="snappy")

def process_day(zip_path: Path):
    print(f"Processing {zip_path.name}...")
    lf = read_zip_csv(zip_path)
    lf = filter_vessels(lf)
    df = lf.collect()  # materialise after filtering

    if df.is_empty():
        return

    # Group by vessel and write each group
    for mmsi, group_df in df.group_by("MMSI"):
        if len(group_df) < MIN_ROWS_PER_VESSEL:
            continue
        table = group_df.to_arrow()
        append_to_vessel_parquet(mmsi, table)

# --- Main loop ---
zip_files = sorted(ZIP_DIR.glob("*.zip"))
for zip_file in zip_files:
    process_day(zip_file)

print(f"Done. Per-vessel parquet files written to {OUT_DIR}/")

