"""Reads zip files of daily AIS data, filters for relevant ship types, and writes per-vessel per day parquet files."""

import io
import sys
import zipfile
from datetime import date
from pathlib import Path

import polars as pl

sys.path.append(str(Path("..").resolve()))
from fleetsense.config import DATA_RAW, DATA_VESSEL, SHIP_TYPES

# --- Config ---
ZIP_DIR = DATA_RAW
OUT_DIR = DATA_VESSEL
OUT_DIR.mkdir(exist_ok=True)


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
    return lf.filter(pl.col("Ship type").is_in(SHIP_TYPES))


def merge_vessel_files():
    # Find all unique imos
    imo_set = set(p.stem.split("_")[0] for p in OUT_DIR.glob("*.parquet"))

    for imo in imo_set:
        files = sorted(OUT_DIR.glob(f"{imo}_*.parquet"))
        # Read and combine one at a time to keep memory low
        combined = pl.concat([pl.read_parquet(f) for f in files])
        combined.write_parquet(OUT_DIR / f"{imo}.parquet", compression="snappy")
        # Delete the daily files
        for f in files:
            f.unlink()


def process_day(zip_path: Path):
    print(f"Processing {zip_path.name}...")
    lf = read_zip_csv(zip_path)
    lf = filter_vessels(lf)
    df = lf.collect()

    if df.is_empty():
        return

    date_str = zip_path.stem.replace("aisdk-", "")  # e.g. "2025-06-01"
    for (imo,), group_df in df.group_by("IMO"):
        out_path = OUT_DIR / f"{imo}_{date_str}.parquet"
        group_df.write_parquet(out_path, compression="snappy")


def process_range(start_date: date, end_date: date):
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    zip_files = [f for f in zip_files if start_str <= f.stem.replace("aisdk-", "") <= end_str]

    for zip_file in zip_files:
        process_day(zip_file)


if __name__ == "__main__":
    START_DATE = date(2026, 6, 10)  # set to None to process all
    END_DATE = date(2026, 6, 26)
    zip_files = sorted(ZIP_DIR.glob("*.zip"))
    if START_DATE:
        start_str = START_DATE.strftime("%Y-%m-%d")
        end_str = END_DATE.strftime("%Y-%m-%d")
        zip_files = [f for f in zip_files if start_str <= f.stem.replace("aisdk-", "") <= end_str]

    for zip_file in zip_files:
        process_day(zip_file)
