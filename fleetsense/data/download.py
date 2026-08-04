"""This script downloads daily AIS data from the specified URL for a given date range and saves it to the "data/raw"
directory. The data is downloaded in ZIP format."""

import os
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

sys.path.append(str(Path("..").resolve()))
from fleetsense.config import DATA_RAW

os.makedirs(DATA_RAW, exist_ok=True)


def download_ais_data(start: date, end: date) -> None:
    total_days = (end - start).days + 1
    d = start

    for _ in range(total_days):
        filename_date = d.strftime("%Y-%m-%d")
        url = f"http://aisdata.ais.dk/aisdk-{filename_date}.zip"
        filename = Path(DATA_RAW) / f"aisdk-{filename_date}.zip"

        if filename.exists():
            print(f"  Already exists, skipping: {filename}")
        else:
            print(f"Downloading {url}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"  Saved: {filename}")
            except urllib.error.HTTPError as e:
                print(f"  Failed ({e.code}): {url}")
            except Exception as e:
                print(f"  Error: {e}")
            time.sleep(1)

        d += timedelta(days=1)


if __name__ == "__main__":
    start = date(2025, 12, 31)
    end = date(2026, 6, 30)
    download_ais_data(start, end)
