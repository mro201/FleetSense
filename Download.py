"""This script downloads daily AIS data from the specified URL for a given date range and saves it to the "data/raw" directory. The data is downloaded in ZIP format."""
import os
import time
import urllib.request
from datetime import date, timedelta

os.makedirs("data/raw", exist_ok=True)

start = date(2025, 6, 1)
end = date(2025, 8, 31)

d = start
while d <= end:
    filename_date = d.strftime("%Y-%m-%d")
    url = f"http://aisdata.ais.dk/aisdk-{filename_date}.zip"
    filename = f"data/raw/aisdk-{filename_date}.zip"
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
