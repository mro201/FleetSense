from datetime import date, datetime

import argparse
from fleetsense.data import aggregate_per_vessel, download
from fleetsense.features import generate_features

START_DATE = date(2025, 12, 31)
END_DATE = date(2026, 6, 30)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AIS data and generate the FleetSense training dataset.")
    parser.add_argument(
        "--start",
        type=parse_date,
        default=START_DATE,
        help=f"start date for training data (default: {START_DATE})",
    )
    parser.add_argument(
        "--end",
        type=parse_date,
        default=END_DATE,
        help=f"end date for training data (default: {END_DATE})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["overwrite", "append"],
        default="overwrite",
        help="overwrite: regenerate from scratch; append: add new rows to the existing file (default: overwrite)",
    )
    return parser.parse_args()


def main(start, end, mode) -> None:
    if start > end:
        raise ValueError(f"start date {start} is after end date {end}")
    print(f"Downloading AIS data from {start} to {end} ...")
    download.download_ais_data(start, end)

    print("Aggregating per-vessel data ...")
    aggregate_per_vessel.process_range(start, end)

    print("Generating features ...")
    generate_features.generate_dataset(start, end, mode)

    print("Dataset generation complete.")


if __name__ == "__main__":
    args = parse_args()
    main(args.start, args.end, args.mode)
