from pathlib import Path

ROOT = Path(__file__).parent  # DriftAnalysis/

DATA_RAW = ROOT / "data" / "raw"
DATA_VESSEL = ROOT / "data" / "per_vessel"
DATA_DATASET = ROOT / "data" / "dataset"

SHIP_TYPES = ["Cargo", "Tanker", "Fishing", "Tug", "Passenger"]
TIMESTAMP_FMT = "%d/%m/%Y %H:%M:%S"
RANDOM_SEED = 42
MIN_PINGS_PER_WEEK = 100
CLASS_COLUMN = "Ship type"
