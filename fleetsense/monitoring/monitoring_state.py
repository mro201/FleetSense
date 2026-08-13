import json
from datetime import date, datetime, timezone
from pathlib import Path
from fleetsense.model.base_model import LOG_PATH
from scripts.train_model import LAST_TRAINING_PATH
import polars as pl

ROOT = Path(__file__).parent.parent

LAST_MONITORING_PATH = ROOT / "outputs" / "last_monitoring.json"


def load_last_checked() -> datetime | None:
    if not LAST_MONITORING_PATH.exists():
        return None
    with open(LAST_MONITORING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(LAST_TRAINING_PATH, "r", encoding="utf-8") as f:
        data_train = json.load(f)

    checked_up_to = datetime.fromisoformat(data["checked_up_to"])
    if checked_up_to.tzinfo is not None:
        checked_up_to = checked_up_to.astimezone(timezone.utc).replace(tzinfo=None)

    data_end_date = date.fromisoformat(data_train["data_end"])  # parse as date, not datetime
    data_end = datetime.combine(data_end_date, datetime.min.time())  # naive, midnight

    return max(checked_up_to, data_end)


def save_last_checked(up_to: datetime) -> None:
    metadata = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checked_up_to": up_to.isoformat(),
    }
    LAST_MONITORING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_MONITORING_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_new_predictions(since: datetime | None) -> pl.DataFrame:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"No prediction log found at {LOG_PATH}")

    valid_lines = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                valid_lines.append(line)
            except json.JSONDecodeError:
                continue  # skip malformed entries

    df = pl.DataFrame([json.loads(line) for line in valid_lines])
    df = df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S"))

    if since is not None:
        df = df.filter(pl.col("timestamp") > since)

    return df
