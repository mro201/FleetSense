from datetime import date


from fleetsense.data import aggregate_per_vessel, download
from fleetsense.features import generate_features

START_DATE = date(2025, 12, 31)
END_DATE = date(2026, 6, 30)

download.download_ais_data(START_DATE, END_DATE)
aggregate_per_vessel.process_range(START_DATE, END_DATE)
generate_features.generate_dataset()
