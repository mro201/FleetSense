from pathlib import Path


from fleetsense.features.data_loader import get_dataset

ROOT = Path(__file__).parent.parent  # DriftAnalysis/

df = get_dataset().to_pandas()

sample = df.groupby("ship_type", group_keys=False).apply(lambda x: x.sample(n=min(len(x), 200), random_state=42))
output_dir = ROOT / "data" / "dataset"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "vessel_weekly_features_sample.csv"
sample.to_csv(output_path, index=False)
print("sample dataset saved at", output_path)
