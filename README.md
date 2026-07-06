# FleetSense: Machine Learning for Maritime Behaviour and Drift Monitoring

A machine learning pipeline for classifying maritime vessel types from AIS (Automatic Identification System) data, with a focus on understanding and detecting model drift in production.

## Overview
This project uses real-world AIS data from the Danish Maritime Authority to train a vessel type classifier, then deliberately introduces distribution shifts to study how and why ML models degrade after deployment.

The classifier predicts vessel type (Cargo, Tanker, Fishing, Passenger, or Tug) from behavioral features derived from raw AIS signals (position, speed, course, MMSI). Because raw AIS data contains thousands of rows per vessel, the pipeline aggregates signals into a single feature vector per vessel per week before training.

## Core Topics

- Feature engineering from time-series AIS signals into per-vessel behavioral profiles
- Baseline classification using a Random Forest model
- Drift experiments across three axes:
    - **Temporal drift** — training on summer data, testing on different months
    - **Geographic drift** — training on a specific area, testing on another
    - **Composition drift** — balanced training set vs. real-world class imbalance
- Drift detection via performance monitoring and distribution-based methods (PSI, KL divergence)
- Explainability using SHAP to identify which features drive misclassifications under drift

## Motivation
Machine learning models in production often degrade silently as real-world data
shifts away from what they were trained on. This project explores that problem
end-to-end: training a working classifier, inducing realistic
drift, then learning to detect it (via performance and distribution monitoring)
and explain it (via SHAP) before it causes silent failures. The goal is to
build transferable skills in drift detection and explainability. The same
techniques apply directly to production ML systems like fraud detection,
churn prediction, or demand forecasting.

## Repository Structure

```
FleetSense/
├── fleetsense/              # main package
│   ├── data/                # download + per-vessel aggregation
│   ├── features/            # feature engineering
│   ├── model/               # baseline model training/evaluation
│   ├── monitoring/          # drift experiment splits + PSI/KL detection
│   ├── evaluation/          # SHAP explainability (in progress)
├── notebooks/               # exploratory analysis, one per project stage
├── scripts/                 # CLI entry points (e.g. full dataset generation)
├── tests/                   # unit tests
├── docs/                    # design decision notes
├── data/                    # raw/intermediate data (not committed — see data/README.md)
├── pyproject.toml
├── uv.lock
└── config.py
```


## Dataset
This project uses real-world AIS (Automatic Identification System) data from
the Danish Maritime Authority to train a vessel type classifier. AIS data is
well-suited for studying drift: vessel types have clear behavioral signatures,
drift scenarios are realistic (seasonal traffic, regional shipping differences),
and features like speed and stopping patterns are intuitively interpretable.


## Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't already have it:

   ​```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ​```

2. Clone the repository:

   ​```bash
   git clone https://github.com/mro201/FleetSense.git
   cd FleetSense
   ​```

3. Install dependencies:

   ​```bash
   uv sync
   ​```
## Data Pipeline

Danish Maritime Authority AIS data is structured as daily logs; each vessel reports at a varying frequency both between vessels and across days. The pipeline processes this in three stages:

​```bash
# 1. Download raw AIS data
uv run python -m fleetsense.data.download

# 2. Aggregate raw logs per vessel (memory-constrained streaming, one day at a time)
uv run python -m fleetsense.data.aggregate_per_vessel

# 3. Generate the final per-vessel feature dataset
uv run python -m fleetsense.features.generate_features
​```

Or run the full pipeline in one step for a given time interval:

​```bash
uv run scripts/generate_dataset.py --start 2024-06-01 --end 2024-07-31
​```

## Running the Project

Run commands inside the project's environment with `uv run` — no activation needed:

```bash
uv run jupyter lab
uv run ruff check .
uv run mypy
uv run pytest
```

Alternatively, activate the virtual environment directly:

```powershell
.venv\Scripts\activate
```

## Example Outputs

_(placeholder — fill in once drift experiments are complete)_

- Baseline model accuracy: ~87% (i.i.d. train/test split)
- Accuracy under temporal drift: ~79%
- Accuracy under geographic drift: ~74%
- Accuracy under composition drift: ~68% (72% F1)

## Main Findings

_(coming soon — to be written up once drift and explainability analysis is complete)_

## Future Work

- Retraining strategies to recover performance after detected drift
