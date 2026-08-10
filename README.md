# FleetSense: Vessel Classification with Drift Detection and Monitoring

A production-style system for classifying maritime vessel types from AIS (Automatic Identification System) data: a trained classifier served behind a versioned API, with a monitoring layer that watches both the input data and the model's own predictions for drift — including without any ground truth labels — and flags it before it becomes a silent failure.

## System at a Glance

- **Model**: Random Forest vessel classifier (Cargo, Tanker, Fishing, Passenger, Tug), trained on weekly per-vessel behavioral features (position, speed, course, draught) derived from raw AIS signals, with the final feature set chosen automatically via permutation importance
- **Serving**: containerized FastAPI service (`/predict`, `/health`), with the model and its feature schema saved and version-checked together as one artifact
- **Monitoring**: PSI-based drift detection on both input features and predicted-class mix, per class, against a fixed reference baseline
- **Alarms**: threshold-based drift alarms flagging specific feature/period/class breaches, not just raw PSI numbers
- **Engineering**: automated tests, CI (Ruff, mypy, pytest), and a reproducible train/test split strategy (temporal, geographical, compositional)
- **Frontend**: Streamlit demo (`app/`)

The retraining trigger and importance-weighted drift scoring described below are the current in-progress milestone — see [Architecture](#architecture).

## Architecture
![alt text](<Skjermbilde 2026-07-29 132046.png>)

**Built:** prediction serving, feature and prediction-level PSI monitoring, threshold-based alarms (`check_drift`), permutation-importance-based feature selection.

**In progress:** weighting drift alarms by permutation importance (see [Main Findings](#main-findings) for why this matters), automated prediction logging, and an automated retrain trigger closing the loop.

## Main Findings

Running the drift monitoring against a full year of real AIS data (train: first three months, evaluation: remaining nine) surfaced a broad, seasonal-looking shift concentrated in a small set of features — vessel position (`lat_mean`, `lon_mean`) and reporting frequency — affecting nearly every vessel type at once through autumn and winter.

Despite that, per-class F1 stayed essentially flat over the same period, and the model's predicted class mix barely moved. The reason: the features driving the drift ranked low in permutation importance, so the model was never relying on them much in the first place. This is the core argument for weighting drift alarms by feature importance rather than treating every PSI breach as equally urgent — a large shift in a feature the model ignores is a very different signal than the same shift in one it depends on.

Full write-up, caveats, and next steps: [`docs/design_note_psi_vs_performance.md`](docs/design_note_psi_vs_performance.md).

## Repository Structure

```
FleetSense/
├── fleetsense/              # main package
│   ├── data/                # download + per-vessel aggregation
│   ├── features/            # feature engineering, dataset handling, feature selection
│   ├── model/                # baseline model training/evaluation, artifact + schema saving, inference
│   ├── monitoring/          # drift experiment splits + PSI-based feature and prediction drift monitoring
│   ├── api/                 # FastAPI serving predictions
│   ├── config.py
├── notebooks/               # exploratory analysis, one per project stage
├── scripts/                 # CLI entry points (e.g. full dataset generation)
├── tests/                   # unit tests
├── docs/                    # design decision notes
├── data/                    # raw/intermediate data (not committed — see data/README.md)
├── app/                     # streamlit frontend app
├── Dockerfile               # containerized FastAPI serving
├── pyproject.toml
├── uv.lock
```

## Data

Source: real-world AIS data from the Danish Maritime Authority. Raw AIS data contains thousands of position/voyage rows per vessel; the pipeline aggregates this into one behavioral feature vector per vessel per week before training. AIS data suits studying drift well — vessel types have clear behavioral signatures, and drift scenarios (seasonal traffic, regional shipping differences) are realistic and interpretable rather than synthetic.

A small sample, `data/dataset/vessel_weekly_features_sample.csv` (~2.6MB, stratified 5% per `ship_type`, `random_state=42`), is committed so the model trains immediately after cloning, without a DMA download. **Use it for** fast local training and API demos. **Don't use it for** drift analysis — it's a thin slice of one snapshot with no meaningful temporal/geographic spread, so PSI computed against it won't reflect real drift. The full dataset stays `.gitignore`'d; regenerate via `scripts/generate_dataset.py`.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Clone and install dependencies:
   ```bash
   git clone https://github.com/mro201/FleetSense.git
   cd FleetSense
   uv sync
   ```
3. Run the project (no venv activation needed with `uv run`):
   ```bash
   uv run jupyter lab              # notebooks
   uv run ruff check .             # lint
   uv run mypy                     # type check
   uv run pytest                   # tests
   uv run uvicorn fleetsense.api.main:app --reload   # serve the API locally
   ```
   Visit `http://127.0.0.1:8000/docs` for the interactive API docs once the server's running.

   Or run the containerized API instead:
   ```bash
   docker build -t fleetsense-api .
   docker run -p 8000:8000 fleetsense-api
   ```

   Alternatively, activate the virtual environment directly:
   ```powershell
   .venv\Scripts\activate
   ```

## Motivation

Machine learning models in production often degrade silently as real-world data shifts away from what they were trained on — and without labels, performance metrics can't catch it. This project trains a classifier on three months of real AIS data, then uses the remaining nine months of real, naturally occurring drift — with labels withheld from the model but available for validation — to build and test a distribution-monitoring and alarm system that works without relying on performance metrics.

The same constraint — no reliable labels at prediction time — applies directly to production ML systems like fraud detection, churn prediction, or demand forecasting, which is the broader skill this project is meant to build.

Explainability (e.g. SHAP) is a possible future extension, not yet built.
