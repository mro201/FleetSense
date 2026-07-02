# DriftAnalysis
A machine learning pipeline for classifying maritime vessel types from AIS (Automatic Identification System) data, with a focus on understanding and detecting model drift in production.

## Overview
This project uses real-world AIS data from the Danish Maritime Authority to train a vessel type classifier, then deliberately introduces distribution shifts to study how and why ML models degrade after deployment.
The classifier predicts vessel type (Cargo, Tanker, Fishing, Passenger, or Tug) from behavioral features derived from raw AIS signals (position, speed, course, MMSI). Because raw AIS data contains thousands of rows per vessel, the pipeline aggregates signals into a single feature vector per vessel (e.g. mean speed, percentage of time stopped, vessel length) before training.

## Core Topics

- Feature engineering from time-series AIS signals into per-vessel behavioral profiles
- Baseline classification using a Random Forest (87% accuracy on i.i.d. data)
- Drift experiments across three axes:
    - Temporal drift — training on summer data, testing on different months
    - Geographic drift — training on Kattegat, testing on the Baltic Sea
    - Composition drift — balanced training set vs. real-world class imbalance
- Drift detection via performance monitoring and distribution-based methods (PSI, KL divergence)
- Explainability using SHAP to identify which features drive misclassifications under drift

## Why This Dataset
AIS data is well-suited for learning drift detection: vessel types have clear behavioral signatures, drift scenarios are realistic (seasonal traffic, regional shipping differences), and features like speed and stopping patterns are intuitively interpretable. The methodology — PSI monitoring, SHAP analysis, retraining strategies — transfers directly to other production ML domains such as fraud detection, churn modeling, and equipment failure prediction.

## Data
Raw AIS data: Danish Maritime Authority, 2024.

## Data preprocessing
Dansish maritime Authority AIS data is structured as logs aggregated day by day. each ship has varying frequency both between ships and days.

### Download data from Danish maritime authority
    '''python download.py'''

### Aggregate data for each vessel
    '''python PerVessel.py'''

### Generate dataset
    '''python DatasetGen.py'''
