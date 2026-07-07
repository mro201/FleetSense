# DriftAnalysis: Data Pipeline & Feature Engineering

## Overview

This document describes the data pipeline and feature engineering process used in the DriftAnalysis project. The pipeline ingests raw AIS (Automatic Identification System) maritime vessel data, filters and organizes it per vessel, and produces weekly-aggregated feature tables used to train and evaluate a Random Forest classifier for vessel type prediction.

## Data Source

Raw AIS data is sourced from the Danish Maritime Authority, published as daily ZIP archives. Each archive contains AIS position and voyage reports for vessels operating in Danish waters, at a volume of roughly 10 million rows per day.


## Ingestion Pipeline

The ingestion pipeline consists of a download step and a daily processing step.

### Download

Daily AIS archives are downloaded directly from the Danish Maritime Authority's public data source, one day at a time, over a configured date range (currently covering the end of December 2025 through the end of June 2026). A short delay is introduced between each day's download to avoid overwhelming the server, and any failure for a given day (for example, a missing file) is logged and skipped without stopping the overall run.

### Daily Processing

For each day's archive, the process is:

1. The single data file inside the archive is extracted and loaded into memory, with common representations of missing values normalized to nulls. Loading is done lazily, so filtering can happen before the full day's data is materialized.
2. Records are filtered down to the five ship types of interest — Cargo, Tanker, Fishing, Tug, and Passenger — using a central list of allowed ship types defined once for the whole project.
3. The filtered records for that day are then split by vessel, using the vessel's IMO number (its unique international identifier) as the grouping key, and each vessel's records for that day are written out as their own file.


If a given day has no records matching the target ship types, that day is simply skipped. The identifier used throughout this pipeline is the vessel's IMO number rather than its MMSI, because the IMO number is permanent and never changes for a given ship, whereas the MMSI can change over a vessel's lifetime.

## Feature Engineering

Feature engineering operates on the per-vessel datasets produced by the ingestion pipeline and produces one row per vessel per week, written out to a single combined output file.

More features are generated at this stage than are actually used in the final classifier — several were computed purely for exploratory comparison during feature selection, and not all of them made it into the model input.

3.1 Data Handling Considerations

A few properties of the underlying data are taken into account when generating features:


The number of position reports per vessel varies enormously — some vessels have very few, others have thousands.
The distribution of ship types is unbalanced.
Some features have missing values, particularly draught and navigational status.
Reporting frequency for an individual vessel can itself change a lot over time, with periods of inactivity or lower reporting frequency.
