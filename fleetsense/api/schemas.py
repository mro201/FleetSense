"""Request/response models for the vessel classification API.

VesselFeatures mirrors FEATURE_COLUMNS from fleetsense.models.baseline_rf — kept
as an explicit pydantic model (rather than a generic dict) so FastAPI validates
incoming requests and auto-generates the /docs schema.
"""

from pydantic import BaseModel


class VesselFeatures(BaseModel):
    mean_moving_speed: float
    max_speed: float
    std_speed: float
    sog_p10: float
    sog_median: float
    sog_p90: float
    frac_time_slow: float
    cog_variability: float
    rot_mean_abs: float
    rot_std: float
    heading_cog_diff_mean: float
    fishing_ratio: float
    anchor_ratio: float
    underway_engine_ratio: float
    moored_ratio: float
    lat_std: float
    lon_std: float
    lat_mean: float
    lon_mean: float
    bbox_area: float
    length: float
    width: float
    length_beam_ratio: float
    max_draught: float
    min_draught: float
    draught_variability: float
    draught_length_ratio: float
    CargoY_ratio: float
    CargoZ_ratio: float
    CargoOS_ratio: float
    CargoX_ratio: float
    CargoReserved_ratio: float
    n_pings: float
    time_span_seconds: float
    mean_ping_interval_seconds: float
    # Note: adjust to match your final 34-feature list exactly once decided


class PredictionResponse(BaseModel):
    vessel_type: str
    probabilities: dict[str, float]
