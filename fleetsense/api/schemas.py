"""Request/response models for the vessel classification API.

VesselFeatures mirrors FEATURE_COLUMNS from fleetsense.models.baseline_rf — kept
as an explicit pydantic model (rather than a generic dict) so FastAPI validates
incoming requests and auto-generates the /docs schema.
"""

from pydantic import BaseModel


class VesselFeatures(BaseModel):
    length_beam_ratio: float
    draught_length_ratio: float
    width: float
    length: float
    min_draught: float
    max_draught: float
    lat_mean: float
    sog_p90: float
    CargoY_ratio: float
    lon_mean: float
    max_speed: float
    draught_variability: float
    sog_median: float
    fishing_ratio: float
    CargoZ_ratio: float
    anchor_ratio: float
    moored_ratio: float
    lon_std: float
    time_span_seconds: float
    rot_std: float
    frac_time_slow: float
    rot_mean_abs: float
    CargoOS_ratio: float
    n_pings: float
    cog_variability: float
    mean_ping_interval_seconds: float
    lat_std: float
    mean_moving_speed: float
    sog_p10: float


class PredictionResponse(BaseModel):
    vessel_type: str
    probabilities: dict[str, float]
