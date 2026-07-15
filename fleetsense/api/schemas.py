from pydantic import BaseModel, create_model
from fleetsense.model.base_model import load_schema

_schema = load_schema()


def dtype_to_python_type(dtype_str: str) -> type:
    if dtype_str.startswith("float"):
        return float
    if dtype_str.startswith("int"):
        return int
    if dtype_str == "bool":
        return bool
    if dtype_str.startswith("object") or dtype_str.startswith("string"):
        return str
    raise ValueError(f"Unrecognized dtype {dtype_str!r} in saved schema — add a mapping for it.")


VesselFeatures = create_model(
    "VesselFeatures",
    **{col: (dtype_to_python_type(dtype), ...) for col, dtype in _schema["dtypes"].items()},
)


class PredictionResponse(BaseModel):
    vessel_type: str
    probabilities: dict[str, float]
