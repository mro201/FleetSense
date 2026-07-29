"""Unit tests for prediction logging in predict_baseline_proba."""

import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pytest
from fleetsense.model import base_model


def _fit_dummy_model(columns: list[str]) -> RandomForestClassifier:
    X = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], columns=columns)
    y = ["A", "B", "A", "B"]
    model = RandomForestClassifier(n_estimators=2, random_state=0)
    model.fit(X, y)
    return model


def _patch_model(monkeypatch, tmp_path, columns):
    model = _fit_dummy_model(columns)
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)
    monkeypatch.setattr(base_model, "MODEL_PATH", model_path)
    monkeypatch.setattr(base_model, "load_schema", lambda: {"columns": columns})
    monkeypatch.setattr(base_model, "FEATURES", columns)


def test_predict_baseline_proba_writes_valid_jsonl(tmp_path, monkeypatch):
    _patch_model(monkeypatch, tmp_path, ["f1", "f2"])
    log_path = tmp_path / "predict_log.json"
    monkeypatch.setattr(base_model, "LOG_PATH", log_path)

    base_model.predict_baseline_proba({"f1": 1, "f2": 2})
    base_model.predict_baseline_proba({"f1": 3, "f2": 4})

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        entry = json.loads(line)  # raises if not valid single-line JSON
        assert "timestamp" in entry
        assert "vessel_type" in entry
        assert "probabilities" in entry


def test_predict_baseline_proba_returns_prediction_when_log_write_fails(tmp_path, monkeypatch):
    _patch_model(monkeypatch, tmp_path, ["f1", "f2"])
    # Point LOG_PATH somewhere that can't be written (parent is a file, not a dir)
    bad_parent = tmp_path / "not_a_directory"
    bad_parent.write_text("blocking file")
    monkeypatch.setattr(base_model, "LOG_PATH", bad_parent / "log.json")

    result = base_model.predict_baseline_proba({"f1": 1, "f2": 2})

    assert "vessel_type" in result
    assert "probabilities" in result


def test_load_baseline_model_succeeds_when_schema_matches(tmp_path, monkeypatch):
    model = _fit_dummy_model(["f1", "f2"])
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    monkeypatch.setattr(base_model, "MODEL_PATH", model_path)
    monkeypatch.setattr(base_model, "load_schema", lambda: {"columns": ["f1", "f2"]})

    loaded = base_model.load_baseline_model()
    assert list(loaded.feature_names_in_) == ["f1", "f2"]


def test_load_baseline_model_raises_on_schema_mismatch(tmp_path, monkeypatch):
    model = _fit_dummy_model(["f1", "f2"])
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    monkeypatch.setattr(base_model, "MODEL_PATH", model_path)
    monkeypatch.setattr(base_model, "load_schema", lambda: {"columns": ["f1", "different_column"]})

    with pytest.raises(ValueError, match="drifted out of sync"):
        base_model.load_baseline_model()
