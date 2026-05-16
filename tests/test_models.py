"""tests/test_models.py — Smoke tests for model loading and inference."""

import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

import config
from utils.models import (
    load_models,
    predict,
    classify_risk,
    run_simulation,
    get_ridge_feature_importance,
    build_report_df,
)


@pytest.fixture(scope="module")
def loaded_models():
    return load_models()


@pytest.fixture(scope="module")
def sample_input():
    return pd.DataFrame({
        "road_age_years":      [10],
        "daily_traffic":       [25000],
        "heavy_vehicle_pct":   [20],
        "rainfall_mm":         [200],
        "waterlogging_events": [5],
        "drainage_score":      [5],
        "asphalt_quality":     [5],
        "repair_count":        [2],
        "avg_temperature":     [25],
        "soil_stability":      [5],
    })


# ── Load tests ─────────────────────────────────────────────────────────────────

def test_models_load(loaded_models):
    ridge, lasso, scaler, features = loaded_models
    assert ridge is not None
    assert lasso is not None
    assert scaler is not None
    assert len(features) == 10


def test_feature_names_correct(loaded_models):
    _, _, _, features = loaded_models
    expected = [
        "road_age_years", "daily_traffic", "heavy_vehicle_pct",
        "rainfall_mm", "waterlogging_events", "drainage_score",
        "asphalt_quality", "repair_count", "avg_temperature", "soil_stability",
    ]
    assert features == expected


# ── Prediction tests ───────────────────────────────────────────────────────────

def test_predict_returns_float(loaded_models, sample_input):
    ridge, _, scaler, _ = loaded_models
    score = predict(ridge, scaler, sample_input)
    assert isinstance(score, float)


def test_predict_in_range(loaded_models, sample_input):
    ridge, _, scaler, _ = loaded_models
    score = predict(ridge, scaler, sample_input)
    assert 0 <= score <= 100, f"Score {score} out of expected range"


def test_predict_high_damage(loaded_models):
    """Very old, high traffic, no maintenance should score high."""
    ridge, _, scaler, _ = loaded_models
    bad = pd.DataFrame({
        "road_age_years":      [20],
        "daily_traffic":       [50000],
        "heavy_vehicle_pct":   [60],
        "rainfall_mm":         [500],
        "waterlogging_events": [20],
        "drainage_score":      [1],
        "asphalt_quality":     [1],
        "repair_count":        [10],
        "avg_temperature":     [45],
        "soil_stability":      [1],
    })
    score = predict(ridge, scaler, bad)
    assert score > 50, f"Expected high score for worst-case input, got {score}"


def test_predict_low_damage(loaded_models):
    """New road, low traffic, excellent maintenance should score low."""
    ridge, _, scaler, _ = loaded_models
    good = pd.DataFrame({
        "road_age_years":      [1],
        "daily_traffic":       [1000],
        "heavy_vehicle_pct":   [5],
        "rainfall_mm":         [50],
        "waterlogging_events": [0],
        "drainage_score":      [10],
        "asphalt_quality":     [10],
        "repair_count":        [0],
        "avg_temperature":     [20],
        "soil_stability":      [10],
    })
    score = predict(ridge, scaler, good)
    assert score < 50, f"Expected low score for best-case input, got {score}"


# ── Risk classification tests ──────────────────────────────────────────────────

def test_classify_healthy():
    assert classify_risk(10.0) == "HEALTHY"
    assert classify_risk(34.9) == "HEALTHY"


def test_classify_moderate():
    assert classify_risk(35.0) == "MODERATE"
    assert classify_risk(69.9) == "MODERATE"


def test_classify_critical():
    assert classify_risk(70.0) == "CRITICAL"
    assert classify_risk(95.0) == "CRITICAL"


# ── Simulation tests ───────────────────────────────────────────────────────────

def test_simulation_reduces_score(loaded_models, sample_input):
    ridge, _, scaler, _ = loaded_models
    targets = {"drainage_score": 9, "asphalt_quality": 9, "heavy_vehicle_pct": 20}
    original, improved, reduction = run_simulation(ridge, scaler, sample_input, targets)
    assert improved <= original, "Improved simulation should not increase score"
    assert abs(reduction - (original - improved)) < 1e-6


def test_simulation_empty_targets(loaded_models, sample_input):
    """No targets → improved score equals original."""
    ridge, _, scaler, _ = loaded_models
    original, improved, reduction = run_simulation(ridge, scaler, sample_input, {})
    assert abs(original - improved) < 1e-6


# ── Feature importance tests ───────────────────────────────────────────────────

def test_ridge_feature_importance(loaded_models):
    ridge, _, _, features = loaded_models
    coeffs = get_ridge_feature_importance(ridge, features)
    assert len(coeffs) == 10
    assert all(isinstance(v, (float, np.floating)) for v in coeffs.values)


# ── Report tests ───────────────────────────────────────────────────────────────

def test_build_report_df(sample_input):
    df = build_report_df(sample_input, 55.0, "MODERATE", "waterlogging_events",
                         40.0, 15.0)
    assert "damage_score"   in df.columns
    assert "risk_level"     in df.columns
    assert "top_driver"     in df.columns
    assert "improved_score" in df.columns
    assert df["risk_level"].iloc[0] == "MODERATE"
    assert df["reduction_pct"].iloc[0] == pytest.approx(27.3, 0.1)
