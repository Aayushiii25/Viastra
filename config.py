"""
config.py — Central constants for VIASTRA.
All thresholds, paths, and defaults live here.
"""

# ── Model paths ────────────────────────────────────────────────────────────────
MODEL_RIDGE  = "models/ridge_cv_model.pkl"
MODEL_LASSO  = "models/lasso_cv_model.pkl"
MODEL_SCALER = "models/scaler.pkl"
MODEL_FEATS  = "models/feature_names.pkl"

# ── Risk thresholds (Damage Index, 0–100) ─────────────────────────────────────
THRESHOLD_HEALTHY   = 35   # score < 35  → HEALTHY
THRESHOLD_MODERATE  = 70   # score < 70  → MODERATE; else CRITICAL

# ── Risk display config ────────────────────────────────────────────────────────
RISK_CONFIG = {
    "HEALTHY":  {"class": "ok",     "dot": "🟢", "hex": "#8ce4e4"},
    "MODERATE": {"class": "warn",   "dot": "🟡", "hex": "#f5c842"},
    "CRITICAL": {"class": "danger", "dot": "🔴", "hex": "#ff6b6b"},
}

# ── Model performance (from training, sklearn 1.6.1, 5-fold CV) ───────────────
MODEL_METRICS = {
    "ridge": {"r2": 0.9827, "rmse": 4.97, "mae": 3.96, "alpha": 1.0,  "cv": 5},
    "lasso": {"r2": 0.9827, "rmse": 4.97, "mae": 3.96, "alpha": 0.01, "cv": 5},
}

# ── Simulation defaults ────────────────────────────────────────────────────────
SIM_DEFAULTS = {
    "drainage_score":    9,
    "asphalt_quality":   9,
    "heavy_vehicle_pct": 20,
}

# ── Feature display names ──────────────────────────────────────────────────────
FEATURE_LABELS = {
    "road_age_years":      "Road Age (Years)",
    "daily_traffic":       "Daily Traffic",
    "heavy_vehicle_pct":   "Heavy Vehicle %",
    "rainfall_mm":         "Rainfall (mm)",
    "waterlogging_events": "Waterlogging Events",
    "drainage_score":      "Drainage Score",
    "asphalt_quality":     "Asphalt Quality",
    "repair_count":        "Repair Count",
    "avg_temperature":     "Temperature (°C)",
    "soil_stability":      "Soil Stability",
}
