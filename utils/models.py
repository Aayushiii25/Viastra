"""utils/models.py — Model loading, prediction, and simulation logic."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model._coordinate_descent import LassoCV
from sklearn.linear_model._ridge import RidgeCV
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


def load_models() -> Tuple[RidgeCV, LassoCV, StandardScaler, list]:
    """Load persisted model artefacts from disk.

    Returns
    -------
    ridge_model, lasso_model, scaler, feature_names
    """
    def _load(path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(p, "rb") as f:
            return pickle.load(f)

    ridge_model   = _load(config.MODEL_RIDGE)
    lasso_model   = _load(config.MODEL_LASSO)
    scaler        = _load(config.MODEL_SCALER)
    feature_names = _load(config.MODEL_FEATS)

    logger.info("Models loaded successfully.")
    return ridge_model, lasso_model, scaler, feature_names


def predict(
    ridge_model: RidgeCV,
    scaler: StandardScaler,
    input_df: pd.DataFrame,
) -> float:
    """Return damage score for a single input row.

    Parameters
    ----------
    ridge_model : trained RidgeCV
    scaler      : fitted StandardScaler
    input_df    : 1-row DataFrame with all 10 feature columns

    Returns
    -------
    Damage index (float, 0–100 nominal range).
    """
    scaled = scaler.transform(input_df)
    score  = ridge_model.predict(scaled)[0]
    return float(np.clip(score, 0, 100))


def classify_risk(score: float) -> str:
    """Map a damage score to a risk tier label."""
    if score < config.THRESHOLD_HEALTHY:
        return "HEALTHY"
    if score < config.THRESHOLD_MODERATE:
        return "MODERATE"
    return "CRITICAL"


def get_ridge_feature_importance(
    ridge_model: RidgeCV,
    feature_names: list,
) -> pd.Series:
    """Return Ridge model coefficients as a named Series.

    NOTE: These coefficients describe the **Ridge** prediction, not Lasso.
    They are the correct basis for explaining individual predictions.
    """
    return pd.Series(ridge_model.coef_, index=feature_names)


def run_simulation(
    ridge_model: RidgeCV,
    scaler: StandardScaler,
    input_df: pd.DataFrame,
    targets: Dict[str, float],
) -> Tuple[float, float, float]:
    """Simulate the effect of infrastructure improvements.

    Parameters
    ----------
    targets : dict mapping feature names to their improved values.

    Returns
    -------
    (original_score, improved_score, reduction)
    """
    original = predict(ridge_model, scaler, input_df)

    improved_df = input_df.copy()
    for feat, val in targets.items():
        if feat in improved_df.columns:
            improved_df[feat] = val

    improved  = predict(ridge_model, scaler, improved_df)
    reduction = original - improved
    return original, improved, reduction


def build_report_df(
    input_df: pd.DataFrame,
    score: float,
    risk: str,
    top_driver: str,
    improved_score: float,
    reduction: float,
) -> pd.DataFrame:
    """Assemble a complete report DataFrame for CSV download."""
    row = input_df.copy()
    row["damage_score"]    = round(score, 2)
    row["risk_level"]      = risk
    row["top_driver"]      = top_driver
    row["improved_score"]  = round(improved_score, 2)
    row["score_reduction"] = round(reduction, 2)
    pct = round((reduction / score) * 100, 1) if score > 0 else 0.0
    row["reduction_pct"]   = pct
    return row
