# Model Card — VIASTRA ML Models

## Overview

VIASTRA uses a two-model pipeline trained on 5,000 synthetic road segment records:

- **RidgeCV** — primary prediction model (Damage Index 0–100)
- **LassoCV** — feature selection reference (sparse coefficient analysis)

Both models use `StandardScaler`-normalised inputs.

---

## RidgeCV — Primary Prediction Model

| Metric | Value |
|--------|-------|
| R² (full dataset) | **0.9827** |
| RMSE | 4.97 |
| MAE | 3.96 |
| Best alpha | 1.0 |
| Alpha search space | [0.01, 0.1, 1, 10, 100] |
| CV folds | 5 |

### Coefficient Table (standardised)

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| waterlogging_events | +17.37 | ↑ Increases damage |
| road_age_years | +14.47 | ↑ Increases damage |
| daily_traffic | +14.19 | ↑ Increases damage |
| heavy_vehicle_pct | +12.71 | ↑ Increases damage |
| rainfall_mm | +7.89 | ↑ Increases damage |
| repair_count | +5.81 | ↑ Increases damage |
| avg_temperature | +4.14 | ↑ Increases damage |
| soil_stability | -11.45 | ↓ Reduces damage |
| asphalt_quality | -11.46 | ↓ Reduces damage |
| drainage_score | -14.22 | ↓ Reduces damage |

---

## LassoCV — Feature Selection Reference

| Metric | Value |
|--------|-------|
| R² (full dataset) | 0.9827 |
| RMSE | 4.97 |
| MAE | 3.96 |
| Best alpha | 0.01 |
| CV folds | 5 |

---

## Training Environment

| Item | Value |
|------|-------|
| scikit-learn | 1.6.1 |
| Python | 3.10+ |
| Training samples | 5,000 |
| Features | 10 |
| Serialisation | pickle |

> ⚠️ Models serialised with scikit-learn 1.6.1. Use `requirements.txt` to match versions exactly.

---

## Limitations

- Trained on **synthetic** data. Performance on real-world road survey data requires re-training.
- No spatial or temporal features — assumes each road segment is independent.
- Heavy vehicle % capped at 60%; extrapolation beyond training range is unreliable.
