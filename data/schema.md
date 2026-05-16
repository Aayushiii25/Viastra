# Data Schema — VIASTRA Training Dataset

**File:** `data/viastra_training.csv`  
**Rows:** 5,000 synthetic road segment records  
**Target:** `road_damage_index` (continuous, 0–100)

## Column Definitions

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `road_age_years` | int | 1–20 | Age of road surface in years |
| `daily_traffic` | int | 1,000–50,000 | Average daily vehicle count |
| `heavy_vehicle_pct` | float | 5–60 | Percentage of heavy vehicles (trucks, buses) |
| `rainfall_mm` | int | 50–500 | Annual rainfall in millimetres |
| `waterlogging_events` | int | 0–20 | Number of flood/waterlogging events per year |
| `drainage_score` | int | 1–10 | Infrastructure drainage quality (10 = excellent) |
| `asphalt_quality` | int | 1–10 | Pavement material quality (10 = new/premium) |
| `repair_count` | int | 0–10 | Number of past maintenance/repair events |
| `avg_temperature` | int | 10–45 | Mean annual temperature in °C |
| `soil_stability` | int | 1–10 | Subgrade soil stability index (10 = most stable) |
| `road_damage_index` | float | 0–100 | **Target** — composite road damage score |
| `predicted_damage` | float | 0–100 | RidgeCV model predicted score (training artifact) |
| `risk_level` | str | — | Categorical label: Healthy / Moderate / Critical |
| `x` | int | — | Simulated grid coordinate (not used in modelling) |
| `y` | int | — | Simulated grid coordinate (not used in modelling) |

## Risk Tier Thresholds

| Level | Score Range |
|-------|-------------|
| HEALTHY | 0–34 |
| MODERATE | 35–69 |
| CRITICAL | 70–100 |

## Data Generation

Synthetic data generated using correlated multivariate distributions calibrated against  
civil engineering literature on road pavement degradation factors.  
See `notebooks/01_model_training.ipynb` for full generation and validation details.
