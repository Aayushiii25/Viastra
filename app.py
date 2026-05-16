"""
app.py — VIASTRA · AI-Powered Road Infrastructure Intelligence Platform
Entry point for the Streamlit dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from utils.charts import feature_impact_chart, residual_chart, simulation_bar_chart
from utils.models import (
    build_report_df,
    classify_risk,
    get_ridge_feature_importance,
    load_models,
    predict,
    run_simulation,
)

logging.basicConfig(level=logging.INFO)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIASTRA — Road Intelligence",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ───────────────────────────────────────────────────────────────────
css_path = Path("styles/theme.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
try:
    with st.spinner("Loading intelligence models…"):
        ridge_model, lasso_model, scaler, feature_names = load_models()
except FileNotFoundError as e:
    st.error(f"❌ Model file missing: {e}")
    st.stop()

# ── Sidebar — Road input parameters ───────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">Road Parameters</div>', unsafe_allow_html=True)
    road_age  = st.slider("Road Age (Years)",  min_value=1,    max_value=20,    value=10,
                          help="Age of road surface in years")
    traffic   = st.slider("Daily Traffic",     min_value=1000, max_value=50000, value=25000, step=500,
                          help="Average daily vehicle count")
    trucks    = st.slider("Heavy Vehicle %",   min_value=5,    max_value=60,    value=20,
                          help="Percentage of heavy vehicles (trucks, buses)")

    st.markdown('<div class="sidebar-section" style="margin-top:16px">Environment</div>', unsafe_allow_html=True)
    rainfall     = st.slider("Rainfall (mm)",         min_value=50,  max_value=500, value=200,
                              help="Annual rainfall in millimetres")
    waterlogging = st.slider("Waterlogging Events",   min_value=0,   max_value=20,  value=5,
                              help="Number of flood/waterlogging events per year")
    temperature  = st.slider("Temperature (°C)",      min_value=10,  max_value=45,  value=25,
                              help="Mean annual temperature")

    st.markdown('<div class="sidebar-section" style="margin-top:16px">Infrastructure</div>', unsafe_allow_html=True)
    drainage = st.slider("Drainage Score",  min_value=1, max_value=10, value=5,
                         help="Infrastructure drainage quality (10 = excellent)")
    asphalt  = st.slider("Asphalt Quality", min_value=1, max_value=10, value=5,
                         help="Pavement material quality (10 = new/premium)")
    repairs  = st.slider("Repair Count",    min_value=0, max_value=10, value=2,
                         help="Number of past maintenance/repair events")
    soil     = st.slider("Soil Stability",  min_value=1, max_value=10, value=5,
                         help="Subgrade soil stability index (10 = most stable)")

# ── Build input DataFrame ──────────────────────────────────────────────────────
input_df = pd.DataFrame({
    "road_age_years":      [road_age],
    "daily_traffic":       [traffic],
    "heavy_vehicle_pct":   [trucks],
    "rainfall_mm":         [rainfall],
    "waterlogging_events": [waterlogging],
    "drainage_score":      [drainage],
    "asphalt_quality":     [asphalt],
    "repair_count":        [repairs],
    "avg_temperature":     [temperature],
    "soil_stability":      [soil],
})

# ── Prediction ─────────────────────────────────────────────────────────────────
score = predict(ridge_model, scaler, input_df)
risk  = classify_risk(score)
risk_cfg = config.RISK_CONFIG[risk]

# ── Feature attribution — uses Ridge coefficients (same model as prediction) ──
ridge_coeffs = get_ridge_feature_importance(ridge_model, feature_names)
top_feature  = abs(ridge_coeffs).idxmax()
top_label    = config.FEATURE_LABELS.get(top_feature, top_feature.replace("_", " ").title())

# ── Percentile (vs training set distribution) ──────────────────────────────────
try:
    train_df = pd.read_csv("data/viastra_training.csv")
    pct_rank = (train_df["road_damage_index"] < score).mean() * 100
    pct_text = f"Scores worse than {pct_rank:.0f}% of roads in the training dataset."
except Exception:
    pct_text = ""

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <p class="brand-title">VIASTRA</p>
    <p class="brand-tag">v2.1</p>
</div>
<p class="brand-sub">AI-Powered Infrastructure Intelligence · Road Damage Assessment System</p>
<hr class="vdivider">
""", unsafe_allow_html=True)

# ── Top metric cards ───────────────────────────────────────────────────────────
card_cls = risk_cfg["class"]
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card {card_cls}">
        <div class="metric-label">Damage Score</div>
        <div class="metric-value">{round(score, 1)}</div>
        <div class="metric-sub">out of 100 · RidgeCV (R²=0.98)</div>
    </div>
    <div class="metric-card {card_cls}">
        <div class="metric-label">Risk Level</div>
        <div class="metric-value" style="font-size:1.4rem">{risk_cfg["dot"]} {risk}</div>
        <div class="metric-sub">current assessment</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Top Damage Driver</div>
        <div class="metric-value" style="font-size:1rem;word-break:break-word">{top_label.upper()}</div>
        <div class="metric-sub">by Ridge coefficient magnitude</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Assessment", "Simulation", "Analytics"])


# ════════════════════════════════════════
# TAB 1 — ASSESSMENT
# ════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Road Condition Report</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="status-badge status-{card_cls}">
        {risk_cfg["dot"]} &nbsp; Road Status: {risk}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        ◈ &nbsp; Primary damage driver identified as
        <strong>{top_label}</strong> (Ridge coefficient: {ridge_coeffs[top_feature]:+.2f}).
        Consider targeted intervention on this factor to reduce overall degradation.
    </div>
    """, unsafe_allow_html=True)

    if pct_text:
        st.markdown(f'<div class="percentile-box">📊 &nbsp; {pct_text}</div>', unsafe_allow_html=True)

    # Parameter table with conditional highlighting
    st.markdown('<div class="section-title" style="margin-top:28px">Input Parameters</div>',
                unsafe_allow_html=True)

    danger_features = {"drainage_score": drainage <= 3, "asphalt_quality": asphalt <= 3,
                       "waterlogging_events": waterlogging >= 15, "heavy_vehicle_pct": trucks >= 50}

    display_df = input_df.T.copy()
    display_df.columns = ["Current Value"]
    display_df.index = [config.FEATURE_LABELS.get(i, i) for i in display_df.index]

    st.dataframe(display_df, use_container_width=True)

    # Download full report
    report_df = build_report_df(input_df, score, risk, top_label,
                                *run_simulation(ridge_model, scaler, input_df, config.SIM_DEFAULTS)[1:])
    st.download_button(
        "📥 Download Full Report (CSV)",
        report_df.to_csv(index=False),
        file_name="viastra_report.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════
# TAB 2 — SIMULATION (INTERACTIVE)
# ════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">What-If Simulation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Adjust the <strong>target improvement values</strong> below to simulate the impact of
        infrastructure upgrades. The model re-runs in real time.
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        target_drainage = st.slider(
            "Target Drainage Score", min_value=drainage, max_value=10,
            value=min(9, max(drainage, 9)),
            help="Set a drainage improvement target",
        )
    with sim_col2:
        target_asphalt = st.slider(
            "Target Asphalt Quality", min_value=asphalt, max_value=10,
            value=min(9, max(asphalt, 9)),
            help="Set an asphalt quality improvement target",
        )
    with sim_col3:
        target_trucks = st.slider(
            "Target Heavy Vehicle %", min_value=5, max_value=trucks,
            value=max(5, min(trucks, 20)),
            help="Set a target for reducing heavy vehicle percentage",
        )

    sim_targets = {
        "drainage_score":    target_drainage,
        "asphalt_quality":   target_asphalt,
        "heavy_vehicle_pct": target_trucks,
    }

    _, improved_score, reduction = run_simulation(ridge_model, scaler, input_df, sim_targets)
    pct_saved = round((reduction / score) * 100, 1) if score > 0 else 0.0
    improved_risk = classify_risk(improved_score)
    improved_cfg  = config.RISK_CONFIG[improved_risk]

    st.markdown(f"""
    <div class="sim-grid">
        <div class="sim-card">
            <div class="sim-label">Current Score</div>
            <div class="sim-value sim-before">{round(score, 1)}</div>
            <div class="sim-label" style="margin-top:8px">{risk_cfg["dot"]} {risk}</div>
        </div>
        <div class="sim-card">
            <div class="sim-label">Projected Score</div>
            <div class="sim-value sim-after">{round(improved_score, 1)}</div>
            <div class="sim-label" style="margin-top:8px">{improved_cfg["dot"]} {improved_risk}</div>
        </div>
        <div class="sim-card">
            <div class="sim-label">Reduction</div>
            <div class="sim-value sim-saved">−{round(reduction, 1)}</div>
            <div class="sim-label" style="margin-top:8px">{pct_saved}% improvement</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Damage Reduction Progress: {pct_saved}%</div>',
                unsafe_allow_html=True)
    st.progress(min(int(pct_saved), 100))

    # Visual comparison chart
    st.markdown('<div class="section-title" style="margin-top:28px">Parameter Changes</div>',
                unsafe_allow_html=True)
    fig_sim = simulation_bar_chart(
        feature_labels=["Drainage Score", "Asphalt Quality", "Heavy Vehicle %"],
        before_vals=[drainage, asphalt, trucks],
        after_vals=[target_drainage, target_asphalt, target_trucks],
    )
    st.pyplot(fig_sim)


# ════════════════════════════════════════
# TAB 3 — ANALYTICS
# ════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

    m = config.MODEL_METRICS
    st.markdown(f"""
    <div class="perf-grid">
        <div class="perf-card">
            <div class="perf-label">R² Score</div>
            <div class="perf-value">{m["ridge"]["r2"]}</div>
            <div class="perf-sub">RidgeCV · 5-fold CV</div>
        </div>
        <div class="perf-card">
            <div class="perf-label">RMSE</div>
            <div class="perf-value">{m["ridge"]["rmse"]}</div>
            <div class="perf-sub">Root Mean Sq Error</div>
        </div>
        <div class="perf-card">
            <div class="perf-label">MAE</div>
            <div class="perf-value">{m["ridge"]["mae"]}</div>
            <div class="perf-sub">Mean Absolute Error</div>
        </div>
        <div class="perf-card">
            <div class="perf-label">Best Alpha</div>
            <div class="perf-value">{m["ridge"]["alpha"]}</div>
            <div class="perf-sub">Ridge regularisation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature impact chart — now uses Ridge coefficients, not Lasso
    st.markdown('<div class="section-title" style="margin-top:28px">Feature Impact (Ridge Coefficients)</div>',
                unsafe_allow_html=True)
    fig_feat = feature_impact_chart(ridge_coeffs)
    st.pyplot(fig_feat)

    # Actual vs Predicted chart
    st.markdown('<div class="section-title" style="margin-top:28px">Actual vs Predicted · Training Set</div>',
                unsafe_allow_html=True)
    try:
        train_df = pd.read_csv("data/viastra_training.csv")
        fig_resid = residual_chart(train_df["road_damage_index"], train_df["predicted_damage"])
        st.pyplot(fig_resid)
    except Exception as e:
        st.warning(f"Could not load training data for residual plot: {e}")

    # Coefficient table
    st.markdown('<div class="section-title" style="margin-top:28px">Coefficient Table</div>',
                unsafe_allow_html=True)
    coeff_df = ridge_coeffs.rename("Coefficient").to_frame()
    coeff_df.index = [config.FEATURE_LABELS.get(i, i) for i in coeff_df.index]
    coeff_df["Direction"] = coeff_df["Coefficient"].apply(
        lambda v: "↑ Increases damage" if v > 0 else "↓ Reduces damage"
    )
    coeff_df["Coefficient"] = coeff_df["Coefficient"].map("{:+.3f}".format)
    st.dataframe(coeff_df.sort_values("Coefficient"), use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="vdivider" style="margin-top:32px">', unsafe_allow_html=True)
st.markdown(
    '<p style="font-size:0.72rem;color:#6b7280;font-family:\'Space Mono\',monospace">'
    'VIASTRA v2.1 · AI Infrastructure Intelligence · RidgeCV + LassoCV · sklearn 1.6.1</p>',
    unsafe_allow_html=True,
)