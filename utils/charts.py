"""utils/charts.py — Matplotlib figure builders for VIASTRA."""

from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Shared style constants ─────────────────────────────────────────────────────
BG    = "#13161d"
CYAN  = "#8ce4e4"
RED   = "#ff6b6b"
GOLD  = "#f5c842"
MUTED = "#6b7280"
BORD  = "#1e2330"
TEXT  = "#e8eaf0"
MINT  = "#cdefe3"


def _base_fig(w: float = 9, h: float = 5) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


def feature_impact_chart(coefficients: pd.Series) -> plt.Figure:
    """Horizontal bar chart of Ridge model coefficients."""
    sorted_c = coefficients.sort_values()
    colors   = [CYAN if v < 0 else RED for v in sorted_c.values]

    fig, ax = _base_fig(9, 5)
    ax.barh(
        [i.replace("_", " ").title() for i in sorted_c.index],
        sorted_c.values,
        color=colors,
        edgecolor="none",
        height=0.6,
    )

    ax.set_xlabel("Coefficient Weight", color=MUTED, fontsize=9, labelpad=10)
    ax.set_title(
        "Feature Impact · Ridge Damage Coefficients",
        color=CYAN, fontsize=11, fontfamily="monospace", pad=14,
    )
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[["top", "right", "bottom", "left"]].set_color(BORD)
    ax.axvline(0, color=BORD, linewidth=1)

    patch_pos = mpatches.Patch(color=RED,  label="Increases damage")
    patch_neg = mpatches.Patch(color=CYAN, label="Reduces damage")
    ax.legend(handles=[patch_pos, patch_neg],
              facecolor=BG, labelcolor=TEXT, fontsize=8, framealpha=0.6)

    plt.tight_layout()
    return fig


def residual_chart(actual: pd.Series, predicted: pd.Series) -> plt.Figure:
    """Actual vs Predicted scatter with residual line."""
    fig, ax = _base_fig(8, 5)

    ax.scatter(actual, predicted, color=CYAN, alpha=0.3, s=8, edgecolors="none")

    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], color=GOLD, linewidth=1.2, linestyle="--",
            label="Perfect prediction")

    ax.set_xlabel("Actual Damage Index",    color=MUTED, fontsize=9)
    ax.set_ylabel("Predicted Damage Index", color=MUTED, fontsize=9)
    ax.set_title(
        "Actual vs Predicted · RidgeCV",
        color=CYAN, fontsize=11, fontfamily="monospace", pad=14,
    )
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[["top", "right", "bottom", "left"]].set_color(BORD)
    ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=8, framealpha=0.6)

    plt.tight_layout()
    return fig


def simulation_bar_chart(
    feature_labels: list[str],
    before_vals: list[float],
    after_vals: list[float],
) -> plt.Figure:
    """Side-by-side bar chart comparing before/after simulation targets."""
    x     = np.arange(len(feature_labels))
    width = 0.35

    fig, ax = _base_fig(7, 4)
    ax.bar(x - width / 2, before_vals, width, color=RED,  alpha=0.8, label="Current",  edgecolor="none")
    ax.bar(x + width / 2, after_vals,  width, color=CYAN, alpha=0.8, label="Simulated", edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, color=MUTED, fontsize=8)
    ax.set_ylabel("Parameter Value", color=MUTED, fontsize=9)
    ax.set_title("Simulation Targets · Before vs After",
                 color=CYAN, fontsize=11, fontfamily="monospace", pad=14)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[["top", "right", "bottom", "left"]].set_color(BORD)
    ax.legend(facecolor=BG, labelcolor=TEXT, fontsize=8, framealpha=0.6)

    plt.tight_layout()
    return fig
