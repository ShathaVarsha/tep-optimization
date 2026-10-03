"""Publication-Grade Plot Generator for Tennessee Eastman Process Optimization.

Generates high-resolution (300 DPI) visual figures for academic reports and presentations:
  1. surrogate_cv_r2_comparison.png: 7-Variable vs 8-Variable Cross-Validation R² scores.
  2. optimization_objectives_comparison.png: Objective improvements across Production, Cost, and Energy.
  3. manipulated_variables_settings.png: Baseline vs Optimized valve openings (%).
  4. process_responses_comparison.png: Predicted process responses under optimal policies.
  5. dominant_sensitivities_ranking.png: Largest local what-if finite-difference gradients.

Author: Team B4 (23MNG336: Operational Research)
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set clean, professional visual styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR = Path("OR_final_visuals")
VIS_DIR.mkdir(parents=True, exist_ok=True)


def plot_surrogate_comparison():
    """Plot 1: 7-variable vs 8-variable Cross-Validation R2 comparison."""
    targets = ["Product Flow\n(XMEAS_17)", "Reactor Pressure\n(XMEAS_07)", "Purge Flow\n(XMEAS_10)", "Steam Flow\n(XMEAS_19)", "Compressor Work\n(XMEAS_20)"]
    r2_7 = [-0.0627, 0.3036, 0.8780, 0.7742, 0.3590]
    r2_8 = [0.9951, 0.3005, 0.8775, 0.7733, 0.3602]

    x = np.arange(len(targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width/2, r2_7, width, label="7 Variables (Standard)", color="#E57373", edgecolor="#C62828", alpha=0.9)
    rects2 = ax.bar(x + width/2, r2_8, width, label="8 Variables (+XMV_11)", color="#4CAF50", edgecolor="#2E7D32", alpha=0.9)

    ax.set_ylabel("Cross-Validated R² Score (TimeSeriesSplit k=5)")
    ax.set_title("Surrogate Model Cross-Validation Performance: Impact of Condenser CW (XMV_11)")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylim(-0.2, 1.15)
    ax.legend(frameon=True, facecolor="white", loc="upper left")

    # Annotate the breakthrough on product flow
    ax.annotate(
        "Breakthrough: R² jumps from -0.06 to +0.995\nvia Condenser CW Coupling",
        xy=(0 + width/2, 0.995),
        xytext=(0.4, 0.82),
        arrowprops=dict(facecolor="#2E7D32", shrink=0.08, width=1.5, headwidth=7),
        fontsize=9.5,
        fontweight="bold",
        color="#1B5E20",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="#4CAF50", facecolor="#E8F5E9"),
    )

    for bar in rects1:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3 if h >= 0 else -10),
                    textcoords="offset points", ha="center", va=va, fontsize=8.5)

    for bar in rects2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "surrogate_cv_r2_comparison.png")
    fig.savefig(VIS_DIR / "surrogate_cv_r2_comparison.png")
    plt.close(fig)
    print("  [OK] Saved surrogate_cv_r2_comparison.png")


def plot_optimization_objectives():
    """Plot 2: Baseline vs Optimized for all three LP problem formulations."""
    problems = ["Production Maximization\n(Product Flow, m³/hr)", "Operating Cost Minimization\n(Approx. Cost, $/hr)", "Total Energy Minimization\n(Equivalent Power, kW)"]
    baseline = [22.9093, 106.4015, 485.8436]
    optimized = [24.6813, 97.7254, 469.0334]
    deltas = ["+7.73%", "-8.15%", "-3.46%"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    colors_base = ["#90CAF9", "#FFE082", "#A5D6A7"]
    colors_opt = ["#1976D2", "#FFA000", "#388E3C"]

    for i in range(3):
        ax = axes[i]
        bars = ax.bar(["Baseline", "Optimized"], [baseline[i], optimized[i]], color=[colors_base[i], colors_opt[i]], width=0.55, edgecolor="black", alpha=0.9)
        ax.set_title(problems[i])
        ax.set_ylim(0, max(baseline[i], optimized[i]) * 1.22)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.annotate(
            f"Improvement: {deltas[i]}",
            xy=(0.5, 0.88),
            xycoords="axes fraction",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color="#0D47A1" if i == 0 else ("#E65100" if i == 1 else "#1B5E20"),
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="#FAFAFA"),
        )

    plt.suptitle("Linear Programming Optimization: Baseline vs Optimal Solutions (HiGHS Dual Simplex)", y=1.03)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "optimization_objectives_comparison.png")
    fig.savefig(VIS_DIR / "optimization_objectives_comparison.png")
    plt.close(fig)
    print("  [OK] Saved optimization_objectives_comparison.png")


def plot_manipulated_variables():
    """Plot 3: Baseline vs Optimal Manipulated Decision Variable Setpoints."""
    vars_names = ["XMV_01\n(D Feed)", "XMV_02\n(E Feed)", "XMV_03\n(A Feed)", "XMV_04\n(A+C Feed)", "XMV_05\n(Recycle)", "XMV_06\n(Purge)", "XMV_09\n(Steam)", "XMV_11\n(Condenser)"]
    u_base = [63.031, 54.006, 24.721, 61.322, 22.252, 40.082, 47.461, 18.216]
    u_prod = [64.436, 55.111, 33.682, 57.903, 20.811, 36.385, 42.162, 14.009]
    u_cost = [61.296, 54.990, 17.781, 57.903, 21.147, 36.128, 42.162, 18.259]

    x = np.arange(len(vars_names))
    width = 0.28

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - width, u_base, width, label="Baseline Setpoint", color="#B0BEC5", edgecolor="#455A64")
    ax.bar(x, u_prod, width, label="Production Max Setpoint (+7.73%)", color="#29B6F6", edgecolor="#0288D1")
    ax.bar(x + width, u_cost, width, label="Cost Min Setpoint (-8.15%)", color="#AB47BC", edgecolor="#7B1FA2")

    ax.set_ylabel("Manipulated Valve Opening Setpoint (%)")
    ax.set_title("Optimal Manipulated Valve Settings across Optimization Formulations")
    ax.set_xticks(x)
    ax.set_xticklabels(vars_names)
    ax.set_ylim(0, 85)
    ax.legend(frameon=True, facecolor="white", loc="upper right")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "manipulated_variables_settings.png")
    fig.savefig(VIS_DIR / "manipulated_variables_settings.png")
    plt.close(fig)
    print("  [OK] Saved manipulated_variables_settings.png")


def plot_process_responses():
    """Plot 4: State responses across baseline, production max, and cost min."""
    responses = ["Product Flow\n(m³/hr)", "Reactor Pressure\n(kPa gauge)", "Purge Rate\n(kscmh)", "Steam Flow\n(kg/hr)", "Compressor Work\n(kW)"]
    # Normalized to baseline = 100%
    base_norm = [100.0, 100.0, 100.0, 100.0, 100.0]
    prod_norm = [24.6813/22.9093*100, 2702.68/2705.40*100, 0.3051/0.3376*100, 210.55/230.28*100, 337.77/341.47*100]
    cost_norm = [22.9093/22.9093*100, 2698.67/2705.40*100, 0.3051/0.3376*100, 209.37/230.28*100, 337.77/341.47*100]

    x = np.arange(len(responses))
    width = 0.28

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(x - width, base_norm, width, label="Baseline (100%)", color="#CFD8DC", edgecolor="#78909C")
    ax.bar(x, prod_norm, width, label="Production Max Policy", color="#26A69A", edgecolor="#00796B")
    ax.bar(x + width, cost_norm, width, label="Cost Min Policy", color="#FF7043", edgecolor="#D84315")

    ax.set_ylabel("Normalized Response (% of Baseline)")
    ax.set_title("Predicted Process Response Comparison Normalized to Nominal Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(responses)
    ax.axhline(100.0, color="gray", linestyle=":", linewidth=1.2)
    ax.set_ylim(80, 115)
    ax.legend(frameon=True, facecolor="white", loc="upper left")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "process_responses_comparison.png")
    fig.savefig(VIS_DIR / "process_responses_comparison.png")
    plt.close(fig)
    print("  [OK] Saved process_responses_comparison.png")


def plot_dominant_sensitivities():
    """Plot 5: Dominant Local What-If Sensitivity per Process Response."""
    labels = [
        "Product Flow (XMEAS_17)\n← XMV_11 (Condenser CW)",
        "Reactor Pressure (XMEAS_07)\n← XMV_05 (Compressor Recycle)",
        "Purge Rate (XMEAS_10)\n← XMV_06 (Purge Valve)",
        "Steam Flow (XMEAS_19)\n← XMV_09 (Steam Valve)",
        "Compressor Power (XMEAS_20)\n← XMV_05 (Compressor Recycle)",
    ]
    slopes = [-0.4128, 8.4188, 0.0075, 3.6876, 2.4757]
    colors = ["#D32F2F" if s < 0 else "#1976D2" for s in slopes]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.barh(labels, slopes, color=colors, alpha=0.85, edgecolor="black", height=0.55)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Local Surrogate Sensitivity Slope (Δy / Δu)")
    ax.set_title("Dominant Local What-If Sensitivity Influencers per Process Response")

    for bar in bars:
        w = bar.get_width()
        ha = "left" if w >= 0 else "right"
        offset = 0.15 if w >= 0 else -0.15
        ax.annotate(f"{w:+.4f}", xy=(w + offset, bar.get_y() + bar.get_height() / 2),
                    va="center", ha=ha, fontsize=10, fontweight="bold")

    ax.set_xlim(-2.0, 10.5)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "dominant_sensitivities_ranking.png")
    fig.savefig(VIS_DIR / "dominant_sensitivities_ranking.png")
    plt.close(fig)
    print("  [OK] Saved dominant_sensitivities_ranking.png")


def main():
    print("Generating high-resolution publication figures...")
    plot_surrogate_comparison()
    plot_optimization_objectives()
    plot_manipulated_variables()
    plot_process_responses()
    plot_dominant_sensitivities()
    print("All figures successfully rendered to results/plots/ and OR_final_visuals/.")


if __name__ == "__main__":
    main()
