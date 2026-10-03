"""
benchmark.core.plotting
=======================
Publication-quality plotting engine for synthetic acoustic benchmark results.
Generates all 14 standard research and diagnostic figures.
"""

from __future__ import annotations
import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.edgecolor": "#cccccc",
        "axes.grid": True,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "lines.linewidth": 1.75,
        "lines.markersize": 6,
    })


def plot_spatial_accuracy_heatmap(
    az_grid: np.ndarray,
    el_grid: np.ndarray,
    error_matrix: np.ndarray,
    save_path: str,
    title: str = "Spatial Localization Error Heatmap (deg)"
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    c = ax.pcolormesh(az_grid, el_grid, error_matrix, cmap="viridis", shading="auto", edgecolors="none")
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Angular Error (deg)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("True Azimuth (deg)")
    ax.set_ylabel("True Elevation (deg)")
    ax.axvline(90.0, color="red", linestyle=":", alpha=0.6, label="Broadside (90°)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_snr_curves(
    snr_levels: List[float],
    white_errors: List[float],
    pink_errors: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(snr_levels, white_errors, "o-", color="#1f77b4", label="White Gaussian Noise")
    ax.plot(snr_levels, pink_errors, "s--", color="#d62728", label="1/f Pink Noise")
    ax.axhline(1.0, color="gray", linestyle=":", label="Tolerance Threshold (1.0°)")
    ax.set_title("Localization Error vs SNR", fontweight="bold")
    ax.set_xlabel("Signal-to-Noise Ratio (dB)")
    ax.set_ylabel("Median Angular Error (deg)")
    ax.set_ylim(bottom=0.0)
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_frequency_sweep(
    freqs: List[float],
    errors: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(freqs, errors, "o-", color="#2ca02c")
    ax.axvspan(500, 3000, color="#e2f0d9", alpha=0.5, label="Active MUSIC Band (500–3000 Hz)")
    ax.set_title("Localization Error across Frequency Spectrum", fontweight="bold")
    ax.set_xlabel("Tone Frequency (Hz)")
    ax.set_ylabel("Angular Error (deg)")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_csm_alpha_tradeoff(
    alphas: List[float],
    static_errors: List[float],
    dynamic_latencies: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    ax2 = ax1.twinx()
    l1 = ax1.plot(alphas, static_errors, "o-", color="#1f77b4", label="Static Noise Error (deg)")
    l2 = ax2.plot(alphas, dynamic_latencies, "s--", color="#ff7f0e", label="Dynamic Step Lag (frames)")
    ax1.set_xlabel("EMA Alpha (α)")
    ax1.set_ylabel("Static Error (deg)", color="#1f77b4")
    ax2.set_ylabel("Step Response Lag (frames)", color="#ff7f0e")
    ax1.set_title("CSM Smoothing Tradeoff: Noise Immunity vs Dynamic Lag", fontweight="bold")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_band_weighting_comparison(
    variant_names: List[str],
    errors: List[float],
    contrasts: List[float],
    save_path: str
) -> None:
    setup_style()
    x = np.arange(len(variant_names))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax2 = ax1.twinx()
    rects1 = ax1.bar(x - width/2, errors, width, label="Angular Error (deg)", color="#3470a3")
    rects2 = ax2.bar(x + width/2, contrasts, width, label="Peak Contrast (P/mean)", color="#e07a5f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(variant_names, rotation=15, ha="right")
    ax1.set_ylabel("Angular Error (deg)", color="#3470a3")
    ax2.set_ylabel("Peak Contrast Ratio", color="#e07a5f")
    ax1.set_title("Band Weighting Schemes: Error & Peak Contrast", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_microphone_fault_curves(
    num_dead_mics: List[int],
    errors: List[float],
    contrasts: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax2 = ax1.twinx()
    ax1.plot(num_dead_mics, errors, "o-", color="#d95f02", label="Error (deg)")
    ax2.plot(num_dead_mics, contrasts, "s--", color="#7570b3", label="Peak Contrast")
    ax1.set_xlabel("Number of Dead Microphones")
    ax1.set_ylabel("Angular Error (deg)", color="#d95f02")
    ax2.set_ylabel("Peak Contrast", color="#7570b3")
    ax1.set_title("Graceful Degradation Under Microphone Failure", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_multipath_curves(
    sirs_db: List[float],
    errors: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(sirs_db, errors, "^-", color="#9467bd")
    ax.axhline(1.0, color="gray", linestyle=":", label="1.0° Threshold")
    ax.set_xlabel("Signal-to-Interference Ratio (SIR dB)")
    ax.set_ylabel("Angular Error (deg)")
    ax.set_title("Robustness to Acoustic Multipath Reflection", fontweight="bold")
    ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_profiling_breakdown(
    stages: List[str],
    durations_ms: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    colors = ["#2b5c8f", "#4682b4", "#5c9ebc", "#87ceeb"]
    bars = ax.barh(stages, durations_ms, color=colors[:len(stages)])
    ax.set_xlabel("Execution Time (ms)")
    ax.set_title("Acoustic Localization Pipeline Latency Breakdown", fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height()/2, f"{w:.2f} ms", va="center")
    ax.set_xlim(right=max(durations_ms) * 1.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_multi_source_resolution(
    separations_deg: List[float],
    resolved_fraction: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(separations_deg, [f * 100.0 for f in resolved_fraction], "d-", color="#e7298a")
    ax.set_xlabel("Angular Separation (deg)")
    ax.set_ylabel("Resolution Success Rate (%)")
    ax.set_title("Two-Source Angular Resolution Limit", fontweight="bold")
    ax.set_ylim(-5, 105)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_motion_tracking(
    speeds_dps: List[float],
    errors: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(speeds_dps, errors, "v-", color="#1b9e77")
    ax.set_xlabel("Source Angular Speed (deg/s)")
    ax.set_ylabel("Dynamic Tracking Lag / Error (deg)")
    ax.set_title("Moving Source Dynamic Tracking Error", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_queue_behavior(
    frame_indices: List[int],
    fifo_latencies_ms: List[float],
    latest_latencies_ms: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(frame_indices, fifo_latencies_ms, "r--", label="Legacy FIFO Buffer (Unbounded Backlog)")
    ax.plot(frame_indices, latest_latencies_ms, "b-", label="Production Latest-Frame Queue (Constant Real-Time)")
    ax.set_xlabel("Audio Frame Sequence")
    ax.set_ylabel("Age of Processed Audio (ms)")
    ax.set_title("Real-Time Latency: Unbounded FIFO vs Latest-Frame Queue", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_contrast_vs_snr(
    snrs: List[float],
    contrasts: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(snrs, contrasts, "o-", color="#ff7f00")
    ax.set_xlabel("Signal-to-Noise Ratio (dB)")
    ax.set_ylabel("Peak-to-Floor Contrast Ratio")
    ax.set_title("Acoustic Map Contrast vs SNR", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_elevation_bias_distribution(
    elevations: List[float],
    biases: List[float],
    save_path: str
) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(elevations, biases, "s-", color="#66a61e")
    ax.axhline(0.0, color="gray", linestyle=":")
    ax.set_xlabel("True Elevation (deg)")
    ax.set_ylabel("Elevation Estimation Bias (deg)")
    ax.set_title("Elevation Estimation Bias vs Aperture Angle", fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_variant_timing_comparison(
    variant_names: List[str],
    median_times_ms: List[float],
    p95_times_ms: List[float],
    save_path: str
) -> None:
    setup_style()
    x = np.arange(len(variant_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.bar(x - width/2, median_times_ms, width, label="Median Time (ms)", color="#4575b4")
    ax.bar(x + width/2, p95_times_ms, width, label="P95 Time (ms)", color="#d73027")
    ax.axhline(85.3, color="gray", linestyle="--", label="Audio Frame Duration (85.3 ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, rotation=15, ha="right")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("Computational Latency Across Algorithm Variants", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
