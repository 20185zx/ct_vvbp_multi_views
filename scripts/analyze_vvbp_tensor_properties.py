"""VVBP-Tensor properties analysis.

Validates why local patch (especially 3x3) improves model performance,
and characterises the effect of patch size on VVBP distributions.

Run:
    python scripts/analyze_vvbp_tensor_properties.py --config configs/cto_multirate.json
    python scripts/analyze_vvbp_tensor_properties.py --slice_idx 0 --sparse_views 72 --high_views 240
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_run_config
from src.geometry import load_or_generate_geo, FanBeamVVBPExtractor, build_linfbp_geo
from src.geometry.fanbeam import compute_deltas_cube_np as _compute_deltas_cube_np
from src.data.local_rank import compute_local_rank, compute_local_rank_sorted

try:
    import astra
except ImportError:
    astra = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vvbp_analysis")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AnalysisConfig:
    experiment_name: str = "vvbp_property_analysis"
    slice_idx: int = 0
    sparse_views: int = 72
    high_views: int = 240
    image_size: int = 256
    n_detec: int = 672
    d_detec: float = 1.0
    d_voxel: float = 1.0
    DSO: float = 595.0
    DOD: float = 480.0
    region: Tuple[int, int, int, int] = (32, 224, 32, 224)  # center region with margin
    num_pixels: int = 1000
    patch_sizes: Tuple[int, ...] = (1, 3, 5, 7)
    n_interp: int = 240  # quantile grid points for sorted curve comparison
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "outputs/vvbp_tensor_property_analysis"
    dicom_folder: Optional[str] = None
    results_folder: Optional[str] = None
    checkpoint_dir: Optional[str] = None  # dir with model .pt + stats_cached.pt
    model_name: str = "local rank center integral mlp, 10 epochs"


# ---------------------------------------------------------------------------
# ASTRA sinogram generation (reuses project conventions)
# ---------------------------------------------------------------------------
def generate_sinogram_astra(
    image_np: np.ndarray,
    views: int,
    n_detec: int,
    d_detec: float,
    DSO: float,
    DOD: float,
) -> np.ndarray:
    """Forward project a 2D attenuation image using ASTRA fan-beam.

    Args:
        image_np: [H, W] attenuation map.
        views: number of projection views.
        n_detec: detector elements.
        d_detec: detector pitch.
        DSO, DOD: source-origin / origin-detector distances.

    Returns:
        sino: [views, n_detec] sinogram.
    """
    if astra is None:
        raise ImportError("ASTRA toolbox is required for sinogram generation.")

    H, W = image_np.shape
    vol_geom = astra.create_vol_geom(H, W)
    angles = np.linspace(0.0, 2.0 * np.pi, int(views), endpoint=False).astype(np.float32)
    proj_geom = astra.create_proj_geom("fanflat", d_detec, n_detec, angles, DSO, DOD)
    projector_id = astra.create_projector("line_fanflat", proj_geom, vol_geom)

    sino_id, sino = astra.create_sino(
        np.ascontiguousarray(image_np, dtype=np.float32), projector_id
    )
    astra.data2d.delete(sino_id)
    astra.projector.delete(projector_id)
    return sino.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# VVBP extraction wrapper
# ---------------------------------------------------------------------------
def compute_vvbp(
    sino: np.ndarray,
    geo: dict,
    device: torch.device,
) -> torch.Tensor:
    """Run VVBP extractor on a sinogram.

    Args:
        sino: [V, D] sinogram.
        geo: LInFBP geometry dict with w1, filter, w2, indices.
        device: target device.

    Returns:
        vvbp: [1, 1, H, W, V] tensor.
    """
    sino_t = torch.from_numpy(sino[None, None, :, :].astype(np.float32)).to(device)
    with torch.no_grad():
        vvbp = FanBeamVVBPExtractor(geo).to(device).eval()(sino_t)
    return vvbp


# ---------------------------------------------------------------------------
# Patch gathering utilities
# ---------------------------------------------------------------------------
def gather_raw_patch_values(
    vvbp: torch.Tensor,
    xs: torch.Tensor,
    ys: torch.Tensor,
    patch_size: int,
) -> np.ndarray:
    """Gather raw (unsorted) VVBP values for each pixel's local patch.

    Args:
        vvbp: [1, 1, H, W, V] VVBP tensor.
        xs, ys: [P] pixel coordinates.
        patch_size: odd integer (1, 3, 5, ...).

    Returns:
        values: [P, patch_size**2, V] raw VVBP values.
    """
    r = patch_size // 2
    H, W = vvbp.shape[2], vvbp.shape[3]
    V = vvbp.shape[4]
    P = xs.shape[0]
    out = torch.zeros(P, patch_size * patch_size, V, dtype=vvbp.dtype, device=vvbp.device)

    idx = 0
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            out[:, idx, :] = vvbp[0, 0, xs + du, ys + dv, :]
            idx += 1
    return out.cpu().numpy()


def gather_center_values(vvbp: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> np.ndarray:
    """Gather per-view center-pixel VVBP values.

    Returns:
        [P, V] array.
    """
    return vvbp[0, 0, xs, ys, :].cpu().numpy()


# ---------------------------------------------------------------------------
# Detector coordinate computation (fan-beam geometry, exact)
# ---------------------------------------------------------------------------
def compute_detector_indices(geo: dict, xs: np.ndarray, ys: np.ndarray,
                             views: int) -> np.ndarray:
    """Compute fractional detector index I(pixel, view) using fan-beam geometry.

    Reuses compute_deltas_cube_np from src/geometry/fanbeam.py for the exact
    detector intersection math used by pixel_index_cal_numpy.

    Args:
        geo: LInFBP geometry dict (needs DSO, DSD, nDetecU, start_angle, end_angle,
             sVoxelX, sVoxelY, dVoxelX, dVoxelY, offOriginX, offOriginY, mode).
        xs, ys: [P] integer pixel coordinates.
        views: number of projection views.

    Returns:
        det_idx: [views, P] fractional detector index for each view × pixel.
    """
    nDetecU = int(geo["nDetecU"])
    alphas = np.linspace(geo["start_angle"], geo["end_angle"], int(views), endpoint=False)
    P = len(xs)

    det_idx = np.zeros((int(views), P), dtype=np.float32)

    for angle_idx in range(int(views)):
        alpha = -alphas[angle_idx]
        origin, deltaX, deltaY = _compute_deltas_cube_np(geo, alpha)

        # P_x = origin_x + ix * deltaX_x + iy * deltaY_x
        P_x = (origin["x"]
               + xs.astype(np.float32) * deltaX["x"]
               + ys.astype(np.float32) * deltaY["x"])
        P_y = (origin["y"]
               + xs.astype(np.float32) * deltaX["y"]
               + ys.astype(np.float32) * deltaY["y"])

        S_x = float(geo["DSO"])
        S_y = 0.0
        vectX = P_x - S_x
        vectY = P_y - S_y
        t = (geo["DSO"] - geo["DSD"] - S_x) / vectX
        y_proj = vectY * t + S_y
        det_idx[angle_idx, :] = y_proj + nDetecU / 2.0 - 0.5

    return det_idx  # [V, P]


# ---------------------------------------------------------------------------
# Sorted curve interpolation & distances
# ---------------------------------------------------------------------------
def interpolate_sorted_curve(values: np.ndarray, n_points: int = 240) -> np.ndarray:
    """Sort 1D values, then interpolate to a uniform quantile grid.

    Args:
        values: [N] 1D array.
        n_points: number of quantile grid points.

    Returns:
        [n_points] interpolated sorted curve.
    """
    sorted_vals = np.sort(values)
    orig_pos = np.linspace(0.0, 1.0, len(sorted_vals))
    target_pos = np.linspace(0.0, 1.0, n_points)
    return np.interp(target_pos, orig_pos, sorted_vals)


def compute_curve_distances(curve: np.ndarray, ref: np.ndarray) -> dict:
    """Compute distribution distances between two sorted curves.

    Args:
        curve, ref: [N] arrays on the same quantile grid.

    Returns:
        dict with MSE, MAE, Wasserstein-1, KS.
    """
    mse = float(np.mean((curve - ref) ** 2))
    mae = float(np.mean(np.abs(curve - ref)))
    # Wasserstein-1 on the quantile-uniform grid ≈ L1 mean
    wass = mae
    # KS: max absolute CDF difference
    # On the quantile grid, the empirical CDF at each point is the quantile
    # But we want sup|F_curve(v) - F_ref(v)|.  We evaluate both CDFs on a
    # combined set of sorted value positions.
    combined = np.sort(np.concatenate([curve, ref]))
    cdf_curve = np.searchsorted(np.sort(curve), combined, side="right") / len(curve)
    cdf_ref = np.searchsorted(np.sort(ref), combined, side="right") / len(ref)
    ks = float(np.max(np.abs(cdf_curve - cdf_ref)))
    return {"MSE": mse, "MAE": mae, "Wasserstein1": wass, "KS": ks}


# ---------------------------------------------------------------------------
# Analysis A: local patch vs center high-view distribution
# ---------------------------------------------------------------------------
def analysis_A(
    px_coords: np.ndarray,
    center_sparse: np.ndarray,
    center_high: np.ndarray,
    patch_values: dict[int, np.ndarray],
    n_interp: int,
    save_dir: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compare sorted VVBP curves: center sparse, center high, local patches.

    Output:
        - representative sorted curve comparison plot
        - distance summary CSV
        - distance boxplot
    """
    log.info("=== Analysis A: Local patch vs center high-view distribution ===")
    P = len(px_coords)
    dist_keys = ["MSE", "MAE", "Wasserstein1", "KS"]
    methods = ["center_sparse"] + [f"local{p}P" for p in sorted(patch_values.keys())]

    # Per-pixel distances
    all_dists: dict[str, list] = {m: {k: [] for k in dist_keys} for m in methods}

    # Interpolate each pixel's sorted curves
    for p_idx in range(P):
        ref_curve = interpolate_sorted_curve(center_high[p_idx], n_interp)

        # Center sparse
        cs_curve = interpolate_sorted_curve(center_sparse[p_idx], n_interp)
        d = compute_curve_distances(cs_curve, ref_curve)
        for k in dist_keys:
            all_dists["center_sparse"][k].append(d[k])

        # Local patches
        for ps, pv in patch_values.items():
            # Pool all patch values and interpolate
            pooled = pv[p_idx].ravel()  # [ps**2 * V]
            lp_curve = interpolate_sorted_curve(pooled, n_interp)
            d = compute_curve_distances(lp_curve, ref_curve)
            mkey = f"local{ps}P"
            for k in dist_keys:
                all_dists[mkey][k].append(d[k])

    # Build summary dataframe
    rows = []
    for m in methods:
        for k in dist_keys:
            vals = np.array(all_dists[m][k])
            rows.append({
                "method": m,
                "metric": k,
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p75": float(np.percentile(vals, 75)),
            })
    df_summary = pd.DataFrame(rows)

    # ---- Plot: representative sorted curves (pick a median-MSE pixel) ----
    mse_center = np.array(all_dists["center_sparse"]["MSE"])
    med_idx = int(np.argmin(np.abs(mse_center - np.median(mse_center))))
    # Select a few representative pixels
    rep_indices = [med_idx]
    if P > 3:
        # Add low and high MSE pixels
        low_idx = int(np.argmin(mse_center))
        high_idx = int(np.argmax(mse_center))
        rep_indices = sorted(set([low_idx, med_idx, high_idx]))

    fig, axes = plt.subplots(1, len(rep_indices), figsize=(7 * len(rep_indices), 5))
    if len(rep_indices) == 1:
        axes = [axes]

    for ax, p_idx in zip(axes, rep_indices):
        q_grid = np.linspace(0, 1, n_interp)
        ax.plot(q_grid, interpolate_sorted_curve(center_high[p_idx], n_interp),
                label="Center High-View", linewidth=2, color="black", linestyle="--")
        ax.plot(q_grid, interpolate_sorted_curve(center_sparse[p_idx], n_interp),
                label=f"Center Sparse (MSE={all_dists['center_sparse']['MSE'][p_idx]:.4e})",
                linewidth=1.5, alpha=0.8)
        for ps in sorted(patch_values.keys()):
            pooled = patch_values[ps][p_idx].ravel()
            curve = interpolate_sorted_curve(pooled, n_interp)
            mkey = f"local{ps}P"
            ax.plot(q_grid, curve,
                    label=f"Local {ps}x{ps} (MSE={all_dists[mkey]['MSE'][p_idx]:.4e})",
                    linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Quantile", fontsize=13)
        ax.set_ylabel("VVBP Value", fontsize=13)
        ax.set_title(f"Pixel ({px_coords[p_idx,0]}, {px_coords[p_idx,1]})", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "A_sorted_curves_representative.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- Boxplot of MSE across pixels ----
    fig, ax = plt.subplots(figsize=(8, 5))
    mse_data = [np.array(all_dists[m]["MSE"]) for m in methods]
    bp = ax.boxplot(mse_data, labels=methods, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(methods)))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("MSE (sorted curve)", fontsize=13)
    ax.set_title("Distribution Distance to Center High-View Curve", fontsize=14)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "A_distance_boxplot.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # CSV
    csv_path = os.path.join(save_dir, "A_distance_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    log.info("  Saved: %s", csv_path)

    # Per-pixel distances for follow-up analysis
    pixel_rows = []
    for p_idx in range(P):
        row = {"pixel_id": p_idx, "x": int(px_coords[p_idx, 0]), "y": int(px_coords[p_idx, 1])}
        for m in methods:
            row[f"{m}_MSE"] = all_dists[m]["MSE"][p_idx]
            row[f"{m}_MAE"] = all_dists[m]["MAE"][p_idx]
        pixel_rows.append(row)
    df_pixel = pd.DataFrame(pixel_rows)
    csv_path = os.path.join(save_dir, "A_pixel_distances.csv")
    df_pixel.to_csv(csv_path, index=False)
    log.info("  Saved: %s", csv_path)

    return df_summary, df_pixel


# ---------------------------------------------------------------------------
# Analysis B: patch size effect
# ---------------------------------------------------------------------------
def analysis_B(
    px_coords: np.ndarray,
    center_high: np.ndarray,
    patch_values: dict[int, np.ndarray],
    all_dists_A: dict,
    n_interp: int,
    save_dir: str,
) -> pd.DataFrame:
    """Analyse how patch size affects distribution distance.

    Output:
        - patch size vs median/mean distance line plot
        - per-pixel best patch size statistics
        - CSV with per-pixel distances for all patch sizes
    """
    log.info("=== Analysis B: Patch size effect ===")
    P = len(px_coords)
    patch_sizes = sorted(patch_values.keys())
    methods = ["center_sparse"] + [f"local{p}P" for p in patch_sizes]

    # Build per-pixel distances from each method
    pixel_dists = {}
    for m in methods:
        pixel_dists[m] = np.array(all_dists_A[m]["MSE"])

    # ---- Line plot: patch size vs distance ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["MSE", "MAE"]):
        means = []
        medians = []
        labels = ["1x1"] + [f"{p}x{p}" for p in patch_sizes if p > 1]
        # center sparse = 1x1 case
        means.append(np.mean(pixel_dists["center_sparse"]))
        medians.append(np.median(pixel_dists["center_sparse"]))
        for ps in patch_sizes:
            if ps == 1:
                continue
            mkey = f"local{ps}P"
            vals = np.array(all_dists_A[mkey][metric])
            means.append(np.mean(vals))
            medians.append(np.median(vals))

        x_pos = np.arange(len(labels))
        ax.plot(x_pos, means, "o-", label="Mean", linewidth=2, markersize=8)
        ax.plot(x_pos, medians, "s--", label="Median", linewidth=2, markersize=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_xlabel("Patch Size", fontsize=13)
        ax.set_ylabel(f"{metric} to High-View Curve", fontsize=13)
        ax.set_title(f"Patch Size vs {metric}", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "B_patch_size_effect.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- Best patch size per pixel ----
    # Which method gives the lowest MSE for each pixel?
    mse_matrix = np.column_stack([pixel_dists[m] for m in methods])  # [P, n_methods]
    best_indices = np.argmin(mse_matrix, axis=1)
    best_counts = pd.Series(best_indices).value_counts().sort_index()
    best_ratio = best_counts / P

    fig, ax = plt.subplots(figsize=(8, 5))
    method_names_short = ["1x1"] + [f"{p}x{p}" for p in patch_sizes if p > 1]
    ax.bar(range(len(method_names_short)),
           [best_ratio.get(i, 0) * 100 for i in range(len(method_names_short))],
           color=plt.cm.viridis(np.linspace(0.2, 0.9, len(method_names_short))))
    ax.set_xticks(range(len(method_names_short)))
    ax.set_xticklabels(method_names_short, fontsize=12)
    ax.set_ylabel("Best Method Ratio (%)", fontsize=13)
    ax.set_title("Per-Pixel Best Patch Size", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "B_best_patch_ratio.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- Per-pixel CSV ----
    pixel_rows = []
    for p_idx in range(P):
        row = {
            "pixel_id": p_idx,
            "x": int(px_coords[p_idx, 0]),
            "y": int(px_coords[p_idx, 1]),
        }
        for m in methods:
            row[f"{m}_MSE"] = float(pixel_dists[m][p_idx])
        row["best_method"] = methods[int(best_indices[p_idx])]
        pixel_rows.append(row)
    df = pd.DataFrame(pixel_rows)
    csv_path = os.path.join(save_dir, "B_pixel_distances.csv")
    df.to_csv(csv_path, index=False)
    log.info("  Saved: %s", csv_path)

    return df


# ---------------------------------------------------------------------------
# Analysis C: gradient-based grouping
# ---------------------------------------------------------------------------
def analysis_C(
    target_image: np.ndarray,
    px_coords: np.ndarray,
    all_dists_A: dict,
    patch_sizes: list,
    save_dir: str,
) -> pd.DataFrame:
    """Group pixels by gradient magnitude and repeat B-style analysis.

    Output:
        - grouped boxplot
        - grouped patch size effect line plot
        - grouped best patch ratio bar plot
        - CSV summary
    """
    log.info("=== Analysis C: Gradient-based grouping ===")

    # Compute gradient magnitude
    gy, gx = np.gradient(target_image)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Per-pixel gradient
    pixel_grad = grad_mag[px_coords[:, 0], px_coords[:, 1]]

    # Split into three groups
    low_cut = np.percentile(pixel_grad, 33)
    high_cut = np.percentile(pixel_grad, 66)
    labels = ["Low-Gradient", "Mid-Gradient", "High-Gradient"]
    mask_low = pixel_grad <= low_cut
    mask_mid = (pixel_grad > low_cut) & (pixel_grad <= high_cut)
    mask_high = pixel_grad > high_cut
    masks = [mask_low, mask_mid, mask_high]

    methods = ["center_sparse"] + [f"local{p}P" for p in patch_sizes if p > 1]
    metrics = ["MSE", "MAE"]

    # ---- Grouped boxplot (MSE per method, faceted by group) ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, mask, grp_label in zip(axes, masks, labels):
        data = [np.array(all_dists_A[m]["MSE"])[mask] for m in methods]
        bp = ax.boxplot(data, labels=methods, patch_artist=True)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(methods)))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_title(grp_label, fontsize=14)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Distribution Distance by Gradient Group", fontsize=15)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "C_gradient_boxplot.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- Grouped patch size effect line plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_i, metric in enumerate(["MSE", "MAE"]):
        ax = axes[ax_i]
        for mask, grp_label in zip(masks, labels):
            means = []
            for m in methods:
                vals = np.array(all_dists_A[m][metric])[mask]
                means.append(np.mean(vals))
            x_pos = np.arange(len(methods))
            ax.plot(x_pos, means, "o-", label=grp_label, linewidth=2, markersize=7)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=11, rotation=30)
        ax.set_xlabel("Method", fontsize=13)
        ax.set_ylabel(f"Mean {metric}", fontsize=13)
        ax.set_title(f"Mean {metric} by Gradient Group", fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "C_gradient_patch_effect.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- Grouped best patch ratio bar plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    n_methods = len(methods)
    x = np.arange(n_methods)
    width = 0.25
    for i, (mask, grp_label) in enumerate(zip(masks, labels)):
        mse_matrix = np.column_stack([np.array(all_dists_A[m]["MSE"])[mask] for m in methods])
        best = np.argmin(mse_matrix, axis=1)
        ratios = np.bincount(best, minlength=n_methods) / max(len(best), 1) * 100
        ax.bar(x + i * width, ratios, width, label=grp_label)
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=30, fontsize=11)
    ax.set_ylabel("Best Method Ratio (%)", fontsize=13)
    ax.set_title("Best Patch Size by Gradient Group", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "C_gradient_best_ratio.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- CSV summary ----
    rows = []
    for grp_label, mask in zip(labels, masks):
        for m in methods:
            for met in metrics:
                vals = np.array(all_dists_A[m][met])[mask]
                mse_matrix_grp = np.column_stack(
                    [np.array(all_dists_A[mm]["MSE"])[mask] for mm in methods]
                )
                best_grp = np.argmin(mse_matrix_grp, axis=1) if len(mask) > 0 else np.array([])
                methods_list = list(methods)
                best_ratio = np.mean(best_grp == methods_list.index(m)) * 100 if len(best_grp) > 0 else 0.0
                rows.append({
                    "group": grp_label,
                    "method": m,
                    "metric": met,
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)),
                    "p25": float(np.percentile(vals, 25)),
                    "p75": float(np.percentile(vals, 75)),
                    "best_ratio": float(best_ratio),
                })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "C_gradient_summary.csv")
    df.to_csv(csv_path, index=False)
    log.info("  Saved: %s", csv_path)

    return df


# ---------------------------------------------------------------------------
# Analysis D: trajectory offset linear fit (exact detector coordinates)
# ---------------------------------------------------------------------------
def analysis_D(
    geo_sparse: dict,
    vvbp_sparse: torch.Tensor,
    px_coords: np.ndarray,
    patch_values: dict,
    sparse_views: int,
    save_dir: str,
    device: torch.device,
):
    """Fit Q_{p+δ}(θ) - Q_p(θ) ≈ ΔI(p,δ,v) · G_p(θ).

    Uses exact fractional detector index I(p,v) computed from fan-beam geometry
    (compute_deltas_cube_np) instead of the parallel-beam approximation.

    For each (pixel, view), use all patch δ offsets to perform a through-origin
    linear regression with the exact detector index change ΔI as predictor.

    Output:
        - R² histogram (compare w/ parallel-beam equivalent)
        - R² map
        - CSV
    """
    log.info("=== Analysis D: Trajectory offset (exact detector coords) ===")
    P = len(px_coords)
    V = int(sparse_views)
    image_size = vvbp_sparse.shape[2]

    # Pick a representative patch size for the fit
    patch_size_d = 3
    if patch_size_d not in patch_values:
        patch_size_d = min(patch_values.keys(), key=lambda k: abs(k - 3))
        if patch_size_d == 1:
            log.warning("  No patch size > 1 available for trajectory analysis, skipping.")
            return None
    log.info("  Using patch size %d for trajectory fit", patch_size_d)

    r = patch_size_d // 2
    P_actual = min(P, 500)

    # Build the list of neighbour offsets (du,dv) excluding centre
    deltas: list[tuple] = []
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            if du == 0 and dv == 0:
                continue
            deltas.append((du, dv))
    n_deltas = len(deltas)

    # Precompute detector indices for every sampled pixel + its patch neighbours.
    # We get indices for the centre pixels first, then for each offset.
    centre_xs = px_coords[:P_actual, 0].astype(np.int64)
    centre_ys = px_coords[:P_actual, 1].astype(np.int64)

    I_centre = compute_detector_indices(geo_sparse, centre_xs, centre_ys, V)  # [V, P_actual]

    # Gather all unique (x+du, y+dv) coordinates to compute I for them
    neighbour_coords: dict[tuple, int] = {}  # (du,dv) → idx
    all_nbr_coords = []
    for du, dv in deltas:
        nbr_xs = centre_xs + du
        nbr_ys = centre_ys + dv
        nbr_xs_clipped = np.clip(nbr_xs, 0, image_size - 1)
        nbr_ys_clipped = np.clip(nbr_ys, 0, image_size - 1)
        coords = np.column_stack([nbr_xs_clipped, nbr_ys_clipped])
        all_nbr_coords.append(coords)

    # Get VVBP centre values  [H, W, V] on CPU
    center_vals = vvbp_sparse[0, 0, :, :, :].cpu().numpy()

    rows = []
    r2_map = np.full((image_size, image_size), np.nan, dtype=np.float32)

    for p_idx in range(P_actual):
        x, y = int(centre_xs[p_idx]), int(centre_ys[p_idx])

        for v_idx in range(V):
            Qc = center_vals[x, y, v_idx]
            Ic = I_centre[v_idx, p_idx]

            dI_list = []
            dQ_list = []
            for k, (du, dv) in enumerate(deltas):
                nx = np.clip(int(x + du), 0, image_size - 1)
                ny = np.clip(int(y + dv), 0, image_size - 1)
                # Recompute I for this neighbour pixel
                In_single = compute_detector_indices(
                    geo_sparse, np.array([nx], dtype=np.int64),
                    np.array([ny], dtype=np.int64), V,
                )
                dI = In_single[v_idx, 0] - Ic
                Qn = center_vals[nx, ny, v_idx]
                dI_list.append(dI)
                dQ_list.append(Qn - Qc)

            dI_arr = np.array(dI_list, dtype=np.float32)
            dQ_arr = np.array(dQ_list, dtype=np.float32)

            if len(dI_arr) < 2:
                continue

            # Through-origin OLS: G = sum(dI * dQ) / sum(dI^2 + eps)
            denom = np.sum(dI_arr ** 2) + 1e-12
            G = np.sum(dI_arr * dQ_arr) / denom
            pred = G * dI_arr
            ss_res = np.sum((dQ_arr - pred) ** 2)
            ss_tot = np.sum(dQ_arr ** 2) + 1e-12
            R2 = 1.0 - ss_res / ss_tot
            residual_mse = ss_res / max(len(dI_arr), 1)

            rows.append({
                "pixel_id": p_idx,
                "x": x,
                "y": y,
                "view_id": v_idx,
                "R2": float(R2),
                "residual_mse": float(residual_mse),
                "G": float(G),
            })

        pixel_rows = [r for r in rows if r["pixel_id"] == p_idx]
        if pixel_rows:
            r2_map[x, y] = float(np.mean([r["R2"] for r in pixel_rows]))

        if (p_idx + 1) % 100 == 0:
            log.info("  Trajectory fit: %d / %d pixels", p_idx + 1, P_actual)

    df = pd.DataFrame(rows)
    log.info("  Total fit points: %d", len(df))

    # ---- R² histogram ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["R2"].values, bins=80, alpha=0.7, color="darkorange", edgecolor="white")
    ax.set_xlabel("R² (exact detector ΔI)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Trajectory Offset Fit R² (Exact Detector Coordinates)", fontsize=14)
    ax.axvline(x=np.median(df["R2"].values), color="red", linestyle="--",
               label=f"Median R²={np.median(df['R2'].values):.4f}")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "D_R2_histogram_exact.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- R² map ----
    fig, ax = plt.subplots(figsize=(8, 8))
    valid_mask = ~np.isnan(r2_map)
    if valid_mask.any():
        im = ax.imshow(
            np.where(valid_mask, r2_map, np.nan),
            cmap="viridis", origin="upper", vmin=0, vmax=1,
        )
        plt.colorbar(im, ax=ax, label="Mean R²")
    ax.set_title("Per-Pixel Mean Trajectory Fit R² (Exact Detector)", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(save_dir, "D_R2_map_exact.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- CSV ----
    csv_path = os.path.join(save_dir, "D_trajectory_fit_exact.csv")
    df.to_csv(csv_path, index=False)
    log.info("  Saved: %s", csv_path)

    return df


# ---------------------------------------------------------------------------
# Analysis E: local CDF / rank coordinate
# ---------------------------------------------------------------------------
def analysis_E(
    patch_values: dict[int, np.ndarray],
    px_coords: np.ndarray,
    center_sparse: np.ndarray,
    sparse_views: int,
    save_dir: str,
) -> pd.DataFrame:
    """Compute empirical CDF coordinate u_{p,v} = F_p(Q_p(θ_v)).

    For patch_size s, F_p pools all s² × V_sparse values.

    Output:
        - u distribution histogram per patch size
        - representative pixel u-vs-sorted-index curves
        - CSV
    """
    log.info("=== Analysis E: Local CDF / rank coordinate ===")
    P = len(px_coords)
    V = int(sparse_views)
    patch_sizes = sorted(patch_values.keys())

    # For each pixel and view, compute u = F_p(Q_p(θ_v))
    all_u: dict[int, np.ndarray] = {}
    for ps in patch_sizes:
        pooled = patch_values[ps]  # [P, ps**2, V]
        P_actual, J, _ = pooled.shape
        merged = pooled.reshape(P_actual, J * V)  # [P, J*V]

        # Compute u for each pixel and view
        u_vals = np.zeros((P_actual, V), dtype=np.float32)
        for p_idx in range(min(P_actual, 2000)):  # limit for speed
            center_v = center_sparse[p_idx]  # [V]
            patch_v = merged[p_idx]  # [J*V]
            sorted_patch = np.sort(patch_v)
            for v_idx in range(V):
                # Count how many pooled values ≤ center value
                # Use searchsorted
                pos_left = np.searchsorted(sorted_patch, center_v[v_idx], side="left")
                pos_right = np.searchsorted(sorted_patch, center_v[v_idx], side="right") - 1
                pos_left = max(0, pos_left)
                pos_right = max(pos_left, pos_right)
                u_val = 0.5 * (pos_left + pos_right) / (len(sorted_patch) - 1)
                u_vals[p_idx, v_idx] = u_val

        all_u[ps] = u_vals

    # ---- u distribution histogram ----
    fig, axes = plt.subplots(1, len(patch_sizes), figsize=(6 * len(patch_sizes), 5))
    if len(patch_sizes) == 1:
        axes = [axes]
    for ax, ps in zip(axes, patch_sizes):
        ax.hist(all_u[ps].ravel(), bins=80, alpha=0.7, color="steelblue", edgecolor="white")
        ax.set_xlabel("u (CDF Coordinate)", fontsize=13)
        ax.set_ylabel("Count", fontsize=13)
        ax.set_title(f"Patch {ps}x{ps}, Mean u={np.mean(all_u[ps]):.4f}", fontsize=14)
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="u=0.5")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "E_u_distribution.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- Representative pixel u vs sorted-index curves ----
    # Pick a few pixels; plot u vs view sorted-by-value index
    fig, axes = plt.subplots(1, min(3, P), figsize=(6 * min(3, P), 5))
    if min(3, P) == 1:
        axes = [axes]
    n_rep = min(3, P)
    for ax_i in range(n_rep):
        ax = axes[ax_i]
        p_idx = int(px_coords.shape[0] * (ax_i + 1) // (n_rep + 1))
        # Sort views by value
        sort_idx = np.argsort(center_sparse[p_idx])
        for ps in patch_sizes:
            ax.plot(all_u[ps][p_idx][sort_idx], label=f"{ps}x{ps}", linewidth=1.5)
        ax.set_xlabel("View Index (sorted by VVBP value)", fontsize=13)
        ax.set_ylabel("u (CDF Coordinate)", fontsize=13)
        ax.set_title(f"Pixel ({int(px_coords[p_idx,0])}, {int(px_coords[p_idx,1])})", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "E_u_vs_sorted_index.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- CSV ----
    rows = []
    for p_idx in range(min(P, 2000)):
        for v_idx in range(V):
            row = {
                "pixel_id": p_idx,
                "x": int(px_coords[p_idx, 0]),
                "y": int(px_coords[p_idx, 1]),
                "view_id": v_idx,
                "Q_center": float(center_sparse[p_idx, v_idx]),
            }
            for ps in patch_sizes:
                row[f"u_{ps}x{ps}"] = float(all_u[ps][p_idx, v_idx])
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, "E_local_cdf.csv")
    df.to_csv(csv_path, index=False)
    log.info("  Saved: %s", csv_path)

    return df, all_u


# ---------------------------------------------------------------------------
# Analysis F: VVBP properties vs reconstruction error
# ---------------------------------------------------------------------------
def analysis_F(
    px_coords: np.ndarray,
    center_sparse: np.ndarray,
    patch_values: dict,
    all_u: dict,
    all_dists_A: dict,
    target_image: np.ndarray,
    sparse_views: int,
    checkpoint_dir: str,
    model_name: str,
    cfg,
    device: torch.device,
    fig_dir: str,
    csv_dir: str,
) -> pd.DataFrame:
    """Correlate VVBP properties with model reconstruction error.

    For each pixel computes:
      - dist_to_highview (MSE from Analysis A)
      - local CDF perturbation  |u - 0.5|
      - gradient magnitude
      - model absolute error  |pred - target|

    Then runs correlation heatmap and scatter plots.

    Requires a trained model checkpoint + stats_cached.pt.
    """
    if checkpoint_dir is None:
        log.info("=== Analysis F: SKIPPED (no --checkpoint_dir) ===")
        return None

    log.info("=== Analysis F: VVBP properties vs reconstruction error ===")

    # ---- Load model and stats ----
    from src.data.local_rank import compute_local_rank_sorted

    # Import model class directly, bypassing src/models/__init__.py which pulls in
    # CTO-adapted (needs triton).  We load the module file manually.
    _model_cls = None
    for _mod_file, _cls_name in [
        ("local_rank_center_integral_mlp", "LocalRankCenterIntegralMLPNet"),
        ("local_rank_center_mlp", "LocalRankCenterMLPNet"),
    ]:
        _mod_path = os.path.join(PROJECT_ROOT, "src", "models", f"{_mod_file}.py")
        if not os.path.exists(_mod_path):
            continue
        try:
            import importlib.util
            _spec = importlib.util.spec_from_file_location(
                f"vvbp_analysis.{_mod_file}", _mod_path,
                submodule_search_locations=[],
            )
            _mod = importlib.util.module_from_spec(_spec)
            # Inject the sub-dependencies this module needs (they don't go through __init__)
            sys.modules[f"vvbp_analysis.{_mod_file}"] = _mod
            _spec.loader.exec_module(_mod)
            _model_cls = getattr(_mod, _cls_name, None)
            if _model_cls is not None:
                break
        except Exception as exc:
            log.debug("  Could not load %s: %s", _mod_file, exc)
            continue

    if _model_cls is None:
        log.warning("  Cannot import model class (triton missing?), skipping F")
        return None

    stats_path = os.path.join(checkpoint_dir, "stats_cached.pt")
    if not os.path.exists(stats_path):
        log.warning("  stats_cached.pt not found at %s, skipping F", stats_path)
        return None

    stats_loaded = torch.load(stats_path, map_location=device, weights_only=False)
    stats_cached = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in stats_loaded.items()
    }
    v_mean = stats_cached["v_mean"].to(device)
    v_std = stats_cached["v_std"].to(device)
    target_mean = stats_cached["target_mean"].to(device)
    target_std = stats_cached["target_std"].to(device)

    # Build model with the directly imported class
    model = _model_cls().to(device)
    safe_name = model_name.replace(", ", "_").replace(" ", "_")
    model_path = os.path.join(checkpoint_dir, f"{safe_name}.pt")
    if not os.path.exists(model_path):
        log.warning("  Model checkpoint not found: %s", model_path)
        return None
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    log.info("  Loaded model from %s", model_path)

    # ---- Run model on sampled pixels ----
    # Need values_sorted: [P, 9, V] sorted per-patch-position VVBP values
    P = len(px_coords)
    V = int(sparse_views)
    ps_model = 3  # model was trained with 3x3

    # Get raw patch values for 3x3
    raw_3x3 = patch_values.get(3)  # [P, 9, V]
    if raw_3x3 is None:
        log.warning("  No 3x3 patch values, skipping F")
        return None

    # Sort each position's VVBP values across views
    raw_3x3_t = torch.from_numpy(raw_3x3).float().to(device)  # [P, 9, V]
    values_sorted = torch.sort(raw_3x3_t, dim=-1).values  # [P, 9, V]

    # Batch inference
    batch_size = 2048
    preds = []
    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        vs = values_sorted[start:end]  # [B, 9, V]
        q_sorted, center_sorted = compute_local_rank_sorted(vs)
        value_norm = (center_sorted - v_mean) / v_std
        tokens = torch.stack([q_sorted, value_norm], dim=-1)
        h = model.point_mlp(tokens)
        point_w = model._nonuniform_trapezoid_weights(q_sorted).unsqueeze(-1)
        pooled = torch.pi * (h * point_w).sum(dim=1)
        pred_norm = model.out_mlp(pooled)
        pred = pred_norm * target_std + target_mean
        preds.append(pred.detach().cpu().numpy().ravel())

    pred_arr = np.concatenate(preds)  # [P]
    target_arr = target_image[px_coords[:, 0], px_coords[:, 1]]  # [P]
    abs_error = np.abs(pred_arr - target_arr)

    # ---- Per-pixel VVBP properties ----
    # dist_to_highview (MSE of center_sparse sorted curve vs center_high)
    center_sparse_mse = np.array(all_dists_A["center_sparse"]["MSE"])  # [P]

    # CDF perturbation for 3x3, 5x5, 7x7
    cdf_perturb: dict = {}
    for ps in sorted(all_u.keys()):
        u_vals = all_u[ps]  # [P, V]
        cdf_perturb[ps] = np.mean(np.abs(u_vals - 0.5), axis=1)  # [P]

    # Gradient magnitude
    gy, gx = np.gradient(target_image)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    pixel_grad = grad_mag[px_coords[:, 0], px_coords[:, 1]]

    # ---- Build per-pixel dataframe ----
    rows = []
    for p_idx in range(P):
        row = {
            "pixel_id": p_idx,
            "x": int(px_coords[p_idx, 0]),
            "y": int(px_coords[p_idx, 1]),
            "dist_center_sparse_MSE": float(center_sparse_mse[p_idx]),
            "gradient_mag": float(pixel_grad[p_idx]),
            "target_value": float(target_arr[p_idx]),
            "pred_value": float(pred_arr[p_idx]),
            "abs_error": float(abs_error[p_idx]),
        }
        for ps in sorted(all_u.keys()):
            row[f"cdf_perturb_{ps}x{ps}"] = float(cdf_perturb[ps][p_idx])
        rows.append(row)
    df = pd.DataFrame(rows)

    # ---- Correlation heatmap ----
    corr_cols = ["dist_center_sparse_MSE", "gradient_mag", "abs_error"]
    for ps in sorted(all_u.keys()):
        corr_cols.append(f"cdf_perturb_{ps}x{ps}")

    corr_mat = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(corr_cols, fontsize=10)
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr_mat.values[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(corr_mat.values[i, j]) < 0.6 else "white")
    plt.colorbar(im, ax=ax)
    ax.set_title("VVBP Properties vs Reconstruction Error", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "F_correlation_heatmap.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- Scatter: key predictors vs abs_error ----
    predictors = ["dist_center_sparse_MSE", "gradient_mag"]
    for ps in sorted(all_u.keys()):
        predictors.append(f"cdf_perturb_{ps}x{ps}")

    fig, axes = plt.subplots(1, len(predictors), figsize=(5 * len(predictors), 5))
    if len(predictors) == 1:
        axes = [axes]
    for ax, col in zip(axes, predictors):
        ax.scatter(df[col], df["abs_error"], alpha=0.3, s=4, c="steelblue")
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("Abs Error", fontsize=11)
        r = float(np.corrcoef(df[col], df["abs_error"])[0, 1])
        ax.set_title(f"r={r:.4f}", fontsize=12)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "F_error_scatter.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved: %s", fig_path)

    # ---- CSV ----
    csv_path = os.path.join(csv_dir, "F_vvbp_vs_error.csv")
    df.to_csv(csv_path, index=False)
    log.info("  Saved: %s", csv_path)

    # Log key correlations
    log.info("  Pearson r with abs_error:")
    for col in corr_cols:
        if col == "abs_error":
            continue
        r = float(np.corrcoef(df[col], df["abs_error"])[0, 1])
        log.info("    %s: r=%.4f", col, r)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="VVBP-Tensor properties analysis.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment JSON config (optional).")
    parser.add_argument("--slice_idx", type=int, default=0,
                        help="Slice index in DICOM folder.")
    parser.add_argument("--sparse_views", type=int, default=72,
                        help="Number of sparse views (V_sparse).")
    parser.add_argument("--high_views", type=int, default=240,
                        help="Number of high/ dense views (V_high).")
    parser.add_argument("--region", type=int, nargs=4,
                        default=[32, 224, 32, 224],
                        help="Region of interest: x0 x1 y0 y1.")
    parser.add_argument("--num_pixels", type=int, default=1000,
                        help="Number of random pixels to sample.")
    parser.add_argument("--patch_sizes", type=int, nargs="+", default=[1, 3, 5, 7],
                        help="List of odd patch sizes to evaluate.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Torch device (cuda / cpu).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/vvbp_tensor_property_analysis",
                        help="Root output directory.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Sub-directory name for this run.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Model checkpoint directory (with .pt + stats_cached.pt).")
    parser.add_argument("--model_name", type=str,
                        default="local rank center integral mlp, 10 epochs",
                        help="Registered model name for loading.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ----- Config -----
    cfg = AnalysisConfig(
        slice_idx=args.slice_idx,
        sparse_views=args.sparse_views,
        high_views=args.high_views,
        region=tuple(args.region),
        num_pixels=args.num_pixels,
        patch_sizes=tuple(args.patch_sizes),
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    # If --config provided, load parameters from it.
    # Supports two formats:
    #   1) Flat JSON  — keys match AnalysisConfig fields directly, OR
    #   2) RunConfig  — nested {"experiment": {...}, "dicom_folder": ..., "results_folder": ...}
    if args.config is not None:
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if "experiment" in raw or "experiment_name" not in raw:
            # RunConfig format
            run_cfg = load_run_config(args.config)
            exp_cfg = run_cfg.experiment
            cfg.image_size = int(exp_cfg.image_size)
            cfg.n_detec = int(exp_cfg.n_detec)
            cfg.d_detec = float(exp_cfg.d_detec)
            cfg.d_voxel = float(exp_cfg.d_voxel)
            cfg.DSO = float(exp_cfg.DSO)
            cfg.DOD = float(exp_cfg.DOD)
            cfg.dicom_folder = str(run_cfg.dicom_folder) if run_cfg.dicom_folder else None
            cfg.results_folder = str(run_cfg.results_folder) if run_cfg.results_folder else None
            if "experiment_name" in raw:
                cfg.experiment_name = raw["experiment_name"]
        else:
            # Flat AnalysisConfig format
            for key in vars(cfg):
                if key in raw:
                    val = raw[key]
                    if key == "region":
                        val = tuple(val)
                    elif key == "patch_sizes":
                        val = tuple(val)
                    setattr(cfg, key, val)
            cfg.dicom_folder = str(raw["dicom_folder"]) if raw.get("dicom_folder") else None
            cfg.results_folder = str(raw["results_folder"]) if raw.get("results_folder") else None
        log.info("Loaded config: %s", args.config)

    if args.experiment_name is not None:
        cfg.experiment_name = args.experiment_name

    # ----- Device -----
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device != "cpu" else "cpu")
    log.info("Device: %s", device)
    log.info("Config: image_size=%d, sparse_views=%d, high_views=%d, num_pixels=%d",
             cfg.image_size, cfg.sparse_views, cfg.high_views, cfg.num_pixels)
    log.info("Region: %s", cfg.region)
    log.info("Patch sizes: %s", cfg.patch_sizes)

    # ----- Seed -----
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # ----- Output directories -----
    save_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)

    # Save analysis config
    with open(os.path.join(save_dir, "logs", "analysis_config.json"), "w") as f:
        json.dump({k: str(v) if not isinstance(v, (int, float, list, tuple, str))
                   else v for k, v in vars(cfg).items()}, f, indent=2)

    fig_dir = os.path.join(save_dir, "figures")
    csv_dir = os.path.join(save_dir, "csv")

    # ----- DICOM loading -----
    if cfg.dicom_folder is None:
        # Try default path relative to project
        default_dicom = os.path.join(PROJECT_ROOT, "full_1mm", "L067", "full_1mm")
        if os.path.isdir(default_dicom):
            cfg.dicom_folder = default_dicom
            log.info("Using default DICOM folder: %s", default_dicom)
        else:
            log.error("No DICOM folder specified and default not found: %s", default_dicom)
            log.error("Please provide --dicom_folder in the config file.")
            sys.exit(1)

    if cfg.results_folder is None:
        cfg.results_folder = os.path.join(PROJECT_ROOT, "Results_analysis")
    os.makedirs(cfg.results_folder, exist_ok=True)

    # Load slice
    from src.data.dicom_dataset import _read_hu, _hu_to_attenuation
    files = [
        os.path.join(cfg.dicom_folder, f)
        for f in sorted(os.listdir(cfg.dicom_folder))
        if f.lower().endswith((".ima", ".dcm"))
    ]
    if not files:
        log.error("No .IMA/.dcm files found in %s", cfg.dicom_folder)
        sys.exit(1)
    if cfg.slice_idx >= len(files):
        log.warning("slice_idx %d >= %d slices, using 0", cfg.slice_idx, len(files))
        cfg.slice_idx = 0
    hu = _read_hu(files[cfg.slice_idx])
    target_image = _hu_to_attenuation(hu, water_mu=0.0192).astype(np.float32)
    log.info("Loaded slice %d: shape=%s, range=[%.4f, %.4f]",
             cfg.slice_idx, target_image.shape, target_image.min(), target_image.max())

    # Resize if needed
    if target_image.shape[0] != cfg.image_size or target_image.shape[1] != cfg.image_size:
        from torch.nn.functional import interpolate
        img_t = torch.from_numpy(target_image)[None, None, :, :]
        img_t = interpolate(img_t, size=(cfg.image_size, cfg.image_size),
                            mode="bilinear", align_corners=False)
        target_image = img_t.squeeze().numpy().astype(np.float32)
        log.info("Resized to %dx%d", cfg.image_size, cfg.image_size)

    # ----- Generate sinograms -----
    log.info("Generating %d-view sinogram...", cfg.high_views)
    sino_high = generate_sinogram_astra(
        target_image, cfg.high_views, cfg.n_detec, cfg.d_detec, cfg.DSO, cfg.DOD,
    )
    log.info("  high sino shape: %s", sino_high.shape)

    log.info("Generating %d-view sinogram...", cfg.sparse_views)
    sino_sparse = generate_sinogram_astra(
        target_image, cfg.sparse_views, cfg.n_detec, cfg.d_detec, cfg.DSO, cfg.DOD,
    )
    log.info("  sparse sino shape: %s", sino_sparse.shape)

    # ----- Build geometry + VVBP extractors -----
    log.info("Building geometry for high views (%d)...", cfg.high_views)
    geo_high = load_or_generate_geo(
        views=cfg.high_views,
        results_folder=cfg.results_folder,
        device=device,
        image_size=cfg.image_size,
        n_detec=cfg.n_detec,
        d_detec=cfg.d_detec,
        d_voxel=cfg.d_voxel,
        DSO=cfg.DSO,
        DOD=cfg.DOD,
    )
    log.info("Building geometry for sparse views (%d)...", cfg.sparse_views)
    geo_sparse = load_or_generate_geo(
        views=cfg.sparse_views,
        results_folder=cfg.results_folder,
        device=device,
        image_size=cfg.image_size,
        n_detec=cfg.n_detec,
        d_detec=cfg.d_detec,
        d_voxel=cfg.d_voxel,
        DSO=cfg.DSO,
        DOD=cfg.DOD,
    )

    # ----- VVBP extraction -----
    log.info("Computing VVBP for high views...")
    t0 = time.time()
    vvbp_high = compute_vvbp(sino_high, geo_high, device)
    log.info("  done in %.1f s, shape=%s", time.time() - t0, tuple(vvbp_high.shape))

    log.info("Computing VVBP for sparse views...")
    t0 = time.time()
    vvbp_sparse = compute_vvbp(sino_sparse, geo_sparse, device)
    log.info("  done in %.1f s, shape=%s", time.time() - t0, tuple(vvbp_sparse.shape))

    # ----- Sample pixel coordinates -----
    x0, x1, y0, y1 = cfg.region
    margin = max(cfg.patch_sizes) // 2
    x0 = max(x0, margin)
    x1 = min(x1, cfg.image_size - margin)
    y0 = max(y0, margin)
    y1 = min(y1, cfg.image_size - margin)

    xs = torch.randint(low=x0, high=x1, size=(cfg.num_pixels,), device=device)
    ys = torch.randint(low=y0, high=y1, size=(cfg.num_pixels,), device=device)
    px_coords = torch.stack([xs, ys], dim=1).cpu().numpy()  # [P, 2]
    P = len(px_coords)
    log.info("Sampled %d pixels in region [%d:%d, %d:%d]",
             P, x0, x1, y0, y1)

    # ----- Gather VVBP values -----
    log.info("Gathering center VVBP values...")
    center_sparse = gather_center_values(vvbp_sparse, xs, ys)      # [P, V_sparse]
    center_high = gather_center_values(vvbp_high, xs, ys)           # [P, V_high]
    log.info("  center_sparse shape=%s, center_high shape=%s",
             center_sparse.shape, center_high.shape)

    patch_values = {}
    for ps in cfg.patch_sizes:
        if ps == 1:
            continue  # 1x1 is equivalent to center sparse
        log.info("Gathering patch %dx%d VVBP values...", ps, ps)
        pv = gather_raw_patch_values(vvbp_sparse, xs, ys, ps)       # [P, ps**2, V_sparse]
        patch_values[ps] = pv
        log.info("  shape=%s", pv.shape)

    # Clear VVBP tensors to free GPU memory
    del vvbp_high, vvbp_sparse
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ======================================================================
    # Run analyses
    # ======================================================================
    results = {}

    # Analysis A
    df_A_summary, df_A_pixel = analysis_A(
        px_coords, center_sparse, center_high, patch_values,
        cfg.n_interp, fig_dir, rng,
    )
    # Save CSVs to csv_dir as well
    df_A_summary.to_csv(os.path.join(csv_dir, "A_distance_summary.csv"), index=False)
    df_A_pixel.to_csv(os.path.join(csv_dir, "A_pixel_distances.csv"), index=False)

    # Reconstruct all_dists_A from df_A_pixel for downstream analyses
    all_dists_A: dict = {}
    method_cols = [c for c in df_A_pixel.columns if c.endswith("_MSE") or c.endswith("_MAE")]
    for col in method_cols:
        base = col.replace("_MSE", "").replace("_MAE", "")
        metric = "_MSE" if col.endswith("_MSE") else "_MAE"
        if base not in all_dists_A:
            all_dists_A[base] = {}
        if metric == "_MSE":
            all_dists_A[base]["MSE"] = df_A_pixel[col].values.tolist()
        else:
            all_dists_A[base]["MAE"] = df_A_pixel[col].values.tolist()

    # Analysis B
    df_B = analysis_B(
        px_coords, center_high, patch_values, all_dists_A,
        cfg.n_interp, fig_dir,
    )
    df_B.to_csv(os.path.join(csv_dir, "B_pixel_distances.csv"), index=False)

    # Analysis C
    df_C = analysis_C(
        target_image, px_coords, all_dists_A,
        [p for p in cfg.patch_sizes if p > 1],
        fig_dir,
    )
    df_C.to_csv(os.path.join(csv_dir, "C_gradient_summary.csv"), index=False)

    # Analysis D (needs vvbp_sparse tensor + geo for exact detector coords)
    log.info("Recomputing VVBP for trajectory analysis (Analysis D)...")
    vvbp_sparse_d = compute_vvbp(sino_sparse, geo_sparse, device)
    df_D = analysis_D(
        geo_sparse, vvbp_sparse_d, px_coords, patch_values,
        cfg.sparse_views, fig_dir, device,
    )
    if df_D is not None:
        df_D.to_csv(os.path.join(csv_dir, "D_trajectory_fit_exact.csv"), index=False)
    del vvbp_sparse_d
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Analysis E
    df_E, all_u_E = analysis_E(
        patch_values, px_coords, center_sparse,
        cfg.sparse_views, fig_dir,
    )
    df_E.to_csv(os.path.join(csv_dir, "E_local_cdf.csv"), index=False)

    # Analysis F: VVBP properties vs model reconstruction error (optional)
    if cfg.checkpoint_dir or args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir or cfg.checkpoint_dir
        df_F = analysis_F(
            px_coords, center_sparse, patch_values, all_u_E,
            all_dists_A, target_image, cfg.sparse_views,
            ckpt_dir, args.model_name, cfg, device, fig_dir, csv_dir,
        )
        if df_F is not None:
            df_F.to_csv(os.path.join(csv_dir, "F_vvbp_vs_error.csv"), index=False)

    # ======================================================================
    # Summary
    # ======================================================================
    log.info("")
    log.info("=" * 60)
    log.info("Analysis complete!")
    log.info("=" * 60)
    log.info("Output directory: %s", save_dir)
    log.info("  figures/  -- all comparison plots (300 dpi PNG)")
    log.info("  csv/      -- all numeric results")
    log.info("  logs/     -- config and run log")
    log.info("")
    log.info("Generated files:")
    for root, dirs, files in os.walk(save_dir):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), save_dir)
            size_kb = os.path.getsize(os.path.join(root, f)) / 1024
            log.info("  %s (%.1f KB)", rel, size_kb)


if __name__ == "__main__":
    main()
