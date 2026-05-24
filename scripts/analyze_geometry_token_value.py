"""Geometry-token value analysis.

Loads original local-rank MLP + exact-detector geometry MLP predictions
and answers:
  1. Do G / R2 correlate with image edges?
  2. Does the geometry model improve more where G/R2 are strong?
"""

from __future__ import annotations

# Suppress ASTRA + MKL OpenMP conflict on Windows (must be before scipy import)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_run_config
from src.geometry import load_or_generate_geo, FanBeamVVBPExtractor
from src.geometry.fanbeam import compute_deltaI_patch
from src.data.local_vvbp import sample_random_coords, gather_sorted_vvbp_patch, gather_raw_vvbp_patch
from src.data.local_rank import compute_local_rank
from src.training.trainer import _compute_GR2_raw
from src.evaluation.metrics import compute_metrics_np


def safe_model_name(name: str) -> str:
    return name.replace(", ", "_").replace(" ", "_")


def _import_model_class(mod_file: str, cls_name: str):
    _mp = os.path.join(PROJECT_ROOT, "src", "models", f"{mod_file}.py")
    if not os.path.exists(_mp):
        return None
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        f"_load_{mod_file}", _mp, submodule_search_locations=[]
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[f"_load_{mod_file}"] = _mod
    _spec.loader.exec_module(_mod)
    return getattr(_mod, cls_name, None)


def load_model(model_name: str, checkpoint_dir: str, device: torch.device,
               model_type: str = "auto"):
    """Load a model from checkpoint.

    ``model_type``: "base" → 2-dim LocalRankCenterIntegralMLPNet,
                    "geo"  → 4-dim ExactDetectorGeometryLocalRankIntegralMLPNet,
                    "auto" → guess from model_name.
    """
    if model_type == "auto":
        model_type = "geo" if "exact" in model_name.lower() or "geometry" in model_name.lower() else "base"

    if model_type == "base":
        _cls = _import_model_class("local_rank_center_integral_mlp",
                                    "LocalRankCenterIntegralMLPNet")
    else:
        _cls = _import_model_class("exact_detector_geometry_local_rank_integral_mlp",
                                    "ExactDetectorGeometryLocalRankIntegralMLPNet")

    if _cls is None:
        raise RuntimeError(f"Cannot load model class for {model_name} (type={model_type})")

    model = _cls().to(device).eval()
    safe = safe_model_name(model_name)
    ckpt = os.path.join(checkpoint_dir, f"{safe}.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    stats_path = os.path.join(checkpoint_dir, "stats_cached.pt")
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location=device, weights_only=False)
    else:
        stats = None  # will be computed from eval data
    return model, stats


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/multirate_selected_models_60view_exactdetector_token.json")
    parser.add_argument("--base_checkpoint", type=str,
                        default="outputs/multirate_vvbp_local_rank")
    parser.add_argument("--geo_checkpoint", type=str,
                        default="outputs/multirate_vvbp_local_rank_exactdetector_token")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/geometry_token_value_analysis")
    args = parser.parse_args()

    run_cfg = load_run_config(args.config)
    exp_cfg = run_cfg.experiment
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    region = tuple(run_cfg.region)
    x0, x1, y0, y1 = region
    Hreg, Wreg = x1 - x0, y1 - y0
    V = 60
    print(f"Region: {region}, size: {Hreg}x{Wreg}, views: {V}")

    # ---- Output dirs ----
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "csv"), exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    csv_dir = os.path.join(args.output_dir, "csv")

    # ---- Geometry ----
    geo = load_or_generate_geo(
        V, str(run_cfg.results_folder or "cache/fanbeam_geometry"),
        device, image_size=exp_cfg.image_size, n_detec=exp_cfg.n_detec,
        d_detec=exp_cfg.d_detec, d_voxel=exp_cfg.d_voxel,
        DSO=exp_cfg.DSO, DOD=exp_cfg.DOD,
    )
    extractor = FanBeamVVBPExtractor(geo).to(device).eval()

    # ---- Dataset ----
    from src.data.dicom_dataset import LInFBPAlignedDataset, build_dataloaders
    dataset, train_indices, test_indices, _ = build_dataloaders(
        dicom_folder=str(run_cfg.dicom_folder or "full_1mm/L067/full_1mm"),
        cfg=exp_cfg,
    )
    test_idx = int(test_indices[0])
    print(f"Test slice: {test_idx}")

    # ---- Load both models ----
    print("\nLoading base model ...")
    base_model, base_stats = load_model(
        "local rank center integral mlp, 10 epochs",
        args.base_checkpoint, device, model_type="base",
    )

    print("Loading geometry model ...")
    geo_model, geo_stats = load_model(
        "exact detector geometry local rank center integral mlp, 10 epochs",
        args.geo_checkpoint, device, model_type="geo",
    )

    # ---- Get G stats from geo stats ----
    # stats_cached has v_mean, v_std, target_mean, target_std.
    # For G stats, we need to estimate them.  We'll directly read from the
    # geo_stats — but geo model's stats don't include G stats.
    # We'll compute G stats on-the-fly from the evaluation data.
    G_stats = {}

    # ---- Generate sino + VVBP ----
    sino_t, img_t = dataset[test_idx]
    sino_full = sino_t.squeeze(0)
    target_full = img_t.squeeze(0)
    target_region = target_full[x0:x1, y0:y1].cpu().numpy()

    step = sino_full.shape[0] // V
    sino_sparse = sino_full[::step, :].unsqueeze(0).unsqueeze(0).to(device)
    vvbp = extractor(sino_sparse)

    # ---- Eval: process all pixels in region ----
    coords = [(x, y) for x in range(x0, x1) for y in range(y0, y1)]
    chunk_size = exp_cfg.chunk_size_eval

    base_pred_chunks = []
    geo_pred_chunks = []

    # Collect G/R2 per pixel per view
    all_G = []       # list of [P_chunk, K]
    all_R2 = []      # list of [P_chunk, K]
    all_G_aligned = []
    all_R2_aligned = []

    # ---- Estimate stats if not available from checkpoint ----
    if base_stats is None:
        print("  Estimating base stats from eval data ...")
        with torch.no_grad():
            xs_s = torch.randint(x0, x1, (2048,), device=device)
            ys_s = torch.randint(y0, y1, (2048,), device=device)
            vs_s = gather_sorted_vvbp_patch(vvbp, xs_s, ys_s, patch_size=3, mode="3x3")
            vs_s = vs_s.reshape(2048, vs_s.shape[2], vs_s.shape[3])
            tg_s = target_full[xs_s.cpu(), ys_s.cpu()].to(device).reshape(2048, 1)
        v_mean_s = vs_s.mean()
        v_std_s = vs_s.std().clamp_min(1e-8)
        tgt_mean_s = tg_s.mean()
        tgt_std_s = tg_s.std().clamp_min(1e-8)
        base_stats = {"v_mean": v_mean_s, "v_std": v_std_s,
                       "target_mean": tgt_mean_s, "target_std": tgt_std_s}
        print(f"  v_mean={float(v_mean_s):.6e} tgt_mean={float(tgt_mean_s):.6e}")

    if geo_stats is None:
        print("  Estimating geo stats from eval data ...")
        geo_stats = dict(base_stats)  # same data, same VVBP

    target_mean = base_stats["target_mean"].to(device)
    target_std = base_stats["target_std"].to(device)
    # G stats — estimate from first chunk
    G_stats_estimated = False

    print(f"\nEvaluating {len(coords)} pixels in region ...")

    for start_i in range(0, len(coords), chunk_size):
        chunk = coords[start_i:start_i + chunk_size]
        xs_t = torch.tensor([c[0] for c in chunk], dtype=torch.long, device=device)
        ys_t = torch.tensor([c[1] for c in chunk], dtype=torch.long, device=device)
        Pc = xs_t.numel()

        # Values sorted (for model)
        values_sorted = gather_sorted_vvbp_patch(
            vvbp, xs_t, ys_t, patch_size=3, mode="3x3",
        )
        values_sorted = values_sorted.reshape(Pc, values_sorted.shape[2], values_sorted.shape[3])

        # Raw patch (for G/R2)
        raw_patch = gather_raw_vvbp_patch(vvbp, xs_t, ys_t, patch_size=3)
        raw_patch = raw_patch.reshape(Pc, raw_patch.shape[2], raw_patch.shape[3])

        # DeltaI
        xs_np = xs_t.cpu().numpy().astype(np.int64)
        ys_np = ys_t.cpu().numpy().astype(np.int64)
        di_np = compute_deltaI_patch(geo, xs_np, ys_np, V, patch_size=3)
        di_t = torch.from_numpy(di_np).to(device)

        # G/R2 (view order)
        G_val, R2_val = _compute_GR2_raw(raw_patch, di_t, eps=1e-8)
        all_G.append(G_val.cpu().numpy())
        all_R2.append(R2_val.cpu().numpy())

        # Aligned G/R2 (by centre value) — keep torch for model, numpy for storage
        centre_idx = raw_patch.shape[1] // 2
        raw_centre = raw_patch[:, centre_idx, :]
        _, c_sort = torch.sort(raw_centre, dim=-1)
        G_al_t = torch.gather(G_val, dim=-1, index=c_sort)
        R2_al_t = torch.gather(R2_val, dim=-1, index=c_sort)
        all_G_aligned.append(G_al_t.cpu().numpy())
        all_R2_aligned.append(R2_al_t.cpu().numpy())

        # ---- G stats (first chunk) ----
        if not G_stats_estimated:
            G_stats["G_mean"] = G_val.mean()
            G_stats["G_std"] = G_val.std().clamp_min(1e-8)
            G_stats_estimated = True
            print(f"  G stats: mean={float(G_stats['G_mean']):.6e}, std={float(G_stats['G_std']):.6e}")

        # ---- Base model inference ----
        base_batch_stats = {
            "v_mean": base_stats["v_mean"].to(device),
            "v_std": base_stats["v_std"].to(device),
            "target_mean": target_mean,
            "target_std": target_std,
        }
        base_pred_norm = base_model(values_sorted, base_batch_stats)
        base_pred = (base_pred_norm * target_std + target_mean).detach().cpu().numpy().ravel()
        base_pred_chunks.append(base_pred)

        # ---- Geo model inference ----
        geo_batch_stats = {
            "v_mean": geo_stats["v_mean"].to(device),
            "v_std": geo_stats["v_std"].to(device),
            "target_mean": target_mean,
            "target_std": target_std,
            "G_mean": G_stats["G_mean"].to(device),
            "G_std": G_stats["G_std"].to(device),
            "G": G_al_t,
            "R2": R2_al_t,
        }
        geo_pred_norm = geo_model(values_sorted, geo_batch_stats)
        geo_pred = (geo_pred_norm * target_std + target_mean).detach().cpu().numpy().ravel()
        geo_pred_chunks.append(geo_pred)

        if (start_i // chunk_size + 1) % 4 == 0:
            print(f"  {start_i + Pc}/{len(coords)} pixels")

    # ---- Concatenate ----
    base_pred = np.concatenate(base_pred_chunks).reshape(Hreg, Wreg)
    geo_pred = np.concatenate(geo_pred_chunks).reshape(Hreg, Wreg)

    G_all = np.concatenate(all_G, axis=0)           # [N, K] view order
    R2_all = np.concatenate(all_R2, axis=0)         # [N, K]
    G_aligned_all = np.concatenate(all_G_aligned, axis=0)   # [N, K] value-sorted
    R2_aligned_all = np.concatenate(all_R2_aligned, axis=0)

    N, K = G_all.shape
    print(f"\nCollected: G {G_all.shape}, R2 {R2_all.shape}")
    print(f"Base  PSNR={compute_metrics_np(base_pred, target_region)['PSNR']:.4f}")
    print(f"Geo   PSNR={compute_metrics_np(geo_pred, target_region)['PSNR']:.4f}")

    # ==================================================================
    # Analysis A: Geometry feature distribution
    # ==================================================================
    print("\n=== A: Feature distribution ===")
    absG = np.abs(G_all).ravel()
    R2_flat = R2_all.ravel()
    absGR2 = absG * R2_flat

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, data, title in zip(axes,
        [G_all.ravel(), absG, R2_flat, absGR2],
        ["G", "|G|", "R2", "|G| * R2"]):
        ax.hist(data, bins=100, alpha=0.7, color="steelblue", edgecolor="white")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(title, fontsize=12)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "A_feature_distribution.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)

    def _summarize(name, arr):
        a = arr.ravel()
        return {"metric": name, "mean": float(np.mean(a)),
                "median": float(np.median(a)), "std": float(np.std(a)),
                "p25": float(np.percentile(a,25)), "p75": float(np.percentile(a,75)),
                "p90": float(np.percentile(a,90)), "p95": float(np.percentile(a,95))}
    rows = [_summarize("G", G_all), _summarize("abs(G)", absG),
            _summarize("R2", R2_all), _summarize("abs(G)*R2", absGR2)]
    dfA = pd.DataFrame(rows)
    dfA.to_csv(os.path.join(csv_dir, "A_feature_summary.csv"), index=False)
    print(dfA.to_string())

    # ==================================================================
    # B: Per-pixel aggregated maps
    # ==================================================================
    print("\n=== B: Per-pixel maps ===")
    mean_absG = np.mean(np.abs(G_all), axis=1).reshape(Hreg, Wreg)
    mean_R2 = np.mean(R2_all, axis=1).reshape(Hreg, Wreg)
    mean_absGR2 = np.mean(np.abs(G_all) * R2_all, axis=1).reshape(Hreg, Wreg)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title in zip(axes,
        [mean_absG, mean_R2, mean_absGR2],
        ["mean |G|", "mean R2", "mean |G|*R2"]):
        im = ax.imshow(data, cmap="viridis")
        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "B_per_pixel_maps.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)

    # ==================================================================
    # C: vs target edge
    # ==================================================================
    print("\n=== C: G/R2 vs target edges ===")
    gy, gx = np.gradient(target_region)
    grad_target = np.sqrt(gy**2 + gx**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].imshow(grad_target, cmap="inferno")
    axes[0].set_title("Target Gradient", fontsize=14)
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(mean_absGR2, cmap="viridis")
    axes[1].set_title("mean |G|*R2", fontsize=14)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Hexbin: mean_absGR2 vs grad
    hb = axes[2].hexbin(mean_absGR2.ravel(), grad_target.ravel(),
                          gridsize=40, cmap="Blues", mincnt=1)
    axes[2].set_xlabel("mean |G|*R2", fontsize=12)
    axes[2].set_ylabel("Target Gradient", fontsize=12)
    axes[2].set_title("Hexbin", fontsize=14)
    plt.colorbar(hb, ax=axes[2], fraction=0.046)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "C_edge_correlation.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)

    # Correlations
    from scipy.stats import pearsonr, spearmanr
    rp, pp = pearsonr(mean_absGR2.ravel(), grad_target.ravel())
    rs, ps = spearmanr(mean_absGR2.ravel(), grad_target.ravel())
    print(f"  Pearson r={rp:.4f} (p={pp:.2e}), Spearman r={rs:.4f} (p={ps:.2e})")

    # ==================================================================
    # D: vs model error / improvement
    # ==================================================================
    print("\n=== D: Geometry feature vs error improvement ===")
    err_base = np.abs(base_pred - target_region)
    err_geo = np.abs(geo_pred - target_region)
    improve = err_base - err_geo  # positive = geo better
    print(f"  err_base  mean={float(np.mean(err_base)):.6e}  median={float(np.median(err_base)):.6e}")
    print(f"  err_geo   mean={float(np.mean(err_geo)):.6e}  median={float(np.median(err_geo)):.6e}")
    print(f"  improve   mean={float(np.mean(improve)):.6e}  median={float(np.median(improve)):.6e}")

    # Error maps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title, cmap_name in zip(axes,
        [err_base, err_geo, improve],
        ["Base Abs Error", "Geo Abs Error", "Improve (base - geo)"],
        ["hot", "hot", "RdBu_r"]):
        vmax = max(np.percentile(np.abs(data), 99), 1e-10)
        vmin = -vmax if "Improve" in title else 0
        im = ax.imshow(data, cmap=cmap_name, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "D_error_maps.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  Saved D_error_maps.png")

    # Scatter: mean_absGR2 vs improve
    fig, ax = plt.subplots(figsize=(8, 5))
    mG = mean_absGR2.ravel()
    imp = improve.ravel()
    # Sample if too many points
    if len(mG) > 20000:
        idx = np.random.default_rng(42).choice(len(mG), 20000, replace=False)
        mG_plt, imp_plt = mG[idx], imp[idx]
    else:
        mG_plt, imp_plt = mG, imp
    ax.scatter(mG_plt, imp_plt, alpha=0.15, s=1, c="steelblue")
    ax.set_xlabel("mean |G|*R2", fontsize=12)
    ax.set_ylabel("Improve (base - geo)", fontsize=12)
    ax.set_title("Geometry Feature vs Improvement", fontsize=14)
    ax.axhline(0, color="black", linewidth=0.5)
    # trend line
    if len(mG_plt) > 1:
        coeffs = np.polyfit(mG_plt, imp_plt, 1)
        xs_line = np.linspace(mG_plt.min(), mG_plt.max(), 100)
        ax.plot(xs_line, np.polyval(coeffs, xs_line), "r-", linewidth=2,
                label=f"trend (slope={coeffs[0]:.2e})")
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "D_improve_vs_feature.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  Saved D_improve_vs_feature.png")

    # Binned analysis — robust binning
    n_bins = 10
    flat_mG = mean_absGR2.ravel()
    sort_idx = np.argsort(flat_mG)
    bin_size = len(flat_mG) // n_bins
    bin_rows = []
    for bi in range(n_bins):
        start = bi * bin_size
        end = (bi + 1) * bin_size if bi < n_bins - 1 else len(flat_mG)
        idx_bin = sort_idx[start:end]
        imp_bin = imp[idx_bin]
        mG_bin = flat_mG[idx_bin]
        bin_rows.append({
            "bin": f"Q{bi}",
            "N": int(len(imp_bin)),
            "mean_absGR2_range": f"[{float(mG_bin.min()):.4e}, {float(mG_bin.max()):.4e}]",
            "mean_improve": float(np.mean(imp_bin)),
            "median_improve": float(np.median(imp_bin)),
            "improve_pos_ratio": float(np.mean(imp_bin > 0)),
        })
    dfD = pd.DataFrame(bin_rows)
    dfD.to_csv(os.path.join(csv_dir, "D_binned_improve.csv"), index=False)
    print(dfD.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["green" if v > 0 else "red" for v in dfD["mean_improve"].values]
    ax.bar(range(len(dfD)), dfD["mean_improve"].values, color=colors)
    ax.set_xticks(range(len(dfD)))
    ax.set_xticklabels(dfD["bin"], fontsize=11)
    ax.set_xlabel(f"mean |G|*R2  decile", fontsize=13)
    ax.set_ylabel("Mean Improve", fontsize=13)
    ax.set_title("Improvement by Geometry-Feature Decile", fontsize=14)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "D_binned_improve.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  Saved D_binned_improve.png")

    # ==================================================================
    # E: Grouped analysis
    # ==================================================================
    print("\n=== E: Grouped analysis ===")
    features = {
        "grad_target": grad_target.ravel(),
        "mean_R2": mean_R2.ravel(),
        "mean_absG": mean_absG.ravel(),
        "mean_absG_R2": mean_absGR2.ravel(),
    }
    group_rows = []
    for feat_name, feat_vals in features.items():
        lo = np.percentile(feat_vals, 33)
        hi = np.percentile(feat_vals, 67)
        for grp, mask_fn in [("low", lambda v: v <= lo),
                              ("mid", lambda v: (v > lo) & (v <= hi)),
                              ("high", lambda v: v > hi)]:
            mask = mask_fn(feat_vals)
            if mask.sum() == 0:
                continue
            imp_grp = improve.ravel()[mask]
            group_rows.append({
                "feature": feat_name, "group": grp,
                "N": int(mask.sum()),
                "base_mean_error": float(np.mean(err_base.ravel()[mask])),
                "geo_mean_error": float(np.mean(err_geo.ravel()[mask])),
                "mean_improve": float(np.mean(imp_grp)),
                "improve_pos_ratio": float(np.mean(imp_grp > 0)),
            })
    dfE = pd.DataFrame(group_rows)
    dfE.to_csv(os.path.join(csv_dir, "E_grouped_summary.csv"), index=False)
    print(dfE.to_string())

    # Bar chart: grouped improve
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, feat_name in zip(axes, features.keys()):
        sub = dfE[dfE["feature"] == feat_name]
        colors = ["red" if v < 0 else "green" for v in sub["mean_improve"]]
        ax.bar(sub["group"], sub["mean_improve"], color=colors)
        ax.set_title(feat_name, fontsize=13)
        ax.set_ylabel("Mean Improve", fontsize=12)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "E_grouped_improve.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("  Saved E_grouped_improve.png")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Pearson r (mean|G|*R2 vs grad): {rp:.4f}")
    print(f"  Base  PSNR: {compute_metrics_np(base_pred, target_region)['PSNR']:.4f}")
    print(f"  Geo   PSNR: {compute_metrics_np(geo_pred, target_region)['PSNR']:.4f}")
    print(f"  Mean improve: {float(np.mean(improve)):.6e}")
    print(f"  Improve > 0 ratio: {float(np.mean(improve > 0)):.4f}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
