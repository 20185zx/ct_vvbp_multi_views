"""Gated geometry-token value analysis: original LR MLP vs gated model."""

from __future__ import annotations
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse, json, sys, importlib.util
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_run_config
from src.geometry import load_or_generate_geo, FanBeamVVBPExtractor
from src.geometry.fanbeam import compute_deltaI_patch
from src.data.local_vvbp import gather_sorted_vvbp_patch, gather_raw_vvbp_patch
from src.training.trainer import _compute_GR2_raw
from src.evaluation.metrics import compute_metrics_np


def _import_cls(mod_file, cls_name):
    mp = os.path.join(PROJECT_ROOT, "src", "models", f"{mod_file}.py")
    if not os.path.exists(mp):
        return None
    spec = importlib.util.spec_from_file_location(f"_x_{mod_file}", mp, submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_x_{mod_file}"] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name, None)


def load_model(model_type, ckpt, device, lambda_geo=None):
    """Load a model checkpoint.  ``model_type``: 'base', 'gated', or 'gated_residual'."""
    kwargs = {}
    if lambda_geo is not None:
        kwargs["lambda_geo"] = lambda_geo
    if model_type == "base":
        cls = _import_cls("local_rank_center_integral_mlp", "LocalRankCenterIntegralMLPNet")
    elif model_type == "gated_residual":
        cls = _import_cls("exact_detector_geometry_gated_residual_local_rank_integral_mlp",
                           "ExactDetectorGeometryGatedResidualLocalRankIntegralMLPNet")
    else:
        cls = _import_cls("exact_detector_geometry_gated_local_rank_integral_mlp",
                           "ExactDetectorGeometryGatedLocalRankIntegralMLPNet")
    model = cls(**kwargs).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/multirate_selected_models_60view_exactdetector_gated.json")
    p.add_argument("--base_checkpoint",
                   default="outputs/multirate_vvbp_local_rank_60view/local_rank_center_integral_mlp_10_epochs.pt")
    p.add_argument("--gated_checkpoint",
                   default="outputs/multirate_vvbp_local_rank_exactdetector_gated_residual/exact_detector_geometry_gated_residual_local_rank_center_integral_mlp_10_epochs.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="outputs/geometry_token_value_analysis_gated")
    p.add_argument("--stats_json", type=str, default=None,
                   help="Path to training_stats.json from experiment. Auto-detected from --gated_checkpoint dir if not given.")
    args = p.parse_args()

    run_cfg = load_run_config(args.config)
    exp_cfg = run_cfg.experiment
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    region = tuple(run_cfg.region)
    x0, x1, y0, y1 = region
    Hreg, Wreg = x1 - x0, y1 - y0
    V = 60

    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "csv"), exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    csv_dir = os.path.join(args.output_dir, "csv")

    # ---- Geometry + dataset ----
    geo = load_or_generate_geo(V, str(run_cfg.results_folder or "Results_512_multirate_60view"),
                               device, image_size=exp_cfg.image_size, n_detec=exp_cfg.n_detec,
                               d_detec=exp_cfg.d_detec, d_voxel=exp_cfg.d_voxel,
                               DSO=exp_cfg.DSO, DOD=exp_cfg.DOD)
    extractor = FanBeamVVBPExtractor(geo).to(device).eval()

    from src.data.dicom_dataset import build_dataloaders
    dataset, _, test_indices, _ = build_dataloaders(
        dicom_folder=str(run_cfg.dicom_folder or "full_1mm/L067/full_1mm"), cfg=exp_cfg)
    test_idx = int(test_indices[0])
    print(f"Test slice: {test_idx}  Region: {region}  Views: {V}")

    # ---- Load training stats from experiment (BEFORE model loading) ----
    stats_json_path = args.stats_json
    if stats_json_path is None:
        stats_json_path = os.path.join(os.path.dirname(args.gated_checkpoint), "training_stats.json")
    if not os.path.exists(stats_json_path):
        stats_json_path = os.path.join(os.path.dirname(args.base_checkpoint), "training_stats.json")

    G_mean_load, G_std_load, s_mean_load, s_std_load = None, None, None, None
    gr_mean_load, gr_std_load = None, None

    if os.path.exists(stats_json_path):
        print(f"Loading training stats from: {stats_json_path}")
        with open(stats_json_path) as _f:
            _sj = json.loads(_f.read())
        tgt_mean = torch.tensor(_sj["target_mean"])
        tgt_std = torch.tensor(_sj["target_std"])
        v_mean = torch.tensor(_sj.get(f"v_mean_V{V}", _sj.get("v_mean", 0.0)))
        v_std = torch.tensor(_sj.get(f"v_std_V{V}", _sj.get("v_std", 1.0)))
        G_mean_load = _sj.get("G_mean", None)
        G_std_load = _sj.get("G_std", None)
        s_mean_load = _sj.get("s_mean", None)
        s_std_load = _sj.get("s_std", None)
        gr_mean_load = _sj.get("gr_mean", None)
        gr_std_load = _sj.get("gr_std", None)
        lambda_geo_val = float(_sj.get("lambda_geo", 0.01))
    else:
        raise FileNotFoundError(
            f"training_stats.json not found at {stats_json_path}. "
            f"Run the experiment first to generate this file, or pass --stats_json."
        )

    print(f"  lambda_geo={lambda_geo_val}")
    print(f"  v_mean={float(v_mean):.8e}  v_std={float(v_std):.8e}")
    print(f"  tgt_mean={float(tgt_mean):.8e}  tgt_std={float(tgt_std):.8e}")
    print(f"  lambda_geo={lambda_geo_val}")
    if G_mean_load is not None:
        print(f"  G_mean={G_mean_load:.6e}  G_std={G_std_load:.6e}")
        print(f"  s_mean={s_mean_load:.6e}  s_std={s_std_load:.6e}")

    # ---- Load models ----
    print(f"\nBase ckpt: {args.base_checkpoint}")
    base_model = load_model("base", args.base_checkpoint, device)
    print(f"Gated ckpt: {args.gated_checkpoint}")
    gated_model = load_model("gated_residual", args.gated_checkpoint, device,
                              lambda_geo=lambda_geo_val)

    # ---- Sino + VVBP ----
    sino_t, img_t = dataset[test_idx]
    sino_full = sino_t.squeeze(0)
    target_full = img_t.squeeze(0)
    target_region = target_full[x0:x1, y0:y1].cpu().numpy()
    sino_sparse = sino_full[::sino_full.shape[0]//V, :].unsqueeze(0).unsqueeze(0).to(device)
    vvbp = extractor(sino_sparse)

    # ---- Eval all pixels ----
    coords = [(x, y) for x in range(x0, x1) for y in range(y0, y1)]
    chunk_sz = exp_cfg.chunk_size_eval

    base_chunks, gated_chunks = [], []
    all_G, all_R2, all_G_al, all_R2_al = [], [], [], []

    # G stats: load from training_stats.json if available, else estimate
    if G_mean_load is not None:
        G_stats = {"G_mean": torch.tensor(G_mean_load), "G_std": torch.tensor(G_std_load),
                   "s_mean": torch.tensor(s_mean_load), "s_std": torch.tensor(s_std_load),
                   "gr_mean": torch.tensor(gr_mean_load or 0.0), "gr_std": torch.tensor(gr_std_load or 1.0)}
    else:
        G_stats = {}
    print(f"\nEvaluating {len(coords)} pixels ...")

    for si in range(0, len(coords), chunk_sz):
        ch = coords[si:si + chunk_sz]
        xst = torch.tensor([c[0] for c in ch], dtype=torch.long, device=device)
        yst = torch.tensor([c[1] for c in ch], dtype=torch.long, device=device)
        Pc = xst.numel()

        values_sorted = gather_sorted_vvbp_patch(vvbp, xst, yst, patch_size=3, mode="3x3")
        values_sorted = values_sorted.reshape(Pc, values_sorted.shape[2], values_sorted.shape[3])

        raw_patch = gather_raw_vvbp_patch(vvbp, xst, yst, patch_size=3)
        raw_patch = raw_patch.reshape(Pc, raw_patch.shape[2], raw_patch.shape[3])

        xs_np = xst.cpu().numpy().astype(np.int64)
        ys_np = yst.cpu().numpy().astype(np.int64)
        di_np = compute_deltaI_patch(geo, xs_np, ys_np, V, patch_size=3)
        di_t = torch.from_numpy(di_np).to(device)

        Gv, R2v = _compute_GR2_raw(raw_patch, di_t, eps=1e-8)
        all_G.append(Gv.cpu().numpy())
        all_R2.append(R2v.cpu().numpy())

        centre_idx = raw_patch.shape[1] // 2
        raw_centre = raw_patch[:, centre_idx, :]
        _, c_sort = torch.sort(raw_centre, dim=-1)
        G_al = torch.gather(Gv, dim=-1, index=c_sort)
        R2_al = torch.gather(R2v, dim=-1, index=c_sort)
        all_G_al.append(G_al.cpu().numpy())
        all_R2_al.append(R2_al.cpu().numpy())

        if not G_stats:
            s_all = Gv.abs().ravel() * R2v.ravel()
            gr_all = Gv.ravel() * R2v.ravel()
            G_stats = {"G_mean": Gv.mean(), "G_std": Gv.std().clamp_min(1e-8),
                       "s_mean": s_all.mean(), "s_std": s_all.std().clamp_min(1e-8),
                       "gr_mean": gr_all.mean(), "gr_std": gr_all.std().clamp_min(1e-8)}

        # Base model
        bs_base = {"v_mean": v_mean, "v_std": v_std}
        bp = base_model(values_sorted, bs_base) * tgt_std + tgt_mean
        base_chunks.append(bp.detach().cpu().numpy().ravel())

        # Gated model
        bs_gated = {"v_mean": v_mean, "v_std": v_std,
                    "G": G_al, "R2": R2_al,
                    "G_mean": G_stats["G_mean"], "G_std": G_stats["G_std"],
                    "s_mean": G_stats["s_mean"], "s_std": G_stats["s_std"],
                    "gr_mean": G_stats["gr_mean"], "gr_std": G_stats["gr_std"]}
        gp = gated_model(values_sorted, bs_gated) * tgt_std + tgt_mean
        gated_chunks.append(gp.detach().cpu().numpy().ravel())

        if (si // chunk_sz + 1) % 4 == 0:
            print(f"  {si + Pc}/{len(coords)}")

    base_pred = np.concatenate(base_chunks).reshape(Hreg, Wreg)
    gated_pred = np.concatenate(gated_chunks).reshape(Hreg, Wreg)
    G_all = np.concatenate(all_G, axis=0)
    R2_all = np.concatenate(all_R2, axis=0)
    print(f"\nBase  PSNR={compute_metrics_np(base_pred, target_region)['PSNR']:.4f}")
    print(f"Gated PSNR={compute_metrics_np(gated_pred, target_region)['PSNR']:.4f}")

    # Per-pixel features
    mean_absG = np.mean(np.abs(G_all), axis=1).reshape(Hreg, Wreg)
    mean_R2 = np.mean(R2_all, axis=1).reshape(Hreg, Wreg)
    mean_absGR2 = np.mean(np.abs(G_all) * R2_all, axis=1).reshape(Hreg, Wreg)
    gy, gx = np.gradient(target_region)
    grad_target = np.sqrt(gy**2 + gx**2)

    err_base = np.abs(base_pred - target_region)
    err_gated = np.abs(gated_pred - target_region)
    improve = err_base - err_gated  # >0 means gated better

    # ---- Summary CSV ----
    bm = compute_metrics_np(base_pred, target_region)
    gm = compute_metrics_np(gated_pred, target_region)
    summary_rows = [{"original_psnr": bm["PSNR"], "original_ssim": bm["SSIM"],
                      "original_mae": bm["MAE"], "original_mse": bm["MSE"],
                      "gated_psnr": gm["PSNR"], "gated_ssim": gm["SSIM"],
                      "gated_mae": gm["MAE"], "gated_mse": gm["MSE"]}]
    pd.DataFrame(summary_rows).to_csv(os.path.join(csv_dir, "gated_vs_original_summary.csv"), index=False)

    # ===== D: Error maps =====
    print("\n=== D: Error maps ===")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title, cmap in zip(axes,
        [err_base, err_gated, improve],
        ["Original LR MLP Abs Error", "Gated Abs Error", "Improve (original - gated)"],
        ["hot", "hot", "RdBu_r"]):
        vm = max(np.percentile(np.abs(data), 99), 1e-10)
        vmin = -vm if "Improve" in title else 0
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vm)
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "D_error_maps_gated.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)

    # D: Improve vs feature
    mG = mean_absGR2.ravel()
    imp = improve.ravel()
    fig, ax = plt.subplots(figsize=(8, 5))
    idx = np.random.default_rng(42).choice(len(mG), min(20000, len(mG)), replace=False)
    ax.scatter(mG[idx], imp[idx], alpha=0.15, s=2, c="steelblue")
    if len(idx) > 1:
        cfs = np.polyfit(mG[idx], imp[idx], 1)
        xs = np.linspace(mG[idx].min(), mG[idx].max(), 100)
        ax.plot(xs, np.polyval(cfs, xs), "r-", lw=2, label=f"slope={cfs[0]:.2e}")
        ax.legend(fontsize=10)
    ax.set_xlabel("mean |G|*R2", fontsize=13)
    ax.set_ylabel("Improve (original - gated)", fontsize=13)
    ax.set_title("Gated: Feature vs Improvement", fontsize=14)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "D_improve_vs_feature_gated.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)

    # D: Binned
    n_bins = 10
    srt = np.argsort(mG)
    bs = len(mG) // n_bins
    bin_rows = []
    for bi in range(n_bins):
        ib = srt[bi * bs: (bi + 1) * bs if bi < n_bins - 1 else len(mG)]
        imp_b = imp[ib]; mG_b = mG[ib]
        bin_rows.append({"bin_id": f"Q{bi}",
                         "feature_min": float(mG_b.min()), "feature_max": float(mG_b.max()),
                         "pixel_count": int(len(imp_b)),
                         "mean_improve": float(np.mean(imp_b)),
                         "median_improve": float(np.median(imp_b)),
                         "improve_positive_ratio": float(np.mean(imp_b > 0)),
                         "base_mean_error": float(np.mean(err_base.ravel()[ib])),
                         "gated_mean_error": float(np.mean(err_gated.ravel()[ib]))})
    dfD = pd.DataFrame(bin_rows)
    dfD.to_csv(os.path.join(csv_dir, "D_binned_improve_gated.csv"), index=False)
    print(dfD.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    clrs = ["green" if v > 0 else "red" for v in dfD["mean_improve"]]
    ax.bar(range(n_bins), dfD["mean_improve"], color=clrs)
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(dfD["bin_id"], fontsize=11)
    ax.set_xlabel("mean |G|*R2  decile", fontsize=13)
    ax.set_ylabel("Mean Improve", fontsize=13)
    ax.set_title("Gated: Improvement by Feature Decile", fontsize=14)
    ax.axhline(0, color="black", lw=0.5); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "D_binned_improve_gated.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)

    # ===== E: Grouped =====
    print("\n=== E: Grouped ===")
    features = {"target_gradient": grad_target.ravel(),
                "mean_absG": mean_absG.ravel(),
                "mean_R2": mean_R2.ravel(),
                "mean_absG_R2": mean_absGR2.ravel()}
    group_rows = []
    for fn, fv in features.items():
        lo = np.percentile(fv, 33); hi = np.percentile(fv, 67)
        for gn, mf in [("low", lambda v: v <= lo),
                        ("mid", lambda v: (v > lo) & (v <= hi)),
                        ("high", lambda v: v > hi)]:
            mk = mf(fv)
            if mk.sum() == 0: continue
            imp_g = improve.ravel()[mk]
            group_rows.append({"group_by": fn, "group_name": gn,
                               "pixel_count": int(mk.sum()),
                               "base_mean_error": float(np.mean(err_base.ravel()[mk])),
                               "gated_mean_error": float(np.mean(err_gated.ravel()[mk])),
                               "mean_improve": float(np.mean(imp_g)),
                               "median_improve": float(np.median(imp_g)),
                               "improve_positive_ratio": float(np.mean(imp_g > 0))})
    dfE = pd.DataFrame(group_rows)
    dfE.to_csv(os.path.join(csv_dir, "E_grouped_summary_gated.csv"), index=False)
    print(dfE.to_string())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, fn in zip(axes, features.keys()):
        sub = dfE[dfE["group_by"] == fn]
        clrs = ["red" if v < 0 else "green" for v in sub["mean_improve"]]
        ax.bar(sub["group_name"], sub["mean_improve"], color=clrs)
        ax.set_title(fn, fontsize=12)
        ax.set_ylabel("Mean Improve", fontsize=11)
        ax.axhline(0, color="black", lw=0.5); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(fig_dir, "E_grouped_improve_gated.png")
    fig.savefig(fp, dpi=300, bbox_inches="tight"); plt.close(fig)

    # ===== A / C (reference) =====
    absG = np.abs(G_all).ravel()
    r2f = R2_all.ravel()
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, d, t in zip(axes, [G_all.ravel(), absG, r2f, absG * r2f],
                         ["G", "|G|", "R2", "|G|*R2"]):
        ax.hist(d, bins=100, alpha=0.7, color="steelblue", edgecolor="white")
        ax.set_title(t, fontsize=14); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "A_feature_distribution.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, d, t in zip(axes, [mean_absG, mean_R2, mean_absGR2],
                         ["mean |G|", "mean R2", "mean |G|*R2"]):
        im = ax.imshow(d, cmap="viridis"); ax.set_title(t, fontsize=14)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "B_per_pixel_maps.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    rp, pp = pearsonr(mean_absGR2.ravel(), grad_target.ravel())
    rs, ps = spearmanr(mean_absGR2.ravel(), grad_target.ravel())
    print(f"\n  Pearson r (mean|G|*R2 vs grad): {rp:.4f}  Spearman: {rs:.4f}")

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("GATED ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Original PSNR={bm['PSNR']:.4f}  SSIM={bm['SSIM']:.6f}  MAE={bm['MAE']:.6e}")
    print(f"  Gated    PSNR={gm['PSNR']:.4f}  SSIM={gm['SSIM']:.6f}  MAE={gm['MAE']:.6e}")
    print(f"  Mean improve: {float(np.mean(improve)):.6e}")
    print(f"  Improve > 0 ratio: {float(np.mean(improve > 0)):.4f}")

    # Binned insights
    low_q = dfD.iloc[:3]["mean_improve"].mean()
    high_q = dfD.iloc[-3:]["mean_improve"].mean()
    print(f"  Low  |G|*R2 (Q0-Q2) mean improve: {low_q:.6e}")
    print(f"  High |G|*R2 (Q7-Q9) mean improve: {high_q:.6e}")

    high_grad = dfE[(dfE["group_by"] == "target_gradient") & (dfE["group_name"] == "high")]["mean_improve"].values
    if len(high_grad):
        print(f"  High-gradient group mean improve: {high_grad[0]:.6e}")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
