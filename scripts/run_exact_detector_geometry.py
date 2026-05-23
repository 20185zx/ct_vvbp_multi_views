"""Experiment B: exact-detector geometry-aware token.

Adds G_{p,v} and R2_{p,v} to each view token in the
local rank centre integral MLP, computed from exact fan-beam
detector coordinates.

Run:
    python scripts/run_exact_detector_geometry.py \
        --config configs/multirate_selected_models_60view_exactdetector_token.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_run_config, save_run_config
from src.experiments.project_setup import prepare_multirate_context
from src.evaluation.fbp_baseline import compute_fbp_baselines
from src.evaluation.visualization import plot_comparison_grid
from src.models import build_model
from src.training import estimate_multirate_stats
from src.training.trainer import train_multirate_model_geometry_token
from src.geometry.fanbeam import compute_deltaI_patch
from src.evaluation.metrics import compute_metrics_np
from src.data.local_vvbp import sample_random_coords, gather_sorted_vvbp_patch, gather_raw_vvbp_patch
from src.training.trainer import _compute_GR2_raw


def safe_model_name(name: str) -> str:
    return name.replace(", ", "_").replace(" ", "_")


def short_label(name: str) -> str:
    return safe_model_name(name).replace("_10_epochs", "")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment B: exact-detector geometry token."
    )
    parser.add_argument("--config", type=str,
                        default="configs/multirate_selected_models_60view_exactdetector_token.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--orig_checkpoint", type=str,
                        default="outputs/multirate_vvbp_local_rank_60view/local_rank_center_integral_mlp_10_epochs.pt",
                        help="Path to original local-rank MLP checkpoint for comparison.")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    run_cfg = load_run_config(args.config)
    exp_cfg = run_cfg.experiment

    exp_cfg.ensure_dirs()
    save_run_config(run_cfg, os.path.join(exp_cfg.save_dir, "config_used.json"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Multi-rate context ----
    ctx = prepare_multirate_context(
        dicom_folder=run_cfg.dicom_folder,
        results_folder=run_cfg.results_folder,
        save_dir=exp_cfg.save_dir,
        cache_dir=exp_cfg.cache_dir,
        cfg=exp_cfg,
        device=device,
        seed=exp_cfg.seed,
    )
    geo_dict = ctx["geo_dict"]
    extractors = ctx["extractors"]
    train_loader = ctx["train_loader"]
    eval_dataset = ctx["eval_dataset"]
    test_indices = ctx["test_indices"]
    sparse_views = exp_cfg.sparse_views
    region = tuple(run_cfg.region)

    print("\nSparse views:", sparse_views)
    print("Region:", region)

    # ---- Estimate stats (target, v_mean, v_std) ----
    print("\n" + "=" * 60)
    print("Estimating normalisation statistics ...")
    print("=" * 60)
    target_stats_raw, v_stats_raw = estimate_multirate_stats(
        train_loader, extractors, exp_cfg, device, num_stats_batches=4,
    )
    target_stats = {k: v.to(device) for k, v in target_stats_raw.items()}
    v_stats = {V: {k: v.to(device) for k, v in s.items()} for V, s in v_stats_raw.items()}

    # ---- Save stats for analysis scripts ----
    _stats_json = {
        "target_mean": float(target_stats["target_mean"].cpu()),
        "target_std": float(target_stats["target_std"].cpu()),
    }
    for _V, _vs in v_stats.items():
        _stats_json[f"v_mean_V{_V}"] = float(_vs["v_mean"].cpu())
        _stats_json[f"v_std_V{_V}"] = float(_vs["v_std"].cpu())
    import json as _json
    _sp = os.path.join(exp_cfg.save_dir, "training_stats.json")
    with open(_sp, "w") as _f:
        _json.dump(_stats_json, _f, indent=2)
    print(f"Saved training stats: {_sp}")

    # ---- Estimate G stats from training data ----
    # G is the detector-direction slope per view.  We estimate its mean/std
    # from the same stats batches that were used for v_stats.
    print("\nEstimating G statistics ...")
    G_all = []
    R2_all = []
    geo = geo_dict[sparse_views[0]]
    for sino_batch, img_batch in train_loader:
        sino_batch = sino_batch.to(device)
        img_batch = img_batch.to(device)
        V = sino_batch.shape[-2]
        extractor = extractors[V]
        vvbp = extractor(sino_batch)
        B, _, H, W, _ = vvbp.shape
        xs, ys = sample_random_coords(H, W, 1024, margin=exp_cfg.patch_size // 2, device=device)
        # Gather RAW (unsorted) patch for G stats
        raw_patch = gather_raw_vvbp_patch(vvbp, xs, ys, patch_size=exp_cfg.patch_size)
        N = B * 1024
        raw_patch = raw_patch.reshape(N, raw_patch.shape[2], raw_patch.shape[3])
        xs_np = xs.cpu().numpy().astype(np.int64)
        ys_np = ys.cpu().numpy().astype(np.int64)
        di = compute_deltaI_patch(geo, xs_np, ys_np, int(V), patch_size=exp_cfg.patch_size)
        di_t = torch.from_numpy(di).to(device)
        G, R2 = _compute_GR2_raw(raw_patch, di_t, eps=1e-8)
        G_all.append(G.flatten().cpu())
        R2_all.append(R2.flatten().cpu())
        if len(G_all) >= 4:
            break
    G_cat = torch.cat(G_all)
    R2_cat = torch.cat(R2_all)
    G_mean = G_cat.mean().to(device)
    G_std = (G_cat.std() + 1e-8).to(device)

    # s = abs(G) * R2 stats
    s_all = G_cat.abs() * R2_cat
    s_mean = s_all.mean().to(device)
    s_std = (s_all.std() + 1e-8).to(device)
    gr_all = G_cat * R2_cat
    gr_mean = gr_all.mean().to(device)
    gr_std = (gr_all.std() + 1e-8).to(device)
    G_stats = {"G_mean": G_mean, "G_std": G_std,
               "s_mean": s_mean, "s_std": s_std,
               "gr_mean": gr_mean, "gr_std": gr_std}
    print(f"[G STATS] G_mean={float(G_mean):.6e}  G_std={float(G_std):.6e}  "
          f"s_mean={float(s_mean):.6e}  s_std={float(s_std):.6e}")
    print(f"[G STATS] gr_mean={float(gr_mean):.6e}  gr_std={float(gr_std):.6e}")

    # Append G/s stats to training_stats.json
    _sp2 = os.path.join(exp_cfg.save_dir, "training_stats.json")
    if os.path.exists(_sp2):
        import json as _json2
        with open(_sp2) as _f:
            _sj = _json2.loads(_f.read())
        _sj["G_mean"] = float(G_mean.cpu())
        _sj["G_std"] = float(G_std.cpu())
        _sj["s_mean"] = float(s_mean.cpu())
        _sj["s_std"] = float(s_std.cpu())
        _sj["gr_mean"] = float(gr_mean.cpu())
        _sj["gr_std"] = float(gr_std.cpu())
        # lambda_geo from raw config
        _raw2 = json.loads(Path(args.config).read_text(encoding="utf-8"))
        _gc2 = _raw2.get("experiment", {}).get("geometry_residual", {}) or \
               _raw2.get("experiment", {}).get("geometry_gate", {}) or {}
        _sj["lambda_geo"] = float(_gc2.get("lambda_geo", 0.01))
        with open(_sp2, "w") as _f:
            _json2.dump(_sj, _f, indent=2)
        print(f"Updated training stats with G/s: {_sp2}")

    # ---- Model ----
    model_names = run_cfg.model_names or ["exact detector geometry local rank center integral mlp, 10 epochs"]
    global_test_idx = int(test_indices[0])

    all_model_preds = {}
    all_model_metrics = {}
    baseline_preds = None
    baseline_metrics_raw = None
    target_arr = None

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        # Read geometry residual params from raw config
        _raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        _geom_cfg = _raw.get("experiment", {}).get("geometry_residual", {}) or \
                    _raw.get("experiment", {}).get("geometry_gate", {}) or {}
        _model_kwargs = {}
        if _geom_cfg.get("lambda_geo") is not None:
            _model_kwargs["lambda_geo"] = float(_geom_cfg["lambda_geo"])
        if _geom_cfg.get("gate_a_init") is not None:
            _model_kwargs["gate_a_init"] = float(_geom_cfg["gate_a_init"])
        if _geom_cfg.get("gate_b_init") is not None:
            _model_kwargs["gate_b_init"] = float(_geom_cfg["gate_b_init"])
        model = build_model(model_name, **_model_kwargs)

        if args.eval_only:
            model = model.to(device)
            ckpt_path = args.checkpoint or os.path.join(
                exp_cfg.save_dir, f"{safe_model_name(model_name)}.pt"
            )
            print(f"Loading checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            t_start = time.time()
            train_log, comp_log = train_multirate_model_geometry_token(
                model=model,
                train_loader=train_loader,
                extractors=extractors,
                geo=geo,
                target_stats=target_stats,
                v_stats=v_stats,
                G_stats=G_stats,
                model_name=model_name,
                num_epochs=exp_cfg.num_epochs,
                patch_size=exp_cfg.patch_size,
                pixels_per_batch=exp_cfg.cache_pixels_per_slice,
                lr=exp_cfg.lr,
                weight_decay=exp_cfg.weight_decay,
                grad_clip=exp_cfg.grad_clip,
                device=device,
                train_region=exp_cfg.train_region,
            )
            train_time = time.time() - t_start
            print(f"Training time: {train_time / 60:.1f} min")

            model_path = os.path.join(exp_cfg.save_dir, f"{safe_model_name(model_name)}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved model: {model_path}")

            log_df = pd.DataFrame(comp_log)
            log_path = os.path.join(exp_cfg.save_dir,
                                     f"train_log_{safe_model_name(model_name)}.csv")
            log_df.to_csv(log_path, index=False)
            print(f"Saved train log: {log_path}")

        # ---- Evaluate ----
        model = model.to(device)
        model.eval()

        print(f"\nEvaluation on test slice {global_test_idx}, region {region}")

        # For evaluation, we need to inject deltaI_patch into the stats
        # that the model receives.  evaluate_multirate builds batch_stats
        # internally, so we modify v_stats to carry G stats + a placeholder.
        # The model reads deltaI from batch_stats during inference.
        # Since evaluate_multirate calls gather_sorted_vvbp_patch for each chunk
        # but doesn't compute deltaI, we need to handle this differently.
        #
        # Solution: subclass or replace evaluate_multirate to also compute deltaI.
        # For now, evaluate directly on the eval slice with custom code.

        sino_full_tensor, img_tensor = eval_dataset[global_test_idx]
        sino_full = sino_full_tensor.squeeze(0)
        target_full = img_tensor.squeeze(0)
        x0, x1, y0, y1 = region
        target_region = target_full[x0:x1, y0:y1].cpu().numpy()
        Hreg, Wreg = x1 - x0, y1 - y0

        model_metrics = {}
        model_preds_region = {}

        for V_val in sparse_views:
            print(f"  Evaluating V={V_val} ...")
            step = sino_full.shape[0] // int(V_val)
            sino_sparse = sino_full[::step, :].unsqueeze(0).unsqueeze(0).to(device)

            extractor = extractors[V_val]
            vvbp = extractor(sino_sparse)

            coords = [(x, y) for x in range(x0, x1) for y in range(y0, y1)]
            chunk_size = exp_cfg.chunk_size_eval
            pred_chunks = []

            for start_i in range(0, len(coords), chunk_size):
                chunk = coords[start_i:start_i + chunk_size]
                xs_t = torch.tensor([c[0] for c in chunk], dtype=torch.long, device=device)
                ys_t = torch.tensor([c[1] for c in chunk], dtype=torch.long, device=device)

                values_sorted = gather_sorted_vvbp_patch(
                    vvbp, xs_t, ys_t, patch_size=exp_cfg.patch_size, mode="3x3",
                )
                P_chunk = xs_t.numel()
                values_sorted = values_sorted.reshape(P_chunk, values_sorted.shape[2], values_sorted.shape[3])

                # Raw (unsorted) patch for G/R2
                raw_patch = gather_raw_vvbp_patch(vvbp, xs_t, ys_t, patch_size=exp_cfg.patch_size)
                raw_patch = raw_patch.reshape(P_chunk, raw_patch.shape[2], raw_patch.shape[3])

                # Compute deltaI
                xs_np = xs_t.cpu().numpy().astype(np.int64)
                ys_np = ys_t.cpu().numpy().astype(np.int64)
                di_np = compute_deltaI_patch(geo, xs_np, ys_np, int(V_val),
                                             patch_size=exp_cfg.patch_size)
                di_t = torch.from_numpy(di_np).to(device)

                # Compute G/R2 from raw VVBP (view order)
                G_val, R2_val = _compute_GR2_raw(raw_patch, di_t, eps=1e-8)

                # Reorder G/R2 to align with values_sorted centre value order.
                # values_sorted[:, centre, :] are sorted by VVBP value, not view.
                # We need G/R2 in the SAME order so tokens stay aligned.
                centre_idx = raw_patch.shape[1] // 2
                raw_centre = raw_patch[:, centre_idx, :]   # [P, V] view order
                _, c_sort = torch.sort(raw_centre, dim=-1)  # sort index by centre value
                G_aligned = torch.gather(G_val, dim=-1, index=c_sort)
                R2_aligned = torch.gather(R2_val, dim=-1, index=c_sort)

                vs_val = v_stats[V_val]
                batch_stats = {
                    "target_mean": target_stats["target_mean"],
                    "target_std": target_stats["target_std"],
                    "v_mean": vs_val["v_mean"].to(device),
                    "v_std": vs_val["v_std"].to(device),
                    "G_mean": G_stats["G_mean"],
                    "G_std": G_stats["G_std"],
                    "s_mean": G_stats["s_mean"],
                    "s_std": G_stats["s_std"],
                    "gr_mean": G_stats["gr_mean"],
                    "gr_std": G_stats["gr_std"],
                    "G": G_aligned.detach(),
                    "R2": R2_aligned.detach(),
                }

                if getattr(model, "input_mode", "features") == "values_sorted":
                    pred_norm = model(values_sorted, batch_stats)
                else:
                    from src.data.feature_builder import make_model_features_from_values
                    features = make_model_features_from_values(
                        values_sorted=values_sorted, stats=batch_stats,
                        use_coord=getattr(model, "use_coord", False),
                        patch_size=exp_cfg.patch_size,
                    )
                    pred_norm = model(features)

                pred_vals = pred_norm * target_stats["target_std"] + target_stats["target_mean"]
                pred_chunks.append(pred_vals.detach().cpu())

            pred_full = torch.cat(pred_chunks, dim=0).numpy().reshape(Hreg, Wreg)

            model_metrics[V_val] = compute_metrics_np(pred_full, target_region)
            model_preds_region[V_val] = pred_full
            print(f"    PSNR={model_metrics[V_val]['PSNR']:.4f} dB  "
                  f"SSIM={model_metrics[V_val]['SSIM']:.6f}")

        all_model_preds[model_name] = model_preds_region
        mdf = pd.DataFrame(model_metrics).T
        mdf.index.name = "V"
        all_model_metrics[model_name] = mdf

        metrics_path = os.path.join(exp_cfg.save_dir,
                                     f"per_v_metrics_{safe_model_name(model_name)}.csv")
        mdf.to_csv(metrics_path)
        print(f"Saved per-V metrics: {metrics_path}")
        print(mdf[["PSNR", "SSIM"]])

        target_arr = target_region

        # Compute parameter-free baselines directly (no model needed).
        # Re-use sinogram + VVBP already extracted above.
        if baseline_preds is None:
            baseline_metrics_raw = {"Local-rank closed": {}}
            baseline_preds = {"Local-rank closed": {}}

            for V_val in sparse_views:
                step = sino_full.shape[0] // int(V_val)
                sino_sparse = sino_full[::step, :].unsqueeze(0).unsqueeze(0).to(device)
                extractor = extractors[V_val]
                vvbp = extractor(sino_sparse)

                coords = [(x, y) for x in range(x0, x1) for y in range(y0, y1)]
                chunk_size = exp_cfg.chunk_size_eval
                lr_chunks = []

                for start_i in range(0, len(coords), chunk_size):
                    chunk = coords[start_i:start_i + chunk_size]
                    xs_t = torch.tensor([c[0] for c in chunk], dtype=torch.long, device=device)
                    ys_t = torch.tensor([c[1] for c in chunk], dtype=torch.long, device=device)

                    values_sorted = gather_sorted_vvbp_patch(
                        vvbp, xs_t, ys_t, patch_size=exp_cfg.patch_size, mode="3x3",
                    )
                    Pc = xs_t.numel()
                    values_sorted = values_sorted.reshape(Pc, values_sorted.shape[2], values_sorted.shape[3])

                    # Local-rank closed integral baseline
                    from src.data.local_rank import compute_local_rank_closed_integral
                    lr_pred = compute_local_rank_closed_integral(values_sorted)
                    lr_chunks.append(lr_pred.cpu())

                lr_full = torch.cat(lr_chunks, dim=0).numpy().reshape(Hreg, Wreg)
                baseline_metrics_raw["Local-rank closed"][V_val] = compute_metrics_np(lr_full, target_region)
                baseline_preds["Local-rank closed"][V_val] = lr_full

            print(f"  Local-rank closed PSNR={baseline_metrics_raw['Local-rank closed'][sparse_views[0]]['PSNR']:.4f} dB")

    # ---- FBP baselines + comparison figure (same as original) ----
    x0, x1, y0, y1 = region
    precomputed_fbp = compute_fbp_baselines(
        eval_dataset=eval_dataset, geo_dict=geo_dict,
        sparse_views=sparse_views, test_idx=global_test_idx,
        region=region, device=device,
    )
    fbp_metrics = precomputed_fbp["fbp_metrics"]
    fbp_preds = precomputed_fbp["full_fbp_preds"]

    # ---- Load original local-rank MLP for comparison (optional) ----
    orig_preds = None
    orig_psnr = None
    orig_ckpt = args.orig_checkpoint
    if os.path.exists(orig_ckpt):
        print("\n[ORIG COMPARE] Loading original local-rank MLP ...")
        print(f"  Checkpoint: {orig_ckpt}")
        try:
            orig_model = build_model("local rank center integral mlp, 10 epochs").to(device).eval()
            orig_model.load_state_dict(torch.load(orig_ckpt, map_location=device))
            print(f"  Model class: {type(orig_model).__name__}")
            print(f"  Eval slice: {global_test_idx}")
            print(f"  Eval region: {region}")
            print(f"  target_mean={float(target_stats['target_mean']):.6e}  "
                  f"target_std={float(target_stats['target_std']):.6e}")

            orig_preds = {}
            orig_psnr = {}
            for V_val in sparse_views:
                step = sino_full.shape[0] // int(V_val)
                sino_s = sino_full[::step, :].unsqueeze(0).unsqueeze(0).to(device)
                ex = extractors[V_val]
                vv = ex(sino_s)
                vsv = v_stats[V_val]
                print(f"  V={V_val}: v_mean={float(vsv['v_mean']):.6e}  "
                      f"v_std={float(vsv['v_std']):.6e}")
                cs = [(x, y) for x in range(x0, x1) for y in range(y0, y1)]
                pc = []
                for si in range(0, len(cs), exp_cfg.chunk_size_eval):
                    ch = cs[si:si + exp_cfg.chunk_size_eval]
                    xst = torch.tensor([c[0] for c in ch], dtype=torch.long, device=device)
                    yst = torch.tensor([c[1] for c in ch], dtype=torch.long, device=device)
                    vss = gather_sorted_vvbp_patch(vv, xst, yst, patch_size=3, mode="3x3")
                    vss = vss.reshape(xst.numel(), vss.shape[2], vss.shape[3])
                    bs = {"v_mean": vsv["v_mean"].to(device), "v_std": vsv["v_std"].to(device)}
                    pn = orig_model(vss, bs)
                    pv = pn * target_stats["target_std"] + target_stats["target_mean"]
                    pc.append(pv.detach().cpu())
                op = torch.cat(pc, dim=0).numpy().reshape(Hreg, Wreg)
                orig_preds[V_val] = op
                orig_psnr[V_val] = compute_metrics_np(op, target_arr)["PSNR"]
                print(f"    pred min={float(op.min()):.6e}  max={float(op.max()):.6e}  "
                      f"mean={float(op.mean()):.6e}  std={float(op.std()):.6e}")
                print(f"    PSNR={orig_psnr[V_val]:.4f} dB  "
                      f"SSIM={compute_metrics_np(op, target_arr)['SSIM']:.6f}")

            # Compare with original per_v_metrics if available
            orig_csv_dir = os.path.dirname(orig_ckpt)
            orig_csv = os.path.join(orig_csv_dir,
                                     "per_v_metrics_local_rank_center_integral_mlp_10_epochs.csv")
            if os.path.exists(orig_csv):
                orig_df = pd.read_csv(orig_csv, index_col=0)
                print(f"  Original per_v_metrics from {orig_csv}:")
                print(orig_df.to_string())
        except Exception as e:
            import traceback
            print(f"  Could not load original model: {e}")
            traceback.print_exc()
    else:
        print(f"\n[ORIG COMPARE] Original checkpoint not found at {orig_ckpt}, skipping")

    # Build comparison grid
    preds_by_method = {}
    psnr_by_method = {}
    col_labels = ["FBP"]

    fbp_region_preds = {V: fbp_preds[V][x0:x1, y0:y1] for V in sparse_views}
    preds_by_method["FBP"] = fbp_region_preds
    psnr_by_method["FBP"] = {V: fbp_metrics[V]["PSNR"] for V in sparse_views}

    if baseline_preds is not None:
        preds_by_method["Local-rank closed"] = baseline_preds["Local-rank closed"]
        psnr_by_method["Local-rank closed"] = {
            V: baseline_metrics_raw["Local-rank closed"][V]["PSNR"] for V in sparse_views
        }
        col_labels.append("Local-rank closed")

    # Original local-rank MLP
    if orig_preds is not None:
        preds_by_method["Original LR MLP"] = orig_preds
        psnr_by_method["Original LR MLP"] = orig_psnr
        col_labels.append("Original LR MLP")

    for model_name in model_names:
        short = short_label(model_name)[:24]
        preds_by_method[short] = all_model_preds[model_name]
        psnr_by_method[short] = {
            V: float(all_model_metrics[model_name].loc[V, "PSNR"]) for V in sparse_views
        }
        col_labels.append(short)

    fig_path = os.path.join(exp_cfg.save_dir, "geometry_token_comparison.png")
    plot_comparison_grid(
        target=target_arr,
        preds_by_method=preds_by_method,
        psnr_by_method=psnr_by_method,
        col_labels=col_labels,
        sparse_views=sparse_views,
        save_path=fig_path,
        show=False,
    )
    print(f"Saved comparison: {fig_path}")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT B COMPLETE")
    print("=" * 60)
    print(f"Output dir: {exp_cfg.save_dir}")


if __name__ == "__main__":
    main()
