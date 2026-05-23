"""Train/evaluate CTO-adapted multi-rate sparse-view CT with ASTRA data consistency.

This script uses the strict data-consistency update

    x <- x - eta * A^*(A x - y) + lambda * NO_i(x)

where A/A^* are ASTRA fan-beam forward/unfiltered backprojection operators.

Run:
    python scripts/run_cto_multirate.py --config configs/cto_multirate.json
    python scripts/run_cto_multirate.py --config configs/cto_multirate.json --epochs 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state.*"
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_run_config, save_run_config
from src.experiments.project_setup import prepare_multirate_context
from src.geometry import LInFBPFixedLinearFBPBatch
from src.models import CTOAdaptedNet
from src.evaluation.fbp_baseline import compute_fbp_baselines
from src.evaluation.metrics import compute_metrics_np
from src.evaluation.visualization import plot_comparison_grid
from src.geometry.astra_sparse_projector import AstraSparseFanBeamProjector


def infer_views(sino: torch.Tensor) -> int:
    return int(sino.shape[2])


def train_one_epoch(model, train_loader, optimizer, device, grad_clip=None, max_batches=None):
    model.train()
    losses = []
    t0 = time.time()
    for it, (sino_sparse, img) in enumerate(train_loader):
        if max_batches is not None and it >= max_batches:
            break
        sino_sparse = sino_sparse.to(device, non_blocking=True)  # [B,1,V,D]
        img = img.to(device, non_blocking=True)                  # [B,1,H,W]

        pred = model(sino_sparse)
        loss = F.mse_loss(pred, img)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        if (it + 1) % 10 == 0:
            print(f"    iter {it + 1:04d}: loss={losses[-1]:.8e}, V={infer_views(sino_sparse)}")

    return sum(losses) / max(len(losses), 1), time.time() - t0


def _build_metrics_rows(eval_out, sparse_views, epoch=None):
    rows = []
    for V in sparse_views:
        for method, metrics in [
            ("FBP", eval_out["fbp_metrics"][V]),
            ("CTO-adapted Sparse-DC", eval_out["model_metrics"][V]),
        ]:
            row = {
                "V": V,
                "method": method,
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "PSNR": metrics["PSNR"],
                "SSIM": metrics["SSIM"],
            }
            if epoch is not None:
                row["epoch"] = epoch
            rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_cto_per_v(model, eval_dataset, geo_dict, sparse_views, test_idx, region, device, precomputed_fbp=None):
    model.eval()
    sino_full_tensor, img_tensor = eval_dataset[test_idx]
    sino_full = sino_full_tensor.squeeze(0)  # [720,D]
    target = img_tensor.squeeze(0).numpy()
    x0, x1, y0, y1 = region
    target_region = target[x0:x1, y0:y1]

    model_metrics = {}
    model_preds_region = {}
    full_model_preds = {}

    if precomputed_fbp is not None:
        fbp_metrics = precomputed_fbp["fbp_metrics"]
        fbp_preds_region = precomputed_fbp["fbp_preds_region"]
        full_fbp_preds = precomputed_fbp["full_fbp_preds"]
    else:
        fbp_metrics = {}
        fbp_preds_region = {}
        full_fbp_preds = {}

    for V in sparse_views:
        print(f"[EVAL CTO] V={V}")
        step = sino_full.shape[0] // int(V)
        sino_sparse = sino_full[::step, :].unsqueeze(0).unsqueeze(0).to(device)

        pred = model(sino_sparse)[0, 0].detach().cpu().numpy()
        pred_region = pred[x0:x1, y0:y1]
        model_metrics[V] = compute_metrics_np(pred_region, target_region)
        model_preds_region[V] = pred_region
        full_model_preds[V] = pred

        if precomputed_fbp is None:
            fbp = LInFBPFixedLinearFBPBatch(geo_dict[int(V)]).to(device).eval()
            fbp_img = fbp(sino_sparse)[0, 0].detach().cpu().numpy()
            fbp_region = fbp_img[x0:x1, y0:y1]
            fbp_metrics[V] = compute_metrics_np(fbp_region, target_region)
            fbp_preds_region[V] = fbp_region
            full_fbp_preds[V] = fbp_img

        print(f"  CTO PSNR={model_metrics[V]['PSNR']:.4f} dB SSIM={model_metrics[V]['SSIM']:.6f}")
        print(f"  FBP PSNR={fbp_metrics[V]['PSNR']:.4f} dB SSIM={fbp_metrics[V]['SSIM']:.6f}")

    return {
        "target_region": target_region,
        "target_full": target,
        "model_metrics": model_metrics,
        "fbp_metrics": fbp_metrics,
        "model_preds_region": model_preds_region,
        "fbp_preds_region": fbp_preds_region,
        "model_preds_full": full_model_preds,
        "fbp_preds_full": full_fbp_preds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cto_multirate.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None, help="Optional debug limit per epoch.")
    args = parser.parse_args()

    run_cfg = load_run_config(args.config)
    exp_cfg = run_cfg.experiment
    if args.epochs is not None:
        exp_cfg.num_epochs = int(args.epochs)

    exp_cfg.ensure_dirs()
    save_run_config(run_cfg, os.path.join(exp_cfg.save_dir, "config_used.json"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ctx = prepare_multirate_context(
        dicom_folder=run_cfg.dicom_folder,
        results_folder=run_cfg.results_folder,
        save_dir=exp_cfg.save_dir,
        cache_dir=exp_cfg.cache_dir,
        cfg=exp_cfg,
        device=device,
        seed=exp_cfg.seed,
    )
    sparse_views = [int(v) for v in exp_cfg.sparse_views]
    region = tuple(run_cfg.region)
    global_test_idx = int(ctx["test_indices"][0])

    projector = AstraSparseFanBeamProjector(
        image_size=int(getattr(exp_cfg, "image_size", 256)),
        n_detec=int(getattr(exp_cfg, "n_detec", 672)),
        d_detec=float(getattr(exp_cfg, "d_detec", 1.0)),
        d_voxel=float(getattr(exp_cfg, "d_voxel", 1.0)),
        DSO=float(getattr(exp_cfg, "DSO", 595.0)),
        DOD=float(getattr(exp_cfg, "DOD", 480.0)),
        views_list=sparse_views,
        angle_range=str(getattr(exp_cfg, "angle_range", "2pi")),
        device=device,
        use_cache=True,
    )

    model = CTOAdaptedNet(
        geo_dict=ctx["geo_dict"],
        projector=projector,
        sparse_views=sparse_views,
        image_size=int(getattr(exp_cfg, "image_size", 256)),
        n_detec=int(getattr(exp_cfg, "n_detec", 672)),
        sino_hidden=getattr(exp_cfg, "cto_sino_hidden", 32),
        image_hidden=getattr(exp_cfg, "cto_image_hidden", 32),
        cascades=getattr(exp_cfg, "cto_cascades", 3),
        sino_residual_scale=getattr(exp_cfg, "cto_sino_residual_scale", 0.01),
        dc_step=getattr(exp_cfg, "cto_dc_step", 1e-7),
        image_reg_scale=getattr(exp_cfg, "cto_image_reg_scale", 0.01),
        udno_pools=getattr(exp_cfg, "udno_pools", 4),
        udno_radius_cutoff=getattr(exp_cfg, "udno_radius_cutoff", 0.02),
        udno_kernel_shape=tuple(getattr(exp_cfg, "udno_kernel_shape", [6, 7])),
        udno_drop_prob=getattr(exp_cfg, "udno_drop_prob", 0.0),
        sino_padding_mode=getattr(exp_cfg, "sino_padding_mode", "sino_circular"),
        image_padding_mode=getattr(exp_cfg, "image_padding_mode", "reflect"),
    ).to(device)

    print("\nCTO forward sanity check...")
    model.eval()
    with torch.no_grad():
        V_test = int(sparse_views[-1])
        dummy_sino = torch.randn(
            1, 1, V_test, int(getattr(exp_cfg, "n_detec", 672)),
            device=device,
            dtype=torch.float32,
        ) * 0.01
        dummy_out = model(dummy_sino)
        print("  dummy_sino:", tuple(dummy_sino.shape))
        print("  dummy_out :", tuple(dummy_out.shape))

    ckpt_path = args.checkpoint or os.path.join(exp_cfg.save_dir, "cto_adapted_sparse_dc.pt")

    # Precompute FBP baselines once (deterministic, independent of model)
    print("\nPrecomputing FBP baselines...")
    precomputed_fbp = compute_fbp_baselines(
        eval_dataset=ctx["eval_dataset"],
        geo_dict=ctx["geo_dict"],
        sparse_views=sparse_views,
        test_idx=global_test_idx,
        region=region,
        device=device,
    )

    def run_eval_and_log(epoch=None):
        eval_out = evaluate_cto_per_v(
            model=model,
            eval_dataset=ctx["eval_dataset"],
            geo_dict=ctx["geo_dict"],
            sparse_views=sparse_views,
            test_idx=global_test_idx,
            region=region,
            device=device,
            precomputed_fbp=precomputed_fbp,
        )

        metrics_df = _build_metrics_rows(eval_out, sparse_views, epoch=epoch)

        if epoch is None:
            metrics_path = os.path.join(exp_cfg.save_dir, "cto_all_methods_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            print("Saved metrics:", metrics_path)

        return metrics_df, eval_out

    if args.eval_only:
        print("Loading checkpoint:", ckpt_path)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=exp_cfg.lr,
            weight_decay=exp_cfg.weight_decay,
        )
        log_rows = []
        print("\nTraining CTO-adapted with sparse-matrix ASTRA DC")
        print("Sparse views:", sparse_views)
        print("Epochs:", exp_cfg.num_epochs)
        epoch_eval_rows = []
        all_epoch_metrics_rows = []
        best_avg_psnr = -1e9
        for epoch in range(1, exp_cfg.num_epochs + 1):
            loss, sec = train_one_epoch(
                model,
                ctx["train_loader"],
                optimizer,
                device,
                grad_clip=exp_cfg.grad_clip,
                max_batches=args.max_train_batches,
            )
            log_rows.append({"epoch": epoch, "loss": loss, "seconds": sec})
            print(f"Epoch {epoch:03d}/{exp_cfg.num_epochs}: loss={loss:.8e}, time={sec/60:.2f} min")
            # Evaluate after each epoch
            metrics_df_epoch, eval_out = run_eval_and_log(epoch=epoch)

            # Collect epoch-level metrics
            cto_df = metrics_df_epoch[metrics_df_epoch["method"] == "CTO-adapted Sparse-DC"]
            fbp_df = metrics_df_epoch[metrics_df_epoch["method"] == "FBP"]

            for _, row in cto_df.iterrows():
                V = row["V"]
                fbp_row = fbp_df[fbp_df["V"] == V].iloc[0]

                epoch_eval_rows.append({
                    "epoch": epoch,
                    "V": V,
                    "PSNR": row["PSNR"],
                    "SSIM": row["SSIM"],
                    "FBP_PSNR": fbp_row["PSNR"],
                    "FBP_SSIM": fbp_row["SSIM"],
                    "PSNR_gain_over_FBP": row["PSNR"] - fbp_row["PSNR"],
                    "SSIM_gain_over_FBP": row["SSIM"] - fbp_row["SSIM"],
                })

            all_epoch_metrics_rows.extend(metrics_df_epoch.to_dict("records"))

            epoch_eval_df = pd.DataFrame(epoch_eval_rows)
            epoch_eval_path = os.path.join(exp_cfg.save_dir, "cto_epoch_eval_log.csv")
            epoch_eval_df.to_csv(epoch_eval_path, index=False)
            print("Saved epoch eval log:", epoch_eval_path)

            # Save best checkpoint and visualization by average PSNR over all sparse rates
            avg_psnr = cto_df["PSNR"].mean()

            if avg_psnr > best_avg_psnr:
                best_avg_psnr = avg_psnr
                best_path = os.path.join(exp_cfg.save_dir, "cto_best_avg_psnr.pt")
                torch.save(model.state_dict(), best_path)
                print(f"Saved best checkpoint: {best_path}, avg PSNR={best_avg_psnr:.4f}")

                # Save visualization for this best epoch
                best_fig_path = os.path.join(exp_cfg.save_dir, f"cto_best_epoch_{epoch:03d}_comparison.png")
                preds_by_method = {
                    "FBP": eval_out["fbp_preds_region"],
                    "CTO-adapted Sparse-DC": eval_out["model_preds_region"],
                }
                psnr_by_method = {
                    "FBP": {V: eval_out["fbp_metrics"][V]["PSNR"] for V in sparse_views},
                    "CTO-adapted Sparse-DC": {V: eval_out["model_metrics"][V]["PSNR"] for V in sparse_views},
                }
                plot_comparison_grid(
                    target=eval_out["target_region"],
                    preds_by_method=preds_by_method,
                    psnr_by_method=psnr_by_method,
                    col_labels=["FBP", "CTO-adapted Sparse-DC"],
                    sparse_views=sparse_views,
                    save_path=best_fig_path,
                    show=False,
                )
                print(f"Saved best epoch visualization: {best_fig_path}")


        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint:", ckpt_path)
        train_log_path = os.path.join(exp_cfg.save_dir, "cto_train_log.csv")
        pd.DataFrame(log_rows).to_csv(train_log_path, index=False)
        print("Saved train log:", train_log_path)
        all_epoch_metrics_path = os.path.join(exp_cfg.save_dir, "cto_all_epoch_metrics.csv")
        pd.DataFrame(all_epoch_metrics_rows).to_csv(all_epoch_metrics_path, index=False)
        print("Saved all epoch metrics:", all_epoch_metrics_path)

    # Evaluation
    eval_out = evaluate_cto_per_v(
        model=model,
        eval_dataset=ctx["eval_dataset"],
        geo_dict=ctx["geo_dict"],
        sparse_views=sparse_views,
        test_idx=global_test_idx,
        region=region,
        device=device,
        precomputed_fbp=precomputed_fbp,
    )

    metrics_df = _build_metrics_rows(eval_out, sparse_views)
    metrics_path = os.path.join(exp_cfg.save_dir, "cto_all_methods_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print("Saved metrics:", metrics_path)

    fig_path = os.path.join(exp_cfg.save_dir, "cto_multirate_comparison.png")
    preds_by_method = {
        "FBP": eval_out["fbp_preds_region"],
        "CTO-adapted Sparse-DC": eval_out["model_preds_region"],
    }
    psnr_by_method = {
        "FBP": {V: eval_out["fbp_metrics"][V]["PSNR"] for V in sparse_views},
        "CTO-adapted Sparse-DC": {V: eval_out["model_metrics"][V]["PSNR"] for V in sparse_views},
    }
    plot_comparison_grid(
        target=eval_out["target_region"],
        preds_by_method=preds_by_method,
        psnr_by_method=psnr_by_method,
        col_labels=["FBP", "CTO-adapted Sparse-DC"],
        sparse_views=sparse_views,
        save_path=fig_path,
        show=False,
    )
    print("Saved comparison:", fig_path)

    results_path = os.path.join(exp_cfg.save_dir, "cto_multirate_results.pt")
    torch.save(eval_out, results_path)
    print("Saved results:", results_path)

    print("\nFINAL SUMMARY")
    print(metrics_df.pivot(index="V", columns="method", values=["PSNR", "SSIM"]))


if __name__ == "__main__":
    main()
