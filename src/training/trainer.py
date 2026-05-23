import time
import math
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.feature_builder import make_model_features_from_values
from src.data.local_vvbp import sample_random_coords, gather_sorted_vvbp_patch, gather_raw_vvbp_patch
from src.training.losses import compute_total_loss


def _compute_GR2_raw(raw_patch: torch.Tensor, deltaI_patch: torch.Tensor,
                      eps: float = 1e-8):
    """Compute G and R² from RAW (unsorted, per-view) patch VVBP + exact ΔI.

    Through-origin OLS; SS_tot is uncentered (null model = zero).
    See ``ExactDetectorGeometryLocalRankIntegralMLPNet.compute_G_R2_from_raw``.
    """
    B, J, K = raw_patch.shape
    centre_idx = J // 2

    Qc = raw_patch[:, centre_idx, :]         # [B, K]
    Y = raw_patch - Qc[:, None, :]           # [B, J, K]
    D = deltaI_patch                         # [B, J, K]

    DS = (D * D).sum(dim=1)                  # [B, K]
    DY = (D * Y).sum(dim=1)                  # [B, K]
    G = DY / (DS + eps)                      # [B, K]

    Yh = D * G[:, None, :]                   # [B, J, K]
    SS_res = ((Y - Yh) ** 2).sum(dim=1)      # [B, K]
    SS_tot = (Y * Y).sum(dim=1) + eps       # [B, K]  uncentered
    R2 = (1.0 - SS_res / SS_tot).clamp(0.0, 1.0)

    return G, R2


def train_direct_model_cached(
    model,
    cached_loader,
    stats,
    model_name="model",
    num_epochs=20,
    patch_size=3,
    lr=1e-3,
    weight_decay=1e-6,
    grad_clip=None,
    device="cuda",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_log = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0
        for values_sorted, target, _, _ in cached_loader:
            values_sorted = values_sorted.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            y_norm = (target - stats["target_mean"]) / stats["target_std"]

            if getattr(model, "input_mode", "features") == "values_sorted":
                pred_norm = model(values_sorted, stats)
            else:
                features = make_model_features_from_values(
                    values_sorted=values_sorted,
                    stats=stats,
                    use_coord=model.use_coord,
                    patch_size=patch_size,
                )
                pred_norm = model(features)
            loss = criterion(pred_norm, y_norm)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            bs = y_norm.shape[0]
            total_loss += loss.item() * bs
            total_count += bs
        avg_loss = total_loss / max(total_count, 1)
        train_log.append(avg_loss)
        print(f"{model_name} | Epoch [{epoch:03d}/{num_epochs}] loss={avg_loss:.6f} time={time.time() - t0:.1f}s")
    return train_log


@torch.no_grad()
def estimate_multirate_stats(
    train_loader,
    extractors,
    cfg,
    device,
    num_stats_batches=4,
):
    """Estimate target mean/std and per-V v_mean/v_std from training data.

    Iterates up to num_stats_batches; continues until every V in extractors
    has at least one sample or the loader is exhausted.

    Returns:
        target_stats: dict with target_mean, target_std (V-independent).
        v_stats: dict mapping V -> {v_mean, v_std}.
    """
    target_values = []
    v_values = {V: [] for V in extractors.keys()}
    max_batches = max(num_stats_batches, len(extractors) * 2)

    for bi, (sino_batch, img_batch) in enumerate(train_loader):
        sino_batch = sino_batch.to(device)
        img_batch = img_batch.to(device)
        V = sino_batch.shape[-2]

        target_values.append(img_batch.flatten().cpu())

        extractor = extractors[V]
        vvbp = extractor(sino_batch)
        v_values[V].append(vvbp.flatten().cpu())

        all_covered = all(len(v_values[v]) > 0 for v in extractors.keys())
        if bi >= max_batches - 1 and all_covered:
            break

    all_targets = torch.cat(target_values)
    target_mean = all_targets.mean()
    target_std = all_targets.std().clamp_min(1e-8)

    print(f"[STATS] target_mean={float(target_mean):.6f}  target_std={float(target_std):.6f}")

    v_stats = {}
    for V, values in v_values.items():
        if len(values) == 0:
            raise RuntimeError(f"No stats collected for V={V}. Increase num_stats_batches.")
        all_v = torch.cat(values)
        v_stats[V] = {
            "v_mean": all_v.mean(),
            "v_std": all_v.std().clamp_min(1e-8),
        }
        print(f"[STATS] V={V}: v_mean={float(v_stats[V]['v_mean']):.6f}  v_std={float(v_stats[V]['v_std']):.6f}")

    return {"target_mean": target_mean, "target_std": target_std}, v_stats


def train_multirate_model(
    model,
    train_loader,
    extractors,
    target_stats,
    v_stats,
    model_name="multirate_model",
    num_epochs=10,
    patch_size=3,
    pixels_per_batch=8192,
    lr=1e-3,
    weight_decay=1e-6,
    grad_clip=None,
    device="cuda",
    train_region=None,
):
    """Train one model with on-the-fly multi-rate VVBP extraction.

    Each batch comes from a MultiRateFanbeamDataset (train=True) and
    contains a sinogram subsampled to a random V ∈ sparse_views.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_log = []

    target_mean = target_stats["target_mean"].to(device)
    target_std = target_stats["target_std"].to(device)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0

        for sino_batch, img_batch in train_loader:
            sino_batch = sino_batch.to(device, non_blocking=True)
            img_batch = img_batch.to(device, non_blocking=True)

            V = sino_batch.shape[-2]

            # Select per-V extractor and stats.
            extractor = extractors[V]
            vs = v_stats[V]
            batch_stats = {
                "target_mean": target_mean,
                "target_std": target_std,
                "v_mean": vs["v_mean"].to(device),
                "v_std": vs["v_std"].to(device),
            }

            # On-the-fly VVBP extraction.
            vvbp = extractor(sino_batch)
            B, _, H, W, _ = vvbp.shape

            xs, ys = sample_random_coords(
                H, W, num_pixels=pixels_per_batch,
                margin=patch_size // 2, device=device,
                train_region=train_region,
            )
            values_sorted = gather_sorted_vvbp_patch(
                vvbp, xs, ys, patch_size=patch_size, mode="3x3",
            )  # [B, P, J, K] with B=1 (single sinogram)
            N = B * pixels_per_batch
            values_sorted = values_sorted.reshape(N, values_sorted.shape[2], values_sorted.shape[3])
            target = img_batch[:, 0, xs, ys].reshape(N, 1)

            y_norm = (target - batch_stats["target_mean"]) / batch_stats["target_std"]

            if getattr(model, "input_mode", "features") == "values_sorted":
                pred_norm = model(values_sorted, batch_stats)
            else:
                features = make_model_features_from_values(
                    values_sorted=values_sorted,
                    stats=batch_stats,
                    use_coord=model.use_coord,
                    patch_size=patch_size,
                )
                pred_norm = model(features)

            loss = criterion(pred_norm, y_norm)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = y_norm.shape[0]
            total_loss += loss.item() * bs
            total_count += bs

        avg_loss = total_loss / max(total_count, 1)
        train_log.append(avg_loss)
        print(f"{model_name} | Epoch [{epoch:03d}/{num_epochs}] loss={avg_loss:.6f} time={time.time() - t0:.1f}s")

    return train_log


# ---------------------------------------------------------------------------
# HF-loss trainer: trains on 2D patches from a densely-cached single region.
# The region cache provides row-major pixel ordering, enabling extraction of
# rectangular patches with genuine spatial adjacency for Sobel / Laplacian.
# ---------------------------------------------------------------------------
def _extract_2d_patch(cache: dict, x0: int, y0: int,
                       patch_h: int, patch_w: int,
                       Hreg: int, Wreg: int, device: torch.device):
    vs_parts, tg_parts = [], []
    for dx in range(patch_h):
        ix = max(0, min(x0 + dx, Hreg - 1))
        start = ix * Wreg + max(0, y0)
        end = start + patch_w
        idx = torch.arange(start, min(end, Hreg * Wreg), device="cpu")
        vs_parts.append(cache["values_sorted"][idx])
        tg_parts.append(cache["target"][idx])
    vs = torch.cat(vs_parts, dim=0).to(device, non_blocking=True)
    tg = torch.cat(tg_parts, dim=0).to(device, non_blocking=True)
    # Pad if needed
    need = patch_h * patch_w
    if vs.shape[0] < need:
        pad = torch.zeros(need - vs.shape[0], vs.shape[1], vs.shape[2],
                          dtype=vs.dtype, device=device)
        vs = torch.cat([vs, pad], dim=0)
        pad_tg = torch.zeros(need - tg.shape[0], tg.shape[1],
                             dtype=tg.dtype, device=device)
        tg = torch.cat([tg, pad_tg], dim=0)
    tg_2d = tg.view(patch_h, patch_w).unsqueeze(0).unsqueeze(0)
    return vs, tg, tg_2d


def train_direct_model_cached_hf(
    model,
    cached_loader,
    stats,
    loss_cfg: Optional[dict] = None,
    model_name: str = "model",
    num_epochs: int = 20,
    patch_size: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    grad_clip: Optional[float] = None,
    device: str = "cuda",
    hf_patch_dim: int = 32,
    hf_steps_per_epoch: int = 128,
    hf_region_cache: Optional[dict] = None,
):
    """Train with pixel-wise MSE + optional 2D-patch HF loss.

    Uses two data sources interleaved per epoch:
      1. Pixel batches from ``cached_loader`` (diverse, multi-slice) for
         standard MSE (full pass).
      2. 2D patches extracted from ``hf_region_cache`` (spatially coherent,
         single slice) for MSE + gradient + Laplacian loss.
         hf_steps_per_epoch random patches per epoch.

    Args:
        cached_loader: DataLoader over CachedSortedVVBPDataset (diverse train cache).
        loss_cfg: dict with keys ``lambda_grad``, ``lambda_lap``.
        hf_patch_dim: side length (pixels) of square 2D HF patches.
        hf_steps_per_epoch: number of random 2D-patch training steps per epoch.
        hf_region_cache: dict with ``values_sorted`` [N,J,K], ``target`` [N,1],
            where N == Hreg * Wreg (row-major) for a single region.
            If None and HF enabled, falls back to inferring from cached_loader.

    Returns:
        train_log: list of float (avg total loss per epoch).
        comp_log: list of dict with loss_img, loss_grad, loss_lap per epoch.
    """
    if loss_cfg is None:
        loss_cfg = {}
    lambda_grad = float(loss_cfg.get("lambda_grad", 0.0))
    lambda_lap = float(loss_cfg.get("lambda_lap", 0.0))
    hf_enabled = (lambda_grad > 0 or lambda_lap > 0)

    # Determine 2D layout for HF patches.
    # If hf_region_cache is provided, use it directly (single region, row-major).
    # Otherwise fall back to inferring from the dataset.
    if hf_region_cache is not None:
        cache_dict = {
            "values_sorted": hf_region_cache["values_sorted"],
            "target": hf_region_cache["target"],
        }
        N = cache_dict["target"].shape[0]
        # Try metadata
        _meta = hf_region_cache.get("metadata", {})
        Hreg = int(_meta.get("Hreg", 0))
        Wreg = int(_meta.get("Wreg", 0))
        if Hreg <= 0 or Wreg <= 0:
            Hreg = int(math.sqrt(N))
            Wreg = N // Hreg
        print(f"[HF] Using provided hf_region_cache: {Hreg}x{Wreg} (N={N})")
    else:
        ds = cached_loader.dataset
        cache_dict = {
            "values_sorted": ds.values_sorted,
            "target": ds.target,
        }
        N = cache_dict["target"].shape[0]
        Hreg = int(math.sqrt(N))
        Wreg = N // Hreg
        if Hreg * Wreg != N:
            raise ValueError(
                f"Cannot infer square region from N={N}. "
                f"Pass hf_region_cache explicitly."
            )
        print(f"[HF] Inferred region from cached_loader: {Hreg}x{Wreg} (N={N})")

    hf_patch_dim = min(hf_patch_dim, Hreg, Wreg, 64)
    if hf_enabled and (Hreg < hf_patch_dim or Wreg < hf_patch_dim):
        print(f"[HF] Region ({Hreg}x{Wreg}) too small for {hf_patch_dim}x{hf_patch_dim}.  Disabling HF loss.")
        hf_enabled = False

    print(f"[HF] lambda_grad={lambda_grad}, lambda_lap={lambda_lap}, enabled={hf_enabled}")
    if hf_enabled:
        print(f"[HF] patch_dim={hf_patch_dim}, hf_steps/epoch={hf_steps_per_epoch}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    target_mean = stats["target_mean"].to(device)
    target_std = stats["target_std"].to(device)

    train_log = []
    comp_log = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        accum_img = 0.0
        accum_grad = 0.0
        accum_lap = 0.0
        total_count = 0

        # ---- Pass 1: pixel-wise MSE (full pass through cached loader) ----
        for vs, target, _, _ in cached_loader:
            vs = vs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            y_norm = (target - target_mean) / target_std

            if getattr(model, "input_mode", "features") == "values_sorted":
                pred_norm = model(vs, stats)
            else:
                features = make_model_features_from_values(
                    values_sorted=vs, stats=stats,
                    use_coord=getattr(model, "use_coord", False),
                    patch_size=patch_size,
                )
                pred_norm = model(features)

            loss = nn.functional.mse_loss(pred_norm, y_norm)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = y_norm.shape[0]
            accum_img += loss.item() * bs
            total_count += bs

        # ---- Pass 2: 2D-patch HF-loss steps (random patches from the region) ----
        if hf_enabled:
            pH = pW = hf_patch_dim
            margin = pH // 2
            for _ in range(hf_steps_per_epoch):
                x0 = random.randint(margin, max(margin + 1, Hreg - pH - margin))
                y0 = random.randint(margin, max(margin + 1, Wreg - pW - margin))

                vs_patch, tg_1d, tg_2d = _extract_2d_patch(
                    cache_dict, x0, y0, pH, pW, Hreg, Wreg, device,
                )

                y_norm_1d = (tg_1d - target_mean) / target_std

                if getattr(model, "input_mode", "features") == "values_sorted":
                    pred_norm_1d = model(vs_patch, stats)
                else:
                    features = make_model_features_from_values(
                        values_sorted=vs_patch, stats=stats,
                        use_coord=getattr(model, "use_coord", False),
                        patch_size=patch_size,
                    )
                    pred_norm_1d = model(features)

                # Denormalise to compute spatial HF loss on physical values
                pred_1d = pred_norm_1d * target_std + target_mean
                pred_2d = pred_1d.view(1, 1, pH, pW)

                loss_dict = compute_total_loss(
                    pred_2d, tg_2d,
                    lambda_grad=lambda_grad,
                    lambda_lap=lambda_lap,
                )

                optimizer.zero_grad()
                loss_dict["loss_total"].backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                bp = tg_1d.shape[0]
                accum_img += loss_dict["loss_img"].item() * bp
                accum_grad += loss_dict["loss_grad"].item() * bp
                accum_lap += loss_dict["loss_lap"].item() * bp
                total_count += bp

        avg_img = accum_img / max(total_count, 1)
        avg_grad = accum_grad / max(total_count, 1)
        avg_lap = accum_lap / max(total_count, 1)
        avg_total = avg_img + lambda_grad * avg_grad + lambda_lap * avg_lap

        train_log.append(float(avg_total))
        comp_log.append({
            "epoch": epoch,
            "loss_img": float(avg_img),
            "loss_grad": float(avg_grad),
            "loss_lap": float(avg_lap),
            "total_loss": float(avg_total),
            "lambda_grad": lambda_grad,
            "lambda_lap": lambda_lap,
        })

        elapsed = time.time() - t0
        part = ""
        if hf_enabled:
            part = f" grad={avg_grad:.6f} lap={avg_lap:.6f}"
        print(f"{model_name} | Epoch [{epoch:03d}/{num_epochs}] "
              f"img={avg_img:.6f}{part} total={avg_total:.6f} time={elapsed:.1f}s")

    return train_log, comp_log if hf_enabled else None


# ---------------------------------------------------------------------------
# Geometry-token trainer: on-the-fly VVBP + exact detector ΔI  →  G, R²
# ---------------------------------------------------------------------------
def train_multirate_model_geometry_token(
    model,
    train_loader,
    extractors,
    geo,
    target_stats,
    v_stats,
    G_stats,
    model_name: str = "geometry_token",
    num_epochs: int = 10,
    patch_size: int = 3,
    pixels_per_batch: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    grad_clip: Optional[float] = None,
    device: str = "cuda",
    train_region: Optional[list] = None,
):
    """Train with on-the-fly VVBP + geometry token (G, R²).

    Differs from ``train_multirate_model`` by computing exact detector-index
    offsets ΔI from pixel coords + fan-beam geometry, which the model uses to
    derive per-view G and R² tokens.

    Args:
        geo: LInFBP geometry dict used for detector-index computation.
        G_stats: dict with ``G_mean``, ``G_std`` (pre-computed statistics).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_log = []
    comp_log = []

    target_mean = target_stats["target_mean"].to(device)
    target_std = target_stats["target_std"].to(device)
    G_mean_val = G_stats["G_mean"].to(device)
    G_std_val = G_stats["G_std"].to(device)
    s_mean_val = G_stats.get("s_mean", torch.tensor(0.0))
    s_std_val = G_stats.get("s_std", torch.tensor(1.0))
    gr_mean_val = G_stats.get("gr_mean", torch.tensor(0.0))
    gr_std_val = G_stats.get("gr_std", torch.tensor(1.0))
    if hasattr(s_mean_val, "to"):
        s_mean_val = s_mean_val.to(device)
        s_std_val = s_std_val.to(device)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0
        epoch_G_vals = []
        epoch_R2_vals = []

        for sino_batch, img_batch in train_loader:
            sino_batch = sino_batch.to(device, non_blocking=True)
            img_batch = img_batch.to(device, non_blocking=True)

            V = sino_batch.shape[-2]
            extractor = extractors[V]
            vs = v_stats[V]

            batch_stats = {
                "target_mean": target_mean,
                "target_std": target_std,
                "v_mean": vs["v_mean"].to(device),
                "v_std": vs["v_std"].to(device),
                "G_mean": G_mean_val,
                "G_std": G_std_val,
                "s_mean": s_mean_val,
                "s_std": s_std_val,
                "gr_mean": gr_mean_val,
                "gr_std": gr_std_val,
            }

            # On-the-fly VVBP extraction
            vvbp = extractor(sino_batch)
            B, _, H, W, _ = vvbp.shape

            xs, ys = sample_random_coords(
                H, W, num_pixels=pixels_per_batch,
                margin=patch_size // 2, device=device,
                train_region=train_region,
            )
            values_sorted = gather_sorted_vvbp_patch(
                vvbp, xs, ys, patch_size=patch_size, mode="3x3",
            )
            N = B * pixels_per_batch
            values_sorted = values_sorted.reshape(N, values_sorted.shape[2], values_sorted.shape[3])

            # ---- Raw (unsorted) patch for G/R2 computation ----
            raw_patch = gather_raw_vvbp_patch(vvbp, xs, ys, patch_size=patch_size)
            raw_patch = raw_patch.reshape(N, raw_patch.shape[2], raw_patch.shape[3])  # [N, J, V]

            target = img_batch[:, 0, xs, ys].reshape(N, 1)

            # ---- Compute ΔI from exact detector coords ----
            xs_np = xs.cpu().numpy().astype(np.int64)
            ys_np = ys.cpu().numpy().astype(np.int64)
            deltaI_np = compute_deltaI_patch(geo, xs_np, ys_np, int(V), patch_size=patch_size)
            deltaI = torch.from_numpy(deltaI_np).to(device, non_blocking=True)  # [N, J, V]

            # ---- Compute G/R2 from RAW values + exact ΔI ----
            with torch.no_grad():
                G_vals, R2_vals = _compute_GR2_raw(raw_patch, deltaI, eps=1e-8)
                # Reorder G/R2 by centre VVBP value so they align with
                # values_sorted (whose centre row is sorted by value).
                centre_idx = raw_patch.shape[1] // 2
                raw_centre = raw_patch[:, centre_idx, :]     # [N, V] view order
                _, c_sort = torch.sort(raw_centre, dim=-1)   # value-sort index
                G_aligned = torch.gather(G_vals, dim=-1, index=c_sort)
                R2_aligned = torch.gather(R2_vals, dim=-1, index=c_sort)
            batch_stats["G"] = G_aligned.detach()    # [N, K] aligned with values_sorted
            batch_stats["R2"] = R2_aligned.detach()

            # Normalise target
            y_norm = (target - batch_stats["target_mean"]) / batch_stats["target_std"]

            if getattr(model, "input_mode", "features") == "values_sorted":
                pred_norm = model(values_sorted, batch_stats)
            else:
                features = make_model_features_from_values(
                    values_sorted=values_sorted,
                    stats=batch_stats,
                    use_coord=getattr(model, "use_coord", False),
                    patch_size=patch_size,
                )
                pred_norm = model(features)

            loss = criterion(pred_norm, y_norm)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = y_norm.shape[0]
            total_loss += float(loss.item()) * bs
            total_count += bs

            # Log G/R2 diagnostics
            epoch_G_vals.append(float(G_vals.mean().cpu()))
            epoch_R2_vals.append(float(R2_vals.mean().cpu()))

        avg_loss = total_loss / max(total_count, 1)
        train_log.append(avg_loss)
        g_avg = sum(epoch_G_vals) / max(len(epoch_G_vals), 1) if epoch_G_vals else 0.0
        r2_avg = sum(epoch_R2_vals) / max(len(epoch_R2_vals), 1) if epoch_R2_vals else 0.0
        r2_med = float(np.median(epoch_R2_vals)) if epoch_R2_vals else 0.0

        # Gate epoch stats
        gate_info = {}
        if hasattr(model, "get_gate_epoch_stats"):
            gate_info = model.get_gate_epoch_stats()

        comp_log.append({
            "epoch": epoch,
            "loss": float(avg_loss),
            "G_mean": g_avg,
            "R2_mean": r2_avg,
            "R2_median": r2_med,
            "G_std_global": float(G_std_val.cpu()),
            **gate_info,
        })

        gate_str = ""
        if gate_info:
            gate_str = (f" s_batch={gate_info.get('s_batch_mean',0):.4e}"
                        f" gate=[{gate_info.get('gate_mean',0):.3f},"
                        f"{gate_info.get('gate_min',0):.3f},"
                        f"{gate_info.get('gate_max',0):.3f}]"
                        f" a={gate_info.get('gate_a',0):.3f}"
                        f" b={gate_info.get('gate_b',0):.3f}")
        print(f"{model_name} | Epoch [{epoch:03d}/{num_epochs}] "
              f"loss={avg_loss:.6f} G={g_avg:.4e} R2={r2_avg:.4f}"
              f"{gate_str} time={time.time() - t0:.1f}s")

    return train_log, comp_log


# Re-export for convenience
from src.geometry.fanbeam import compute_deltaI_patch
