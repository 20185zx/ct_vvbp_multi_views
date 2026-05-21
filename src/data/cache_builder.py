import os
import time
import math
import torch

from .local_vvbp import extract_cache_items_from_batch, gather_sorted_vvbp_patch


def _sparse_views_repr(cfg, views=None):
    """String representation of sparse_views for cache filenames."""
    if views is not None:
        return str(views)
    sv = cfg.sparse_views
    if isinstance(sv, list):
        return "_".join(str(v) for v in sv)
    return str(sv)


def train_cache_path(cfg, views=None):
    nb = "ALL" if cfg.cache_num_batches is None else str(cfg.cache_num_batches)
    v_str = _sparse_views_repr(cfg, views)
    return os.path.join(cfg.cache_dir, f"train_cache_3x3_ps{cfg.cache_pixels_per_slice}_nb{nb}_patch{cfg.patch_size}_v{v_str}.pt")


def build_or_load_train_cache(loader, extractor, cfg, device="cuda", views=None):
    path = train_cache_path(cfg, views=views)
    if os.path.exists(path) and not cfg.rebuild_train_cache:
        print(f"[CACHE] Load train cache: {path}")
        cache = torch.load(path, map_location="cpu", weights_only=False)
        for k, v in cache.items():
            if torch.is_tensor(v):
                print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        return cache

    print(f"[CACHE] Build train cache and save to: {path}")
    t0 = time.time()
    values_list, target_list, center_list, local3x3_list = [], [], [], []
    for bi, (sino_batch, img_batch) in enumerate(loader):
        if cfg.cache_num_batches is not None and bi >= cfg.cache_num_batches:
            break
        print(f"  extracting batch {bi + 1} ...")

        train_region = getattr(cfg, "train_region", None)
        item = extract_cache_items_from_batch(
            sino_batch=sino_batch,
            img_batch=img_batch,
            extractor=extractor,
            pixels_per_slice=cfg.cache_pixels_per_slice,
            patch_size=cfg.patch_size,
            device=device,
            train_region=train_region,
        )
        values_list.append(item["values_sorted"])
        target_list.append(item["target"])
        center_list.append(item["center_base"])
        local3x3_list.append(item["local_3x3_base"])

    cache = {
        "values_sorted": torch.cat(values_list, dim=0).contiguous(),
        "target": torch.cat(target_list, dim=0).contiguous(),
        "center_base": torch.cat(center_list, dim=0).contiguous(),
        "local_3x3_base": torch.cat(local3x3_list, dim=0).contiguous(),
        "metadata": {
            "patch_size": cfg.patch_size,
            "cache_pixels_per_slice": cfg.cache_pixels_per_slice,
            "cache_num_batches": cfg.cache_num_batches,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(cache, path)
    print(f"[CACHE] Saved train cache: {path}")
    print(f"[CACHE] Total samples: {cache['target'].shape[0]}")
    print(f"[CACHE] Time: {(time.time() - t0) / 60:.2f} min")
    print("[CACHE] train_region:", getattr(cfg, "train_region", None))
    return cache


def estimate_stats_from_train_cache(cache, device="cuda"):
    values = cache["values_sorted"]
    target = cache["target"]
    stats = {
        "v_mean": values.mean().to(device),
        "v_std": (values.std() + 1e-8).to(device),
        "target_mean": target.mean().to(device),
        "target_std": (target.std() + 1e-8).to(device),
    }
    print("Normalization statistics from cache:")
    for k, v in stats.items():
        print(f"  {k}: {float(v):.8f}")
    return stats


def region_to_tag(region):
    return f"x{region[0]}-{region[1]}_y{region[2]}-{region[3]}"


def eval_cache_path(cfg, global_idx, region_name, region, views=None):
    tag = region_to_tag(region)
    v_str = _sparse_views_repr(cfg, views)
    return os.path.join(cfg.cache_dir, f"eval_cache_g{global_idx}_{region_name}_{tag}_patch{cfg.patch_size}_v{v_str}.pt")


@torch.no_grad()
def build_or_load_region_cache(dataset, extractor, global_idx, region_name, region, cfg, device="cuda", views=None):
    path = eval_cache_path(cfg, global_idx, region_name, region, views=views)
    if os.path.exists(path) and not cfg.rebuild_eval_cache:
        return torch.load(path, map_location="cpu", weights_only=False)

    print(f"[CACHE] Build eval cache: global_idx={global_idx}, region={region_name}, {region}")
    sino_tensor, img_target_tensor = dataset[global_idx]
    sino_batch = sino_tensor.unsqueeze(0).to(device)
    img_batch = img_target_tensor.unsqueeze(0).to(device)
    extractor.eval()
    vvbp = extractor(sino_batch)

    x_start, x_end, y_start, y_end = region
    coords = [(x, y) for x in range(x_start, x_end) for y in range(y_start, y_end)]
    values_list, target_list, center_list, local3x3_list = [], [], [], []
    for start in range(0, len(coords), cfg.chunk_size_eval):
        chunk = coords[start : start + cfg.chunk_size_eval]
        xs = torch.tensor([c[0] for c in chunk], dtype=torch.long, device=device)
        ys = torch.tensor([c[1] for c in chunk], dtype=torch.long, device=device)
        P = xs.numel()
        values_sorted = gather_sorted_vvbp_patch(vvbp, xs, ys, patch_size=cfg.patch_size, mode="3x3")
        target = img_batch[:, 0, xs, ys].reshape(P, 1)
        center_vals = vvbp[:, 0, xs, ys, :]
        center_base = center_vals.mean(dim=-1).reshape(P, 1) * math.pi
        local_3x3_base = values_sorted.mean(dim=(2, 3)).reshape(P, 1) * math.pi
        values_list.append(values_sorted.reshape(P, values_sorted.shape[2], values_sorted.shape[3]).cpu().float())
        target_list.append(target.cpu().float())
        center_list.append(center_base.cpu().float())
        local3x3_list.append(local_3x3_base.cpu().float())

    cache = {
        "values_sorted": torch.cat(values_list, dim=0).contiguous(),
        "target": torch.cat(target_list, dim=0).contiguous(),
        "center_base": torch.cat(center_list, dim=0).contiguous(),
        "local_3x3_base": torch.cat(local3x3_list, dim=0).contiguous(),
        "metadata": {
            "global_idx": global_idx,
            "region_name": region_name,
            "region": region,
            "Hreg": x_end - x_start,
            "Wreg": y_end - y_start,
            "patch_size": cfg.patch_size,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(cache, path)
    print("[CACHE] Saved eval cache:", path)
    del vvbp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache
