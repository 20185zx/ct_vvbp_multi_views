import os
import torch

from src.utils.seed import set_seed
from src.utils.config import ExperimentConfig, infer_project_paths
from src.geometry import load_or_generate_geo, FanBeamVVBPExtractor
from src.data import build_dataloaders, build_multirate_dataloaders


def _prepare_base_context(cfg, device, seed, dicom_folder, results_folder, save_dir, cache_dir):
    """Common base setup shared by single-V and multi-rate context preparation."""
    set_seed(seed)
    paths = infer_project_paths()
    cfg = cfg or ExperimentConfig()
    if save_dir is not None:
        cfg.save_dir = str(save_dir)
    else:
        cfg.save_dir = str(paths["output_dir"])
    if cache_dir is not None:
        cfg.cache_dir = str(cache_dir)
    else:
        cfg.cache_dir = str(paths["shared_cache_dir"])
    cfg.ensure_dirs()

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dicom_folder = str(dicom_folder or paths["dicom_folder"])
    results_folder = str(results_folder or paths["results_folder"])
    os.makedirs(results_folder, exist_ok=True)

    print("Device:", device)
    print("DICOM folder:", dicom_folder)
    print("Results folder:", results_folder)
    print("Save dir:", cfg.save_dir)
    print("Cache dir:", cfg.cache_dir)

    return {
        "cfg": cfg,
        "device": device,
        "dicom_folder": dicom_folder,
        "results_folder": results_folder,
        "paths": paths,
    }


def prepare_project_context(
    dicom_folder=None,
    results_folder=None,
    save_dir=None,
    cache_dir=None,
    cfg=None,
    device=None,
    seed=42,
):
    """Initialize the CT-specific context previously kept in notebook cells.

    Returns a dict with cfg, device, geo_sparse, dataset_sparse, train_loader_sparse,
    train_indices, test_indices, extractor.
    """
    base = _prepare_base_context(cfg, device, seed, dicom_folder, results_folder, save_dir, cache_dir)
    cfg = base["cfg"]

    # sparse_views can be a list (multi-rate) or int (single-V, legacy).
    sv = cfg.sparse_views
    geo_views = sv[0] if isinstance(sv, list) else sv

    geo_sparse = load_or_generate_geo(
        geo_views, base["results_folder"], base["device"],
        image_size=cfg.image_size,
        n_detec=cfg.n_detec,
        d_detec=cfg.d_detec,
        d_voxel=cfg.d_voxel,
        DSO=cfg.DSO,
        DOD=cfg.DOD,
    )
    dataset_sparse, train_indices, test_indices, train_loader_sparse = build_dataloaders(
        base["dicom_folder"], cfg
    )
    print("Train slices:", len(train_indices))
    print("Test slices:", len(test_indices))

    extractor = FanBeamVVBPExtractor(geo_sparse).to(base["device"])
    extractor.eval()

    return {
        "cfg": cfg,
        "device": base["device"],
        "geo_sparse": geo_sparse,
        "dataset_sparse": dataset_sparse,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_loader_sparse": train_loader_sparse,
        "extractor": extractor,
        "paths": base["paths"],
    }


def prepare_multirate_context(
    dicom_folder=None,
    results_folder=None,
    save_dir=None,
    cache_dir=None,
    cfg=None,
    device=None,
    seed=42,
):
    """Initialize multi-rate context: geo/extractor per V + multi-rate dataloader.

    Returns:
        cfg, device, geo_dict (V→geo), extractors (V→extractor),
        train_loader (random-V per batch), eval_dataset (full 720-view),
        train_indices, test_indices, paths.
    """
    base = _prepare_base_context(cfg, device, seed, dicom_folder, results_folder, save_dir, cache_dir)
    cfg = base["cfg"]

    sparse_views = cfg.sparse_views
    if not isinstance(sparse_views, list):
        raise ValueError("sparse_views must be a list for multi-rate training")

    print("Sparse views:", sparse_views)

    # --- Build geo + extractor for each V ---
    geo_dict = {}
    extractors = {}
    for V in sparse_views:
        print(f"\n[GEO] Building geo for V={V} ...")
        geo = load_or_generate_geo(
            V, base["results_folder"], base["device"],
            image_size=cfg.image_size,
            n_detec=cfg.n_detec,
            d_detec=cfg.d_detec,
            d_voxel=cfg.d_voxel,
            DSO=cfg.DSO,
            DOD=cfg.DOD,
        )
        geo_dict[V] = geo
        extractors[V] = FanBeamVVBPExtractor(geo).to(base["device"])
        extractors[V].eval()

    # --- Multi-rate dataloaders ---
    train_dataset, eval_dataset, train_indices, test_indices, train_loader = \
        build_multirate_dataloaders(base["dicom_folder"], cfg)
    print("Train slices:", len(train_indices))
    print("Test slices:", len(test_indices))

    return {
        "cfg": cfg,
        "device": base["device"],
        "geo_dict": geo_dict,
        "extractors": extractors,
        "train_loader": train_loader,
        "eval_dataset": eval_dataset,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "paths": base["paths"],
    }
