from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from src.utils.config import load_run_config, save_run_config, RunConfig
from src.experiments.project_setup import prepare_project_context
from src.experiments.selected_models import run_selected_models_experiment


def run_experiment_from_config(config: Union[str, os.PathLike, RunConfig]):
    """Run selected-model experiment from a JSON/YAML RunConfig.

    Stage 4 keeps the experiment type focused on selected-model fair comparison.
    Later stages can add coord-ablation, patch-size ablation, and multi-region modes.
    """
    run_cfg = load_run_config(config) if not isinstance(config, RunConfig) else config
    exp_cfg = run_cfg.experiment
    print("[CONFIG] train_region:", getattr(exp_cfg, "train_region", None))

    # If save_dir/cache_dir are not specified in the config, create a named output folder.
    if run_cfg.save_dir is None:
        exp_cfg.save_dir = os.path.join("outputs", run_cfg.experiment_name)
    if run_cfg.cache_dir is not None:
        exp_cfg.cache_dir = run_cfg.cache_dir
    exp_cfg.region = tuple(run_cfg.region)
    exp_cfg.ensure_dirs()

    used_cfg_path = Path(exp_cfg.save_dir) / "config_used.json"
    save_run_config(run_cfg, used_cfg_path)
    print("Saved config copy:", used_cfg_path)

    ctx = prepare_project_context(
        dicom_folder=run_cfg.dicom_folder,
        results_folder=run_cfg.results_folder,
        save_dir=exp_cfg.save_dir,
        cache_dir=exp_cfg.cache_dir,
        cfg=exp_cfg,
        seed=exp_cfg.seed,
    )

    result = run_selected_models_experiment(
        dataset=ctx["dataset_sparse"],
        train_loader=ctx["train_loader_sparse"],
        extractor=ctx["extractor"],
        test_indices=ctx["test_indices"],
        cfg=ctx["cfg"],
        device=ctx["device"],
        model_names=run_cfg.model_names,
        region_name=run_cfg.region_name,
    )
    return result
