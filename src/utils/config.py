from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import json
import os


@dataclass
class ExperimentConfig:
    full_views: int = 720
    sparse_views: list = None          # [9, 18, 36, 72] for multi-rate; use first for eval
    image_size: int = 256
    n_detec: int = 672
    d_detec: float = 1.0
    d_voxel: float = 1.0
    DSO: float = 595.0
    DOD: float = 480.0
    train_batch_size: int = 1
    patch_size: int = 3
    cache_num_batches: Optional[int] = None
    cache_pixels_per_slice: int = 8192
    cached_batch_size: int = 8192
    num_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: Optional[float] = None
    region: Tuple[int, int, int, int] = (64, 192, 64, 192)
    chunk_size_eval: int = 8192
    rebuild_train_cache: bool = False
    rebuild_eval_cache: bool = False
    save_dir: str = "outputs"
    cache_dir: str = "cache/vvbp_patches"
    seed: int = 42
    train_region: Optional[list] = None
    eval_only: bool = False
    eval_every_epoch: bool = True
    checkpoint_dir: str = ""

    # CTO-adapted baseline parameters
    cto_sino_hidden: int = 32
    cto_image_hidden: int = 32
    cto_cascades: int = 3
    cto_sino_residual_scale: float = 0.05
    cto_dc_step: float = 1e-6
    cto_image_reg_scale: float = 0.05

    def __post_init__(self):
        if self.sparse_views is None:
            self.sparse_views = [9, 18, 36, 72]
        # Normalize legacy cache_dir paths to the consolidated cache/ directory.
        if isinstance(self.cache_dir, str) and self.cache_dir.startswith("cached_direct_vvbp_results/"):
            self.cache_dir = "cache/vvbp_patches"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def ensure_dirs(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)


@dataclass
class RunConfig:
    """Top-level experiment config loaded from JSON/YAML.

    The nested `experiment` section maps to ExperimentConfig.
    Other fields control project paths, selected models, and evaluation regions.
    """
    experiment_name: str = "selected_models_10epoch"
    experiment: ExperimentConfig = None
    model_names: List[str] = None
    region_name: str = "single_center"
    region: Tuple[int, int, int, int] = (64, 192, 64, 192)
    test_slice_mode: str = "first"  # currently: first
    dicom_folder: Optional[str] = None
    results_folder: Optional[str] = None
    save_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    archive_stamp: str = "2026-05-17_17-04"

    def __post_init__(self):
        if self.cache_dir is not None and self.cache_dir.startswith("cached_direct_vvbp_results/"):
            self.cache_dir = "cache/vvbp_patches"
        if isinstance(self.results_folder, str) and os.path.basename(self.results_folder.replace("\\", "/")) == "Results_analysis":
            self.results_folder = "cache/fanbeam_geometry"
        if self.experiment is None:
            self.experiment = ExperimentConfig()
        if self.model_names is None:
            self.model_names = ["local rank center integral mlp, 10 epochs"]
        # Keep ExperimentConfig.region synchronized with top-level region.
        self.experiment.region = tuple(self.region)
        if self.save_dir is not None:
            self.experiment.save_dir = self.save_dir
        if self.cache_dir is not None:
            self.experiment.cache_dir = self.cache_dir


def _read_config_file(path: str | os.PathLike) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in [".json"]:
        return json.loads(text)
    if suffix in [".yaml", ".yml"]:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config files. Either install pyyaml or use JSON config."
            ) from exc
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config file suffix: {suffix}. Use .json, .yaml, or .yml")


def _filter_dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


def load_run_config(path: str | os.PathLike) -> RunConfig:
    """Load JSON/YAML into a RunConfig object.

    Example schema:
    {
      "experiment_name": "selected_models_10epoch",
      "experiment": {"num_epochs": 10, "patch_size": 3},
      "model_names": ["local_rank_center_integral_mlp"],
      "region": [192, 320, 192, 320]
    }
    """
    raw = _read_config_file(path)
    exp_raw = raw.get("experiment", {}) or {}
    exp = ExperimentConfig(**_filter_dataclass_kwargs(ExperimentConfig, exp_raw))

    run_kwargs = _filter_dataclass_kwargs(RunConfig, raw)
    run_kwargs["experiment"] = exp
    if "region" in run_kwargs and run_kwargs["region"] is not None:
        run_kwargs["region"] = tuple(run_kwargs["region"])
    cfg = RunConfig(**run_kwargs)
    return cfg


def save_run_config(cfg: RunConfig, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg)
    # tuples are JSON-serializable as lists through json.dump
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def infer_project_paths(archive_stamp: str = "2026-05-17_17-04") -> Dict[str, Path]:
    """Infer paths used by the notebook archive layout.

    This keeps the old notebook behavior but makes it explicit and reusable.
    If your directory layout differs, pass paths manually in your experiment config.
    """
    try:
        archive_dir = Path(__file__).resolve().parent
    except NameError:
        archive_dir = Path.cwd().resolve()

    if archive_dir.name != archive_stamp and (archive_dir / "test codes" / archive_stamp).exists():
        archive_dir = archive_dir / "test codes" / archive_stamp
    elif archive_dir.name == "test codes":
        archive_dir = archive_dir / archive_stamp

    project_root = archive_dir.parents[1] if len(archive_dir.parents) >= 2 else archive_dir
    return {
        "archive_dir": archive_dir,
        "project_root": project_root,
        "dicom_folder": project_root / "full_1mm" / "L067" / "full_1mm",
        "results_folder": project_root / "Results",
        "shared_cache_dir": project_root / "cache" / "vvbp_patches",
        "output_dir": archive_dir / "outputs",
    }
