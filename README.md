# CT VVBP Multi-rate Reconstruction (ASTRA CTO-adapted)

Sparse-view CT reconstruction experiments using VVBP (volume voxel back projection)-based local-rank methods and CTO-adapted deep learning with ASTRA-based strict projection/backprojection.

## Current focus

The active experiments span two tracks:

**Track 1 — Local-rank VVBP reconstruction** (patch-based):
- `center base`
- `local-rank center integral closed` parameter-free baseline
- selected learnable model (e.g. `local rank center integral mlp, 10 epochs`)

**Track 2 — CTO-adapted multirate reconstruction** (ASTRA-based):
- CTO-adapted model with strict sparse-matrix projection/backprojection
- UDNO and dynamic-UDNO variants
- Multirate evaluation at 8, 24, 48, 80 sparse views

## Project structure

```text
configs/                      Experiment JSON files
scripts/                      Entry-point scripts
src/data/                     DICOM loading, VVBP patch extraction, cache building
src/evaluation/               Metrics, region prediction, visualization
src/experiments/              Config-driven experiment runner
src/geometry/                 Fan-beam geometry, FBP, VVBP extractor, ASTRA projectors
src/models/                   Model implementations (local-rank + CTO + UDNO)
src/training/                 Cached training loop
src/utils/                    Config and seed utilities
```

`cache/` 用来放缓存文件（可以重新生成），`configs/` 用来放 JSON 配置文件，`outputs/` 用来放实验结果（训练输出、指标、图片等）。

## Commands

Local-rank VVBP experiments:

```bash
python scripts/run_local_rank_center_integral.py --config configs/patch_size_3_local_rank_center_integral_mlp.json
```

CTO-adapted multirate experiments:

```bash
python scripts/run_cto_multirate.py --config configs/cto_multirate.json
```

Multirate VVBP baseline:

```bash
python scripts/run_multirate_vvbp.py --config configs/multirate_selected_models.json
```

FBP baseline:

```bash
python scripts/run_fbp_baseline.py --config configs/multirate_fbp_baseline.json
```

Sinogram upsampling baseline:

```bash
python scripts/run_sino_upsample_baseline.py --config configs/sino_60_to_240_baseline.json
```

## Active configs

- `configs/patch_size_3_local_rank_center_integral_mlp.json`
- `configs/patch_size_3_local_rank_center_integral_mlp_v120.json`
- `configs/patch_size_3_local_rank_center_closed_mlp.json`
- `configs/patch_size_3_local_rank_center_mlp.json`
- `configs/patch_size_3_merged_local_sorted.json`
- `configs/cto_multirate.json`
- `configs/multirate_selected_models.json`
- `configs/multirate_fbp_baseline.json`
- `configs/sino_60_to_240_baseline.json`

## Notes

Large generated files are intentionally not included in the cleaned archive:

- `.git/`
- `outputs/`
- `cache/vvbp_patches/`
- `Results/`
- `full_1mm/`

Regenerate caches and outputs by running the relevant config.
