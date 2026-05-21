# CT VVBP Local-Rank Reconstruction

This repository contains the cleaned VVBP-based sparse-view CT reconstruction experiments.

## Current focus

The active comparison is:

1. `center base`
2. `local-rank center integral closed` parameter-free baseline
3. one selected learnable model, usually `local rank center integral mlp, 10 epochs`

## Project structure

```text
configs/                      Experiment JSON files
scripts/                      Entry-point scripts
src/data/                     DICOM loading, VVBP patch extraction, cache building
src/evaluation/               Metrics, region prediction, visualization
src/experiments/              Config-driven experiment runner
src/geometry/                 Fan-beam geometry, fixed FBP, VVBP extractor
src/models/                   Active model implementations only
src/training/                 Cached training loop
src/utils/                    Config and seed utilities
```

## Main command

```bash
python scripts/run_from_config.py --config configs/patch_size_3_local_rank_center_integral_mlp.json
```

## Active model configs

- `configs/patch_size_3_local_rank_center_integral_mlp.json`
- `configs/patch_size_3_local_rank_center_closed_mlp.json`
- `configs/patch_size_3_local_rank_center_mlp.json`
- `configs/patch_size_3_merged_local_sorted.json`

The sinogram upsampling baseline is kept as a separate baseline workflow:

```bash
python scripts/run_sino_upsample_baseline.py --config configs/sino_60_to_240_baseline.json
```

## Notes

Large generated files are intentionally not included in the cleaned archive:

- `.git/`
- `outputs/`
- `cached_direct_vvbp_results/`
- `Results/`
- `full_1mm/`

Regenerate caches and outputs by running the relevant config.
