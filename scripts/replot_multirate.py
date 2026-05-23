"""Re-plot multirate comparison from saved results (bypasses scipy import chain)."""
import sys, os, importlib.util

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

# Load visualization module directly, bypassing src.evaluation.__init__.py
viz_path = os.path.join(PROJECT_ROOT, "src", "evaluation", "visualization.py")
spec = importlib.util.spec_from_file_location("visualization", viz_path)
viz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viz)
plot_comparison_grid = viz.plot_comparison_grid

RESULTS_PATH = "outputs/multirate_vvbp_local_rank/multirate_all_results.pt"
SAVE_PATH = "outputs/multirate_vvbp_local_rank/multirate_comparison.png"

d = torch.load(RESULTS_PATH, map_location="cpu", weights_only=False)

def to_np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

target = to_np(d["target"])
sparse_views = d["sparse_views"]
x0, x1, y0, y1 = d["region"]
model_names = d["model_names"]

# FBP preds are full 512x512, crop to region
fbp_preds = d["fbp_preds"]
fbp_metrics = d["fbp_metrics"]
fbp_region = {V: to_np(fbp_preds[V][x0:x1, y0:y1]) for V in sparse_views}

# Baseline preds (already cropped)
baseline_preds = d["baseline_preds"]
baseline_metrics = d["baseline_metrics"]

# Model preds (already cropped)
model_preds = d["model_preds"]
model_metrics = d["model_metrics"]

preds_by_method = {}
psnr_by_method = {}

# FBP
preds_by_method["FBP"] = {V: fbp_region[V] for V in sparse_views}
psnr_by_method["FBP"] = {V: fbp_metrics[V]["PSNR"] for V in sparse_views}

# Local-rank closed
preds_by_method["Local-rank closed"] = {
    V: to_np(baseline_preds["Local-rank closed"][V]) for V in sparse_views
}
psnr_by_method["Local-rank closed"] = {
    V: baseline_metrics["Local-rank closed"][V]["PSNR"] for V in sparse_views
}

# Learned models
col_labels = ["FBP", "Local-rank closed"]
for model_name in model_names:
    short = model_name.replace(", ", "_").replace(" ", "_").replace("_10_epochs", "")
    preds_by_method[short] = {
        V: to_np(model_preds[model_name][V]) for V in sparse_views
    }
    psnr_by_method[short] = {
        V: model_metrics[model_name]["PSNR"][V] for V in sparse_views
    }
    col_labels.append(short)

plot_comparison_grid(
    target=target,
    preds_by_method=preds_by_method,
    psnr_by_method=psnr_by_method,
    col_labels=col_labels,
    sparse_views=sparse_views,
    save_path=SAVE_PATH,
    show=False,
)
print(f"Saved: {SAVE_PATH}")
