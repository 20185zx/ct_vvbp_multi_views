"""Model registry for the current VVBP experiments."""

from __future__ import annotations

from .local_rank_center_closed_mlp import LocalRankCenterClosedMLPNet
from .local_rank_center_integral_mlp import LocalRankCenterIntegralMLPNet
from .exact_detector_geometry_local_rank_integral_mlp import ExactDetectorGeometryLocalRankIntegralMLPNet
from .exact_detector_geometry_gated_local_rank_integral_mlp import ExactDetectorGeometryGatedLocalRankIntegralMLPNet
from .exact_detector_geometry_gated_residual_local_rank_integral_mlp import ExactDetectorGeometryGatedResidualLocalRankIntegralMLPNet
from .local_rank_center_mlp import LocalRankCenterMLPNet
from .merged_local_sorted import MergedLocalSortedVVBPNet


MODEL_FACTORIES = {
    "local rank center mlp, 10 epochs": LocalRankCenterMLPNet,
    "local_rank_center_mlp": LocalRankCenterMLPNet,
    "local rank center integral mlp, 10 epochs": LocalRankCenterIntegralMLPNet,
    "local_rank_center_integral_mlp": LocalRankCenterIntegralMLPNet,
    "local rank center closed mlp, 10 epochs": LocalRankCenterClosedMLPNet,
    "local_rank_center_closed_mlp": LocalRankCenterClosedMLPNet,
    "merged local sorted, 10 epochs": MergedLocalSortedVVBPNet,
    "merged_local_sorted": MergedLocalSortedVVBPNet,
    "exact detector geometry local rank center integral mlp, 10 epochs": ExactDetectorGeometryLocalRankIntegralMLPNet,
    "exact_detector_geometry_local_rank_integral_mlp": ExactDetectorGeometryLocalRankIntegralMLPNet,
    "exact detector geometry gated local rank center integral mlp, 10 epochs": ExactDetectorGeometryGatedLocalRankIntegralMLPNet,
    "exact_detector_geometry_gated_local_rank_integral_mlp": ExactDetectorGeometryGatedLocalRankIntegralMLPNet,
    "exact detector geometry gated residual local rank center integral mlp, 10 epochs": ExactDetectorGeometryGatedResidualLocalRankIntegralMLPNet,
    "exact_detector_geometry_gated_residual_local_rank_integral_mlp": ExactDetectorGeometryGatedResidualLocalRankIntegralMLPNet,
}


def build_model(model_name: str, **kwargs):
    """Build a model from the cleaned registry."""
    if model_name not in MODEL_FACTORIES:
        available = ", ".join(MODEL_FACTORIES)
        raise ValueError(f"Unknown model name: {model_name}. Available models: {available}")
    return MODEL_FACTORIES[model_name](**kwargs)
