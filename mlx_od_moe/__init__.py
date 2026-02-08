"""mlx-od-moe: On-Demand Mixture of Experts for Apple Silicon"""

__version__ = "0.1.0"

from .expert_store import UnifiedMemoryExpertStore
from .shadow_model import ExpertPredictor, ShadowRunner
from .od_moe_layer import ODMoELayer
from .model import (
    KimiODMoEModel, KimiODMoEConfig,
    KVCache, MLACache,
    Attention, MLAttention,
    DenseMLP, TransformerBlock,
)

__all__ = [
    "UnifiedMemoryExpertStore",
    "ExpertPredictor",
    "ShadowRunner",
    "ODMoELayer",
    "KimiODMoEModel",
    "KimiODMoEConfig",
    "KVCache",
    "MLACache",
    "Attention",
    "MLAttention",
    "DenseMLP",
    "TransformerBlock",
]
