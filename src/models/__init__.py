"""Model architectures for QuantumMeta-Sat."""
from .baselines import LSEstimator, WindowedLSEstimator
from .mlp import ClassicalMLP
from .cnn import ClassicalCNN
from .hybrid_model import HybridVQC
from .vqc import HybridVQC as VQC

__all__ = [
    "LSEstimator",
    "WindowedLSEstimator",
    "ClassicalMLP",
    "ClassicalCNN",
    "HybridVQC",
    "VQC",
]
