"""
MoEForge Kernels Package

Optimized GPU kernels for MoE training.
"""

from .triton_kernels import (
    fused_swiglu,
    get_available_kernels,
    TRITON_AVAILABLE,
)

# Re-export from normalization for convenience
from ..layers.normalization import (
    MGRMSNORM_AVAILABLE,
    LIGER_AVAILABLE,
)

__all__ = [
    "fused_swiglu",
    "get_available_kernels",
    "TRITON_AVAILABLE",
    "MGRMSNORM_AVAILABLE", 
    "LIGER_AVAILABLE",
]
