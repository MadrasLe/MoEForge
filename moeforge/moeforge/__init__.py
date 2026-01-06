"""
🔥 MoEForge - Optimized Mixture of Experts Training Library

Fast and memory-efficient MoE training with:
- Custom CUDA kernels (MGRMSNORM)
- Flash Attention 2 support  
- Token shuffling for efficient expert routing
- Shared expert architecture (DeepSeek-style)
- Load balancing with Z-loss

Usage:
    from moeforge import MoEConfig, MoEModel, MoETrainer
    
    config = MoEConfig(
        hidden_dim=1024,
        num_experts=8,
        top_k=2,
        shared_expert=True,
    )
    
    model = MoEModel(config)
    trainer = MoETrainer(model)
    trainer.train(dataset)

Author: Gabriel Yogi (MadrasLe)
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Gabriel Yogi"

from .config import MoEConfig
from .models.transformer import MoEModel
from .layers.moe import MoELayer
from .layers.attention import OptimizedAttention
from .layers.expert import SwiGLUExpert
from .layers.normalization import get_norm_layer

# Kernels (optional - may not be available)
try:
    from .kernels import fused_swiglu, get_available_kernels, TRITON_AVAILABLE
except ImportError:
    TRITON_AVAILABLE = False
    fused_swiglu = None
    get_available_kernels = lambda: {"triton": False}

__all__ = [
    "MoEConfig",
    "MoEModel", 
    "MoELayer",
    "OptimizedAttention",
    "SwiGLUExpert",
    "get_norm_layer",
    "fused_swiglu",
    "get_available_kernels",
    "TRITON_AVAILABLE",
]
