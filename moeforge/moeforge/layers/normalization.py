"""
Normalization layers with custom kernel support
"""

import torch
import torch.nn as nn
from typing import Literal

# Try to import optimized implementations
LIGER_AVAILABLE = False
MGRMSNORM_AVAILABLE = False

try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    LIGER_AVAILABLE = True
except ImportError:
    pass

try:
    import rmsnorm_cuda_ops
    MGRMSNORM_AVAILABLE = True
except ImportError:
    pass


class ManualRMSNorm(nn.Module):
    """
    RMSNorm implementation in pure PyTorch (fallback).
    
    RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
    
    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"


class MGRMSNormFunction(torch.autograd.Function):
    """
    Custom autograd function for MGRMSNORM CUDA kernel.
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5):
        # Ensure float32 for CUDA kernel
        input_f32 = input.float().contiguous()
        weight_f32 = weight.float().contiguous()
        
        # Reshape to 2D
        orig_shape = input_f32.shape
        input_2d = input_f32.view(-1, orig_shape[-1])
        
        output, inv_rms = rmsnorm_cuda_ops.rmsnorm_forward(input_2d, weight_f32, eps)
        
        ctx.save_for_backward(input_2d, weight_f32, inv_rms)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = input.dtype
        
        return output.view(orig_shape).to(input.dtype)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_2d, weight, inv_rms = ctx.saved_tensors
        
        grad_output_f32 = grad_output.float().contiguous()
        grad_2d = grad_output_f32.view(-1, ctx.orig_shape[-1])
        
        grad_input, grad_weight = rmsnorm_cuda_ops.rmsnorm_backward(
            grad_2d, input_2d, weight, inv_rms
        )
        
        return grad_input.view(ctx.orig_shape).to(ctx.orig_dtype), grad_weight.to(ctx.orig_dtype), None


class MGRMSNorm(nn.Module):
    """
    RMSNorm using custom CUDA kernel (MGRMSNORM).
    
    Features:
        - Vectorized memory access (float4)
        - Warp-level reductions
        - Efficient backward pass
    
    Requires: pip install rmsnorm_cuda (from MadrasLe/MGRrmsnorm)
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        if not MGRMSNORM_AVAILABLE:
            raise ImportError(
                "MGRMSNORM not available. Install from: "
                "pip install git+https://github.com/MadrasLe/MGRrmsnorm.git"
            )
        
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return MGRMSNormFunction.apply(x, self.weight, self.eps)
    
    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"


def get_norm_layer(
    hidden_size: int,
    norm_type: Literal["auto", "mgrmsnorm", "liger", "pytorch"] = "auto",
    eps: float = 1e-5,
) -> nn.Module:
    """
    Get the best available normalization layer.
    
    Args:
        hidden_size: Hidden dimension
        norm_type: Type of normalization
            - "auto": Use best available (MGRMSNORM > LigerRMSNorm > PyTorch)
            - "mgrmsnorm": Use custom CUDA kernel
            - "liger": Use Liger Kernel
            - "pytorch": Use pure PyTorch
        eps: Epsilon for numerical stability
        
    Returns:
        Normalization layer module
    """
    if norm_type == "mgrmsnorm":
        if MGRMSNORM_AVAILABLE:
            return MGRMSNorm(hidden_size, eps)
        raise ImportError("MGRMSNORM requested but not available")
    
    elif norm_type == "liger":
        if LIGER_AVAILABLE:
            return LigerRMSNorm(hidden_size, eps)
        raise ImportError("LigerRMSNorm requested but not available")
    
    elif norm_type == "pytorch":
        return ManualRMSNorm(hidden_size, eps)
    
    elif norm_type == "auto":
        # Prefer MGRMSNORM > Liger > PyTorch
        if MGRMSNORM_AVAILABLE:
            return MGRMSNorm(hidden_size, eps)
        elif LIGER_AVAILABLE:
            return LigerRMSNorm(hidden_size, eps)
        else:
            return ManualRMSNorm(hidden_size, eps)
    
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


def get_available_norms() -> dict:
    """Get dictionary of available normalization implementations."""
    return {
        "mgrmsnorm": MGRMSNORM_AVAILABLE,
        "liger": LIGER_AVAILABLE,
        "pytorch": True,
    }
