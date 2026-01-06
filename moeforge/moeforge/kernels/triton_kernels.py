"""
Triton Kernels for MoEForge

Optimized GPU kernels using Triton for:
    - Fused SwiGLU activation
    - Fused RMSNorm (alternative to CUDA kernel)
"""

import torch

# Check Triton availability
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


if TRITON_AVAILABLE:
    # =========================================================================
    # SwiGLU Fused Kernel
    # =========================================================================
    
    @triton.jit
    def _swiglu_fwd_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused SwiGLU forward: out = SiLU(gate) * up"""
        pid = tl.program_id(0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements
        
        gate = tl.load(gate_ptr + offset, mask=mask)
        up = tl.load(up_ptr + offset, mask=mask)
        
        # SiLU(x) = x * sigmoid(x)
        gate_sigmoid = tl.sigmoid(gate)
        silu = gate * gate_sigmoid
        
        out = silu * up
        tl.store(out_ptr + offset, out, mask=mask)
    
    
    @triton.jit
    def _swiglu_bwd_kernel(
        grad_out_ptr,
        gate_ptr,
        up_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused SwiGLU backward"""
        pid = tl.program_id(0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < n_elements
        
        grad_out = tl.load(grad_out_ptr + offset, mask=mask)
        gate = tl.load(gate_ptr + offset, mask=mask)
        up = tl.load(up_ptr + offset, mask=mask)
        
        # Forward values
        gate_sigmoid = tl.sigmoid(gate)
        silu = gate * gate_sigmoid
        
        # Gradients
        # d(SiLU(x))/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        #               = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        d_silu = gate_sigmoid * (1.0 + gate * (1.0 - gate_sigmoid))
        
        grad_up = grad_out * silu
        grad_gate = grad_out * up * d_silu
        
        tl.store(grad_up_ptr + offset, grad_up, mask=mask)
        tl.store(grad_gate_ptr + offset, grad_gate, mask=mask)
    
    
    class TritonSwiGLUFunction(torch.autograd.Function):
        """Triton-accelerated SwiGLU"""
        
        @staticmethod
        def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
            assert gate.is_cuda and up.is_cuda
            assert gate.shape == up.shape
            assert gate.is_contiguous() and up.is_contiguous()
            
            n_elements = gate.numel()
            out = torch.empty_like(gate)
            
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            
            _swiglu_fwd_kernel[grid](
                gate, up, out, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            ctx.save_for_backward(gate, up)
            return out
        
        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            gate, up = ctx.saved_tensors
            grad_out = grad_out.contiguous()
            
            n_elements = gate.numel()
            grad_gate = torch.empty_like(gate)
            grad_up = torch.empty_like(up)
            
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            
            _swiglu_bwd_kernel[grid](
                grad_out, gate, up, grad_gate, grad_up, n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            
            return grad_gate, grad_up
    
    
    def triton_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Apply fused SwiGLU using Triton kernel."""
        return TritonSwiGLUFunction.apply(gate, up)
    
    
    # =========================================================================
    # RMSNorm Triton Kernel
    # =========================================================================
    
    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        inv_rms_ptr,
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMSNorm forward"""
        row_idx = tl.program_id(0)
        
        # Load row
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        
        x_row_ptr = x_ptr + row_idx * n_cols
        x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
        
        # Compute RMS
        x_sq = x * x
        mean_sq = tl.sum(x_sq, axis=0) / n_cols
        inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
        
        # Store inv_rms for backward
        tl.store(inv_rms_ptr + row_idx, inv_rms)
        
        # Normalize and scale
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        out = x * inv_rms * weight
        
        out_row_ptr = out_ptr + row_idx * n_cols
        tl.store(out_row_ptr + col_offsets, out, mask=mask)
    
    
    def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Apply RMSNorm using Triton kernel (forward only for now)."""
        assert x.is_cuda and weight.is_cuda
        
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        n_rows, n_cols = x.shape
        
        out = torch.empty_like(x)
        inv_rms = torch.empty(n_rows, device=x.device, dtype=x.dtype)
        
        # Block size must be >= n_cols
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        _rmsnorm_fwd_kernel[(n_rows,)](
            x, weight, out, inv_rms,
            n_rows, n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out.view(orig_shape)


# =========================================================================
# PyTorch Fallback (when Triton not available)
# =========================================================================

def pytorch_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU using pure PyTorch (fallback)."""
    return torch.nn.functional.silu(gate) * up


# =========================================================================
# Public API
# =========================================================================

def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU activation.
    
    Uses Triton kernel if available, otherwise falls back to PyTorch.
    
    Args:
        gate: Gate tensor from w_gate projection
        up: Up tensor from w_up projection
        
    Returns:
        SiLU(gate) * up
    """
    if TRITON_AVAILABLE and gate.is_cuda:
        return triton_swiglu(gate, up)
    return pytorch_swiglu(gate, up)


def get_available_kernels() -> dict:
    """Get dictionary of available Triton kernels."""
    return {
        "triton": TRITON_AVAILABLE,
        "swiglu_triton": TRITON_AVAILABLE,
        "rmsnorm_triton": TRITON_AVAILABLE,
    }
