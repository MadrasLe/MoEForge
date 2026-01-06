"""
SwiGLU Expert Implementation

Supports optional fused Triton kernel for SwiGLU.
"""

import torch
import torch.nn as nn

# Try to import Triton kernel
try:
    from ..kernels.triton_kernels import fused_swiglu, TRITON_AVAILABLE
except ImportError:
    TRITON_AVAILABLE = False
    fused_swiglu = None


class SwiGLUExpert(nn.Module):
    """
    SwiGLU Feed-Forward Network (Expert).
    
    Architecture: 
        gate = SiLU(W_gate @ x)
        up = W_up @ x
        output = W_down @ (gate * up)
    
    This is the standard FFN used in modern LLMs like Llama, Mistral, etc.
    
    Supports fused Triton kernel for faster training.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int = None,
        hidden_multiplier: float = 5/3,
        dropout: float = 0.0,
        bias: bool = False,
        use_triton: bool = True,
    ):
        super().__init__()
        
        # Calculate intermediate dimension
        if intermediate_dim is None:
            intermediate_dim = int(hidden_dim * hidden_multiplier)
            # Round to multiple of 256 for GPU efficiency
            intermediate_dim = 256 * ((intermediate_dim + 255) // 256)
        
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # Projections
        self.w_gate = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.w_up = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.w_down = nn.Linear(intermediate_dim, hidden_dim, bias=bias)
        
        # Activation (only used when Triton not available)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., hidden_dim)
            
        Returns:
            Output tensor of shape (..., hidden_dim)
        """
        gate_proj = self.w_gate(x)
        up_proj = self.w_up(x)
        
        # Use fused Triton kernel if available
        if self.use_triton and fused_swiglu is not None and x.is_cuda:
            hidden = fused_swiglu(gate_proj, up_proj)
        else:
            hidden = self.activation(gate_proj) * up_proj
        
        hidden = self.dropout(hidden)
        return self.w_down(hidden)


class SharedExpert(SwiGLUExpert):
    """
    Shared Expert - always active for all tokens.
    Used in DeepSeek-V2 style architectures.
    
    Typically smaller than regular experts (0.5x multiplier).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        hidden_multiplier: float = 0.5,  # Smaller than regular experts
        dropout: float = 0.0,
    ):
        # Calculate intermediate dim with smaller multiplier
        intermediate_dim = int(hidden_dim * (5/3) * hidden_multiplier)
        intermediate_dim = 256 * ((intermediate_dim + 255) // 256)
        
        super().__init__(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
        )
