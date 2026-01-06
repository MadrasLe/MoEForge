"""
Optimized Attention with RoPE and Flash Attention support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class ManualRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    Pre-computes sin/cos for efficiency.
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        # Compute position indices
        t = torch.arange(max_seq_len).float()
        
        # Outer product to get frequencies for each position
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # Register as buffers (not parameters)
        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding.
        
        Args:
            x: Input tensor of shape (B, S, H, D)
            
        Returns:
            Tensor with rotary embeddings applied
        """
        B, S, H, D = x.shape
        
        # Get cos/sin for current sequence length
        cos = self.cos[:S].unsqueeze(0).unsqueeze(2)  # (1, S, 1, D//2)
        sin = self.sin[:S].unsqueeze(0).unsqueeze(2)
        
        # Split into even/odd components
        x_reshaped = x.reshape(B, S, H, D // 2, 2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]
        
        # Apply rotation
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        
        return torch.stack((y_even, y_odd), dim=-1).reshape(B, S, H, D)


class OptimizedAttention(nn.Module):
    """
    Multi-Head Attention with RoPE and Flash Attention support.
    
    Features:
        - Rotary Position Embeddings
        - Flash Attention 2 (if available)
        - Fallback to PyTorch SDPA
        - Automatic bf16 handling
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
        use_rope: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        
        # QKV projection (fused for efficiency)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # RoPE
        self.rope = ManualRotaryEmbedding(self.head_dim, max_seq_len) if use_rope else None
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, S, D)
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (B, S, D)
        """
        B, S, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)
        
        # Apply RoPE
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        
        # Attention computation
        if self.use_flash_attn:
            # Flash Attention expects (B, S, H, D) and bf16/fp16
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            
            attn_out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            )
            attn_out = attn_out.to(x.dtype)
        else:
            # Fallback to PyTorch SDPA
            # Transpose to (B, H, S, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
            attn_out = attn_out.transpose(1, 2)
        
        # Reshape and project
        attn_out = attn_out.contiguous().view(B, S, D)
        return self.o_proj(attn_out)
