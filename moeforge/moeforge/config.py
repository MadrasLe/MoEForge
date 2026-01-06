"""
MoEForge Configuration
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class MoEConfig:
    """
    Configuration for MoE Transformer models.
    
    Example:
        config = MoEConfig(
            hidden_dim=1024,
            num_experts=8,
            top_k=2,
            shared_expert=True,
        )
    """
    
    # Model dimensions
    vocab_size: int = 32000
    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 12
    max_seq_len: int = 2048
    
    # MoE configuration
    num_experts: int = 8
    top_k: int = 2
    shared_expert: bool = True
    shared_expert_multiplier: float = 0.5  # Size of shared expert relative to normal
    expert_hidden_multiplier: float = 5/3  # SwiGLU expansion ratio
    
    # Capacity and routing
    capacity_factor: float = 1.25
    gate_temperature: float = 1.0
    
    # Regularization
    dropout: float = 0.05
    expert_dropout: float = 0.05
    
    # Loss weights
    load_balance_alpha: float = 0.01
    z_loss_weight: float = 1e-3
    router_aux_loss_weight: float = 0.01
    
    # Normalization
    norm_type: Literal["auto", "mgrmsnorm", "liger", "pytorch"] = "auto"
    norm_eps: float = 1e-5
    
    # Attention
    use_flash_attention: bool = True
    use_rope: bool = True
    
    # Training
    use_bf16: bool = True
    gradient_checkpointing: bool = False
    
    # Residual scaling (DeepSeek-style)
    residual_scale: float = 1.0
    mlp_residual_scale: float = 0.5  # For shared + MoE combination
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.top_k <= self.num_experts, \
            f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})"
        assert self.capacity_factor >= 1.0, \
            f"capacity_factor ({self.capacity_factor}) must be >= 1.0"
    
    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads
    
    @property
    def expert_hidden_dim(self) -> int:
        """Calculate expert FFN hidden dimension"""
        hidden = int(self.hidden_dim * self.expert_hidden_multiplier)
        # Round to multiple of 256 for efficiency
        return 256 * ((hidden + 255) // 256)
    
    @property
    def shared_expert_hidden_dim(self) -> int:
        """Calculate shared expert FFN hidden dimension"""
        hidden = int(self.hidden_dim * self.expert_hidden_multiplier * self.shared_expert_multiplier)
        return 256 * ((hidden + 255) // 256)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            k: getattr(self, k) for k in self.__dataclass_fields__
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "MoEConfig":
        """Create config from dictionary"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def __repr__(self) -> str:
        return (
            f"MoEConfig(\n"
            f"  hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, num_layers={self.num_layers},\n"
            f"  num_experts={self.num_experts}, top_k={self.top_k}, shared_expert={self.shared_expert},\n"
            f"  capacity_factor={self.capacity_factor}, norm_type='{self.norm_type}'\n"
            f")"
        )


# Preset configurations
SMALL_CONFIG = MoEConfig(
    hidden_dim=512,
    num_heads=16,
    num_layers=6,
    num_experts=4,
    top_k=2,
)

MEDIUM_CONFIG = MoEConfig(
    hidden_dim=1024,
    num_heads=16,
    num_layers=12,
    num_experts=8,
    top_k=2,
)

LARGE_CONFIG = MoEConfig(
    hidden_dim=2048,
    num_heads=32,
    num_layers=24,
    num_experts=8,
    top_k=2,
)
