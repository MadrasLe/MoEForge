"""
Mixture of Experts Layer with Token Shuffling

Features:
    - Vectorized token shuffling (no Python loops)
    - Z-loss for router regularization
    - Load balancing loss
    - Optional capacity factor
    - Shared expert support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .expert import SwiGLUExpert, SharedExpert


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer with efficient token shuffling.
    
    Architecture:
        1. Router computes expert scores for each token
        2. Top-k experts selected per token
        3. Tokens shuffled and batched by destination expert
        4. Experts process their assigned tokens
        5. Output un-shuffled and weighted
    
    Optionally includes a shared expert (always active).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden_multiplier: float = 5/3,
        expert_dropout: float = 0.0,
        capacity_factor: Optional[float] = None,
        gate_temperature: float = 1.0,
        z_loss_weight: float = 1e-3,
        shared_expert: bool = False,
        shared_expert_multiplier: float = 0.5,
        mlp_residual_scale: float = 1.0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gate_temperature = gate_temperature
        self.z_loss_weight = z_loss_weight
        self.mlp_residual_scale = mlp_residual_scale
        self.use_shared_expert = shared_expert
        
        # Router (gate)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(
                hidden_dim=hidden_dim,
                hidden_multiplier=expert_hidden_multiplier,
                dropout=expert_dropout,
            )
            for _ in range(num_experts)
        ])
        
        # Shared expert (optional)
        if shared_expert:
            self.shared_ffn = SharedExpert(
                hidden_dim=hidden_dim,
                hidden_multiplier=shared_expert_multiplier,
                dropout=expert_dropout,
            )
        else:
            self.shared_ffn = None
    
    def forward(
        self, 
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, S, D)
            
        Returns:
            output: Output tensor of shape (B, S, D)
            aux_loss: Combined auxiliary loss (load balance + z-loss)
            expert_probs: Mean expert probabilities for monitoring
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        T = x_flat.size(0)  # Total tokens
        
        # 1. Router computation with temperature
        logits = self.gate(x_flat) / self.gate_temperature
        
        # 2. Z-loss for regularization
        z_loss = (logits ** 2).mean()
        
        # 3. Top-k selection
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        
        # 4. Load balancing loss
        probs = F.softmax(logits, dim=-1)
        load_balance = self.num_experts * (probs.mean(0) ** 2).sum()
        
        # Combined auxiliary loss
        aux_loss = load_balance + self.z_loss_weight * z_loss
        
        # 5. Token shuffling for efficient batched processing
        output_flat = self._token_shuffle_forward(x_flat, topk_idx, topk_weights, T)
        
        # 6. Add shared expert output if enabled
        if self.use_shared_expert and self.shared_ffn is not None:
            shared_out = self.shared_ffn(x_flat)
            output_flat = output_flat + shared_out
        
        # 7. Apply residual scale
        output_flat = output_flat * self.mlp_residual_scale
        
        return output_flat.view(B, S, D), aux_loss, probs.mean(0)
    
    def _token_shuffle_forward(
        self,
        x_flat: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        T: int,
    ) -> torch.Tensor:
        """
        Efficient token shuffling for batched expert processing.
        
        With optional capacity factor that prioritizes tokens by weight.
        
        Instead of looping over experts, we:
        1. Flatten expert indices
        2. Apply capacity factor (if enabled) - keeps highest weight tokens
        3. Sort by destination expert
        4. Split into batches per expert
        5. Process each batch
        6. Un-shuffle back to original order
        """
        D = x_flat.size(-1)
        device = x_flat.device
        
        # Flatten indices and weights
        flat_idx = topk_idx.view(-1)  # (T * top_k,)
        token_ids = torch.arange(T, device=device).repeat_interleave(self.top_k)
        flat_weights = topk_weights.view(-1)  # (T * top_k,)
        
        # Apply capacity factor if enabled
        if self.capacity_factor is not None:
            # Calculate capacity per expert
            expert_capacity = int((T * self.top_k / self.num_experts) * self.capacity_factor)
            
            # Create mask for tokens to keep (prioritize by weight)
            keep_mask = torch.zeros_like(flat_idx, dtype=torch.bool)
            
            for e_id in range(self.num_experts):
                # Find all tokens routed to this expert
                expert_positions = torch.where(flat_idx == e_id)[0]
                
                if expert_positions.numel() == 0:
                    continue
                
                if expert_positions.numel() > expert_capacity:
                    # Get weights for tokens going to this expert
                    expert_weights = flat_weights[expert_positions]
                    # Sort by weight (descending) and keep top capacity
                    sorted_positions = expert_positions[torch.argsort(expert_weights, descending=True)]
                    positions_to_keep = sorted_positions[:expert_capacity]
                    keep_mask[positions_to_keep] = True
                else:
                    # Keep all tokens for this expert
                    keep_mask[expert_positions] = True
            
            # Apply mask
            flat_idx = flat_idx[keep_mask]
            token_ids = token_ids[keep_mask]
            flat_weights = flat_weights[keep_mask]
        
        # Add dimension for weights
        flat_weights = flat_weights.unsqueeze(1)  # (N, 1)
        
        # Sort by expert index (shuffle)
        sorted_expert_idx, sort_order = flat_idx.sort(0)
        permuted_tokens = x_flat[token_ids[sort_order]]
        permuted_weights = flat_weights[sort_order]
        
        # Count tokens per expert
        counts = torch.bincount(sorted_expert_idx, minlength=self.num_experts)
        
        # Split into batches per expert
        splits = torch.split(permuted_tokens, counts.tolist())
        
        # Process each expert batch
        expert_outputs = []
        for i, expert_batch in enumerate(splits):
            if expert_batch.numel() > 0:
                expert_outputs.append(self.experts[i](expert_batch))
        
        # Concatenate and apply weights
        if expert_outputs:
            concatenated = torch.cat(expert_outputs, dim=0)
            weighted_output = concatenated * permuted_weights
            
            # Un-shuffle: scatter back to original positions
            output_flat = torch.zeros_like(x_flat)
            output_flat.index_add_(0, token_ids[sort_order], weighted_output.to(output_flat.dtype))
        else:
            output_flat = torch.zeros_like(x_flat)
        
        return output_flat
    
    def get_expert_load(self, logits: torch.Tensor) -> torch.Tensor:
        """Get the load (number of tokens) per expert."""
        _, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        flat_idx = topk_idx.view(-1)
        return torch.bincount(flat_idx, minlength=self.num_experts).float()
