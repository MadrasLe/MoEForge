"""
MoE Transformer Model

A complete Mixture of Experts transformer with:
    - Optimized attention with RoPE
    - MoE layers with token shuffling
    - Optional shared experts
    - Custom CUDA kernels for normalization
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..config import MoEConfig
from ..layers.attention import OptimizedAttention
from ..layers.moe import MoELayer
from ..layers.normalization import get_norm_layer


class MoEBlock(nn.Module):
    """
    Single transformer block with MoE.
    
    Architecture:
        x -> LN -> Attention -> + -> LN -> MoE -> +
           |__________________|   |______________|
           
    Pre-norm architecture (like Llama, GPT-NeoX).
    """
    
    def __init__(self, config: MoEConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Normalization layers
        self.ln1 = get_norm_layer(config.hidden_dim, config.norm_type, config.norm_eps)
        self.ln2 = get_norm_layer(config.hidden_dim, config.norm_type, config.norm_eps)
        
        # Attention
        self.attn = OptimizedAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            use_rope=config.use_rope,
            use_flash_attn=config.use_flash_attention,
        )
        
        # MoE Layer
        self.moe = MoELayer(
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            expert_hidden_multiplier=config.expert_hidden_multiplier,
            expert_dropout=config.expert_dropout,
            capacity_factor=config.capacity_factor,
            gate_temperature=config.gate_temperature,
            z_loss_weight=config.z_loss_weight,
            shared_expert=config.shared_expert,
            shared_expert_multiplier=config.shared_expert_multiplier,
            mlp_residual_scale=config.mlp_residual_scale,
        )
        
        self.residual_scale = config.residual_scale
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, S, D)
            attention_mask: Optional attention mask
            
        Returns:
            x: Output tensor (B, S, D)
            aux_loss: MoE auxiliary loss
            expert_probs: Expert probability distribution
        """
        # Attention block
        residual = x
        x = self.ln1(x)
        x = residual + self.attn(x, attention_mask) * self.residual_scale
        
        # MoE block
        residual = x
        x = self.ln2(x)
        moe_out, aux_loss, expert_probs = self.moe(x)
        x = residual + moe_out
        
        return x, aux_loss, expert_probs


class MoEModel(nn.Module):
    """
    Complete MoE Transformer Model.
    
    A decoder-only transformer with Mixture of Experts layers,
    optimized for efficient training and inference.
    
    Example:
        config = MoEConfig(hidden_dim=1024, num_experts=8)
        model = MoEModel(config)
        
        output = model(input_ids)
        loss = model(input_ids, labels=labels)
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Dropout
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            MoEBlock(config, layer_idx=i) for i in range(config.num_layers)
        ])
        
        # Final normalization
        self.ln_f = get_norm_layer(config.hidden_dim, config.norm_type, config.norm_eps)
        
        # LM head (weight tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Weight tying
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs (B, S)
            labels: Optional target labels for loss computation (B, S)
            attention_mask: Optional attention mask
            return_aux_loss: Whether to return auxiliary MoE loss
            
        Returns:
            Dictionary with:
                - logits: Output logits (B, S, V)
                - loss: Total loss (if labels provided)
                - lm_loss: Language modeling loss
                - aux_loss: MoE auxiliary loss
                - expert_probs: Mean expert probabilities per layer
        """
        B, S = input_ids.shape
        
        # Embeddings
        x = self.embed_tokens(input_ids)
        x = self.embed_dropout(x)
        
        # Transformer blocks
        total_aux_loss = 0.0
        expert_probs_list = []
        
        for layer in self.layers:
            x, aux_loss, expert_probs = layer(x, attention_mask)
            total_aux_loss += aux_loss
            expert_probs_list.append(expert_probs)
        
        # Final normalization
        x = self.ln_f(x)
        
        # LM head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        lm_loss = None
        loss = None
        
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            lm_loss = self.criterion(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # Combine losses
            avg_aux_loss = total_aux_loss / len(self.layers)
            loss = lm_loss + self.config.router_aux_loss_weight * avg_aux_loss
        
        # Stack expert probs
        expert_probs_stacked = torch.stack(expert_probs_list) if expert_probs_list else None
        
        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "aux_loss": total_aux_loss / len(self.layers) if self.layers else 0,
            "expert_probs": expert_probs_stacked,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Simple greedy/sampling generation.
        
        Args:
            input_ids: Starting token IDs (B, S)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated token IDs (B, S + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate to max sequence length
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self(idx_cond)
                logits = outputs["logits"][:, -1, :]  # Last token
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> dict:
        """Count total and active parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Calculate expert params
        if self.layers:
            params_per_expert = sum(
                p.numel() for p in self.layers[0].moe.experts[0].parameters()
            )
            total_expert_params = (
                params_per_expert * 
                self.config.num_experts * 
                self.config.num_layers
            )
            
            # Shared expert params
            shared_params = 0
            if self.config.shared_expert and self.layers[0].moe.shared_ffn is not None:
                shared_params = sum(
                    p.numel() for p in self.layers[0].moe.shared_ffn.parameters()
                ) * self.config.num_layers
            
            # Non-expert params
            other_params = total_params - total_expert_params - shared_params
            
            # Active params = other + shared + (top_k experts per layer)
            active_params = (
                other_params + 
                shared_params + 
                params_per_expert * self.config.top_k * self.config.num_layers
            )
        else:
            active_params = total_params
        
        return {
            "total": total_params,
            "active": active_params,
            "utilization": active_params / total_params if total_params > 0 else 0,
        }
    
    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"MoEModel(\n"
            f"  config={self.config},\n"
            f"  total_params={params['total']/1e6:.2f}M,\n"
            f"  active_params={params['active']/1e6:.2f}M ({params['utilization']*100:.1f}%)\n"
            f")"
        )
