"""
Example: Training a MoE Model with MoEForge

This example shows how to train a Mixture of Experts model
using MoEForge with DeepSpeed.

Usage:
    python train_moe.py --config small
    deepspeed train_moe.py --config medium --deepspeed
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import time

# MoEForge imports
from moeforge import MoEConfig, MoEModel
from moeforge.config import SMALL_CONFIG, MEDIUM_CONFIG, LARGE_CONFIG


class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, vocab_size: int, seq_len: int, size: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids}


def train_step(model, batch, device):
    """Single training step."""
    input_ids = batch["input_ids"].to(device)
    
    # Forward pass with causal LM objective
    outputs = model(input_ids, labels=input_ids)
    
    return outputs["loss"], outputs["lm_loss"], outputs["aux_loss"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    
    # Select config
    configs = {"small": SMALL_CONFIG, "medium": MEDIUM_CONFIG, "large": LARGE_CONFIG}
    config = configs[args.config]
    
    print(f"\n🔥 MoEForge Training Example")
    print(f"=" * 50)
    print(f"Config: {args.config}")
    print(config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    
    # Create model
    print(f"\n📦 Building model...")
    model = MoEModel(config).to(device).to(dtype)
    
    params = model.count_parameters()
    print(f"Total params: {params['total']/1e6:.2f}M")
    print(f"Active params: {params['active']/1e6:.2f}M ({params['utilization']*100:.1f}%)")
    
    # Dataset and dataloader
    dataset = DummyDataset(config.vocab_size, config.max_seq_len // 4, size=args.steps * args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Training loop
    print(f"\n🚀 Starting training...")
    model.train()
    
    total_tokens = 0
    start_time = time.time()
    
    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        if step >= args.steps:
            break
        
        # Forward + backward
        loss, lm_loss, aux_loss = train_step(model, batch, device)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Track tokens
        total_tokens += batch["input_ids"].numel()
        
        # Update progress bar
        elapsed = time.time() - start_time
        tps = total_tokens / elapsed if elapsed > 0 else 0
        
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "lm": f"{lm_loss.item():.3f}",
            "aux": f"{aux_loss:.4f}",
            "tps": f"{tps:,.0f}",
        })
    
    # Final stats
    total_time = time.time() - start_time
    final_tps = total_tokens / total_time
    
    print(f"\n✅ Training complete!")
    print(f"=" * 50)
    print(f"Time: {total_time:.2f}s")
    print(f"Tokens: {total_tokens:,}")
    print(f"TPS: {final_tps:,.0f}")
    
    if torch.cuda.is_available():
        print(f"Peak Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
