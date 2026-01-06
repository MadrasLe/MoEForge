# 🔥 MoEForge

**Fast and memory-efficient Mixture of Experts training library**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)

MoEForge is a PyTorch library for training Mixture of Experts (MoE) transformers with state-of-the-art optimizations.

## ✨ Features

- 🚀 **Token Shuffling** - Efficient vectorized routing without Python loops
- ⚡ **Flash Attention 2** - Memory-efficient attention with O(N) memory
- 🔧 **Custom CUDA Kernels** - MGRMSNORM for fast RMSNorm
- 🎯 **Shared Experts** - DeepSeek-V2 style architecture
- 📊 **Load Balancing** - Z-loss and auxiliary losses for stable training
- 🔥 **LigerRMSNorm** - Optional Liger Kernel integration
- 💾 **DeepSpeed Ready** - ZeRO optimization support

## 📦 Installation

```bash
# Basic installation
pip install moeforge

# With Flash Attention
pip install moeforge[flash-attn]

# With all optimizations
pip install moeforge[full]

# From source
git clone https://github.com/MadrasLe/moeforge.git
cd moeforge
pip install -e .
```

### Optional: Install MGRMSNORM (Custom CUDA Kernel)

```bash
pip install git+https://github.com/MadrasLe/MGRrmsnorm.git
```

## 🚀 Quick Start

```python
from moeforge import MoEConfig, MoEModel

# Create configuration
config = MoEConfig(
    vocab_size=32000,
    hidden_dim=1024,
    num_heads=16,
    num_layers=12,
    num_experts=8,
    top_k=2,
    shared_expert=True,  # DeepSeek-style
)

# Build model
model = MoEModel(config)
print(model)  # Shows total and active params

# Forward pass
input_ids = torch.randint(0, 32000, (4, 512))
outputs = model(input_ids)

# With labels for training
outputs = model(input_ids, labels=input_ids)
loss = outputs["loss"]  # Includes LM loss + aux loss
loss.backward()
```

## 📊 Benchmark Results

Tested on NVIDIA L4 (24GB) with BF16:

| Config | Params Total | Params Active | TPS | Memory |
|--------|-------------|---------------|-----|--------|
| Small | 77M | 58M (75%) | 52,805 | 3.0 GB |
| Medium | 644M | 248M (38%) | 20,534 | 3.2 GB |

With LigerRMSNorm + Flash Attention:
- **66% faster** than naive implementation
- **47% less memory** usage

## 🏗️ Architecture

```
MoEModel
├── embed_tokens (Embedding)
├── layers (ModuleList)
│   └── MoEBlock
│       ├── ln1 (RMSNorm)
│       ├── attn (OptimizedAttention + RoPE)
│       ├── ln2 (RMSNorm)
│       └── moe (MoELayer)
│           ├── gate (Router)
│           ├── experts (ModuleList[SwiGLUExpert])
│           └── shared_ffn (SharedExpert, optional)
├── ln_f (RMSNorm)
└── lm_head (Linear, weight-tied)
```

## ⚙️ Configuration Options

```python
config = MoEConfig(
    # Model size
    vocab_size=32000,
    hidden_dim=1024,
    num_heads=16,
    num_layers=12,
    max_seq_len=2048,
    
    # MoE settings
    num_experts=8,
    top_k=2,
    shared_expert=True,
    capacity_factor=1.25,
    gate_temperature=1.0,
    
    # Regularization
    dropout=0.05,
    load_balance_alpha=0.01,
    z_loss_weight=1e-3,
    
    # Optimizations
    norm_type="auto",  # auto, mgrmsnorm, liger, pytorch
    use_flash_attention=True,
    use_bf16=True,
)
```

## 🔧 Normalization Options

MoEForge supports multiple RMSNorm implementations:

| Type | Speed | Memory | Requires |
|------|-------|--------|----------|
| `mgrmsnorm` | ⚡⚡⚡ | Low | CUDA kernel |
| `liger` | ⚡⚡ | Low | liger-kernel |
| `pytorch` | ⚡ | Medium | None |

```python
from moeforge.layers import get_norm_layer, get_available_norms

# Check available implementations
print(get_available_norms())
# {'mgrmsnorm': True, 'liger': True, 'pytorch': True}

# Get specific implementation
norm = get_norm_layer(1024, norm_type="liger")
```

## 🎓 Training with DeepSpeed

```python
import deepspeed
from moeforge import MoEConfig, MoEModel

config = MoEConfig(...)
model = MoEModel(config)

# DeepSpeed config
ds_config = {
    "train_batch_size": 1024,
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": 1},
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
)
```

## 📁 Project Structure

```
moeforge/
├── moeforge/
│   ├── __init__.py
│   ├── config.py          # MoEConfig dataclass
│   ├── layers/
│   │   ├── attention.py   # Flash Attention + RoPE
│   │   ├── expert.py      # SwiGLU experts
│   │   ├── moe.py         # MoE layer with token shuffling
│   │   └── normalization.py  # RMSNorm implementations
│   └── models/
│       └── transformer.py # Complete MoE model
├── examples/
│   └── train_moe.py
├── tests/
├── pyproject.toml
└── README.md
```

## 🤝 Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) for shared expert architecture
- [Mixtral](https://arxiv.org/abs/2401.04088) for MoE design inspiration

## 📚 Citation

```bibtex
@software{moeforge2024,
  author = {Gabriel Yogi},
  title = {MoEForge: Fast Mixture of Experts Training},
  year = {2024},
  url = {https://github.com/MadrasLe/moeforge}
}
```

---

**Made with 🔥 by [MadrasLe](https://github.com/MadrasLe)**
