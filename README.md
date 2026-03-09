# Graviton-Native

**500B+ parameters on 32 GB RAM. Through architectural change.**

Graviton-Native trains AI models with efficient architectures **from scratch**. Instead of post-training quantization, the model training itself uses low-bit and sparse representations — so anyone can run large models on their own machine.

**Technical Report:** [opengraviton.github.io/paper.html](https://opengraviton.github.io/paper.html) | [PAPER.md](PAPER.md)

## Vision

| Current State | Graviton-Native Target |
|---------------|------------------------|
| 70B model → 140 GB+ RAM | 70B model → **14 GB** (1.58-bit) |
| 500B model → impossible | 500B MoE → **32 GB** (active params) |
| Quantization = quality loss | Native training = minimal loss |

## Architectural Approaches

### 1. BitNet b1.58 — Ternary Weights
- Weights trained with **{-1, 0, +1}**
- Matrix multiply = add/subtract only (no float multiply)
- ~10x memory savings, ~10x energy savings
- Reference: [BitNet b1.58](https://arxiv.org/abs/2402.17764)

### 2. MoE (Mixture of Experts)
- 500B total params, ~10–20B active per token
- Top-K routing: k experts selected per token
- Inference support in Graviton

### 3. Sparse / Top-K Activation
- Only 30% of neurons fire per layer
- 70% compute savings

## Project Structure

```
graviton-native/
├── graviton_native/
│   ├── models/          # BitNet, MoE architectures
│   ├── training/        # Training pipeline
│   └── quantization/    # Quantization during training
├── scripts/             # Training scripts
├── configs/             # Model configurations
└── README.md
```

## Installation

```bash
cd graviton-native
pip install -e ".[train]"
```

## Quick Start

### Quick demo (2 steps)

```bash
python scripts/train_bitnet.py --model_size 350m --steps 2
```

### Training with real data (WikiText, C4, etc.)

```bash
# 500 steps with WikiText-2
python scripts/train_bitnet_full.py --model_size 350m --steps 500

# Custom JSONL data
python scripts/train_bitnet_full.py --data_path ./data/train.jsonl --model_size 350m
```

### Inference with Graviton

Graviton-Native checkpoints run directly in Graviton:

```bash
graviton-ui
# BitNet: graviton-native/checkpoints/bitnet-350m
# MoE:    graviton-native/checkpoints/moe-small
# Auto-detected
```

### MoE training

```bash
python scripts/train_moe.py --model_size small --steps 100
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Apple Silicon / NVIDIA GPU (optional; runs on CPU too)

## License

Apache-2.0 — Compatible with Graviton.
