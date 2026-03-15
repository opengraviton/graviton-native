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

### Omega — Ultra-sparse (8GB hedef)

```bash
python scripts/train_omega.py --model_size micro --steps 100
```

Omega: k=1 MoE + BitNet. 80M total, ~10M active/token, ~2 MB RAM aktif.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Apple Silicon / NVIDIA GPU (optional; runs on CPU too)

## graviton-train — Single Command (Mac + NVIDIA)

**Memory-efficient:** Gradient checkpointing + 8-bit optimizer (for 7b). Training with low RAM.

**Mac (MPS, 32 GPU core):**
```bash
# 1B or 2B — easy on 64 GB Mac
graviton-train run --num_gpu_cores 32 --model_size 1b --dataset the-stack --steps 5000 --batch_size 2 --save_every 500 --resume

# 7B — gradient checkpoint + 8-bit optimizer (try on 64 GB Mac)
graviton-train run --num_gpu_cores 32 --model_size 7b --dataset the-stack --steps 5000 --batch_size 1 --save_every 500

# 72B on Mac — disk-offload (~15 GB RAM, ~500 GB disk)
cd graviton-native
python3 -m graviton_native.cli run --num_gpu_cores 32 --model_size 72b --disk_offload --steps 5000 --save_every 100
```

**NVIDIA (DeepSpeed, 8 GPU):**
```bash
graviton-train run --num_gpus 8 --model_size 72b --dataset the-stack --steps 100000 --batch_size 1 --grad_accum 64
```

## 72B Code Model

**Option A — Mac (disk-offload):** Train 72B on 64 GB Mac with `--disk_offload`. ~15 GB RAM, ~500 GB disk. See command above.

**Option B — Cloud (8x A100 80GB):** RunPod, Vast.ai, Lambda. See **[CLOUD_72B.md](CLOUD_72B.md)** for full setup.

```bash
# On cloud instance (8x A100)
cd graviton-native
export HF_TOKEN=hf_xxx
bash scripts/run_72b_cloud.sh
```

## License

Apache-2.0 — Compatible with Graviton.
