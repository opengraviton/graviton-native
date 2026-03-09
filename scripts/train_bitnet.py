#!/usr/bin/env python3
"""
Train BitNet b1.58 model from scratch.

Usage:
    python scripts/train_bitnet.py --model_size 350m --data_path ./data --output_dir ./checkpoints

Model sizes:
    350m: ~350M params, fits 8GB GPU
    1b:   ~1B params, fits 16GB GPU
    2b:   ~2B params, fits 24GB GPU
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from graviton_native.models.bitnet import BitNetConfig, BitNetCausalLM


def get_config(size: str) -> BitNetConfig:
    """Preset configs for different model sizes."""
    presets = {
        "350m": BitNetConfig(
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            vocab_size=32000,
            max_position_embeddings=2048,
        ),
        "1b": BitNetConfig(
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            vocab_size=32000,
            max_position_embeddings=2048,
        ),
        "2b": BitNetConfig(
            hidden_size=2560,
            intermediate_size=6912,
            num_hidden_layers=30,
            num_attention_heads=20,
            num_key_value_heads=5,
            vocab_size=128256,
            max_position_embeddings=4096,
        ),
    }
    return presets.get(size, presets["350m"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="350m", choices=["350m", "1b", "2b"])
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    config = get_config(args.model_size)
    model = BitNetCausalLM(config)

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  BitNet {args.model_size}: {n_params / 1e6:.1f}M parameters")
    print(f"  Ternary weights: ~{n_params * 1.58 / 8 / 1e6:.1f} MB (vs {n_params * 2 / 1e6:.0f} MB FP16)\n")

    # Dummy training step (real data loading would use datasets)
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(min(args.steps, 3)):  # Demo: 3 steps
        # Dummy batch
        batch = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            batch[:, 1:].reshape(-1),
            ignore_index=0,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    # Save (HuggingFace-compatible structure for Graviton)
    save_path = output_dir / f"bitnet-{args.model_size}"
    save_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_path / "pytorch_model.bin")
    import json
    config_dict = {f: getattr(config, f) for f in config.__dataclass_fields__}
    (save_path / "config.json").write_text(json.dumps(config_dict, indent=2))

    # Tokenizer (TinyLlama vocab matches 32000)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tok.save_pretrained(str(save_path))
    except Exception as e:
        print(f"  Warning: Could not save tokenizer: {e}")
    print(f"\n  Saved to {save_path}")
    print("  Load in Graviton with: graviton-ui → Model path =", str(save_path))


if __name__ == "__main__":
    main()
