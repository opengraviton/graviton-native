#!/usr/bin/env python3
"""
Train MoE (Mixture of Experts) model.

Usage:
    python scripts/train_moe.py --model_size small --steps 100
"""

import argparse
from pathlib import Path

import torch

from graviton_native.models.moe import MoEConfig, MoECausalLM


def get_moe_config(size: str) -> MoEConfig:
    presets = {
        "small": MoEConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_experts=4,
            top_k=2,
            vocab_size=32000,
        ),
        "medium": MoEConfig(
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=8,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=8,
            top_k=2,
            vocab_size=32000,
        ),
        "large": MoEConfig(
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=16,
            num_attention_heads=16,
            num_key_value_heads=4,
            num_experts=16,
            top_k=2,
            vocab_size=32000,
        ),
    }
    return presets.get(size, presets["small"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    config = get_moe_config(args.model_size)
    model = MoECausalLM(config)

    total = sum(p.numel() for p in model.parameters())
    active_per_token = config.top_k * (config.hidden_size * config.intermediate_size * 3)  # approx
    print(f"\n  MoE {args.model_size}: {total / 1e6:.1f}M total params")
    print(f"  Experts: {config.num_experts}, Top-K: {config.top_k}")
    print(f"  Active per token: ~{active_per_token / 1e6:.0f}M\n")

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    output_path = Path(args.output_dir) / f"moe-{args.model_size}"
    output_path.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
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
        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    torch.save(model.state_dict(), output_path / "pytorch_model.bin")
    import json
    cfg_dict = {f: getattr(config, f) for f in config.__dataclass_fields__}
    (output_path / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
