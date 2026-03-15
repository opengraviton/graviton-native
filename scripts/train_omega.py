#!/usr/bin/env python3
"""
Train Omega — Ultra-sparse MoE + BitNet.

Omega-Micro: 8 expert, k=1, ~800M total, ~100M active.
Runs on 8GB RAM.

Usage:
    python scripts/train_omega.py --model_size micro --steps 100
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from graviton_native.models.omega import OmegaConfig, OmegaCausalLM


def get_config(size: str) -> OmegaConfig:
    if size == "micro":
        return OmegaConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=32000,
            num_experts=8,
            top_k=1,
            expert_intermediate_ratio=4,
        )
    if size == "small":
        return OmegaConfig(
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=16,
            num_attention_heads=12,
            num_key_value_heads=4,
            vocab_size=32000,
            num_experts=16,
            top_k=1,
            expert_intermediate_ratio=4,
        )
    raise ValueError(f"Unknown size: {size}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", default="micro", choices=["micro", "small"])
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    config = get_config(args.model_size)
    model = OmegaCausalLM(config)

    total = sum(p.numel() for p in model.parameters())
    active_approx = total // config.num_experts
    print(f"\n  Omega {args.model_size}: {total/1e6:.1f}M total params")
    print(f"  k=1 → ~{active_approx/1e6:.0f}M active/token")
    print(f"  BitNet 1.58-bit → ~{active_approx*1.58/8/1e6:.1f} MB active RAM\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
        logits = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            ids[:, 1:].reshape(-1),
            ignore_index=0,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step}/{args.steps} loss={loss.item():.4f}")

    out = Path(args.output_dir) / f"omega-{args.model_size}"
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "pytorch_model.bin")
    import json
    (out / "config.json").write_text(json.dumps({
        "model_type": "omega",
        **{f: getattr(config, f) for f in config.__dataclass_fields__}
    }, indent=2))
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
