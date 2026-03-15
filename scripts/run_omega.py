#!/usr/bin/env python3
"""
Omega inference — generate text from checkpoint.

Usage:
    python scripts/run_omega.py --checkpoint checkpoints/omega-micro --prompt "def hello" --max_tokens 50
"""

import argparse
import json
from pathlib import Path

import torch

from graviton_native.models.omega import OmegaConfig, OmegaCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/omega-micro")
    p.add_argument("--prompt", default="Hello")
    p.add_argument("--max_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        return 1

    config_path = ckpt / "config.json"
    if not config_path.exists():
        print("No config.json")
        return 1
    cfg = json.loads(config_path.read_text())

    config = OmegaConfig(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
        rope_theta=cfg.get("rope_theta", 10000.0),
        num_experts=cfg.get("num_experts", 8),
        top_k=cfg.get("top_k", 1),
        expert_intermediate_ratio=cfg.get("expert_intermediate_ratio", 4),
    )

    model = OmegaCausalLM(config)
    model.load_state_dict(
        torch.load(ckpt / "pytorch_model.bin", map_location="cpu", weights_only=True),
        strict=False,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception:
        tok = __import__("transformers").AutoTokenizer.from_pretrained("gpt2")

    ids = tok.encode(args.prompt, return_tensors="pt").to(device)
    print(args.prompt, end="", flush=True)

    with torch.no_grad():
        for _ in range(args.max_tokens - 1):
            logits = model(ids)
            next_logits = logits[0, -1] / args.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            if next_id == tok.eos_token_id:
                break
            print(tok.decode([next_id]), end="", flush=True)
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    print()
    return 0


if __name__ == "__main__":
    exit(main())
