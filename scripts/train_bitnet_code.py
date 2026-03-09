#!/usr/bin/env python3
"""
Graviton-Native: Code model training from scratch.

Pre-training on code data with BitNet architecture.
Runs on your PC — 8–32 GB RAM sufficient.

Usage:
    # Small code dataset (HuggingFace repos)
    python scripts/train_bitnet_code.py --model_size 350m --dataset hf-stack --steps 1000

    # GitHub code (streaming, large)
    python scripts/train_bitnet_code.py --model_size 350m --dataset github-code --steps 5000

    # Custom JSONL (your own repos)
    python scripts/train_bitnet_code.py --data_path ./my_code.jsonl --model_size 350m
"""

import argparse
from pathlib import Path

from graviton_native.training.trainer import get_preset_config, train_bitnet


# Code datasets — HuggingFace
CODE_DATASETS = {
    "hf-stack": ("smangrul/hf-stack-v1", "default", "content"),  # HuggingFace repos
    "github-code": ("codeparrot/github-code", "clean", "code"),  # GitHub code (large)
    "the-stack": ("bigcode/the-stack", "data", "content"),  # The Stack
    "wikitext": ("wikitext", "wikitext-2-raw-v1", "text"),  # Fallback
}


def main():
    parser = argparse.ArgumentParser(
        description="Graviton-Native: Code model training from scratch (BitNet)"
    )
    parser.add_argument("--model_size", default="350m", choices=["350m", "1b", "2b"])
    parser.add_argument(
        "--dataset",
        default="hf-stack",
        choices=list(CODE_DATASETS.keys()),
        help="Code dataset (hf-stack=small, github-code=large)",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Custom JSONL path")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=250)
    args = parser.parse_args()

    if args.data_path:
        dataset_name = "json"
        dataset_config = args.data_path
        print(f"\n  Code data: {args.data_path}")
    else:
        ds_info = CODE_DATASETS[args.dataset]
        dataset_name = ds_info[0]
        dataset_config = ds_info[1]
        print(f"\n  Dataset: {dataset_name} ({dataset_config})")

    print(f"  Model: BitNet {args.model_size}")
    print(f"  Steps: {args.steps}\n")

    train_bitnet(
        model_size=args.model_size,
        data_path=args.data_path,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        save_every=args.save_every,
    )

    print("\n  To run in Graviton:")
    print(f"  graviton-ui → Model: {args.output_dir}/bitnet-{args.model_size}")


if __name__ == "__main__":
    main()
