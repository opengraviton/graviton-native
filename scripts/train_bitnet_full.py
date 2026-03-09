#!/usr/bin/env python3
"""
Full BitNet training with real data (HuggingFace datasets).

Usage:
    # WikiText-2 (small, fast)
    python scripts/train_bitnet_full.py --model_size 350m --steps 500

    # Custom JSONL data
    python scripts/train_bitnet_full.py --data_path ./data/train.jsonl --model_size 350m
"""

import argparse

from graviton_native.training.trainer import train_bitnet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="350m", choices=["350m", "1b", "2b"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset", default="wikitext", help="HF dataset name")
    parser.add_argument("--dataset_config", default="wikitext-2-raw-v1")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=250)
    args = parser.parse_args()

    train_bitnet(
        model_size=args.model_size,
        data_path=args.data_path,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
