#!/usr/bin/env python3
"""
72B BitNet — Disk Offload Training (Mac 64 GB)

Usage:
  graviton-train run --num_gpu_cores 32 --model_size 72b --disk_offload --steps 10 --batch_size 1 --seq_len 128

Or directly:
  python scripts/train_bitnet_72b_disk_offload.py --steps 10 --seq_len 128
"""

import argparse
from pathlib import Path

from graviton_native.training.disk_offload import train_72b_disk_offload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="./checkpoints")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--data_path", default=None)
    p.add_argument("--no_resume", action="store_true", help="Start from scratch (ignore step.txt and checkpoints)")
    args = p.parse_args()

    train_72b_disk_offload(
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        save_every=args.save_every,
        data_path=args.data_path,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
