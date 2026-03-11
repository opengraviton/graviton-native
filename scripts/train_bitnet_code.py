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


# Code datasets — HuggingFace (streaming=True for huge datasets)
# (dataset_name, config, text_key, streaming, data_dir?)
# Note: the-stack requires HuggingFace login + accept terms at huggingface.co/datasets/bigcode/the-stack
CODE_DATASETS = {
    "the-stack": ("bigcode/the-stack", None, "content", True, "data/python"),  # 3TB, gated
    "the-stack-js": ("bigcode/the-stack", None, "content", True, "data/javascript"),
    "hug-stack": ("smangrul/hug_stack", None, "content", False, None),  # 6.5K samples, open
    "wikitext": ("wikitext", "wikitext-2-raw-v1", "text", False, None),
}


def main():
    parser = argparse.ArgumentParser(
        description="Graviton-Native: Code model training from scratch (BitNet)"
    )
    parser.add_argument("--model_size", default="350m", choices=["350m", "1b", "2b", "7b"])
    parser.add_argument(
        "--dataset",
        default="the-stack",
        choices=list(CODE_DATASETS.keys()),
        help="Code dataset (the-stack=3TB gated, hug-stack=open)",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Custom JSONL path")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing (uses more memory)")
    parser.add_argument("--use_8bit_optimizer", action="store_true", help="Use 8-bit Adam (less memory)")
    args = parser.parse_args()

    if args.data_path:
        dataset_name = "json"
        dataset_config = args.data_path
        streaming = False
        print(f"\n  Code data: {args.data_path}")
    else:
        ds_info = CODE_DATASETS[args.dataset]
        dataset_name = ds_info[0]
        dataset_config = ds_info[1] or ""
        streaming = ds_info[3] if len(ds_info) > 3 else False
        data_dir = ds_info[4] if len(ds_info) > 4 else None
        info = f"{dataset_name}"
        if data_dir:
            info += f" ({data_dir})"
        elif dataset_config:
            info += f" ({dataset_config})"
        if streaming:
            info += " [streaming]"
        print(f"\n  Dataset: {info}")

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
        streaming=streaming,
        resume=args.resume,
        data_dir=data_dir if not args.data_path else None,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        use_8bit_optimizer=args.use_8bit_optimizer,
    )

    print("\n  To run in Graviton:")
    print(f"  graviton-ui → Model: {args.output_dir}/bitnet-{args.model_size}")


if __name__ == "__main__":
    main()
