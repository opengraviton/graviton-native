"""CLI for Graviton-Native training."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="graviton-train",
        description="Train efficient LLM architectures (BitNet, MoE) for 32GB RAM.",
    )
    parser.add_argument(
        "model_type",
        choices=["bitnet", "moe"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--model_size",
        default="350m",
        help="Model size preset (350m, 1b, 2b)",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to training data (parquet, jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    args = parser.parse_args()

    print(f"\n  Graviton-Native: Training {args.model_type} ({args.model_size})")
    print(f"  Data: {args.data_path}")
    print(f"  Output: {args.output_dir}")
    print(f"  Steps: {args.steps}")
    print("\n  Run: python scripts/train_bitnet.py for full training.\n")


if __name__ == "__main__":
    main()
