"""
Graviton-Native CLI — Mac (MPS) and NVIDIA (DeepSpeed) unified command.

Mac: graviton-train run --num_gpu_cores 32 --model_size 1b --dataset hug-stack --steps 5000
NVIDIA: graviton-train run --num_gpus 8 --model_size 72b --dataset the-stack --steps 100000
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _is_mac_mps() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    try:
        import torch
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except Exception:
        return False


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def cmd_run(args):
    """Run training — Mac (MPS) or NVIDIA (DeepSpeed) selected automatically."""
    num_gpu_cores = getattr(args, "num_gpu_cores", None)
    num_gpus = getattr(args, "num_gpus", None)
    model_size = getattr(args, "model_size", "1b")
    dataset = getattr(args, "dataset", "hug-stack")
    steps = getattr(args, "steps", 5000)
    batch_size = getattr(args, "batch_size", 2)
    grad_accum = getattr(args, "grad_accum", 32)
    seq_len = getattr(args, "seq_len", 512)
    output_dir = getattr(args, "output_dir", "./checkpoints")
    save_every = getattr(args, "save_every", 500)
    resume = getattr(args, "resume", False)
    hf_token = getattr(args, "hf_token", None)

    # Ensure HF token reaches subprocess
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Mac: num_gpu_cores → MPS (or disk-offload for 72b)
    disk_offload = getattr(args, "disk_offload", False)
    if num_gpu_cores is not None and num_gpu_cores > 0:
        if not _is_mac_mps():
            print("  ⚠️  --num_gpu_cores given but MPS not available. Using CPU.")
        if model_size in ("36b", "72b") and disk_offload:
            from graviton_native.training.disk_offload import train_disk_offload
            disk_gb = 250 if model_size == "36b" else 500
            print(f"\n  {model_size.upper()} disk-offload — ~15 GB RAM, ~{disk_gb} GB disk.\n")
            train_disk_offload(
                model_size=model_size,
                output_dir=output_dir,
                steps=steps,
                batch_size=batch_size,
                seq_len=min(seq_len, 256),
                save_every=save_every,
                resume=resume,
            )
            return 0
        if model_size in ("36b", "72b"):
            print(f"  ⚠️  {model_size.upper()} does not fit on Mac. Use --disk_offload for {model_size.upper()}, or 7b (max with grad checkpoint).")
            model_size = "7b"
        if dataset == "the-stack":
            print("  ⚠️  the-stack is gated. Use --dataset hug-stack, or request access at https://huggingface.co/datasets/bigcode/the-stack")
        script_dir = Path(__file__).resolve().parents[1]
        script = script_dir / "scripts" / "train_bitnet_code.py"
        cmd = [
            sys.executable, str(script),
            "--model_size", model_size,
            "--dataset", dataset,
            "--steps", str(steps),
            "--batch_size", str(batch_size),
            "--seq_len", str(seq_len),
            "--output_dir", output_dir,
            "--save_every", str(save_every),
        ]
        if resume:
            cmd.append("--resume")
        if model_size in ("7b",):
            cmd.append("--use_8bit_optimizer")
        print(f"\n  graviton-train: Mac MPS ({num_gpu_cores} GPU core)")
        print(f"  Model: {model_size} | Dataset: {dataset} | Steps: {steps}\n")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return subprocess.run(cmd, cwd=script_dir).returncode

    # NVIDIA: num_gpus → DeepSpeed
    if num_gpus is not None and num_gpus > 0 and _has_cuda():
        script_dir = Path(__file__).resolve().parents[1]
        script = script_dir / "scripts" / "train_bitnet_72b.py"
        cmd = [
            "deepspeed", f"--num_gpus={num_gpus}", str(script),
            "--model_size", model_size,
            "--dataset", dataset,
            "--steps", str(steps),
            "--batch_size", str(batch_size),
            "--grad_accum", str(grad_accum),
            "--seq_len", str(seq_len),
            "--output_dir", output_dir,
        ]
        print(f"\n  graviton-train: DeepSpeed ({num_gpus} GPU)")
        print(f"  Model: {model_size} | Dataset: {dataset} | Steps: {steps}\n")
        return subprocess.run(cmd, cwd=script_dir).returncode

    # Default: Mac MPS if available
    if _is_mac_mps():
        return cmd_run(argparse.Namespace(
            num_gpu_cores=32, num_gpus=None,
            model_size=model_size, dataset=dataset, steps=steps,
            batch_size=batch_size, grad_accum=grad_accum, seq_len=seq_len,
            output_dir=output_dir, save_every=save_every, resume=resume,
        ))
    if _has_cuda():
        print("  NVIDIA GPU detected. Use: graviton-train run --num_gpus 8 ...")
    print("  graviton-train run --num_gpu_cores 32 ...  (Mac)")
    print("  graviton-train run --num_gpus 8 ...       (NVIDIA)")
    return 1


def main():
    parser = argparse.ArgumentParser(
        prog="graviton-train",
        description="Graviton-Native: BitNet training. Mac (MPS) or NVIDIA (DeepSpeed).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # run
    p_run = subparsers.add_parser("run", help="Start training")
    p_run.add_argument("--num_gpu_cores", type=int, default=None,
        help="Mac: GPU cores (e.g. 32). Uses MPS.")
    p_run.add_argument("--num_gpus", type=int, default=None,
        help="NVIDIA: GPU count (e.g. 8). Uses DeepSpeed.")
    p_run.add_argument("--model_size", default="1b", choices=["350m", "1b", "2b", "7b", "36b", "72b"],
        help="Model size (7b max on Mac, 36b/72b disk-offload on Mac)")
    p_run.add_argument("--dataset", default="hug-stack", choices=["hug-stack", "the-stack"],
        help="Dataset (hug-stack=open, the-stack=gated, needs HF access)")
    p_run.add_argument("--steps", type=int, default=5000)
    p_run.add_argument("--batch_size", type=int, default=2)
    p_run.add_argument("--grad_accum", type=int, default=32)
    p_run.add_argument("--seq_len", type=int, default=512)
    p_run.add_argument("--output_dir", default="./checkpoints")
    p_run.add_argument("--save_every", type=int, default=500)
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--hf_token", type=str, default=None,
        help="HuggingFace token for gated datasets (the-stack). Or set HF_TOKEN env.")
    p_run.add_argument("--disk_offload", action="store_true",
        help="36b/72b on Mac: disk-offload (~15 GB RAM, 36b=~2x faster)")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if args.command == "run":
        sys.exit(args.func(args))
    else:
        parser.print_help()
        print("\nExample (Mac):")
        print("  graviton-train run --num_gpu_cores 32 --model_size 1b --dataset hug-stack --steps 5000")
        print("\nExample (NVIDIA):")
        print("  graviton-train run --num_gpus 8 --model_size 72b --dataset the-stack --steps 100000")


if __name__ == "__main__":
    main()
