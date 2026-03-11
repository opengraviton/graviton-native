#!/usr/bin/env python3
"""
Graviton-Native: 72B BitNet Code Model — Production Training

World-class code model: BitNet ternary (10x inference speed) + CodeLlama architecture.
Requires: 8x A100 80GB (or 8x H100) + DeepSpeed ZeRO-3 + CPU offload.

Usage:
    # Multi-GPU with DeepSpeed (8x A100)
    deepspeed --num_gpus=8 scripts/train_bitnet_72b.py

    # With config
    deepspeed --num_gpus=8 scripts/train_bitnet_72b.py \\
        --dataset the-stack \\
        --steps 100000 \\
        --batch_size 1 \\
        --grad_accum 64
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graviton_native.models.bitnet import BitNetConfig, BitNetCausalLM
from graviton_native.training.trainer import get_preset_config


def get_code_tokenizer():
    """CodeLlama tokenizer — best for code."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", trust_remote_code=True)
    except Exception:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bigcode/starcoder2-3b", trust_remote_code=True)


def create_dataloader(dataset_name: str, tokenizer, batch_size: int, seq_len: int, grad_accum: int):
    """Create DataLoader or generator for code data."""
    from datasets import load_dataset

    eff_batch = batch_size * grad_accum

    if dataset_name == "the-stack":
        # 3TB code — gated, needs HF_TOKEN + access at huggingface.co/datasets/bigcode/the-stack
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
            except Exception:
                pass
        ds = load_dataset(
            "bigcode/the-stack", data_dir="data/python",
            split="train", streaming=True, token=hf_token
        )
        text_key = "content"

        def stream_batches():
            batch_texts = []
            for ex in ds:
                t = ex.get(text_key) or ""
                if t and len(t.strip()) > 50:
                    batch_texts.append(t)
                if len(batch_texts) >= eff_batch:
                    chunk = batch_texts[:eff_batch]
                    batch_texts = batch_texts[eff_batch:]
                    tok = tokenizer(
                        chunk,
                        truncation=True,
                        max_length=seq_len,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    yield {"input_ids": tok["input_ids"], "attention_mask": tok.get("attention_mask")}

        return stream_batches()

    elif dataset_name == "hug-stack":
        ds = load_dataset("smangrul/hug_stack", split="train")
        text_key = "content"

        def tokenize(examples):
            return tokenizer(
                examples[text_key],
                truncation=True,
                max_length=seq_len,
                padding="max_length",
                return_tensors="pt",
            )

        ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return DataLoader(ds, batch_size=eff_batch, shuffle=True, num_workers=0)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Graviton-Native 72B BitNet Code Model")
    parser.add_argument("--model_size", default="72b", choices=["72b"])
    parser.add_argument("--dataset", default="the-stack", choices=["the-stack", "hug-stack"])
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=1, help="Per-GPU batch size")
    parser.add_argument("--grad_accum", type=int, default=64, help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = get_preset_config(args.model_size)
    n_params = sum(
        p.numel() for p in BitNetCausalLM(config).parameters()
    )
    print(f"\n  BitNet {args.model_size}: {n_params / 1e9:.2f}B parameters")
    print(f"  Ternary inference: ~{n_params * 1.58 / 8 / 1e9:.2f} GB")
    print(f"  Dataset: {args.dataset}")
    print(f"  Steps: {args.steps}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum} (per-GPU {args.batch_size} x accum {args.grad_accum})\n")

    # DeepSpeed
    use_deepspeed = "RANK" in os.environ
    if use_deepspeed:
        import deepspeed
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        tokenizer = get_code_tokenizer()
        model = BitNetCausalLM(config)

        # DeepSpeed init — ZeRO-3 + CPU offload for 72B
        ds_config_path = Path(__file__).resolve().parents[1] / "configs" / "ds_config_72b.json"
        with open(ds_config_path) as f:
            ds_config = json.load(f)
        ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
        ds_config["gradient_accumulation_steps"] = args.grad_accum
        ds_config["optimizer"]["params"]["lr"] = args.lr
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )

        output_path = Path(args.output_dir) / f"bitnet-{args.model_size}"
        output_path.mkdir(parents=True, exist_ok=True)

        loader = create_dataloader(
            args.dataset, tokenizer, args.batch_size, args.seq_len, args.grad_accum
        )
        # For small datasets (hug-stack), cycle to reach target steps
        if args.dataset == "hug-stack":
            from itertools import cycle
            loader = cycle(loader)
        step = 0
        model_engine.train()

        for batch in loader:
            if step >= args.steps:
                break
            if isinstance(batch, dict):
                ids = batch["input_ids"].to(model_engine.local_rank)
                mask = batch.get("attention_mask")
                if mask is not None:
                    mask = mask.to(model_engine.local_rank)
            else:
                ids = batch[0].to(model_engine.local_rank)

            logits = model_engine(ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size),
                ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_token_id or 0,
            )
            model_engine.backward(loss)
            model_engine.step()

            if model_engine.local_rank == 0:
                print(f"Step {step} | loss={loss.item():.4f}")

            step += 1
            if step > 0 and step % args.save_every == 0 and model_engine.local_rank == 0:
                save_path = output_path / f"checkpoint-{step}"
                save_path.mkdir(parents=True, exist_ok=True)
                model_engine.save_checkpoint(str(save_path))
                tokenizer.save_pretrained(str(save_path))
                cfg_dict = {f: getattr(config, f) for f in config.__dataclass_fields__}
                cfg_dict["model_type"] = "bitnet"
                (save_path / "config.json").write_text(json.dumps(cfg_dict, indent=2))
                print(f"  Saved checkpoint at step {step}")

        if model_engine.local_rank == 0:
            save_path = output_path / "final"
            save_path.mkdir(parents=True, exist_ok=True)
            model_engine.save_checkpoint(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            cfg_dict = {f: getattr(config, f) for f in config.__dataclass_fields__}
            cfg_dict["model_type"] = "bitnet"
            (save_path / "config.json").write_text(json.dumps(cfg_dict, indent=2))
            print(f"\n  Done. Model saved to {save_path}")
    else:
        print("  ERROR: 72B requires DeepSpeed. Run:")
        print("    deepspeed --num_gpus=8 scripts/train_bitnet_72b.py")
        print("\n  For local testing, use smaller model:")
        print("    python scripts/train_bitnet_code.py --model_size 2b --steps 5000")


if __name__ == "__main__":
    main()
