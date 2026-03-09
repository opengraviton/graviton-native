"""
Graviton-Native Training Pipeline

Trains BitNet models on real data with HuggingFace datasets.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from graviton_native.models.bitnet import BitNetConfig, BitNetCausalLM


def get_preset_config(size: str) -> BitNetConfig:
    """Model size presets."""
    presets = {
        "350m": BitNetConfig(
            hidden_size=1024,
            intermediate_size=2816,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            vocab_size=32000,
            max_position_embeddings=2048,
        ),
        "1b": BitNetConfig(
            hidden_size=2048,
            intermediate_size=5504,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=4,
            vocab_size=32000,
            max_position_embeddings=2048,
        ),
        "2b": BitNetConfig(
            hidden_size=2560,
            intermediate_size=6912,
            num_hidden_layers=30,
            num_attention_heads=20,
            num_key_value_heads=5,
            vocab_size=128256,
            max_position_embeddings=4096,
        ),
    }
    return presets.get(size, presets["350m"])


def train_bitnet(
    model_size: str = "350m",
    data_path: Optional[str] = None,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    output_dir: str = "./checkpoints",
    steps: int = 1000,
    batch_size: int = 4,
    seq_len: int = 512,
    lr: float = 3e-4,
    save_every: int = 500,
    streaming: bool = False,
    resume: bool = False,
    data_dir: Optional[str] = None,
):
    """
    Train BitNet model on HuggingFace dataset.

    Args:
        model_size: 350m, 1b, or 2b
        data_path: Optional local path (overrides dataset_name)
        dataset_name: HuggingFace dataset (e.g. wikitext, c4)
        dataset_config: Dataset config (e.g. wikitext-2-raw-v1)
        output_dir: Checkpoint directory
        steps: Training steps
        batch_size: Per-device batch size
        seq_len: Sequence length
        lr: Learning rate
        save_every: Save checkpoint every N steps
    """
    from datasets import load_dataset

    config = get_preset_config(model_size)
    model = BitNetCausalLM(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  BitNet {model_size}: {n_params / 1e6:.1f}M parameters")
    print(f"  Ternary: ~{n_params * 1.58 / 8 / 1e6:.1f} MB\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Tokenizer (load first, needed for both paths)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if streaming:
        # Streaming: for huge datasets (the-stack, etc.)
        if data_dir:
            ds = load_dataset(dataset_name, split="train", streaming=True, data_dir=data_dir)
        else:
            ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
        text_key = next((k for k in ["text", "content", "code"] if k in ds.column_names), "code")

        def stream_batches():
            batch_texts = []
            for ex in ds:
                t = ex.get(text_key) or ex.get("code") or ""
                if t and len(t.strip()) > 10:
                    batch_texts.append(t)
                if len(batch_texts) >= batch_size:
                    tok = tokenizer(
                        batch_texts,
                        truncation=True,
                        max_length=seq_len,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    yield {"input_ids": tok["input_ids"], "attention_mask": tok.get("attention_mask")}
                    batch_texts = []
            if batch_texts:
                tok = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=seq_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                yield {"input_ids": tok["input_ids"], "attention_mask": tok.get("attention_mask")}

        loader = stream_batches()
    else:
        # Non-streaming: load full dataset
        if data_path and Path(data_path).exists():
            ds = load_dataset("json", data_files=data_path, split="train")
            text_key = "text" if "text" in ds.column_names else (ds.column_names[0] if ds.column_names else "content")
        else:
            try:
                if dataset_config:
                    ds = load_dataset(dataset_name, dataset_config, split="train")
                else:
                    ds = load_dataset(dataset_name, split="train")
            except Exception:
                ds = load_dataset(dataset_name, split="train")
            text_key = next((k for k in ["text", "content", "code"] if k in ds.column_names), ds.column_names[0])

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
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    output_path = Path(output_dir) / f"bitnet-{model_size}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if requested and exists
    step = 0
    ckpt_file = output_path / "pytorch_model.bin"
    step_file = output_path / "training_step.txt"
    if resume and ckpt_file.exists():
        model.load_state_dict(torch.load(ckpt_file, map_location="cpu"), strict=True)
        model = model.to(device)
        if step_file.exists():
            step = int(step_file.read_text().strip())
            print(f"\n  Resumed from step {step}")
        else:
            step = 0  # checkpoint exists but no step file

    model.train()
    pbar = tqdm(total=steps, initial=step, desc="Training")

    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            ids = batch["input_ids"].to(device)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(device)

            logits = model(ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size),
                ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_token_id or 0,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            step += 1

            if step > 0 and step % save_every == 0:
                torch.save(model.state_dict(), output_path / "pytorch_model.bin")
                (output_path / "training_step.txt").write_text(str(step))
                import json
                cfg_dict = {f: getattr(config, f) for f in config.__dataclass_fields__}
                (output_path / "config.json").write_text(json.dumps(cfg_dict, indent=2))
                tokenizer.save_pretrained(str(output_path))
                print(f"\n  Saved checkpoint at step {step}")

    pbar.close()
    torch.save(model.state_dict(), output_path / "pytorch_model.bin")
    (output_path / "training_step.txt").write_text(str(step))
    import json
    cfg_dict = {f: getattr(config, f) for f in config.__dataclass_fields__}
    (output_path / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    tokenizer.save_pretrained(str(output_path))
    print(f"\n  Done. Model saved to {output_path}")
