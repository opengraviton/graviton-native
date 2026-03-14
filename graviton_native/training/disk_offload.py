"""
Disk-Offload Training — 72B on 64 GB Mac

Params, gradients, optimizer states stored on disk. Load one layer at a time.
~15 GB peak RAM. Requires ~500 GB free disk. SLOW but works.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from graviton_native.models.bitnet import BitNetBlock, BitNetConfig, BitNetCausalLM


def train_72b_disk_offload(
    output_dir: str = "./checkpoints",
    offload_dir: Optional[str] = None,
    steps: int = 100,
    batch_size: int = 1,
    seq_len: int = 256,
    lr: float = 1e-5,
    save_every: int = 10,
    data_path: Optional[str] = None,
    dataset_name: str = "smangrul/hug_stack",
    resume: bool = True,
):
    """
    Train 72B BitNet with disk offload. One layer in RAM at a time.
    Peak RAM: ~15 GB. Disk: ~500 GB. Very slow (~hours per step).
    """
    offload_dir = Path(offload_dir or output_dir) / "offload_72b"
    out_path = Path(output_dir)
    offload_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load step count or restore from latest checkpoint
    start_step = 0
    step_file = offload_dir / "step.txt"
    step_file_tmp = offload_dir / "step.txt.tmp"
    layers_dir = offload_dir / "layers"

    def _write_step(s: int):
        """Atomic write of step number."""
        step_file_tmp.write_text(str(s))
        step_file_tmp.replace(step_file)

    def _latest_ckpt(max_step: Optional[int] = None):
        ckpts = sorted(
            out_path.glob("bitnet-72b-step*"),
            key=lambda p: int(p.name.split("step")[1]) if "step" in p.name else 0,
        )
        if not ckpts:
            return None
        if max_step is not None:
            valid = [p for p in ckpts if int(p.name.split("step")[1] if "step" in p.name else 0) <= max_step]
            return valid[-1] if valid else None
        return ckpts[-1]

    if resume:
        if step_file.exists():
            try:
                start_step = int(step_file.read_text().strip())
                print(f"  Resuming from step {start_step}")
            except (ValueError, OSError):
                start_step = 0
                print("  step.txt invalid, will infer from checkpoint")
        elif not (layers_dir / "layer_000.pt").exists():
            latest = _latest_ckpt()
            if latest:
                start_step = int(latest.name.split("step")[1])
                print(f"  Restoring from {latest} (step {start_step})")
                for f in ["embed.pt", "norm.pt", "lm_head.pt", "inv_freq.pt"]:
                    src = latest / f
                    if src.exists():
                        (offload_dir / f).write_bytes(src.read_bytes())
                layers_dir.mkdir(exist_ok=True)
                for p in (latest / "layers").glob("layer_*.pt"):
                    (layers_dir / p.name).write_bytes(p.read_bytes())
                _write_step(start_step)
        else:
            # layers exist but no step.txt — infer from latest checkpoint
            latest = _latest_ckpt()
            if latest:
                start_step = int(latest.name.split("step")[1])
                print(f"  Resuming from step {start_step} (inferred from {latest.name})")

    config = BitNetConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        vocab_size=32016,
        max_position_embeddings=16384,
        rope_theta=1000000.0,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    n_params = 72e9
    print(f"\n  72B BitNet — Disk Offload Training")
    print(f"  Peak RAM: ~15 GB | Disk: ~500 GB | Device: {device}")
    print(f"  WARNING: Very slow (~10-30 min/step). For speed use cloud.\n")

    # Build model layer-by-layer to avoid OOM (72B = ~144 GB)
    layers_dir = offload_dir / "layers"
    layers_dir.mkdir(exist_ok=True)
    if not (layers_dir / "layer_000.pt").exists():
        print("  Initializing 72B model (layer by layer)...")
        for i in range(config.num_hidden_layers):
            layer = BitNetBlock(config, i)
            torch.save(layer.state_dict(), layers_dir / f"layer_{i:03d}.pt")
            del layer
            gc.collect()
        embed = nn.Embedding(config.vocab_size, config.hidden_size)
        norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.hidden_size // config.num_attention_heads, 2).float() / (config.hidden_size // config.num_attention_heads)))
        torch.save(embed.state_dict(), offload_dir / "embed.pt")
        torch.save(norm.state_dict(), offload_dir / "norm.pt")
        torch.save(lm_head.state_dict(), offload_dir / "lm_head.pt")
        torch.save(inv_freq, offload_dir / "inv_freq.pt")
        del embed, norm, lm_head
        gc.collect()
    (offload_dir / "config.json").write_text(json.dumps({f: getattr(config, f) for f in config.__dataclass_fields__}, indent=2))

    # Ensure embed/norm/lm_head/inv_freq exist (e.g. init was interrupted after layers)
    for name in ["embed.pt", "norm.pt", "lm_head.pt", "inv_freq.pt"]:
        if not (offload_dir / name).exists():
            print(f"  Creating missing {name}...")
            embed = nn.Embedding(config.vocab_size, config.hidden_size)
            norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.hidden_size // config.num_attention_heads, 2).float() / (config.hidden_size // config.num_attention_heads)))
            torch.save(embed.state_dict(), offload_dir / "embed.pt")
            torch.save(norm.state_dict(), offload_dir / "norm.pt")
            torch.save(lm_head.state_dict(), offload_dir / "lm_head.pt")
            torch.save(inv_freq, offload_dir / "inv_freq.pt")
            del embed, norm, lm_head
            gc.collect()
            break

    def _restore_from_ckpt(latest):
        """Restore offload_dir from checkpoint."""
        for f in ["embed.pt", "norm.pt", "lm_head.pt", "inv_freq.pt"]:
            src = latest / f
            if src.exists():
                (offload_dir / f).write_bytes(src.read_bytes())
        for p in (latest / "layers").glob("layer_*.pt"):
            (layers_dir / p.name).write_bytes(p.read_bytes())
        return int(latest.name.split("step")[1])

    # Verify layer 0 (quick sanity check)
    def _verify_layers():
        try:
            torch.load(layers_dir / "layer_000.pt", map_location="cpu")
            return True
        except Exception:
            return False

    if resume and not _verify_layers():
        # Prefer checkpoint <= step.txt to avoid jumping backward
        step_from_file = None
        if step_file.exists():
            try:
                step_from_file = int(step_file.read_text().strip())
            except (ValueError, OSError):
                pass
        latest = _latest_ckpt(max_step=step_from_file) if step_from_file is not None else _latest_ckpt()
        if latest:
            print(f"  Corrupted layers detected. Restoring from {latest.name}...")
            start_step = _restore_from_ckpt(latest)
            _write_step(start_step)
            print(f"  Restored to step {start_step}")
        else:
            print("  WARNING: Corrupted layers and no checkpoint. Re-initializing from scratch.")
            for p in layers_dir.glob("layer_*.pt"):
                p.unlink()
            start_step = 0
            _write_step(0)
            for i in range(config.num_hidden_layers):
                layer = BitNetBlock(config, i)
                torch.save(layer.state_dict(), layers_dir / f"layer_{i:03d}.pt")
                del layer
                gc.collect()

    # Load data
    from datasets import load_dataset
    try:
        tok = __import__("transformers").AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    except Exception:
        tok = __import__("transformers").AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if data_path and Path(data_path).exists():
        ds = load_dataset("json", data_files=data_path, split="train")
    else:
        ds = load_dataset(dataset_name, split="train")
    text_key = "content" if "content" in ds.column_names else ds.column_names[0]

    def get_batch():
        texts = [ds[i % len(ds)][text_key] for i in range(batch_size)]
        out = tok(texts, truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt")
        return out["input_ids"].to(device)

    # Optimizer state on disk (per-layer)
    opt_dir = offload_dir / "optimizer"
    opt_dir.mkdir(exist_ok=True)

    # Recreate single layer for compute
    layer_module = BitNetBlock(config, 0).to(device)
    embed = nn.Embedding(config.vocab_size, config.hidden_size).to(device)
    norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device)
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device)

    inv_freq = torch.load(offload_dir / "inv_freq.pt", map_location=device)

    def get_rope(position_ids):
        freqs = torch.outer(position_ids[0].float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

    acts_dir = offload_dir / "activations"
    grads_dir = offload_dir / "gradients"
    acts_dir.mkdir(exist_ok=True)
    grads_dir.mkdir(exist_ok=True)

    # Forward: layer by layer, save activations to disk
    def forward(ids):
        embed.load_state_dict(torch.load(offload_dir / "embed.pt", map_location=device))
        x = embed(ids)
        cos, sin = get_rope(torch.arange(x.size(1), device=device).unsqueeze(0))
        torch.save(x.detach().cpu(), acts_dir / "act_00.pt")
        del x
        gc.collect()

        for i in range(config.num_hidden_layers):
            x = torch.load(acts_dir / f"act_{i:02d}.pt", map_location=device)
            layer_module.load_state_dict(torch.load(layers_dir / f"layer_{i:03d}.pt", map_location=device))
            x = layer_module(x, position_embeddings=(cos, sin))
            torch.save(x.detach().cpu(), acts_dir / f"act_{i+1:02d}.pt")
            del x
            gc.collect()

        x = torch.load(acts_dir / f"act_{config.num_hidden_layers:02d}.pt", map_location=device)
        norm.load_state_dict(torch.load(offload_dir / "norm.pt", map_location=device))
        lm_head.load_state_dict(torch.load(offload_dir / "lm_head.pt", map_location=device))
        x = norm(x)
        logits = lm_head(x)
        return logits

    def backward_and_optimizer(lr):
        """Backward pass + SGD step, one layer at a time."""
        # Last: norm + lm_head
        act = torch.load(acts_dir / f"act_{config.num_hidden_layers:02d}.pt", map_location=device)
        act.requires_grad_(True)
        norm.load_state_dict(torch.load(offload_dir / "norm.pt", map_location=device))
        lm_head.load_state_dict(torch.load(offload_dir / "lm_head.pt", map_location=device))
        x = norm(act)
        logits = lm_head(x)
        targets = torch.load(acts_dir / "targets.pt", map_location=device)
        l = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size), targets, ignore_index=0)
        grads = torch.autograd.grad(l, [act] + list(norm.parameters()) + list(lm_head.parameters()))
        grad_act = grads[0]
        for p, g in zip(list(norm.parameters()) + list(lm_head.parameters()), grads[1:]):
            if g is not None:
                p.data.sub_(g, alpha=lr)
        torch.save(norm.state_dict(), offload_dir / "norm.pt")
        torch.save(lm_head.state_dict(), offload_dir / "lm_head.pt")
        del act, x, logits, l, grads
        gc.collect()

        # Layers 79 down to 0
        for i in range(config.num_hidden_layers - 1, -1, -1):
            act = torch.load(acts_dir / f"act_{i:02d}.pt", map_location=device)
            act.requires_grad_(True)
            layer_module.load_state_dict(torch.load(layers_dir / f"layer_{i:03d}.pt", map_location=device))
            cos, sin = get_rope(torch.arange(act.size(1), device=device).unsqueeze(0))
            out = layer_module(act, position_embeddings=(cos, sin))
            grad_out = grad_act
            grads = torch.autograd.grad(out, [act] + list(layer_module.parameters()), grad_out)
            grad_act = grads[0]
            param_grads = grads[1:]
            # SGD step
            for p, g in zip(layer_module.parameters(), param_grads):
                if g is not None:
                    p.data.sub_(g, alpha=lr)
            torch.save(layer_module.state_dict(), layers_dir / f"layer_{i:03d}.pt")
            del act, out, grads, param_grads
            gc.collect()

        # Embed
        embed.load_state_dict(torch.load(offload_dir / "embed.pt", map_location=device))
        ids = torch.load(acts_dir / "input_ids.pt", map_location=device)
        x = embed(ids)
        ge = torch.autograd.grad(x, list(embed.parameters()), grad_act)
        for p, g in zip(embed.parameters(), ge):
            if g is not None:
                p.data.sub_(g, alpha=lr)
        torch.save(embed.state_dict(), offload_dir / "embed.pt")
        del grad_act, x, ge
        gc.collect()

    # Training loop (resume from start_step)
    total_to_do = steps - start_step
    if total_to_do <= 0:
        print(f"  Already at step {start_step}. Nothing to do.")
        return str(offload_dir)

    step = start_step
    pbar = tqdm(initial=step, total=steps, desc="72B disk-offload")
    while step < steps:
        try:
            # Persist step BEFORE heavy work so resume survives Ctrl+C mid-step
            _write_step(step)

            ids = get_batch()
            torch.save(ids[:, 1:].reshape(-1).cpu(), acts_dir / "targets.pt")
            torch.save(ids.cpu(), acts_dir / "input_ids.pt")

            logits = forward(ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size),
                ids[:, 1:].reshape(-1),
                ignore_index=0,
            )
            loss_val = loss.item()
            del logits
            gc.collect()

            backward_and_optimizer(lr)
            del loss
            gc.collect()

            step += 1
            _write_step(step)
            pbar.update(1)

            if step % 1 == 0:
                pbar.write(f"  step {step}/{steps} loss={loss_val:.4f}")
            if step % save_every == 0:
                ckpt = out_path / f"bitnet-72b-step{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                for f in ["embed.pt", "norm.pt", "lm_head.pt", "inv_freq.pt", "config.json"]:
                    if (offload_dir / f).exists():
                        (ckpt / f).write_bytes((offload_dir / f).read_bytes())
                (ckpt / "layers").mkdir(exist_ok=True)
                for p in layers_dir.glob("layer_*.pt"):
                    (ckpt / "layers" / p.name).write_bytes(p.read_bytes())
                pbar.write(f"  Saved {ckpt}")
            # Light checkpoint every 10 steps for disk offload (more restore points)
            elif step % 10 == 0:
                ckpt = out_path / f"bitnet-72b-step{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                for f in ["embed.pt", "norm.pt", "lm_head.pt", "inv_freq.pt", "config.json"]:
                    if (offload_dir / f).exists():
                        (ckpt / f).write_bytes((offload_dir / f).read_bytes())
                (ckpt / "layers").mkdir(exist_ok=True)
                for p in layers_dir.glob("layer_*.pt"):
                    (ckpt / "layers" / p.name).write_bytes(p.read_bytes())
        except RuntimeError as e:
            if "PytorchStreamReader" in str(e) or "file not found" in str(e).lower():
                step_from_file = None
                if step_file.exists():
                    try:
                        step_from_file = int(step_file.read_text().strip())
                    except (ValueError, OSError):
                        pass
                latest = _latest_ckpt(max_step=step_from_file) if step_from_file is not None else _latest_ckpt()
                if latest:
                    pbar.write(f"  Corrupted layer. Restoring from {latest.name}...")
                    step = _restore_from_ckpt(latest)
                    _write_step(step)
                    pbar.n = step
                    pbar.write(f"  Restored to step {step}. Retrying.")
                else:
                    pbar.write("  No checkpoint to restore. Re-initializing layers...")
                    for p in layers_dir.glob("layer_*.pt"):
                        p.unlink()
                    for i in range(config.num_hidden_layers):
                        layer = BitNetBlock(config, i)
                        torch.save(layer.state_dict(), layers_dir / f"layer_{i:03d}.pt")
                        del layer
                        gc.collect()
                    step = 0
                    _write_step(0)
                    pbar.write("  Re-initialized. Retrying from step 0.")
            else:
                raise

    print(f"\n  Done. Checkpoints in {output_dir}")
    return str(offload_dir)
