"""
Disk-Offload Training — 72B on 64 GB Mac

Params, gradients, optimizer states stored on disk. Load one layer at a time.
~15 GB peak RAM. Requires ~500 GB free disk. Optimized with:
  - Activations in RAM (~1.3 GB) — no disk I/O for 80 layer activations
  - Layer prefetching — overlap disk load with compute
  - Background layer save — overlap disk write with next layer load
"""

from __future__ import annotations

import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import psutil
import torch
import torch.nn as nn
from tqdm import tqdm

from graviton_native.models.bitnet import BitNetBlock, BitNetConfig, BitNetCausalLM


def _get_disk_offload_config(model_size: str) -> BitNetConfig:
    """Config for disk-offload: 72b (80 layers) or 36b (40 layers, ~2x faster)."""
    base = BitNetConfig(
        hidden_size=8192,
        intermediate_size=28672,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        vocab_size=32016,
        max_position_embeddings=16384,
        rope_theta=1000000.0,
    )
    if model_size == "36b":
        return BitNetConfig(
            hidden_size=base.hidden_size,
            intermediate_size=base.intermediate_size,
            num_hidden_layers=40,
            num_attention_heads=base.num_attention_heads,
            num_key_value_heads=base.num_key_value_heads,
            vocab_size=base.vocab_size,
            max_position_embeddings=base.max_position_embeddings,
            rope_theta=base.rope_theta,
        )
    return base


# Approx MB per layer (BitNet 1.58-bit): 72b/80 ≈ 0.9B params → ~180 MB
_LAYER_MB = 300  # conservative estimate for 72b layer


def train_disk_offload(
    model_size: str = "72b",
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
    use_compile: bool = True,
    ram_cache_layers: int = 0,
    ram_cache_gb: Optional[float] = None,
):
    """
    Train BitNet with disk offload. 72b (80 layers) or 36b (40 layers, ~2x faster).
    Peak RAM: ~15 GB (ram_cache=0). Disk: ~250 GB (36b) or ~500 GB (72b).
    ram_cache_layers: 0=minimal RAM. 20=cache 20 layers (~6 GB). 80=all layers (~25 GB).
    ram_cache_gb: Alternative — max GB for layer cache. Auto-computes layers (e.g. 25 → all 80 for 72b).
    """
    config = _get_disk_offload_config(model_size)
    if ram_cache_gb is not None and ram_cache_gb > 0:
        ram_cache_layers = min(config.num_hidden_layers, int(ram_cache_gb * 1024 / _LAYER_MB))
    # Cap RAM cache to avoid OOM: leave 8 GB for activations, PyTorch, system
    if ram_cache_layers > 0:
        try:
            avail_gb = psutil.virtual_memory().available / (1024**3)
            max_cache_gb = max(2.0, avail_gb - 8.0)
            max_layers = int(max_cache_gb * 1024 / _LAYER_MB)
            if ram_cache_layers > max_layers:
                ram_cache_layers = max(0, max_layers)
                print(f"  RAM cache capped to {ram_cache_layers} layers (~{ram_cache_layers * _LAYER_MB / 1024:.1f} GB) — {avail_gb:.1f} GB available")
        except Exception:
            pass
    n_params = 36e9 if model_size == "36b" else 72e9
    offload_dir = Path(offload_dir or output_dir) / f"offload_{model_size}"
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
            out_path.glob(f"bitnet-{model_size}-step*"),
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

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  {model_size.upper()} BitNet — Disk Offload Training")
    disk_gb = 250 if model_size == "36b" else 500
    print(f"  Peak RAM: ~15 GB | Disk: ~{disk_gb} GB | Device: {device}")
    if model_size == "36b":
        print(f"  36b = 40 layers (vs 72b/80) — ~2x faster per step.\n")
    else:
        print(f"  WARNING: Very slow (~10-30 min/step). Try --model_size 36b for ~2x speed.\n")

    # Build model layer-by-layer to avoid OOM
    layers_dir = offload_dir / "layers"
    layers_dir.mkdir(exist_ok=True)
    if not (layers_dir / "layer_000.pt").exists():
        print(f"  Initializing {model_size.upper()} model (layer by layer)...")
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
    if use_compile:
        try:
            layer_module = torch.compile(layer_module, mode="reduce-overhead")
            print("  torch.compile: ON (faster compute)")
        except Exception:
            pass
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

    # Prefetch executor: overlap disk I/O with compute
    _exec = ThreadPoolExecutor(max_workers=2)
    cache_size = min(ram_cache_layers, config.num_hidden_layers) if ram_cache_layers > 0 else 0
    if cache_size > 0:
        cache_gb = cache_size * _LAYER_MB / 1024
        print(f"  RAM cache: {cache_size} layers (~{cache_gb:.1f} GB) — less disk I/O")

    def _load_layer(i: int):
        return torch.load(layers_dir / f"layer_{i:03d}.pt", map_location="cpu", weights_only=True)

    def _save_layer(state_dict, i: int):
        torch.save(state_dict, layers_dir / f"layer_{i:03d}.pt")

    # torch.compile wraps module; saved state_dict has original keys — load/save via _orig_mod
    def _layer_for_io(module):
        return getattr(module, "_orig_mod", module)

    def _load_chunk(start: int, end: int) -> dict:
        """Load layers [start, end) into a dict. Uses parallel loads when cache_size > 1."""
        if end - start <= 1:
            return {start: _load_layer(start)} if start < config.num_hidden_layers else {}
        futures = [_exec.submit(_load_layer, i) for i in range(start, min(end, config.num_hidden_layers))]
        return {start + j: f.result() for j, f in enumerate(futures)}

    # Forward: activations kept in RAM (~1.3 GB), layer prefetching overlaps I/O with compute
    def forward(ids):
        embed.load_state_dict(torch.load(offload_dir / "embed.pt", map_location=device, weights_only=True))
        x = embed(ids)
        cos, sin = get_rope(torch.arange(x.size(1), device=device).unsqueeze(0))
        acts = [x.detach().cpu()]
        del x
        gc.collect()

        if cache_size >= config.num_hidden_layers:
            # Load all layers into RAM once — no disk during forward
            layer_cache = _load_chunk(0, config.num_hidden_layers)
        else:
            layer_cache = {}

        next_load = None if cache_size > 0 else _exec.submit(_load_layer, 0)
        for i in range(config.num_hidden_layers):
            if cache_size >= config.num_hidden_layers:
                layer_sd = layer_cache[i]
            elif cache_size > 0:
                chunk_start = (i // cache_size) * cache_size
                if i == chunk_start:
                    layer_cache.clear()
                    layer_cache.update(_load_chunk(chunk_start, chunk_start + cache_size))
                layer_sd = layer_cache[i]
            else:
                layer_sd = next_load.result()
                if i + 1 < config.num_hidden_layers:
                    next_load = _exec.submit(_load_layer, i + 1)
            _layer_for_io(layer_module).load_state_dict(layer_sd)
            x = acts[-1].to(device)
            x = layer_module(x, position_embeddings=(cos, sin))
            acts.append(x.detach().cpu())
            del x, layer_sd
            gc.collect()

        x = acts[-1].to(device)
        norm.load_state_dict(torch.load(offload_dir / "norm.pt", map_location=device, weights_only=True))
        lm_head.load_state_dict(torch.load(offload_dir / "lm_head.pt", map_location=device, weights_only=True))
        x = norm(x)
        logits = lm_head(x)
        return logits, acts, layer_cache if cache_size >= config.num_hidden_layers else None

    def backward_and_optimizer(lr, acts, input_ids_cpu, targets_cpu, layer_cache: Optional[dict] = None):
        """Backward pass + SGD step. Activations from RAM. Prefetch + background save."""
        # Last: norm + lm_head
        act = acts[config.num_hidden_layers].to(device)
        act.requires_grad_(True)
        norm.load_state_dict(torch.load(offload_dir / "norm.pt", map_location=device, weights_only=True))
        lm_head.load_state_dict(torch.load(offload_dir / "lm_head.pt", map_location=device, weights_only=True))
        x = norm(act)
        logits = lm_head(x)
        targets = targets_cpu.to(device)
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

        # Layers 79 down to 0: prefetch next layer, save current in background
        save_future = None
        next_load = None if (layer_cache is not None and len(layer_cache) >= config.num_hidden_layers) else _exec.submit(_load_layer, config.num_hidden_layers - 1)
        backward_chunk = {}
        for idx, i in enumerate(range(config.num_hidden_layers - 1, -1, -1)):
            if save_future is not None:
                save_future.result()
            if layer_cache is not None and len(layer_cache) >= config.num_hidden_layers:
                layer_sd = layer_cache[i]
            elif cache_size > 0:
                chunk_start = (i // cache_size) * cache_size
                if i not in backward_chunk:
                    backward_chunk.clear()
                    backward_chunk.update(_load_chunk(chunk_start, min(chunk_start + cache_size, config.num_hidden_layers)))
                layer_sd = backward_chunk[i]
            else:
                layer_sd = next_load.result()
                if i > 0:
                    next_load = _exec.submit(_load_layer, i - 1)
            _layer_for_io(layer_module).load_state_dict(layer_sd)
            act = acts[i].to(device)
            act.requires_grad_(True)
            cos, sin = get_rope(torch.arange(act.size(1), device=device).unsqueeze(0))
            out = layer_module(act, position_embeddings=(cos, sin))
            grads = torch.autograd.grad(out, [act] + list(layer_module.parameters()), grad_act)
            grad_act = grads[0]
            for p, g in zip(layer_module.parameters(), grads[1:]):
                if g is not None:
                    p.data.sub_(g, alpha=lr)
            updated_sd = _layer_for_io(layer_module).state_dict()
            if layer_cache is not None and len(layer_cache) >= config.num_hidden_layers:
                layer_cache[i] = updated_sd
            else:
                save_future = _exec.submit(_save_layer, updated_sd, i)
            del act, out, grads, layer_sd
            gc.collect()

        if save_future is not None:
            save_future.result()
        if layer_cache is not None and len(layer_cache) >= config.num_hidden_layers:
            for i, sd in layer_cache.items():
                _save_layer(sd, i)

        # Embed
        embed.load_state_dict(torch.load(offload_dir / "embed.pt", map_location=device, weights_only=True))
        ids = input_ids_cpu.to(device)
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
    pbar = tqdm(initial=step, total=steps, desc=f"{model_size.upper()} disk-offload")
    while step < steps:
        try:
            # Persist step BEFORE heavy work so resume survives Ctrl+C mid-step
            _write_step(step)

            ids = get_batch()
            targets_cpu = ids[:, 1:].reshape(-1).cpu()
            input_ids_cpu = ids.cpu()
            torch.save(targets_cpu, acts_dir / "targets.pt")  # for resume/corruption recovery
            torch.save(input_ids_cpu, acts_dir / "input_ids.pt")

            logits, acts, layer_cache = forward(ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size),
                ids[:, 1:].reshape(-1),
                ignore_index=0,
            )
            loss_val = loss.item()
            del logits
            gc.collect()

            backward_and_optimizer(lr, acts, input_ids_cpu, targets_cpu, layer_cache)
            del loss
            gc.collect()

            step += 1
            _write_step(step)
            pbar.update(1)

            if step % 1 == 0:
                pbar.write(f"  step {step}/{steps} loss={loss_val:.4f}")
            if step % save_every == 0:
                ckpt = out_path / f"bitnet-{model_size}-step{step}"
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
                ckpt = out_path / f"bitnet-{model_size}-step{step}"
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


def train_72b_disk_offload(**kwargs):
    """Backward compatibility: alias for train_disk_offload(model_size="72b")."""
    return train_disk_offload(model_size="72b", **kwargs)
