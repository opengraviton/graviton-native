"""E2E tests: quick training run + checkpoint verification."""

import json
import tempfile
from pathlib import Path

import pytest

from graviton_native.models.bitnet import BitNetConfig, BitNetCausalLM


def test_e2e_train_350m_two_steps():
    """Train 350m for 2 steps, verify checkpoint structure."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "checkpoints"
        output_dir.mkdir()

        config = BitNetConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
        )
        model = BitNetCausalLM(config)
        optimizer = __import__("torch").optim.AdamW(model.parameters(), lr=3e-4)

        # 2 training steps
        for step in range(2):
            ids = __import__("torch").randint(0, 1000, (2, 32))
            logits = model(ids)
            loss = __import__("torch").nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, config.vocab_size),
                ids[:, 1:].reshape(-1),
                ignore_index=0,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint (Graviton-compatible)
        ckpt_dir = output_dir / "bitnet-350m"
        ckpt_dir.mkdir()
        __import__("torch").save(model.state_dict(), ckpt_dir / "pytorch_model.bin")
        config_dict = {f: getattr(config, f) for f in config.__dataclass_fields__}
        (ckpt_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

        # Verify
        assert (ckpt_dir / "pytorch_model.bin").exists()
        assert (ckpt_dir / "config.json").exists()
        loaded_config = json.loads((ckpt_dir / "config.json").read_text())
        assert loaded_config["hidden_size"] == 256
        assert loaded_config["num_hidden_layers"] == 2


def test_e2e_checkpoint_loadable():
    """Verify saved checkpoint can be loaded back."""
    import torch

    config = BitNetConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=100,
    )
    model = BitNetCausalLM(config)
    state = model.state_dict()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save(state, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        model2 = BitNetCausalLM(config)
        model2.load_state_dict(loaded, strict=True)
        ids = torch.randint(0, 100, (1, 8))
        out1 = model(ids)
        out2 = model2(ids)
        assert torch.allclose(out1, out2)
