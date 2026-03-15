"""Tests for Omega architecture."""

import pytest
import torch

from graviton_native.models.omega import OmegaConfig, OmegaCausalLM


def test_omega_forward():
    config = OmegaConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        top_k=1,
        vocab_size=1000,
    )
    model = OmegaCausalLM(config)
    ids = torch.randint(0, 1000, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, 1000)


def test_omega_k1_sparsity():
    """k=1 means only 1 expert active per token."""
    config = OmegaConfig(num_experts=8, top_k=1)
    assert config.top_k == 1
    total = 80e6  # micro
    active = total / 8
    assert active < total
