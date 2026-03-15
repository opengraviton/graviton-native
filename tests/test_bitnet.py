"""Tests for BitNet model architecture."""

import pytest
import torch

from graviton_native.models.bitnet import (
    BitNetBlock,
    BitNetConfig,
    BitNetCausalLM,
    BitLinear,
)


def test_bitnet_config_defaults():
    config = BitNetConfig()
    assert config.hidden_size == 2560
    assert config.num_hidden_layers == 30
    assert config.use_ternary_weights is True


def test_bitlinear_forward_shape():
    layer = BitLinear(256, 512)
    x = torch.randn(2, 10, 256)
    out = layer(x)
    assert out.shape == (2, 10, 512)


def test_bitlinear_ternary_output():
    """Ternary weights should produce bounded outputs (add/subtract only)."""
    layer = BitLinear(64, 32)
    x = torch.randn(1, 5, 64)
    out = layer(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_bitnet_block_forward():
    config = BitNetConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    block = BitNetBlock(config, 0)
    x = torch.randn(2, 8, 256)
    # RoPE cos/sin: (1, 1, seq_len, head_dim)
    head_dim = 256 // 4
    cos = torch.randn(1, 1, 8, head_dim)
    sin = torch.randn(1, 1, 8, head_dim)
    out = block(x, position_embeddings=(cos, sin))
    assert out.shape == x.shape


def test_bitnet_causal_lm_forward():
    config = BitNetConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
    )
    model = BitNetCausalLM(config)
    ids = torch.randint(0, 1000, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, 1000)


def test_bitnet_350m_param_count():
    config = BitNetConfig(
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=32000,
    )
    model = BitNetCausalLM(config)
    n = sum(p.numel() for p in model.parameters())
    assert 300e6 < n < 400e6  # ~350M


def test_bitnet_gradient_flow():
    config = BitNetConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=100,
    )
    model = BitNetCausalLM(config)
    ids = torch.randint(0, 100, (1, 8))
    logits = model(ids)
    loss = logits[:, :-1].reshape(-1, 100).sum()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any()
