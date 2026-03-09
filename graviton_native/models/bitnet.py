"""
BitNet b1.58 — Native 1.58-bit Linear Layer

Weights are ternary {-1, 0, +1}. Matrix multiply = add/subtract only.
Trained from scratch with this quantization — not post-training.

Reference: https://arxiv.org/abs/2402.17764
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BitNetConfig:
    """BitNet model configuration."""
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 30
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    vocab_size: int = 128256
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    # BitNet-specific
    use_ternary_weights: bool = True
    activation: str = "relu2"  # ReLU² as in BitNet paper


class BitLinear(nn.Module):
    """
    BitNet b1.58 Linear Layer — ternary weights {-1, 0, +1}.

    Forward pass uses absmean quantization:
        W_ternary = sign(W) * (|W| > threshold)
        threshold = alpha * mean(|W|)

    Matmul becomes: Y = X @ W_ternary * scale
    No floating-point multiply in the core matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        alpha: float = 0.7,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _quantize_weight(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize to ternary {-1, 0, +1} with per-row scale."""
        w = self.weight
        absmean = w.abs().mean(dim=1, keepdim=True)
        threshold = self.alpha * absmean
        signs = w.sign()
        mask = (w.abs() > threshold).float()
        ternary = (signs * mask).to(w.dtype)
        scale = absmean.squeeze(1)
        return ternary, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Y = (X @ W_ternary^T) * scale

        For ternary weights, this can be implemented as:
        pos = X @ (W == 1).T
        neg = X @ (W == -1).T
        out = (pos - neg) * scale
        """
        w_ternary, scale = self._quantize_weight()
        w_ternary = w_ternary.to(x.device)
        scale = scale.to(x.device)

        # Efficient: separate positive and negative masks
        pos_mask = (w_ternary == 1).float()
        neg_mask = (w_ternary == -1).float()

        # Y = X @ (pos_mask - neg_mask)^T * scale
        out = (
            F.linear(x, pos_mask, None) - F.linear(x, neg_mask, None)
        ) * scale.unsqueeze(0)

        if self.bias is not None:
            out = out + self.bias
        return out


class BitNetBlock(nn.Module):
    """
    Single BitNet transformer block.

    Uses BitLinear for Q/K/V/O and FFN projections.
    ReLU² activation (squared ReLU) as in BitNet paper.
    """

    def __init__(self, config: BitNetConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.intermediate_size = config.intermediate_size

        # Attention
        self.q_proj = BitLinear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = BitLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = BitLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = BitLinear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # FFN (ReLU²)
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size, bias=False)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _relu2(self, x: torch.Tensor) -> torch.Tensor:
        """Squared ReLU: max(0, x)² — BitNet activation."""
        return F.relu(x).pow(2)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        residual = x
        x = self.input_layernorm(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        b, s, _ = x.shape
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Apply RoPE
            q_embed = (q * cos) + (self._rotate_half(q) * sin)
            k_embed = (k * cos) + (self._rotate_half(k) * sin)
            q, k = q_embed, k_embed

        # GQA: repeat kv heads to match q heads
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Simplified attention (no KV cache for training)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        h = torch.matmul(attn, v)
        h = h.transpose(1, 2).contiguous().view(b, s, -1)
        x = residual + self.o_proj(h)

        # FFN with ReLU²
        residual = x
        x = self.post_attention_layernorm(x)
        gate = self._relu2(self.gate_proj(x))
        x = residual + self.down_proj(gate * self.up_proj(x))
        return x

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class BitNetCausalLM(nn.Module):
    """Full BitNet causal LM — embedding + blocks + lm_head."""

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            BitNetBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # RoPE
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.hidden_size // config.num_attention_heads, 2).float() / (config.hidden_size // config.num_attention_heads)))
        self.register_buffer("inv_freq", inv_freq)

    def _get_rope(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.outer(position_ids[0].float(), self.inv_freq.to(position_ids.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        cos, sin = self._get_rope(position_ids)

        for layer in self.layers:
            x = layer(x, position_embeddings=(cos, sin))

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
