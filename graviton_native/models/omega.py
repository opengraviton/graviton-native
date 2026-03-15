"""
Graviton Omega — Ultra-sparse MoE + BitNet.

Hedef: 100B+ efektif kapasite, 8GB RAM'de.
- k=1 routing: Her token sadece 1 expert kullanır
- BitNet ternary: 8x bellek tasarrufu
- 8 expert × 100M = 800M total, 100M active (Omega-Micro)

Ref: docs/OMEGA_ARCHITECTURE.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from graviton_native.models.bitnet import BitLinear


@dataclass
class OmegaConfig:
    """Omega: Ultra-sparse MoE + BitNet."""
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    vocab_size: int = 32000
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    # Omega: k=1, çok expert
    num_experts: int = 8
    top_k: int = 1  # Sadece 1 expert — ultra sparse
    expert_intermediate_ratio: int = 4  # Expert küçük


class Top1Router(nn.Module):
    """k=1 router — her token 1 expert."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        expert_idx = logits.argmax(dim=-1)
        return logits, expert_idx


class OmegaExpert(nn.Module):
    """BitNet FFN expert — ternary weights, ReLU²."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = BitLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = BitLinear(intermediate_size, hidden_size, bias=False)

    def _relu2(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).pow(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self._relu2(self.gate_proj(x)) * self.up_proj(x))


class OmegaMoELayer(nn.Module):
    """k=1 MoE — sadece seçilen expert çalışır."""

    def __init__(self, config: OmegaConfig):
        super().__init__()
        self.router = Top1Router(config.hidden_size, config.num_experts)
        expert_dim = config.intermediate_size // config.expert_intermediate_ratio
        self.experts = nn.ModuleList([
            OmegaExpert(config.hidden_size, expert_dim)
            for _ in range(config.num_experts)
        ])
        self.num_experts = config.num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        _, expert_idx = self.router(x)
        x_flat = x.view(-1, hidden)
        expert_idx_flat = expert_idx.view(-1)

        output = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            mask = (expert_idx_flat == e)
            if mask.any():
                output[mask] = self.experts[e](x_flat[mask])
        return output.view(batch, seq, hidden)


class OmegaBlock(nn.Module):
    """Transformer block: BitNet attention + Omega MoE FFN."""

    def __init__(self, config: OmegaConfig, layer_idx: int):
        super().__init__()
        hidden = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv = config.num_key_value_heads
        head_dim = hidden // num_heads

        self.q_proj = BitLinear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = BitLinear(hidden, num_kv * head_dim, bias=False)
        self.v_proj = BitLinear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = BitLinear(num_heads * head_dim, hidden, bias=False)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv
        self.head_dim = head_dim

        self.moe = OmegaMoELayer(config)
        self.input_layernorm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        residual = x
        x = self.input_layernorm(x)

        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if cos is not None and sin is not None:
            q = (q * cos) + (self._rotate_half(q) * sin)
            k = (k * cos) + (self._rotate_half(k) * sin)

        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if s > 1:
            mask = torch.triu(torch.ones(s, s, device=x.device), diagonal=1).bool()
            attn.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        h = torch.matmul(attn, v)
        h = h.transpose(1, 2).contiguous().view(b, s, -1)
        x = residual + self.o_proj(h)

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.moe(x)
        return x


class OmegaCausalLM(nn.Module):
    """
    Omega: Ultra-sparse MoE + BitNet.

    Omega-Micro: 8 expert × ~100M = 800M total, 100M active
    Bellek: ~20 MB active (1.58-bit)
    """

    def __init__(self, config: OmegaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            OmegaBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False)

        # RoPE
        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (config.rope_theta ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        ))
        self.register_buffer("inv_freq", inv_freq)

    def _get_rope(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        cos, sin = self._get_rope(x.size(1), x.device)
        for layer in self.layers:
            x = layer(x, cos=cos, sin=sin)
        x = self.norm(x)
        return self.lm_head(x)
