"""Shared building blocks, held constant across both arms of the study."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope

KVCache = tuple[torch.Tensor, torch.Tensor]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class SwiGLU(nn.Module):
    """Llama-style gated MLP. hidden rounded to a multiple of 64 for kernel efficiency."""

    def __init__(self, d_model: int, hidden_mult: float, dropout: float = 0.0):
        super().__init__()
        hidden = int(hidden_mult * d_model)
        hidden = 64 * ((hidden + 63) // 64)
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


def _causal_mask(q_pos: torch.Tensor, k_len: int, device) -> torch.Tensor:
    """Boolean keep-mask (True = attend) of shape (T_q, k_len) for cached decode."""
    k_pos = torch.arange(k_len, device=device)
    return k_pos[None, :] <= q_pos[:, None]


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention with RoPE and optional KV cache.

    The one non-standard affordance is ``x_kv``: when given, keys and values
    are computed from it instead of from ``x``. The Free Transformer's
    injected block uses this to show queries the clean stream X while keys and
    values see X + R (Fleuret, arXiv:2510.17558, Sec. "the decoder").
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, past_kv: KVCache | None = None,
                use_cache: bool = False, x_kv: torch.Tensor | None = None):
        B, T, _ = x.shape
        src_kv = x if x_kv is None else x_kv
        past_len = 0 if past_kv is None else past_kv[0].shape[2]
        positions = torch.arange(past_len, past_len + T, device=x.device)

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(src_kv).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(src_kv).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_cache = (k, v) if use_cache else None

        dp = self.dropout if self.training else 0.0
        if past_kv is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dp)
        else:
            mask = _causal_mask(positions, k.shape[2], x.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)

        out = out.transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        return self.o_proj(out), new_cache
