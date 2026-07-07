"""Rotary positional embeddings (RoPE), GPT-NeoX / Llama "rotate_half" convention.

Identical to the mla-gpt implementation so the backbone stays comparable.
"""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each shape (seq_len, dim), for positions [0, seq_len)."""
    assert dim % 2 == 0, "RoPE dim must be even"
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)               # (seq_len, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)        # (seq_len, dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply RoPE to x of shape (B, n_head, T, dim).

    cos/sin are (S, dim) caches. If `positions` (shape (T,)) is given, the
    corresponding rows are gathered (needed for KV-cached incremental decode);
    otherwise the first T rows are used.
    """
    T = x.shape[-2]
    if positions is None:
        cos_t = cos[:T][None, None, :, :]
        sin_t = sin[:T][None, None, :, :]
    else:
        cos_t = cos[positions]
        sin_t = sin[positions]
        if cos_t.dim() == 2:
            cos_t = cos_t[None, None, :, :]
            sin_t = sin_t[None, None, :, :]
        else:
            cos_t = cos_t[:, None, :, :]
            sin_t = sin_t[:, None, :, :]
    cos_t = cos_t.to(x.dtype)
    sin_t = sin_t.to(x.dtype)
    return (x * cos_t) + (rotate_half(x) * sin_t)
