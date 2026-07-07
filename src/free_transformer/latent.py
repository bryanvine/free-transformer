"""The Free Transformer's latent machinery (Fleuret, arXiv:2510.17558).

Z_t is formally a one-hot over 2^H states per token. The approximate posterior
Q(Z_t | S) factorizes over H independent Bernoulli bits, so in practice:

* ``LatentEncoder`` — ONE non-causal transformer block computing Q(Z|S). Its
  queries are a single learned embedding zeta replicated across positions, so a
  position's latent cannot trivially copy that position's content — content
  reaches the latent only through attention over the mid-depth activations,
  and positions are distinguished only via RoPE on the queries. A linear
  readout emits H bit-logits per token.
* ``sample_bits`` — straight-through Bernoulli sampling: hard {0,1} bits
  forward, identity gradient into the probabilities backward.
* ``kl_to_uniform`` — per-token KL(Q || uniform over 2^H) in nats, which
  factorizes to sum_i [log 2 - H_b(p_i)]; ``free_bits_penalty`` applies the
  paper's per-token hinge max(0, KL_t - kappa) (Eq. 5).

Implementation note: the paper's "linear post-sampler" on the 2^16 one-hot is
parameterized here as a linear map on the +-1 bit vector (H -> d_model). A
literal 2^H x d table cannot receive straight-through gradients tractably at
H=16, and the paper's reported ~3% overhead accounts for the encoder block,
not a 268M-parameter table — see paper/RESEARCH_LOG.md for the full argument.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLU
from .rope import apply_rope


def sample_bits(bit_logits: torch.Tensor) -> torch.Tensor:
    """Straight-through Bernoulli sample. Input (..., H) logits, output {0,1} floats."""
    p = torch.sigmoid(bit_logits.float())
    hard = torch.bernoulli(p)
    # parenthesization matters: (p - p.detach()) is exactly 0 elementwise, so
    # the forward value is exactly `hard`; left-to-right (hard + p) - p is not.
    return hard + (p - p.detach())


def random_bits(shape, device) -> torch.Tensor:
    """Prior sample: independent fair coin flips (uniform over the 2^H states)."""
    return torch.randint(0, 2, shape, device=device).float()


def kl_to_uniform(bit_logits: torch.Tensor) -> torch.Tensor:
    """Per-token KL(Q(Z_t|S) || Uniform(2^H)) in nats, shape (..., H) -> (...).

    For a product of Bernoullis, KL to the uniform joint = sum over bits of
    log(2) - H_b(p_i). Computed via logsigmoid for stability.
    """
    l = bit_logits.float()
    p = torch.sigmoid(l)
    ent = -(p * F.logsigmoid(l) + (1.0 - p) * F.logsigmoid(-l))  # binary entropy, nats
    return (math.log(2.0) - ent).sum(dim=-1)


def free_bits_penalty(kl_nats: torch.Tensor, kappa_bits: float) -> torch.Tensor:
    """Paper Eq. 5: mean over tokens of max(0, KL_t - kappa). kappa given in bits."""
    kappa = kappa_bits * math.log(2.0)
    return F.relu(kl_nats - kappa).mean()


class LatentEncoder(nn.Module):
    """One non-causal block: queries from a learned zeta, K/V from mid-depth X."""

    def __init__(self, d_model: int, n_head: int, mlp_hidden_mult: float,
                 latent_bits: int, dropout: float = 0.0, norm_eps: float = 1e-5):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout = dropout

        self.zeta = nn.Parameter(torch.randn(d_model) * 0.02)
        self.norm_q = RMSNorm(d_model, eps=norm_eps)
        self.norm_kv = RMSNorm(d_model, eps=norm_eps)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.mlp = SwiGLU(d_model, mlp_hidden_mult, dropout)
        self.norm_out = RMSNorm(d_model, eps=norm_eps)
        self.readout = nn.Linear(d_model, latent_bits, bias=True)

    def forward(self, x_mid: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x_mid: (B, T, d) mid-depth decoder activations -> (B, T, H) bit logits."""
        B, T, d = x_mid.shape
        e = self.zeta.view(1, 1, d).expand(B, T, d)

        # Queries carry position only (zeta + RoPE); content enters via K/V.
        q = self.q_proj(self.norm_q(e)).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv_in = self.norm_kv(x_mid)
        k = self.k_proj(kv_in).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        dp = self.dropout if self.training else 0.0
        att = F.scaled_dot_product_attention(q, k, v, dropout_p=dp)  # non-causal
        att = att.transpose(1, 2).reshape(B, T, d)

        e = e + self.o_proj(att)
        e = e + self.mlp(self.norm2(e))
        return self.readout(self.norm_out(e))
