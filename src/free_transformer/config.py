"""Model and training configuration.

A single FTConfig drives both arms of the study. The backbone (depth, width,
MLP, normalization, RoPE, attention) is identical across arms; ``model_type``
selects the plain decoder ("baseline") or the Free Transformer ("free"), which
adds a one-block non-causal latent encoder and mid-depth Z injection. The
latent mechanism is the sole independent variable.

The backbone matches mla-gpt (github.com/bryanvine/mla-gpt) so results are
comparable across the two studies.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

ModelType = Literal["baseline", "free"]


@dataclass
class FTConfig:
    # --- vocab / sequence ---
    vocab_size: int = 50304          # GPT-2 BPE (50257) padded to a multiple of 64
    block_size: int = 1024           # max context length

    # --- backbone (held constant across arms) ---
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    mlp_hidden_mult: float = 8 / 3   # SwiGLU expansion (Llama-style); ~4x equiv params
    dropout: float = 0.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

    # --- arm selection ---
    model_type: ModelType = "baseline"

    # --- Free Transformer (Fleuret, arXiv:2510.17558) ---
    # Z_t is formally one-hot over 2^H states per token; the posterior Q(Z_t|S)
    # factorizes over H independent Bernoulli bits, so we sample H bits with a
    # straight-through estimator and inject R = bit_embed(2Z-1) into the K/V
    # stream of block `inject_layer` (queries see X, keys/values see X + R).
    latent_bits: int = 16            # H (paper: 16)
    inject_layer: int = -1           # -1 -> n_layer // 2 (paper: the middle layer)
    kappa_bits: float = 0.5          # free-bits threshold per token, in bits (paper best: 0.5)
    zero_init_post_sampler: bool = True  # R == 0 at init => free CE == baseline CE at init

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        return self.d_model // self.n_head

    @property
    def resolved_inject_layer(self) -> int:
        return self.inject_layer if self.inject_layer >= 0 else self.n_layer // 2

    def validate(self) -> None:
        assert self.model_type in ("baseline", "free"), self.model_type
        assert self.d_model % self.n_head == 0
        if self.model_type == "free":
            assert 1 <= self.latent_bits <= 30
            assert 0 <= self.resolved_inject_layer < self.n_layer
            assert self.kappa_bits >= 0.0

    def to_dict(self) -> dict:
        return asdict(self)
