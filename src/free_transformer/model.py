"""Decoder-only GPT with an optional Free-Transformer latent path.

``model_type="baseline"`` is a plain pre-norm GPT (RMSNorm / SwiGLU / RoPE /
tied embeddings), byte-identical in architecture to the mla-gpt MHA backbone.

``model_type="free"`` adds, per Fleuret (arXiv:2510.17558):
  * a one-block non-causal ``LatentEncoder`` reading the activations after
    block m = n_layer//2, emitting H bit-logits per token;
  * straight-through Bernoulli sampling of Z during training (posterior) and
    fair coin flips during generation (prior);
  * injection of R = post_sampler(2Z-1) into block m's K/V stream only:
    queries see X, keys and values see X + R;
  * loss = cross-entropy + free-bits KL hinge.

With ``zero_init_post_sampler`` (default) R == 0 at initialization, so the
free model's CE loss starts exactly at the baseline's — a correctness
invariant the tests exploit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FTConfig
from .latent import LatentEncoder, free_bits_penalty, kl_to_uniform, random_bits, sample_bits
from .layers import CausalSelfAttention, RMSNorm, SwiGLU
from .rope import build_rope_cache


class Block(nn.Module):
    def __init__(self, config: FTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config.d_model, config.n_head, config.dropout)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.mlp_hidden_mult, config.dropout)

    def forward(self, x, cos, sin, past_kv=None, use_cache=False, r=None):
        # r: Free-Transformer injection. Queries see the clean stream; only
        # keys/values see X + R, per the paper.
        x_kv = self.norm1(x + r) if r is not None else None
        attn_out, new_cache = self.attn(self.norm1(x), cos, sin, past_kv, use_cache, x_kv=x_kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class FreeTransformer(nn.Module):
    def __init__(self, config: FTConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.is_free = config.model_type == "free"
        self.inject_at = config.resolved_inject_layer

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        if self.is_free:
            self.encoder = LatentEncoder(
                config.d_model, config.n_head, config.mlp_hidden_mult,
                config.latent_bits, config.dropout, config.norm_eps,
            )
            self.post_sampler = nn.Linear(config.latent_bits, config.d_model, bias=False)

        cos, sin = build_rope_cache(config.block_size, config.head_dim, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        if self.is_free and config.zero_init_post_sampler:
            nn.init.zeros_(self.post_sampler.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.config.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def _latent_from_posterior(self, x_mid: torch.Tensor):
        """Encoder -> ST bits -> (R, kl_nats_per_token, bit_logits)."""
        bit_logits = self.encoder(x_mid, self.cos, self.sin)
        z = sample_bits(bit_logits)                       # (B, T, H) in {0,1}, ST grads
        r = self.post_sampler((2.0 * z - 1.0).to(x_mid.dtype))
        return r, kl_to_uniform(bit_logits), bit_logits

    def _latent_from_prior(self, B: int, T: int, device, dtype, z_bits: torch.Tensor | None):
        z = random_bits((B, T, self.config.latent_bits), device) if z_bits is None else z_bits.float()
        return self.post_sampler((2.0 * z - 1.0).to(dtype)), z

    def forward(self, idx, targets=None, z_bits=None):
        """Returns (logits, loss, aux). aux carries ce/kl diagnostics as tensors.

        Training/eval (targets given, free model): Z ~ Q(Z|S) via the encoder.
        If ``z_bits`` is given it overrides the latent (used by probes and
        steering experiments; shape (B, T, H), values in {0,1}).
        """
        B, T = idx.shape
        assert T <= self.config.block_size
        x = self.drop(self.tok_emb(idx))

        aux: dict[str, torch.Tensor] = {}
        r = None
        kl = None
        for i, block in enumerate(self.blocks):
            if self.is_free and i == self.inject_at:
                if z_bits is not None:
                    r, _ = self._latent_from_prior(B, T, x.device, x.dtype, z_bits)
                elif targets is not None:
                    r, kl, _ = self._latent_from_posterior(x)
                else:
                    r, _ = self._latent_from_prior(B, T, x.device, x.dtype, None)
                x, _ = block(x, self.cos, self.sin, r=r)
            else:
                x, _ = block(x, self.cos, self.sin)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            aux["ce"] = ce.detach()
            if kl is not None:
                kl_term = free_bits_penalty(kl, self.config.kappa_bits)
                loss = ce + kl_term
                aux["kl_bits_per_token"] = (kl.detach().mean() / math.log(2.0))
                aux["kl_penalty"] = kl_term.detach()
            else:
                loss = ce
            return logits, loss, aux

        logits = self.lm_head(x[:, -1:, :])
        return logits, None, aux

    @torch.no_grad()
    def encode_latent(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Posterior bit probabilities and hard bits for a batch (probing tool)."""
        assert self.is_free
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            if i == self.inject_at:
                break
            x, _ = block(x, self.cos, self.sin)
        bit_logits = self.encoder(x, self.cos, self.sin)
        p = torch.sigmoid(bit_logits.float())
        return p, (p > 0.5).float()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 z_bits: torch.Tensor | None = None, use_posterior_prefill: bool = True):
        """Autoregressive generation with per-layer KV caches.

        Free model: prompt positions take Z from the posterior (encoder over
        the prompt) when ``use_posterior_prefill``, else from the prior; each
        newly generated position draws Z_t from the prior (fair coins) unless
        ``z_bits`` (B, block_size, H) pins them — the steering knob.
        """
        self.eval()
        B = idx.shape[0]
        caches = [None] * len(self.blocks)
        cur = idx
        past_len = 0
        for step in range(max_new_tokens):
            assert past_len + cur.shape[1] < self.config.block_size
            T_cur = cur.shape[1]
            x = self.drop(self.tok_emb(cur))
            new_caches = []
            for i, block in enumerate(self.blocks):
                r = None
                if self.is_free and i == self.inject_at:
                    if z_bits is not None:
                        z = z_bits[:, past_len:past_len + T_cur, :].float()
                        r = self.post_sampler((2.0 * z - 1.0).to(x.dtype))
                    elif step == 0 and T_cur > 1 and use_posterior_prefill:
                        bit_logits = self.encoder(x, self.cos, self.sin)
                        z = torch.bernoulli(torch.sigmoid(bit_logits.float()))
                        r = self.post_sampler((2.0 * z - 1.0).to(x.dtype))
                    else:
                        r, _ = self._latent_from_prior(B, T_cur, x.device, x.dtype, None)
                x, c = block(x, self.cos, self.sin, past_kv=caches[i], use_cache=True, r=r)
                new_caches.append(c)
            caches = new_caches
            past_len += T_cur
            logits = self.lm_head(self.norm_f(x[:, -1:, :])).squeeze(1) / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            cur = nxt
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=use_fused)
