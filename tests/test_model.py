"""Correctness tests for the Free Transformer implementation.

The load-bearing one is ``test_free_equals_baseline_at_init``: with the
post-sampler zero-initialized, R == 0 and the free model must produce exactly
the baseline's logits when the shared weights are copied over — this pins the
K/V-only injection wiring.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from free_transformer.config import FTConfig
from free_transformer.latent import kl_to_uniform, sample_bits
from free_transformer.model import FreeTransformer

CFG = dict(vocab_size=256, block_size=64, n_layer=4, n_head=4, d_model=64)


def tiny(model_type="baseline", **kw):
    torch.manual_seed(0)
    return FreeTransformer(FTConfig(model_type=model_type, **CFG, **kw))


def batch(B=2, T=32, vocab=256):
    torch.manual_seed(1)
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    return x, y


def test_baseline_shapes_and_loss():
    m = tiny()
    x, y = batch()
    logits, loss, aux = m(x, y)
    assert logits.shape == (2, 32, 256)
    assert loss.item() > 0
    assert "kl_bits_per_token" not in aux


def test_free_shapes_loss_and_aux():
    m = tiny("free", latent_bits=8, kappa_bits=0.5)
    x, y = batch()
    logits, loss, aux = m(x, y)
    assert logits.shape == (2, 32, 256)
    assert loss.item() > 0
    assert 0.0 <= aux["kl_bits_per_token"].item() <= 8.0
    assert aux["kl_penalty"].item() >= 0.0


def test_kl_bounds():
    logits = torch.randn(4, 16, 8) * 3
    kl = kl_to_uniform(logits)
    assert (kl >= -1e-5).all()
    assert (kl <= 8 * math.log(2) + 1e-5).all()
    # neutral logits -> KL exactly 0
    assert kl_to_uniform(torch.zeros(2, 3, 8)).abs().max().item() < 1e-6


def test_sample_bits_straight_through():
    logits = torch.randn(2, 8, 4, requires_grad=True)
    z = sample_bits(logits)
    assert set(z.detach().unique().tolist()) <= {0.0, 1.0}
    z.sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_free_equals_baseline_at_init():
    """R == 0 at init => identical logits to a baseline sharing the backbone."""
    free = tiny("free")  # zero_init_post_sampler=True by default
    base = FreeTransformer(FTConfig(model_type="baseline", **CFG))
    base.tok_emb.load_state_dict(free.tok_emb.state_dict())
    base.blocks.load_state_dict(free.blocks.state_dict())
    base.norm_f.load_state_dict(free.norm_f.state_dict())
    free.eval(), base.eval()
    x, y = batch()
    with torch.no_grad():
        lf, lossf, _ = free(x, y)
        lb, lossb, _ = base(x, y)
    assert torch.allclose(lf, lb, atol=1e-6)
    assert abs(lossf.item() - lossb.item()) < 1e-6


def test_encoder_receives_gradient():
    m = tiny("free", zero_init_post_sampler=False, kappa_bits=0.0)
    x, y = batch()
    _, loss, _ = m(x, y)
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in m.encoder.parameters() if p.grad is not None]
    assert sum(g > 0 for g in grads) > 0, "no gradient reached the latent encoder"


def test_param_overhead_is_one_block_plus_heads():
    base, free = tiny(), tiny("free")
    extra = free.num_params() - base.num_params()
    d = CFG["d_model"]
    assert extra > 4 * d * d  # at least the encoder attention
    assert extra < base.num_params() * 0.35


def test_generate_both_arms():
    for mt in ("baseline", "free"):
        m = tiny(mt)
        out = m.generate(torch.randint(0, 256, (2, 8)), max_new_tokens=5)
        assert out.shape == (2, 13)


def test_generate_with_pinned_latent():
    m = tiny("free")
    zb = torch.ones(2, CFG["block_size"], 16)
    out = m.generate(torch.randint(0, 256, (2, 8)), max_new_tokens=5, z_bits=zb)
    assert out.shape == (2, 13)


def test_encode_latent_probe():
    m = tiny("free", latent_bits=8)
    x, _ = batch()
    p, bits = m.encode_latent(x)
    assert p.shape == (2, 32, 8) and bits.shape == (2, 32, 8)
    assert ((p >= 0) & (p <= 1)).all()


def test_kv_only_injection_changes_output():
    """With a non-zero post-sampler, pinning different Z must change logits."""
    m = tiny("free", zero_init_post_sampler=False)
    m.eval()
    x, y = batch()
    with torch.no_grad():
        z0 = torch.zeros(2, 32, 16)
        z1 = torch.ones(2, 32, 16)
        l0, _, _ = m(x, y, z_bits=z0)
        l1, _, _ = m(x, y, z_bits=z1)
    assert not torch.allclose(l0, l1)
