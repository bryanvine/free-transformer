# free-transformer

A **controlled small-scale study** of [**The Free Transformer**](https://arxiv.org/abs/2510.17558)
(François Fleuret, FAIR, Oct 2025) — a decoder-only GPT that *decides before it
speaks*: a VAE-style latent variable **Z**, learned without supervision, is
injected mid-decoder so the model can commit to global properties of its output
before generating a single token.

The paper reports gains on reasoning/code benchmarks at 1.5B and 8B params.
As of July 2026 — nine months and much discussion later — **we could find no
public independent replication at any scale.** This repo is that replication,
plus the small-scale science the paper leaves open:

> **RQ1 — Does it replicate?** At 124M params under a matched token budget,
> does the Free Transformer beat (a) a params-and-data-matched plain GPT and
> (b) a **params-matched 13-layer GPT** (the control the paper doesn't run —
> is it the latent, or just the extra block?) — measured across **multiple
> seeds with error bars**, not N=1?
>
> **RQ2 — What does Z learn?** Probe the 16 bits: do they encode topic,
> style, length, sentiment? Can you **steer** generation by pinning Z?
>
> **RQ3 — Where does it break?** Map the free-bits threshold κ (the paper's
> collapse knob) and latent width H at small scale.

## Why this is interesting

Every token an ordinary GPT emits is a fresh roll of the dice; consistency of
"who is speaking / where this is going" exists only implicitly. The Free
Transformer makes that decision explicit, cheap (~3% overhead), and — if the
paper's story survives contact with independent hardware — steerable. Honest
negative results will be reported as plainly as positive ones.

## Design (paper-faithful, deviations documented)

- **Backbone**: pre-norm GPT — RMSNorm, SwiGLU, RoPE, tied embeddings, GPT-2
  BPE — identical to [mla-gpt](https://github.com/bryanvine/mla-gpt), so
  results are comparable across studies. `model_type` (baseline/free) is the
  only independent variable.
- **Latent**: H=16 Bernoulli bits per token (formally one-hot over 2^16),
  straight-through sampling, KL to the uniform prior with the paper's
  free-bits hinge `max(0, KL_t − κ)`, κ = 0.5 bits.
- **Encoder**: one non-causal block; queries are a learned embedding ζ (+RoPE),
  keys/values read the mid-depth activations; 16-logit readout.
- **Injection at layer L/2**: queries see X, **keys/values see X + R** — the
  paper's asymmetric wiring, pinned by a unit test
  (`test_free_equals_baseline_at_init`).
- Deviations & justifications live in [`paper/RESEARCH_LOG.md`](paper/RESEARCH_LOG.md).

## Hardware

Runs on one consumer GPU. Verified backends:

| Machine | GPU | Stack |
|---|---|---|
| dev box | RTX 5060 8GB (Blackwell) | torch 2.11 + cu130 |
| b70 | Intel Arc Pro B70 32GB (Battlemage) | torch 2.11 + xpu |

The training loop is vendor-neutral (`device: auto`); as far as we can tell
the Arc runs here will be among the first documented LLM *pretraining* runs
on Intel Arc B-series hardware.

## Quickstart

```bash
python3 -m venv .venv --system-site-packages   # inherit the working CUDA/XPU torch
.venv/bin/pip install -e ".[dev]"

.venv/bin/pytest tests/ -v                     # correctness suite
.venv/bin/python scripts/prepare_tinystories.py
.venv/bin/python scripts/train.py configs/smoke.yaml            # 60-second check
.venv/bin/python scripts/train.py configs/dev_tinystories.yaml \
    --set model.model_type=free train.out_dir=runs/dev_free_s1 train.seed=1
```

## Layout

```
configs/           # smoke / dev (~51M, TinyStories) / headline (124M, FineWeb-Edu)
src/free_transformer/
  config.py        # FTConfig — one config, two arms
  layers.py        # RMSNorm, SwiGLU, causal attention (with the K/V-injection hook)
  latent.py        # encoder, straight-through bits, KL + free-bits hinge
  model.py         # FreeTransformer (baseline | free), generation w/ prior or pinned Z
  training.py      # single-GPU loop, CUDA/XPU, CE+KL logging
  data.py          # GPT-2 BPE -> uint16 .bin (mla-gpt convention)
scripts/           # train / prepare_tinystories / prepare_fineweb
tests/             # incl. the init-equivalence invariant
paper/             # RESEARCH_LOG.md (live) -> eventual paper
```

## Status

**Phase 0 complete (2026-07-07)** — scaffold, 11/11 correctness tests, smoke
training verified on **both** backends the same day: RTX 5060 (124M free
model under torch.compile, 4.8 GiB peak) and Arc Pro B70 (identical code via
`device: auto`; as far as public record shows, the first documented LLM
pretraining steps on Arc B-series silicon). Cross-backend val loss agrees to
~3 decimals at matched seed. Next: the dev-scale κ×H sweep. Follow the
[research log](paper/RESEARCH_LOG.md); results tables land here as runs
complete.

## Related work by the same author

- [mla-gpt](https://github.com/bryanvine/mla-gpt) — *Attention, Controlled*:
  MHA/MQA/GQA/MLA at 124M, same backbone and methodology.
- [turboquant-xpu](https://github.com/bryanvine/turboquant-xpu) — KV-cache
  quantization on Intel Arc Pro B70.

## License

MIT. The Free Transformer is Fleuret's idea (arXiv:2510.17558); this is an
independent implementation and study.
