# Research Log — free-transformer

A controlled small-scale study of The Free Transformer (Fleuret,
arXiv:2510.17558). Live document; newest entries at the bottom. Written for
future-us and for readers of the eventual paper: every design decision,
deviation, and result — including the ones that don't work — gets recorded
here with dates.

---

## 2026-07-07 — Phase 0: topic selection, design decisions, scaffold

### Why this topic

A six-report survey of the Jan–Jul 2026 frontier (architecture, training
efficiency, indie-research landscape, consumer-hardware feasibility, data/
sample-efficiency, speedrun ecosystem) surfaced the Free Transformer as the
highest novelty-per-effort target: a famous, simple, load-bearing idea from a
frontier lab with — as far as extensive searching can establish — **zero
public independent replications nine months after publication**. It fits this
lab's brand (single-variable controlled studies, honest negatives, consumer
hardware) and this lab's existing mla-gpt backbone directly.

### Research questions

1. **RQ1 (replication):** at 124M/matched tokens, free vs baseline vs
   params-matched 13L baseline, N≥3 seeds, error bars. The 13L control is our
   addition: the encoder block is ~+6% params at this scale, so "is it the
   latent or the extra block?" is a live confound the paper's scale made
   negligible.
2. **RQ2 (what is Z):** bit-level probes (topic/style/length/sentiment),
   steering by pinning Z during generation, per-Z sample diversity.
3. **RQ3 (stability map):** κ ∈ {1/8, 1/2, 1, 2, 4} bits × H ∈ {4, 16} at dev
   scale; where does posterior collapse / KL-crash happen at 51M?

### Implementation decisions (and why)

- **Factored bit-embedding, not a 2^16 one-hot table.** The paper formalizes
  Z_t as one-hot over 2^16 with a "linear post-sampler". A literal 2^16×d
  table (a) cannot receive straight-through gradients tractably — ST works at
  the bit level; a hard joint index would need REINFORCE/Gumbel over 65,536
  categories or a (B,T,65536) soft one-hot, both absurd here — and (b) is
  ruled out by the paper's own overhead numbers: at 8B, "3.1%" matches one
  extra Llama-style block (~220M params), not a 268M-param table on top of
  it. So: R = W·(2Z−1), W ∈ R^{d×H}. Flagged for verification against the
  paper's pseudocode when we do the close read for the writeup.
- **±1 bit encoding** into the post-sampler (not {0,1}): under the uniform
  prior E[2Z−1]=0, so random Z injects zero-mean signal.
- **Zero-init post-sampler**: R=0 at init ⇒ the free model's CE starts
  exactly at the baseline's (unit-tested equivalence). Mirrors the backbone's
  zero-ish residual-projection init philosophy and removes early-training
  shock from random Z.
- **K/V-only injection** at block L/2: queries see X, keys/values see X+R,
  exactly per the paper's stated wiring. Unit test pins it.
- **Encoder queries = ζ + RoPE.** All positions share the learned query
  embedding; position identity enters only via RoPE on Q — the paper's
  "prevents token-wise mapping" device. Content reaches the latent only
  through attention values over mid-depth activations.
- **Posterior-prefill at generation:** conditioning text gets Z ~ Q(Z|prompt)
  (sampled, matching training); each newly generated position draws Z from
  the prior. `z_bits` overrides everything (the steering knob).
- **KL in nats internally, reported in bits/token.** Free-bits hinge is the
  paper's Eq. 5: mean over tokens of max(0, KL_t − κ).
- **Known paper pathologies to instrument, not discover twice:** KL rapidly
  falling under κ and staying there; unstable performance curves (encoder/
  decoder coupling); collapse at κ=4 bits. metrics.csv logs val CE and KL
  (bits/token) separately at every eval.

### Experiment plan

| Phase | What | Where | Status |
|---|---|---|---|
| 0 | scaffold, tests, smoke (this entry) | 5060 | done when pushed |
| 1 | dev sweep: {baseline, free×κ∈{.125,.5,1,2,4}} × 3 seeds, 51M/TinyStories | 5060 + 4080S | next |
| 1b | same smoke + 1 dev run on Arc B70 (first documented Arc pretraining) | B70 | next |
| 2 | headline: {baseline-12L, baseline-13L, free} × 3 seeds, 124M/FineWeb-Edu ~2.5B tok | 4080S + B70 + weekend RTX 6000 Pro; PACE-ICE if needed | pending |
| 3 | RQ2 probes + steering demos on the best free checkpoint | 5060 | pending |
| 4 | writeup: paper page + LinkedIn series | — | pending |

Rigor standards for every reported comparison: ≥3 seeds, mean ± range,
identical data order where the harness allows, config + commit hash + seed in
every results row, negative results reported.

### Hardware notes

- 5060 (8GB, cu130): primary dev; batch 12 × accum 40 at 1024 ctx planned for
  124M — verify in smoke.
- Arc Pro B70 (32GB, xpu): torch 2.11.0+xpu confirmed working inside the
  `vllm-xpu` container (needs `source /opt/intel/oneapi/setvars.sh`). Known
  XPU caveats to expect: fatal (non-recoverable) OOM errors, possible Triton
  autotune issues under torch.compile — fall back to eager if needed.
- RTX 6000 Pro 96GB: weekends only — reserve for the 3-seed headline batch.

### Param audit (verified on scaffold day)

| Scale | baseline | free | baseline+1L |
|---|---|---|---|
| dev (8L/512d) | 51,454,464 | 54,684,688 (+6.28%) | 54,666,752 (+6.24%) |
| headline (12L/768d) | 123,587,328 | 130,693,648 (+5.75%) | 130,666,752 (+5.73%) |

The +1-layer baseline params-matches the free model to within 0.02% at both
scales — a clean "latent vs. extra depth" control. The 12L baseline is
byte-identical in param count to mla-gpt's MHA arm (123,587,328), so results
plot on the same axes as that study.

### Phase 0 smoke results (same day)

- **Tests**: 11/11 pass. The suite caught one real bug during development: the
  straight-through estimator must be written `hard + (p - p.detach())` —
  left-to-right `(hard + p) - p.detach()` returns values like 0.99999994
  instead of exact bits (float associativity).
- **RTX 5060 (cu130), 17M smoke**: loss 10.84→4.73 in 100 iters, 64k tok/s
  eager. KL trace already shows the paper's pathology in miniature: 0.26b →
  0.48b → 0.02b within 100 iters — the collapse instrument works.
- **RTX 5060, 124M free w/ torch.compile**: peak 4.78 GiB at batch 4×1024ctx
  (with an unrelated 1.5 GiB llama-server resident). batch 8-12 OOMs on the
  full-vocab logits/backward; fix queued: chunked cross-entropy. Free-bits
  arithmetic verified in the wild: at init KL=0.748b > κ=0.5b and
  loss−CE = (0.748−0.5)·ln2 exactly.
- **Arc Pro B70 (torch 2.11+xpu, throwaway `intel/vllm:latest` container),
  17M smoke**: identical code path via `device: auto`, loss 10.84→4.731
  (CUDA got 4.732, same seed — cross-backend agreement to ~3 decimals),
  13.8k tok/s eager, peak 2.56 GiB / 32 GiB. Per the Phase-0 survey, no
  public record of LLM pretraining on Arc B-series exists — these are
  plausibly the first documented Arc Pro B70 pretraining steps. Container
  recipe: `docker run --rm --device /dev/dri --group-add 992 --group-add 44
  -v ~/free-transformer:/work -w /work intel/vllm:latest` + `source
  /opt/intel/oneapi/setvars.sh`.
- 124M on the 5060 at ~45k tok/s (steady-state est.) ⇒ a 2.5B-token
  milestone ≈ 15h/run; the κ sweep belongs at dev scale, headline seeds on
  the 4080S / B70 / weekend RTX 6000 Pro.

### Risks

1. **Effect invisible at 124M on val loss.** Mitigation: the paper's gains
   were on downstream/generative tasks, not raw ppl — RQ2's probes and
   structured-generation evals are first-class, and "no effect at small
   scale, here's the noise floor" is a publishable finding (cf. the
   modifications-don't-transfer literature, arXiv:2605.20798).
2. **VAE fiddliness** (collapse, dead bits): κ/H sweep is Phase 1, not an
   afterthought; KL trace logged from day one.
3. **Attribution uncertainty in implementation details**: the arXiv HTML
   extraction may miss appendix specifics; a close PDF read happens before
   the writeup and any deviation found gets an errata entry here.

---

## 2026-07-07 (later) — Phase 1 first results: the ELBO leak, and Arc contention

### The headline lesson: "val loss" is not comparable across arms

All 12 CUDA dev runs (51M, TinyStories, 131M tokens, ~31 min each on the 5060)
completed cleanly. The logged val losses *looked* sensational for the free
arm — κ=4 reached 0.61 vs baseline 1.51 — but this is the **posterior
evaluation leak**: `model(x, y)` computes Z from the encoder, which reads the
full sequence, so CE-with-posterior-Z is an ELBO-like quantity, not an LM
loss. CPU spot-check (80 val windows, identical for all models):

| run | CE posterior Z | CE prior Z | KL used |
|---|---|---|---|
| baseline s1 | 1.231 | — | — |
| free κ=0.5 s1 | 1.073 | 1.519 | 0.481 b/tok |
| free κ=4 s1 | 0.497 | **3.417** | 3.984 b/tok |

κ=4 is an autoencoder: it routes its full ~4-bit budget through Z and
collapses without the encoder (prior CE 2.8× worse than baseline). This is
the paper's "cross-entropy collapse" quantified from the generative side.
Even κ=0.5 pays: single-sample prior CE 1.52 vs baseline 1.23 — but
single-sample prior CE is only an upper bound on NLL (Jensen). Proper
comparison = K-sample importance-weighted bound, now in
`scripts/eval_prior.py` (`--iwae-k`); full-grid GPU eval runs after the
sweep. Rule from here on: **never compare arms on training/val loss; only on
prior/IWAE NLL and downstream/generative evals.** (This also means the
`best.pt` selection criterion is arm-internal only.)

### Arc Pro B70: first sweep attempt hit resource contention

The B70's sweep share ran ~150× slower than its own smoke test (iter 60 of
4000 after 6h). Suspected cause: the resident vLLM server pre-allocates most
of the 32GB and the driver silently spills training tensors to host RAM
rather than OOMing. Decision: killed the container, moved the full κ grid to
the 5060 (supplementary runs for κ∈{1,2} in flight — total 18 CUDA runs),
and Arc pretraining gets a **dedicated window** (serving containers paused)
before any Arc numbers are reported. The "consumer GPU that also serves the
house" failure mode is itself worklog material.
