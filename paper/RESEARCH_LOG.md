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

---

## 2026-07-07 (evening) — Phase 1 complete: the latent is an NLL tax that scales exactly with κ

18/18 dev runs clean (51M, TinyStories, 131M tokens, 3 seeds per config, all
on the 5060 after the B70 descope, ~31 min/run). Full-val-sweep eval
(`scripts/eval_prior.py`, deterministic identical windows for every model):

| config | post-CE | **honest NLL (ELBO bound)** [seed range] | prior-CE (K=1) | KL b/tok |
|---|---|---|---|---|
| baseline | 1.5103 | **1.5103** (exact; range 1.5089–1.5119) | — | — |
| free κ=0.125 | 1.5042 | **1.5590** [1.5262, 1.5780] | 1.5408 | 0.079 |
| free κ=0.5 | 1.3283 | **1.6641** [1.6614, 1.6664] | 1.8130 | 0.484 |
| free κ=1 | 1.1273 | **1.8191** [1.8138, 1.8283] | 2.1718 | 0.998 |
| free κ=2 | 0.8583 | **2.2416** [2.2413, 2.2420] | 2.9999 | 1.996 |
| free κ=4 | 0.6137 | **3.3792** [3.3414, 3.4324] | 3.7139 | 3.990 |

### Findings

1. **The free-bits budget is always fully spent, and always paid back with
   interest.** At every κ, posterior KL pins almost exactly at κ (the hinge
   saturates), and the honest NLL (posterior-CE + KL, the ELBO bound) is
   *worse* than baseline by roughly the KL spent — monotone in κ. At
   51M/TinyStories the latent channel is a pure tax on language modeling:
   whatever the encoder stuffs into Z, generation must buy back from a
   uniform prior.
2. **Collapse at κ=4 reproduced** (paper's boundary): 4.0 bits/token through
   Z, prior-CE 2.5× baseline — the model is an autoencoder of its own input.
3. **Seed-dependent posterior collapse at κ=0.125**: seed 2 uses 0.006
   b/tok (Z ignored; its prior-CE equals its posterior-CE), seeds 1/3 use
   ~0.117 b/tok — a 20× spread. Latent models at small scale are exactly
   where single-seed results mislead; error bars aren't optional here.
4. **Methodology trap #2 (after the posterior leak): prior-proposal IWAE is
   useless at high per-token KL.** Sequence-level information ≈ κ·T bits
   (≈250 bits at κ=0.5, T=512); K=64 prior draws cannot find the posterior
   region, so `iwae64` barely improves on single-sample prior-CE and even
   sits *above* the ELBO bound. At κ=0.125 IWAE-64 ≈ ELBO (cross-check
   passes). Use the ELBO bound, or implement posterior-proposal IWAE (queued
   for Phase 2).
5. **Qualitative generations at κ=0.5 are coherent** — indistinguishable
   TinyStories-grade text from either arm at temp 0.8 (samples in
   runs/…/best.pt, printed in the session record). The tax is invisible to
   the eye at low κ, consistent with the paper's claims living on downstream
   tasks rather than perplexity.

### What this does and doesn't say about the paper

It does NOT refute Fleuret: the paper never claims better NLL; its gains are
on prompted downstream benchmarks at 1.5B/8B, where conditioning uses the
posterior prefill and "committing to a decision" could help coherence at some
likelihood cost. What Phase 1 establishes: (a) two methodology traps that any
replication comparing on loss will fall into; (b) at 51M on a low-entropy
corpus the latent buys nothing on-distribution; (c) the collapse boundary and
seed instability the paper describes are real and land where it says.

### Phase 2 decisions

- Headline scale (124M/FineWeb-Edu): κ ∈ {0.125, 0.5} only (the low-tax
  regime), vs baseline-12L vs params-matched baseline-13L, 3 seeds each.
- Primary metrics move off-perplexity: RQ2 probes (what Z encodes), steering,
  and structured-generation evals; NLL reported as the honest secondary.
- Implement posterior-proposal IWAE for tight NLL on the free arm.
- Arc B70 gets a dedicated window (serving containers paused) for its anchor
  runs — contention discovery documented above.

---

## 2026-07-07 (night) — Arc window post-mortem: killed by a package upgrade; 57.6k tok/s while it lived

The dedicated window (vLLM paused) ran the baseline anchor at full health —
**57.6k tok/s sustained at dev scale (51M, batch 32×512, eager XPU)**, within
~20% of the RTX 5060's torch.compile throughput on the same config — until
minute 28, when **unattended-upgrades upgraded containerd.io (2.2.4→2.2.5)**;
the containerd service restart SIGKILL'd the training container mid-operation
(rc=137), the xe driver's GuC reset failed (`reset failed (-ETIMEDOUT)`), and
the GPU wedged (forcewake MMIO 0xFFFFFFFF) until the next VM power cycle.

Consolidated hardware lesson for BMG-G31 under vfio passthrough (2 wedges,
1 day, both kill-mid-op): **any ungraceful kill of in-flight Level Zero work
wedges the GPU unrecoverably** — the driver reset path never succeeds, and
only a hypervisor-level power cycle (not a guest reboot) restores the device.
Candidate upstream report for Intel (xe GuC reset failure on BMG-G31/vfio,
kernel 6.17) — fits this lab's history of filing vllm-xpu-kernels issues.
Mitigations adopted: never force-kill (already scripted); container-runtime
package holds during windows (pending owner approval); partial-run artifacts
survive via checkpoint-resume, so the anchor runs restart where they left off.

Salvage: dev_baseline_s1_xpu reached iter 500+ with a clean loss curve
(2.17 @ iter 500, matching the CUDA baseline's trajectory) — the resume
logic continues it in the next window.

---

## 2026-07-08 — Arc window 2 post-mortem: the GPU that came back without its power brain

The resumed window ran correctly but at **~1.1k tok/s** (50× under the same
card's healthy 57.6k) with zero contention. Live sysfs forensics: `cur/min/
max_freq` return empty reads, `act_freq` reports an impossible 8517 MHz,
package temperature reads the 255°C sentinel, energy 0.00 J, GT throttle
status stuck at 1. Diagnosis: the cold boot revived command submission but
**PCODE's power-management interface never came back** — the card is locked
at a low safe clock and all PM telemetry is garbage. Frequency writes are
accepted but ignored. Window closed early (checkpoint at iter 2740/4000
preserved; vLLM + apt timers auto-restored). Next escalation: full power
drain (PSU off ≥60s) and, if the state recurs, a GSC firmware check.

Robustness fix landed as a result: `docker stop` now reaches the trainer —
SIGTERM handler checkpoints and exits 0 at the iteration boundary
(verified live), with bash signal-forwarding in sweep_dev.sh (bash-as-PID1
was swallowing the signal, turning every "graceful" stop into a SIGKILL).

Running tally of the Arc Pro B70 story: fastest consumer-Arc pretraining
throughput on record when healthy; wedged by a force-kill, a package
manager, and its own power firmware in 36 hours; every failure mode now
documented, scripted around, and reproducible. This is the worklog post.

---

## 2026-07-08 (late) — Arc anchors complete: cross-backend parity, hardened window holds

First fully clean Arc window (exit 0, zero watchdog triggers) under the new
saturation hardening: clocks capped 2800→2000 MHz, instantaneous batch halved
at equal tokens/iter, 50ms/iter breather, dmesg watchdog with graceful
checkpoint-stop, vLLM/apt-timers paused and auto-restored. Cost of the whole
stack: ~20% (46.0k tok/s vs 57.6k unhardened).

**Cross-backend anchors (dev scale, seed 1):** free κ=0.5 on XPU lands
posterior-val **1.3224 / KL 0.481b — inside the CUDA 3-seed band**
([1.3155, 1.3356] / [0.477, 0.499]). Baseline on XPU 1.4865 vs CUDA
[1.5089, 1.5119] — 1.5% below the band, but this run's history is
heterogeneous (two wedge interruptions, batch shape 32×2→16×4 mid-run), so
it serves as a backend sanity anchor, not a seed-replica; a clean re-run can
tighten it if the paper needs it. Conclusion: no XPU-specific training
pathology; the free-transformer results are backend-independent.

These are, to our knowledge, the first completed LLM pretraining runs
published from Intel Arc B-series hardware.

---

## 2026-07-09 — Headline surprise: the latent collapses at 124M on FineWeb-Edu

free κ=0.5 s1 at 124M/2.46B tokens: best val **3.1968** — but **KL collapsed
to ~0.001 b/tok** (trace: 0.73 @ init → 0.068 @ 1k → 0.002 @ 2k → dead).
The model is effectively a 130.7M-param plain GPT that ignores its latent.
Contrast dev scale, where κ=0.5 held 0.48 b/tok on 3/3 seeds: **collapse is
scale/data-dependent** (TinyStories' low-entropy structure gives Z early
utility; FineWeb-Edu at 124M does not — CE optimization drowns the encoder
before it finds a use). Note the free-bits hinge is one-sided: it stops the
KL *penalty* below κ but provides no pressure keeping Z alive.

Consequences:
- baseline (123.6M): 3.2092 · free-collapsed (130.7M): 3.1968 — the gap is
  now a pure +6%-params effect. The **params-matched 13L control (130.67M)
  is the deciding run** — launched (s1, same budget).
- The paper's 1.5B/8B runs at κ=0.5 did NOT collapse (KL pinned at κ). Open
  question for the writeup: what keeps Z alive at scale — model capacity,
  data richness, tokens, or luck of the seed (cf. our dev κ=0.125 seed
  split)? Candidate follow-ups: κ=1 at 124M (more free bits early), KL
  warmup schedules, encoder LR multiplier.
- Since Z carries ~nothing, this run's val loss is honest (no posterior
  leak) — directly comparable to baselines.

---

## 2026-07-09 — B70 at 124M×1024: eager is dispatch-bound, compile hits the autotune storm

Attempted the 124M κ=1 arm on the B70 in a hardened window. Two failure
modes, both instructive:

1. **Eager, batch 16×30**: 8.2k tok/s steady (vs ~15-19k expected by scaling
   the dev-scale 46k) with one host core pegged at ~91% — eager XPU dispatch
   is co-limiting at 124M×1024ctx on this 12-vCPU VM. Window would run ~83h
   (authorized: 30h).
2. **torch.compile, batch 32×15**: compiled and trained 10 real iterations,
   then fell into an Inductor/Triton-XPU **autotune-recompile storm** —
   python at 100% CPU generating variants, GPU busy benchmarking them, zero
   iterations in 45+ min. Matches the known BMG autotuner breakage
   (intel-xpu-backend-for-triton #6367 class). No crash, no GPU errors —
   just a compiler that never converges at this shape.

**Division of labor conclusion:** the B70's proven regime is dev-scale
(512ctx, 46k tok/s hardened, clean windows); 124M×1024 belongs on the CUDA
fleet (5060: 29.5k measured; 4080S: est. 75-90k). κ=1 s1 reassigned to the
5060 queue behind the 13L control. Candidate B70 work instead: the RQ3
dev-scale H-sweep and κ-instability seed expansion — same-science, right
tool. (Closing the stuck window via boundary-aware graceful stop; the run
was 10 iterations old, nothing lost.)

---

## 2026-07-10 — B70 window 3 results: H doesn't matter; collapse is a 1-in-3 coin flip

11/11 runs clean (~47 min each, second consecutive zero-incident hardened
window; vLLM auto-restored).

**RQ3a — latent width H at κ=0.5 (dev scale, posterior-val, all XPU):**

| H | val (3 seeds) | KL |
|---|---|---|
| 4 | 1.316–1.347 | ~0.48b |
| 8 | 1.331–1.346 | ~0.48b |
| 16 | 1.322–1.343 | ~0.49b |

All widths statistically indistinguishable; every run pins KL at exactly κ.
**The free-bits budget, not channel width, is the operative dial** — even
H=4 (4 bits max) dwarfs a 0.5-bit budget. Width is slack at this scale.

**RQ3b — κ=0.125 collapse rate, n=6 (3 CUDA + 3 XPU seeds):**
alive at ~0.117–0.118b: s1,s3 (cuda), s4,s5 (xpu); collapsed at ~0.005b:
s2 (cuda), s6 (xpu). **Collapse rate 2/6, strictly bimodal (no middle
ground), one collapse on each backend** — a seed lottery, not a backend
artifact. Collapsed runs' val ≈ baseline+block; alive runs ≈ 0.025 better
on posterior-val (and pay it back at generation, per the ELBO accounting).

Open thread these results sharpen: at 124M/FineWeb κ=0.5 collapsed on its
single seed — is 124M collapse also stochastic (needs free κ=0.5 s2) or
deterministic at this scale/data? κ=1 (4080S, running) probes the budget
axis; free κ=0.5 s2 queued behind baseline s2 on the 5060.
