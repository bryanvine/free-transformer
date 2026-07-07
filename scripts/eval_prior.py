#!/usr/bin/env python3
"""Fair evaluation of trained checkpoints: posterior-CE vs prior-CE.

A free model's training/val loss is ELBO-like: the encoder computes Z from the
*full sequence*, so CE-with-posterior-Z leaks information about the tokens
being predicted and is NOT comparable to a baseline's LM loss (at kappa=4 bits
the model routes so much through Z that "val loss" halves — that is the leak,
not generative skill). The generative comparison is CE with Z drawn from the
prior — what generation actually experiences.

For every runs/<name>/best.pt this script reports, over a deterministic
non-overlapping sweep of val.bin (identical windows for every model):
  * ce_posterior  — CE with Z ~ Q(Z|S) (free) or plain CE (baseline)
  * ce_prior      — CE with Z ~ U (free only; mean over --prior-samples draws)
  * kl_bits       — posterior KL usage, bits/token (free only)

Writes runs/eval_prior.json and prints a markdown table grouped by config.

Usage: python scripts/eval_prior.py [--runs-glob "runs/dev_*"] [--device auto]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from free_transformer.config import FTConfig
from free_transformer.latent import kl_to_uniform
from free_transformer.model import FreeTransformer
from free_transformer.training import resolve_device


def val_windows(data_dir: str, block_size: int):
    data = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")
    n = (len(data) - 1) // block_size
    for i in range(n):
        a = i * block_size
        x = torch.from_numpy(data[a : a + block_size].astype(np.int64))
        y = torch.from_numpy(data[a + 1 : a + 1 + block_size].astype(np.int64))
        yield x, y


def batched(gen, batch_size):
    xs, ys = [], []
    for x, y in gen:
        xs.append(x), ys.append(y)
        if len(xs) == batch_size:
            yield torch.stack(xs), torch.stack(ys)
            xs, ys = [], []
    if xs:
        yield torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def _seq_logprob(model, x, y, z, ctx) -> torch.Tensor:
    """Per-sequence sum log p(y|x,z), float32. Shape (B,)."""
    with ctx:
        logits, _, _ = model(x, y, z_bits=z)
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).sum(dim=-1)


@torch.no_grad()
def iwae_bound(model, cfg, x, y, K: int, ctx) -> torch.Tensor:
    """K-sample importance-weighted NLL bound (nats/token), Z ~ prior.

    With Z from the prior the importance weights are p(x|z) itself:
    -log(1/K sum_k p(x|z_k)) >= -log p(x); tightens as K grows. Shape (B,).
    """
    B, T = x.shape
    lps = []
    for _ in range(K):
        z = torch.randint(0, 2, (B, T, cfg.latent_bits), device=x.device).float()
        lps.append(_seq_logprob(model, x, y, z, ctx))
    lps = torch.stack(lps)                                   # (K, B)
    return -(torch.logsumexp(lps, dim=0) - math.log(K)) / T  # nats/token


@torch.no_grad()
def eval_ckpt(run_dir: Path, data_dir: str, device: str, batch_size: int,
              prior_samples: int, max_batches: int, iwae_k: int = 0,
              iwae_batches: int = 0) -> dict:
    ck = torch.load(run_dir / "best.pt", map_location=device)
    cfg = FTConfig(**ck["model_cfg"])
    model = FreeTransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    ctx = torch.autocast(device_type=device.split(":")[0], dtype=dtype) \
        if device != "cpu" else torch.no_grad()

    ce_post, kl_bits, n = 0.0, 0.0, 0
    ce_prior_total, n_prior = 0.0, 0
    iwae_total, n_iwae = 0.0, 0
    torch.manual_seed(1234)  # prior draws reproducible across models
    for bi, (x, y) in enumerate(batched(val_windows(data_dir, cfg.block_size), batch_size)):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        B, T = x.shape
        with ctx:
            _, loss, aux = model(x, y)
        ce_post += aux["ce"].item() * B
        if cfg.model_type == "free":
            kl_bits += aux["kl_bits_per_token"].item() * B
            for _ in range(prior_samples):
                z = torch.randint(0, 2, (B, T, cfg.latent_bits), device=device).float()
                with ctx:
                    _, loss_p, _ = model(x, y, z_bits=z)
                ce_prior_total += loss_p.item() * B
                n_prior += B
            if iwae_k and (not iwae_batches or bi < iwae_batches):
                iwae_total += iwae_bound(model, cfg, x, y, iwae_k, ctx).sum().item()
                n_iwae += B
        n += B
    out = {"run": run_dir.name, "model_type": cfg.model_type,
           "kappa_bits": cfg.kappa_bits if cfg.model_type == "free" else None,
           "params": model.num_params(), "val_iter": int(ck.get("iter", -1)),
           "ce_posterior": ce_post / n, "n_windows": n}
    if cfg.model_type == "free":
        out["ce_prior"] = ce_prior_total / n_prior
        out["kl_bits"] = kl_bits / n
        if n_iwae:
            out["nll_iwae"] = iwae_total / n_iwae
            out["iwae_k"] = iwae_k
    return out


def config_key(name: str) -> str:
    return re.sub(r"_s\d+", "", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/dev_*")
    ap.add_argument("--data-dir", default="data/tinystories")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--prior-samples", type=int, default=1)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = full val sweep")
    ap.add_argument("--iwae-k", type=int, default=0, help="K prior samples for the IW bound (0 = off)")
    ap.add_argument("--iwae-batches", type=int, default=0, help="limit IWAE to first N batches (0 = all)")
    ap.add_argument("--out", default="runs/eval_prior.json")
    args = ap.parse_args()
    device = resolve_device(args.device)

    results = []
    for d in sorted(glob.glob(args.runs_glob)):
        run_dir = Path(d)
        if not (run_dir / "best.pt").exists():
            continue
        r = eval_ckpt(run_dir, args.data_dir, device, args.batch_size,
                      args.prior_samples, args.max_batches, args.iwae_k, args.iwae_batches)
        results.append(r)
        extra = (f" prior={r['ce_prior']:.4f} kl={r['kl_bits']:.3f}b"
                 + (f" iwae{r['iwae_k']}={r['nll_iwae']:.4f}" if "nll_iwae" in r else "")
                 if r["model_type"] == "free" else "")
        print(f"{r['run']:<28} post={r['ce_posterior']:.4f}{extra}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2))

    groups = defaultdict(list)
    for r in results:
        groups[config_key(r["run"])].append(r)
    print("\n| config | seeds | CE posterior (mean [min,max]) | CE prior | NLL (IWAE) | KL bits/tok |")
    print("|---|---|---|---|---|---|")
    for key in sorted(groups):
        rs = groups[key]
        def agg(field):
            vals = [r[field] for r in rs if r.get(field) is not None]
            if not vals:
                return "—"
            return f"{sum(vals)/len(vals):.4f} [{min(vals):.4f}, {max(vals):.4f}]"
        print(f"| {key} | {len(rs)} | {agg('ce_posterior')} | {agg('ce_prior')} | {agg('nll_iwae')} | {agg('kl_bits')} |")


if __name__ == "__main__":
    main()
