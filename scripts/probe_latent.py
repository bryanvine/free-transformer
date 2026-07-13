#!/usr/bin/env python3
"""RQ2 probes for a live-latent Free Transformer checkpoint.

Three experiments, one JSON out:

A) POSTERIOR STATISTICS over deterministic val windows — per bit:
   mean p, hardness (fraction of tokens with |p-0.5|>0.4), KL contribution
   (bits/token), temporal coherence P(bit_t == bit_{t+1}), and correlations
   with cheap observables (position in window, token starts-with-space,
   token is alphabetic, token is newline/punct-ish).

B) PER-BIT VALUE via ablation — CE with the full posterior Z, then CE with
   bit i replaced by prior noise at every position: delta CE ranks how much
   each bit's information is worth to reconstruction.

C) STEERING — from fixed prompts, sample generations with (i) a fixed
   z-sequence reused across samples (several patterns) and (ii) Z resampled
   per generation. If Z carries global decisions, intra-pattern generations
   should be more similar (word-3-gram Jaccard) than inter-pattern ones.
   Saves example generations for qualitative reading.

Usage:
  python scripts/probe_latent.py --ckpt runs/X/best.pt --data-dir data/fineweb_edu \
      --out runs/probe_X.json [--device auto] [--stat-batches 60] [--ablate-batches 20]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from free_transformer.config import FTConfig
from free_transformer.model import FreeTransformer
from free_transformer.training import resolve_device

LN2 = math.log(2.0)


def val_batches(data_dir, block_size, batch_size, n_batches):
    data = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")
    n = min(n_batches * batch_size, (len(data) - 1) // block_size)
    xs, ys = [], []
    for i in range(n):
        a = i * block_size
        xs.append(torch.from_numpy(data[a:a + block_size].astype(np.int64)))
        ys.append(torch.from_numpy(data[a + 1:a + 1 + block_size].astype(np.int64)))
        if len(xs) == batch_size:
            yield torch.stack(xs), torch.stack(ys)
            xs, ys = [], []


def token_features(enc, ids):
    """Cheap per-token observables for one window (list of ints)."""
    starts_space, is_alpha, is_break = [], [], []
    for t in ids:
        s = enc.decode([int(t)])
        starts_space.append(1.0 if s[:1] == " " else 0.0)
        is_alpha.append(1.0 if s.strip().isalpha() and s.strip() else 0.0)
        is_break.append(1.0 if ("\n" in s or (s.strip() and not s.strip().isalnum())) else 0.0)
    return starts_space, is_alpha, is_break


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stat-batches", type=int, default=60)
    ap.add_argument("--ablate-batches", type=int, default=20)
    ap.add_argument("--gen-prompts", type=int, default=4)
    ap.add_argument("--gen-patterns", type=int, default=3)
    ap.add_argument("--gen-samples", type=int, default=4)
    ap.add_argument("--gen-tokens", type=int, default=120)
    args = ap.parse_args()

    device = resolve_device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    cfg = FTConfig(**ck["model_cfg"])
    assert cfg.model_type == "free", "probe requires a free-model checkpoint"
    model = FreeTransformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    H = cfg.latent_bits
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    ctx = torch.autocast(device_type=device.split(":")[0], dtype=dtype) \
        if device != "cpu" else torch.no_grad()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    torch.manual_seed(1234)

    out = {"ckpt": args.ckpt, "H": H, "kappa_bits": cfg.kappa_bits}

    # ---- A: posterior statistics -------------------------------------------
    sum_p = torch.zeros(H); sum_hard = torch.zeros(H); sum_kl = torch.zeros(H)
    coh_agree = torch.zeros(H); coh_n = 0
    feats = {"position": [], "starts_space": [], "is_alpha": [], "is_break": []}
    bits_acc = [[] for _ in range(H)]
    n_tok = 0
    for bi, (x, y) in enumerate(val_batches(args.data_dir, cfg.block_size,
                                            args.batch_size, args.stat_batches)):
        x = x.to(device)
        with ctx:
            p, hard = model.encode_latent(x)
        p = p.float().cpu()                                  # (B, T, H)
        pc = p.clamp(1e-6, 1 - 1e-6)
        kl_bits = 1.0 + (pc * pc.log2() + (1 - pc) * (1 - pc).log2())  # per bit, bits
        sum_p += p.sum(dim=(0, 1)); n_tok += p.shape[0] * p.shape[1]
        sum_hard += ((p - 0.5).abs() > 0.4).float().sum(dim=(0, 1))
        sum_kl += kl_bits.sum(dim=(0, 1))
        hb = (p > 0.5).float()
        coh_agree += (hb[:, 1:, :] == hb[:, :-1, :]).float().sum(dim=(0, 1))
        coh_n += hb.shape[0] * (hb.shape[1] - 1)
        if bi < 8:  # feature correlations on a subsample
            for b in range(x.shape[0]):
                ids = x[b].cpu().tolist()
                ss, ia, br = token_features(enc, ids)
                T = len(ids)
                feats["position"].extend([t / T for t in range(T)])
                feats["starts_space"].extend(ss)
                feats["is_alpha"].extend(ia)
                feats["is_break"].extend(br)
                for h in range(H):
                    bits_acc[h].extend(hb[b, :, h].tolist())
    out["bit_stats"] = {
        "mean_p": (sum_p / n_tok).tolist(),
        "hardness": (sum_hard / n_tok).tolist(),
        "kl_bits": (sum_kl / n_tok).tolist(),
        "temporal_coherence": (coh_agree / coh_n).tolist(),
    }
    out["bit_feature_corr"] = {
        f: [corr(bits_acc[h], v) for h in range(H)] for f, v in feats.items()
    }

    # ---- B: per-bit ablation ------------------------------------------------
    def ce_with_bits(x, y, override=None):
        """CE using posterior bits, optionally with bit `override` randomized."""
        with ctx:
            p, _ = model.encode_latent(x)
        z = torch.bernoulli(p.float())
        if override is not None:
            z[:, :, override] = torch.randint(0, 2, z.shape[:2], device=z.device).float()
        with ctx:
            _, loss, _ = model(x, y.to(x.device), z_bits=z)
        return loss.item()

    base_ce, abl = 0.0, torch.zeros(H)
    nb = 0
    for x, y in val_batches(args.data_dir, cfg.block_size, args.batch_size,
                            args.ablate_batches):
        x = x.to(device)
        base_ce += ce_with_bits(x, y); nb += 1
        for h in range(H):
            abl[h] += ce_with_bits(x, y, override=h)
    out["ablation"] = {
        "base_ce_posterior_bits": base_ce / nb,
        "delta_ce_per_bit": ((abl / nb) - (base_ce / nb)).tolist(),
    }

    # ---- C: steering ----------------------------------------------------------
    prompts = ["The most important thing to understand about",
               "In this tutorial, we will learn how to",
               "The history of the city begins with",
               "Scientists have recently discovered that"][: args.gen_prompts]
    gens = {}
    T_full = cfg.block_size
    for pi, prompt in enumerate(prompts):
        ptoks = torch.tensor([enc.encode(prompt)], device=device)
        gens[prompt] = {"fixed": [], "resampled": []}
        for pat in range(args.gen_patterns):
            torch.manual_seed(9000 + pat)
            zfix = torch.randint(0, 2, (1, T_full, H), device=device).float()
            outs = []
            for s in range(args.gen_samples):
                torch.manual_seed(100 * pi + 10 * pat + s)
                o = model.generate(ptoks.clone(), max_new_tokens=args.gen_tokens,
                                   temperature=0.8, top_k=50, z_bits=zfix)
                outs.append(enc.decode(o[0, ptoks.shape[1]:].tolist()))
            gens[prompt]["fixed"].append(outs)
        outs = []
        for s in range(args.gen_patterns * args.gen_samples):
            torch.manual_seed(7000 + 10 * pi + s)
            o = model.generate(ptoks.clone(), max_new_tokens=args.gen_tokens,
                               temperature=0.8, top_k=50, use_posterior_prefill=False)
            outs.append(enc.decode(o[0, ptoks.shape[1]:].tolist()))
        gens[prompt]["resampled"] = outs

    def jac3(a, b):
        ga = set(zip(*[a.split()[i:] for i in range(3)]))
        gb = set(zip(*[b.split()[i:] for i in range(3)]))
        return len(ga & gb) / max(1, len(ga | gb))

    intra, inter = [], []
    for prompt in gens:
        pats = gens[prompt]["fixed"]
        for outs in pats:
            intra.extend(jac3(a, b) for a, b in itertools.combinations(outs, 2))
        for oa, ob in itertools.combinations(pats, 2):
            inter.extend(jac3(a, b) for a in oa for b in ob)
    out["steering"] = {
        "intra_pattern_jaccard3_mean": float(np.mean(intra)),
        "inter_pattern_jaccard3_mean": float(np.mean(inter)),
        "n_intra_pairs": len(intra), "n_inter_pairs": len(inter),
        "examples": {p: {"fixed_pattern0": gens[p]["fixed"][0][:2],
                         "fixed_pattern1": gens[p]["fixed"][1][:2],
                         "resampled": gens[p]["resampled"][:2]} for p in gens},
    }

    Path(args.out).write_text(json.dumps(out, indent=1))
    bs = out["bit_stats"]
    print(f"KL by bit (bits/tok): {[round(x,3) for x in bs['kl_bits']]}")
    print(f"coherence by bit:     {[round(x,2) for x in bs['temporal_coherence']]}")
    print(f"ablation dCE by bit:  {[round(x,4) for x in out['ablation']['delta_ce_per_bit']]}")
    print(f"steering: intra={out['steering']['intra_pattern_jaccard3_mean']:.4f} "
          f"inter={out['steering']['inter_pattern_jaccard3_mean']:.4f}")


if __name__ == "__main__":
    main()
