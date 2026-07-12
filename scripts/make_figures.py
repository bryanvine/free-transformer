#!/usr/bin/env python3
"""Generate Paper 1 figures (SVG) from paper/results_all.json.

Style: thin marks, hairline grid, direct labels, one axis per chart; colors
from the validated reference palette (blue #2a78d6 = live latent,
red #e34948 = collapsed, near-black ink = baselines).
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN2 = math.log(2.0)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, RED, AQUA, VIOLET = "#2a78d6", "#e34948", "#1baf7a", "#4a3aa7"
SURFACE = "#fcfcfb"

RES = json.loads(Path("paper/results_all.json").read_text())
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "svg.fonttype": "none",
})


def final(name):
    t = RES[name]["trace"]
    return t[-1]["val"], t[-1]["kl"]


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# ---- Fig 1: honest NLL vs kappa at 124M -------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.0))
style_ax(ax)

base_vals = [final(f"ft124m_baseline_s{s}")[0] for s in (1, 2)]  # s3 pending
ax.axhspan(min(base_vals), max(base_vals), color=GRID, alpha=0.55, lw=0)
ax.axhline(sum(base_vals) / len(base_vals), color=INK, lw=1.0)
ax.annotate("baseline band (12L, seeds)", (2.02, sum(base_vals) / 2 + 0.004),
            color=INK2, fontsize=9, va="bottom", ha="right")
v13, _ = final("ft124m_baseline13L_s1")
ax.plot([0], [v13], marker="D", ms=6, color=INK2, mfc="none", mew=1.2, ls="none")
ax.annotate("13L control", (0.05, v13 - 0.022), color=INK2, fontsize=9)

pts = {0.5: ["ft124m_free_k0.5_s1", "ft124m_free_k0.5_s2", "ft124m_free_k0.5_s3"],
       1.0: ["ft124m_free_k1_s1", "ft124m_free_k1_s2"],
       2.0: ["ft124m_free_k2_s1"]}
for kappa, runs in pts.items():
    for r in runs:
        v, kl = final(r)
        alive = kl > 0.05
        nll = v + kl * LN2
        ax.plot([kappa], [nll], marker="o", ms=7,
                color=BLUE if alive else RED, ls="none",
                mec=SURFACE, mew=1.0)
ax.annotate("collapsed (s1)", (0.56, 3.203), color=RED, fontsize=9, va="center")
ax.annotate("live latent (ELBO bound)", (0.62, 3.44), color=BLUE, fontsize=9)
ax.annotate("the bend:\nmarginal bit no longer free", (1.42, 3.60),
            color=INK2, fontsize=9, ha="center")

ax.set_xticks([0, 0.5, 1, 2])
ax.set_xlabel("free-bits budget κ (bits/token)")
ax.set_ylabel("honest NLL (nats/token, val)")
ax.set_title("The latent tax at 124M: flat to κ=1, expensive past it",
             color=INK, fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig1_elbo_vs_kappa.svg"); fig.savefig(OUT / "fig1_elbo_vs_kappa.png", dpi=150)
plt.close(fig)

# ---- Fig 2: KL traces at 124M ------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.8))
style_ax(ax)
series = [("ft124m_free_k0.5_s1", RED, "κ=0.5 s1 — collapses, stays dead"),
          ("ft124m_free_k0.5_s2", AQUA, "κ=0.5 s2 — dies, resurrects"),
          ("ft124m_free_k0.5_s3", VIOLET, "κ=0.5 s3 — dips, recovers early"),
          ("ft124m_free_k1_s1", BLUE, "κ=1 s1 — pinned at budget")]
for name, color, label in series:
    t = RES[name]["trace"]
    xs = [r["iter"] for r in t]
    ys = [r["kl"] for r in t]
    ax.plot(xs, ys, color=color, lw=2.0, solid_capstyle="round")
    off = {"ft124m_free_k0.5_s3": 0.06, "ft124m_free_k0.5_s2": -0.05}.get(name, 0.0)
    ax.annotate(label, (xs[-1] + 60, ys[-1] + off), color=color, fontsize=9, va="center")
ax.set_xlim(0, 8400)
ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax.set_xlabel("training iteration")
ax.set_ylabel("posterior KL (bits/token)")
ax.set_title("Collapse is a seed lottery, and death is escapable (124M, FineWeb-Edu)",
             color=INK, fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig2_kl_traces.svg"); fig.savefig(OUT / "fig2_kl_traces.png", dpi=150)
plt.close(fig)

# ---- Fig 3: the posterior leak ------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.6))
style_ax(ax)
arms = [("baseline (12L)", *final("ft124m_baseline_s1"), None),
        ("free κ=0.5 s3", *final("ft124m_free_k0.5_s3"), None),
        ("free κ=1 s1", *final("ft124m_free_k1_s1"), None),
        ("free κ=2 s1", *final("ft124m_free_k2_s1"), None)]
x = range(len(arms))
w = 0.36
post = [a[1] for a in arms]
honest = [a[1] + a[2] * LN2 for a in arms]
ax.bar([i - w / 2 for i in x], post, width=w, color=MUTED, label="logged val loss (posterior Z — leaks)",
       edgecolor=SURFACE, linewidth=1)
ax.bar([i + w / 2 for i in x], honest, width=w, color=BLUE, label="honest NLL (ELBO bound)",
       edgecolor=SURFACE, linewidth=1)
for i, (p, h) in enumerate(zip(post, honest)):
    ax.annotate(f"{p:.2f}", (i - w / 2, p + 0.04), ha="center", color=INK2, fontsize=8.5)
    ax.annotate(f"{h:.2f}", (i + w / 2, h + 0.04), ha="center", color=INK2, fontsize=8.5)
ax.set_xticks(list(x))
ax.set_xticklabels([a[0] for a in arms], fontsize=9)
ax.set_ylim(0, 4.3)
ax.set_ylabel("nats/token (val)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Trap #1: the logged loss flatters the free model by its full KL",
             color=INK, fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(OUT / "fig3_leak.svg"); fig.savefig(OUT / "fig3_leak.png", dpi=150)
plt.close(fig)

print("figures written:", sorted(p.name for p in OUT.glob("*.svg")))
