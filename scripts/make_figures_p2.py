#!/usr/bin/env python3
"""Paper 2 figure: how the free-bits budget quantizes into channels."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, SURFACE = "#2a78d6", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "svg.fonttype": "none",
})

panels = [
    ("probe_ft124m_free_k1_s1", "κ=1, pristine run (s1) — the budget lands in exactly 3 channels"),
    ("probe_ft124m_free_k1_s3", "κ=1, died-and-resurrected run (s3) — same budget, diffuse 8-channel code"),
    ("probe_ft124m_free_k0.5_s2", "κ=0.5, resurrected run (s2) — one channel carries everything"),
]

fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.6), sharex=True)
for ax, (name, label) in zip(axes, panels):
    d = json.loads(Path(f"paper/{name}.json").read_text())
    kl = d["bit_stats"]["kl_bits"]
    ax.bar(range(16), kl, color=BLUE, width=0.62, edgecolor=SURFACE, linewidth=0.8)
    ax.set_ylim(0, 0.55)
    ax.set_yticks([0, 0.25, 0.5])
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(label, color=INK, fontsize=10, loc="left", pad=4)
    total = sum(kl)
    ax.annotate(f"Σ = {total:.2f} bits/token", (15.4, 0.42), ha="right",
                color=INK2, fontsize=9)
axes[-1].set_xlabel("latent bit index (of H=16)")
axes[-1].set_xticks(range(0, 16, 2))
axes[1].set_ylabel("KL carried per bit (bits/token)")
fig.suptitle("The free-bits budget quantizes into a few soft channels — and training\nhistory decides how many (124M, FineWeb-Edu)",
             color=INK, fontsize=11, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = Path("docs/figures")
fig.savefig(out / "fig4_bit_allocation.svg")
fig.savefig(out / "fig4_bit_allocation.png", dpi=150)
print("written fig4")
