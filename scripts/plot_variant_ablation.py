#!/usr/bin/env python3
"""Plot the held-out V1/V2/V3 ablation (Bucket 7, Part H)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "variant_ablation"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
NAMES = {"qwen-2.5-1.5b": "Qwen-1.5B", "qwen-2.5-7b": "Qwen-7B",
         "mistral-7b-v0.3": "Mistral", "llama-3.1-8b": "Llama"}
VAR = ["v1", "v2", "v3"]
VLAB = {"v1": "V1 global", "v2": "V2 per-rel", "v3": "V3 per-dom"}
COL = {"v1": "#4477aa", "v2": "#ee6677", "v3": "#228833"}


def main():
    rows = {(r["model"], r["variant"]): r for r in
            csv.DictReader((OUT / "variant_summary.csv").open())}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(MODELS)); w = 0.25
    for j, v in enumerate(VAR):
        rec = [float(rows[(m, v)]["recovery"]) for m in MODELS]
        pa = [float(rows[(m, v)]["pa"]) for m in MODELS]
        off = [i + (j - 1) * w for i in x]
        ax[0].bar(off, rec, w, color=COL[v], label=VLAB[v])
        ax[1].bar(off, pa, w, color=COL[v], label=VLAB[v])
    for a, t, yl in [(ax[0], "Held-out Recovery", "Recovery"), (ax[1], "Held-out PA", "PA")]:
        a.set_xticks(list(x)); a.set_xticklabels([NAMES[m] for m in MODELS], fontsize=8)
        a.set_title(t, fontsize=10); a.set_ylabel(yl, fontsize=9); a.set_ylim(0, 1)
        a.tick_params(labelsize=8); a.legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "variant_ablation.png", dpi=150, bbox_inches="tight")
    fig.savefig(REPO / "TAS_AAAI27_submission" / "figures" / "variant_ablation.png",
                dpi=150, bbox_inches="tight")
    print("Wrote variant_ablation.png")


if __name__ == "__main__":
    main()
