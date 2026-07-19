#!/usr/bin/env python3
"""Multi-panel layer-localization figure + AFR/steering correlations (Bucket 6, G/H).

Reads the split-aware AFR curves, held-out steering-by-layer results, and
geometry, and produces (a) a 4-panel supplemental figure and (b) Spearman
correlations between patching AFR and held-out steering Recovery (and between
temporal-shift alignment and Recovery). All numbers come from saved artifacts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "layer_localization_splitclean"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
NAMES = {"qwen-2.5-1.5b": "Qwen-2.5-1.5B", "qwen-2.5-7b": "Qwen-2.5-7B",
         "mistral-7b-v0.3": "Mistral-7B-v0.3", "llama-3.1-8b": "Llama-3.1-8B"}


def read_csv(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


def main():
    locstats = json.loads((OUT / "localization_statistics.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    corr = {}
    for ax, m in zip(axes.ravel(), MODELS):
        afr = [r for r in read_csv(OUT / m / "afr_by_layer_splitclean.csv")
               if r["split"] == "all"]
        afr.sort(key=lambda r: int(r["layer"]))
        layers = [int(r["layer"]) for r in afr]
        av = [float(r["afr"]) for r in afr]
        lo = [float(r["ci_lo"]) for r in afr]
        hi = [float(r["ci_hi"]) for r in afr]
        # test AFR overlay
        afr_t = {int(r["layer"]): float(r["afr"]) for r in
                 read_csv(OUT / m / "afr_by_layer_splitclean.csv") if r["split"] == "test"}
        ax.plot(layers, av, "-", color="#2b6", lw=1.5, label="AFR (all-data)")
        ax.fill_between(layers, lo, hi, color="#2b6", alpha=0.15)
        ax.plot(sorted(afr_t), [afr_t[L] for L in sorted(afr_t)], ":",
                color="#093", lw=1, label="AFR (test)")

        ell = locstats[m]["ell_star"]
        plat = locstats[m]["per_split"]["all"]["plateau"]["0.05"]
        ax.axvspan(plat["lo"], plat["hi"], color="#fc8", alpha=0.25, label="0.05 plateau")
        ax.axvline(ell, color="#c33", ls="--", lw=1, label=r"$\ell^*$")

        # steering Recovery at evaluated layers (twin axis)
        sw = read_csv(OUT / m / "steering_by_layer_test.csv")
        rec_pairs = []
        if sw:
            ax2 = ax.twinx()
            sl = [int(r["layer"]) for r in sw]
            rc = [float(r["recovery"]) for r in sw]
            ax2.plot(sl, rc, "s-", color="#36c", ms=5, lw=1, label="steering Recovery (test)")
            ax2.set_ylabel("steering Recovery", color="#36c", fontsize=8)
            ax2.tick_params(axis="y", labelcolor="#36c", labelsize=7)
            ax2.set_ylim(0, 1)
            wl = locstats[m]["wrong_layer_control"]["null_layer"]
            ax2.plot([wl], [next((float(r["recovery"]) for r in sw
                                  if int(r["layer"]) == wl), 0)], "v",
                     color="#f80", ms=9, label="off-peak control")
            # correlation over evaluated layers
            afr_at = {int(r["layer"]): float(r["afr"]) for r in afr}
            xs = [afr_at[L] for L in sl if L in afr_at]
            ys = [rc[i] for i, L in enumerate(sl) if L in afr_at]
            rec_pairs = (xs, ys)
        n = locstats[m]["per_split"].get("test", {}).get("n", 0)
        ax.set_title(f"{NAMES[m]}  ($\\ell^*$={ell}, test n={n})", fontsize=9)
        ax.set_xlabel("layer", fontsize=8); ax.set_ylabel("AFR", fontsize=8)
        ax.set_ylim(0, 1); ax.tick_params(labelsize=7)

        if rec_pairs and len(rec_pairs[0]) >= 3:
            rho, p = spearmanr(rec_pairs[0], rec_pairs[1])
            corr[m] = {"spearman_afr_vs_steering_recovery": round(float(rho), 3),
                       "p_value": round(float(p), 3), "n_layers": len(rec_pairs[0])}

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(OUT / "layer_localization.png", dpi=150, bbox_inches="tight")
    fig.savefig(REPO / "TAS_AAAI27_submission" / "figures" / "layer_localization.png",
                dpi=150, bbox_inches="tight")

    # geometry-based correlation: cosine alignment vs steering recovery per layer
    for m in MODELS:
        geo = {int(r["layer"]): float(r["mean_cosine_shift_vs_deltal"])
               for r in read_csv(OUT / m / "geometry_by_layer.csv")}
        sw = read_csv(OUT / m / "steering_by_layer_test.csv")
        if geo and sw:
            xs = [geo[int(r["layer"])] for r in sw if int(r["layer"]) in geo]
            ys = [float(r["recovery"]) for r in sw if int(r["layer"]) in geo]
            if len(xs) >= 3:
                rho, p = spearmanr(xs, ys)
                corr.setdefault(m, {})["spearman_alignment_vs_recovery"] = round(float(rho), 3)
                corr[m]["alignment_p"] = round(float(p), 3)

    (OUT / "correlations.json").write_text(json.dumps(corr, indent=2))
    print("Wrote layer_localization.png + correlations.json")
    for m in MODELS:
        print(f"  {m}: {corr.get(m, {})}")


if __name__ == "__main__":
    main()
