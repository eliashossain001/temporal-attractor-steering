#!/usr/bin/env python3
"""Phase-1 headline figure (main paper, Fig. 1), regenerated de-zoomed.

Two panels on the 8,746-record benchmark:
  (a) Outdated-preference rate (OPR) per model -- similar across evaluated models.
  (b) Filtered PTC rate per model (with 95% bootstrap CIs) increasing across the
      evaluated models, the Kept fraction (knowledge-recovery filter pass rate)
      on a twin axis, and the elicitation gap EG annotated per model.

The models differ in family, tokenizer, corpus, and cutoff as well as size, so
panel (b) is an observed trend across the evaluated set, not controlled scaling.

All bar heights are recomputed from results/phase1/<model>/per_instance.jsonl
(OPR, filtered PTC, Kept, EG) so the figure matches the numbers in the text;
the filtered-PTC error bars use the paper's published percentile-bootstrap CIs.
Rendered at a modest size/font so panel elements are not oversized.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PHASE1 = REPO / "results" / "phase1"
OUT = REPO / "TAS_AAAI27_submission" / "figures" / "phase1_headline.png"

# evaluated-model order (smallest -> largest parameter count; family and cutoff
# also vary along this axis, so the ordering is not a capacity control)
MODELS =["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
NAMES = {"qwen-2.5-1.5b": "Qwen-2.5\n1.5B", "qwen-2.5-7b": "Qwen-2.5\n7B",
         "mistral-7b-v0.3": "Mistral\n7B", "llama-3.1-8b": "Llama-3.1\n8B"}
TAU_REC = -3.0

# Published 95% percentile-bootstrap CIs for filtered PTC (main text).
PTC_CI = {
    "qwen-2.5-1.5b": (0.033, 0.049),
    "qwen-2.5-7b": (0.063, 0.080),
    "mistral-7b-v0.3": (0.078, 0.093),
    "llama-3.1-8b": (0.096, 0.111),
}

BAR = "#3b7dd8"   # OPR / neutral bars
PTCC = "#d1495b"  # filtered PTC bars
KEPT = "#2a9d8f"  # kept fraction line


def compute(model: str) -> dict:
    n = opr = kept = ptc_kept = 0
    eg = 0.0
    for line in (PHASE1 / model / "per_instance.jsonl").open():
        d = json.loads(line)
        n += 1
        s = d["scores"]
        anew_t = s["temporal"]["a_new"]["mean_logprob"]
        anew_s = s["standard"]["a_new"]["mean_logprob"]
        eg += anew_t - anew_s
        if d["prefers_old_under_standard"]:
            opr += 1
        if anew_t >= TAU_REC:
            kept += 1
            if d["is_ptc"]:
                ptc_kept += 1
    return {
        "opr": opr / n,
        "kept": kept / n,
        "ptc": ptc_kept / kept,
        "eg": eg / n,
    }


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8})
    stats = {m: compute(m) for m in MODELS}
    x = list(range(len(MODELS)))
    labels = [NAMES[m] for m in MODELS]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.2, 3.3))

    # ---- Panel (a): OPR, similar across evaluated models ------------------
    opr = [stats[m]["opr"] for m in MODELS]
    axa.bar(x, opr, width=0.62, color=BAR, edgecolor="white")
    mean_opr = sum(opr) / len(opr)
    axa.axhline(mean_opr, ls="--", lw=1, color="#333",
                label=f"mean = {mean_opr:.3f}")
    for xi, v in zip(x, opr):
        axa.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axa.set_ylim(0, 0.66)
    axa.set_xticks(x)
    axa.set_xticklabels(labels)
    axa.set_ylabel("OPR (outdated preference)")
    axa.set_title("(a) OPR is similar across evaluated models")
    axa.legend(loc="upper right", frameon=False)
    axa.spines[["top", "right"]].set_visible(False)

    # ---- Panel (b): filtered PTC + Kept + EG -----------------------------
    ptc = [stats[m]["ptc"] for m in MODELS]
    lo = [ptc[i] - PTC_CI[m][0] for i, m in enumerate(MODELS)]
    hi = [PTC_CI[m][1] - ptc[i] for i, m in enumerate(MODELS)]
    axb.bar(x, ptc, width=0.62, color=PTCC, edgecolor="white",
            yerr=[lo, hi], capsize=3, ecolor="#333",
            error_kw={"lw": 1})
    for xi, m in zip(x, MODELS):
        axb.text(xi, PTC_CI[m][1] + 0.004, f"{stats[m]['ptc']:.3f}",
                 ha="center", va="bottom", fontsize=8)
    axb.set_ylim(0, 0.13)
    axb.set_xticks(x)
    axb.set_xticklabels(labels)
    axb.set_ylabel("Filtered PTC rate", color=PTCC)
    axb.tick_params(axis="y", labelcolor=PTCC)
    axb.set_title("(b) Filtered PTC increases across the evaluated models")
    axb.spines[["top"]].set_visible(False)

    # twin axis: Kept fraction (knowledge-recovery pass rate)
    axk = axb.twinx()
    kept = [100 * stats[m]["kept"] for m in MODELS]
    axk.plot(x, kept, "o-", color=KEPT, lw=1.5, ms=5, label="Kept (%)")
    for xi, v in zip(x, kept):
        axk.annotate(f"{v:.0f}%", (xi, v), textcoords="offset points",
                     xytext=(9, 6), ha="left", va="bottom", fontsize=8,
                     color=KEPT, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white",
                               ec="none", alpha=0.85))
    axk.set_ylim(0, 100)
    axk.set_ylabel("Kept by recovery filter (%)", color=KEPT)
    axk.tick_params(axis="y", labelcolor=KEPT)
    axk.spines[["top"]].set_visible(False)

    # EG footnote line
    eg_txt = "  ".join(
        f"{NAMES[m].replace(chr(10), ' ')}: EG={stats[m]['eg']:+.3f}"
        for m in MODELS)
    fig.text(0.5, -0.02, "Elicitation gap (nats), positive on every model:  "
             + eg_txt, ha="center", va="top", fontsize=7.2, color="#444")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)
    for m in MODELS:
        s = stats[m]
        print(f"  {m:16} OPR={s['opr']:.3f} kept={100*s['kept']:.1f}% "
              f"PTC={s['ptc']:.4f} EG={s['eg']:+.3f}")


if __name__ == "__main__":
    main()
