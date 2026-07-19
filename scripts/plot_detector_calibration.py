#!/usr/bin/env python3
"""Bucket 1: corrected subject-disjoint detector figures.

Reads the per-instance TEST predictions dumped by detector_operating_points.py
(results/detector_splitclean/per_instance_test_<model>.csv) plus the gated-TAS
JSON, and produces corrected calibration/discrimination figures on UNTOUCHED test
subjects. No model forward passes.

Figures (written to TAS_AAAI27_submission/figures/):
  detector_pr_corrected.png            PR curves (all models)
  detector_roc_corrected.png           ROC curves (all models)
  detector_reliability_corrected.png   reliability/calibration curves
  detector_score_dist_corrected.png    PTC vs non-conflict score distributions
  detector_threshold_tpr_fpr.png       threshold vs TPR/FPR
  detector_threshold_gated.png         threshold vs gated Recovery/PA (0.15/0.2/0.3)
  detector_summary_corrected.png       compact multi-model summary (AUROC/AUPRC/FPR)

Usage:  python scripts/plot_detector_calibration.py
"""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

REPO = Path(__file__).resolve().parents[1]
DET = REPO / "results" / "detector_splitclean"
GATED = REPO / "results" / "tas_splitclean"
FIG = REPO / "TAS_AAAI27_submission" / "figures"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
LABEL = {"qwen-2.5-1.5b": "Qwen-2.5-1.5B", "qwen-2.5-7b": "Qwen-2.5-7B",
         "mistral-7b-v0.3": "Mistral-7B-v0.3", "llama-3.1-8b": "Llama-3.1-8B"}
COL = {"qwen-2.5-1.5b": "#4C72B0", "qwen-2.5-7b": "#DD8452",
       "mistral-7b-v0.3": "#55A868", "llama-3.1-8b": "#C44E52"}


def load(model):
    y, p = [], []
    with (DET / f"per_instance_test_{model}.csv").open() as f:
        for r in csv.DictReader(f):
            y.append(int(r["is_ptc"])); p.append(float(r["calibrated_prob"]))
    return np.array(y), np.array(p)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = {m: load(m) for m in MODELS}

    # 1. PR curves
    plt.figure(figsize=(5, 4))
    for m in MODELS:
        y, p = data[m]
        pr, rc, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        plt.plot(rc, pr, color=COL[m], label=f"{LABEL[m]} (AUPRC {ap:.3f})")
        plt.axhline(y.mean(), color=COL[m], ls=":", lw=0.7, alpha=0.5)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Corrected subject-disjoint PR (test)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG / "detector_pr_corrected.png", dpi=200); plt.close()

    # 2. ROC curves
    plt.figure(figsize=(5, 4))
    for m in MODELS:
        y, p = data[m]
        fpr, tpr, _ = roc_curve(y, p)
        au = roc_auc_score(y, p)
        plt.plot(fpr, tpr, color=COL[m], label=f"{LABEL[m]} (AUROC {au:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.7)
    plt.xlabel("False-positive rate"); plt.ylabel("True-positive rate")
    plt.title("Corrected subject-disjoint ROC (test)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG / "detector_roc_corrected.png", dpi=200); plt.close()

    # 3. reliability curves
    plt.figure(figsize=(5, 4))
    bins = np.linspace(0, 1, 11)
    for m in MODELS:
        y, p = data[m]
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(len(bins) - 1):
            mask = idx == b
            if mask.sum() >= 5:
                xs.append(p[mask].mean()); ys.append(y[mask].mean())
        plt.plot(xs, ys, "o-", color=COL[m], ms=3, label=LABEL[m])
    plt.plot([0, 1], [0, 1], "k--", lw=0.7)
    plt.xlabel("Predicted probability"); plt.ylabel("Empirical PTC frequency")
    plt.title("Reliability (test; bins with n$\\geq$5)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(FIG / "detector_reliability_corrected.png", dpi=200); plt.close()

    # 4. score distributions
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    for ax, m in zip(axes.ravel(), MODELS):
        y, p = data[m]
        ax.hist(p[y == 0], bins=30, density=True, alpha=0.5, color="#888", label="non-conflict")
        ax.hist(p[y == 1], bins=30, density=True, alpha=0.6, color=COL[m], label="PTC")
        ax.set_title(LABEL[m], fontsize=8); ax.set_yscale("log"); ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    fig.suptitle("Calibrated detector score distribution (test)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "detector_score_dist_corrected.png", dpi=200); plt.close(fig)

    # 5. threshold vs TPR/FPR
    plt.figure(figsize=(5, 4))
    taus = np.linspace(0.01, 0.99, 99)
    for m in MODELS:
        y, p = data[m]
        pos, neg = (y == 1).sum(), (y == 0).sum()
        tpr = [((p >= t) & (y == 1)).sum() / pos for t in taus]
        fpr = [((p >= t) & (y == 0)).sum() / neg for t in taus]
        plt.plot(taus, tpr, color=COL[m], lw=1.2, label=f"{LABEL[m]} TPR")
        plt.plot(taus, fpr, color=COL[m], lw=1.0, ls="--")
    plt.axvline(0.15, color="k", ls=":", lw=0.8, label="$\\tau$=0.15")
    plt.xlabel("Threshold $\\tau$"); plt.ylabel("Rate (solid=TPR, dashed=FPR)")
    plt.title("Threshold vs TPR/FPR (test)"); plt.legend(fontsize=6); plt.tight_layout()
    plt.savefig(FIG / "detector_threshold_tpr_fpr.png", dpi=200); plt.close()

    # 6. threshold vs gated Recovery/PA (0.15/0.2/0.3 only)
    plt.figure(figsize=(5, 4))
    for m in MODELS:
        p = GATED / m / "detector_gated_test.json"
        if not p.exists():
            continue
        dg = json.loads(p.read_text()).get("detector_gated_tas", {})
        ts, recs, pas = [], [], []
        for k, v in sorted(dg.items()):
            ts.append(float(k.split("_")[1])); recs.append(v["recovery"]); pas.append(v["pa"])
        plt.plot(ts, recs, "o-", color=COL[m], ms=4, label=f"{LABEL[m]} Rec")
        plt.plot(ts, pas, "s--", color=COL[m], ms=3, alpha=0.6)
    plt.xlabel("Threshold $\\tau$"); plt.ylabel("Gated Recovery (solid) / PA (dashed)")
    plt.title("Threshold vs gated Recovery/PA (test)"); plt.legend(fontsize=6); plt.tight_layout()
    plt.savefig(FIG / "detector_threshold_gated.png", dpi=200); plt.close()

    # 7. compact summary (AUROC / AUPRC / FPR@0.15)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(MODELS)); w = 0.25
    auroc, auprc, fpr15 = [], [], []
    for m in MODELS:
        y, p = data[m]
        auroc.append(roc_auc_score(y, p)); auprc.append(average_precision_score(y, p))
        fpr15.append(((p >= 0.15) & (y == 0)).sum() / (y == 0).sum())
    ax.bar(x - w, auroc, w, label="AUROC", color="#4C72B0")
    ax.bar(x, auprc, w, label="AUPRC (true base)", color="#55A868")
    ax.bar(x + w, fpr15, w, label="FPR @ $\\tau$=0.15", color="#C44E52")
    ax.axhline(0.5, color="k", ls=":", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=15, fontsize=7)
    ax.set_title("Corrected detector summary (test)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(FIG / "detector_summary_corrected.png", dpi=200); plt.close(fig)

    print("Wrote 7 corrected detector figures to", FIG)


if __name__ == "__main__":
    main()
