#!/usr/bin/env python3
"""Bucket 1: explicit detector operating-point metrics with subject-level CIs.

Reproduces the authoritative subject-disjoint detector protocol
(``fit_eval_splitclean``: probe on train, isotonic on calibration, thresholds on
VALIDATION, report on TEST) from the cached hidden states -- NO model forward
passes. Derives per-instance TEST predictions, computes the full operating-point
statistic set (TP/FP/TN/FN, precision, recall/TPR, specificity/TNR, FPR, FNR, F1,
balanced accuracy, % gated, % PTC gated, % non-conflict gated), selects thresholds
ONLY on validation, adds subject-level bootstrap 95% CIs (fixed seed), and merges
gated-TAS Recovery/PA + always-on from results/tas_splitclean/.

No threshold is ever selected on the test set.

Outputs (results/detector_splitclean/):
  operating_points.json / .csv / .md
  per_instance_test_<model>.csv   (calibrated prob, label, subject, relation)

Usage:  PYTHONPATH=src python scripts/detector_operating_points.py
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np
import torch

from temporal_conflict import splits as S
from temporal_conflict.steering.detector import fit_eval_splitclean

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "runs" / "tas_large"
OUTDIR = REPO / "results" / "detector_splitclean"
GATED = REPO / "results" / "tas_splitclean"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
LAYER = {"qwen-2.5-1.5b": 23, "qwen-2.5-7b": 23, "mistral-7b-v0.3": 30, "llama-3.1-8b": 31}
SWEEP = [round(0.01 * i, 2) for i in range(1, 100)]   # fine grid for figures/selection
BOOT_B = 10000
BOOT_SEED = 0


def load_split_arrays(model, manifest):
    d = torch.load(CACHE / model / "detector_activations.pt", weights_only=False)
    H = d["H"].float().numpy()
    by = {sp: {"idx": [], "y": [], "subj": [], "rel": []} for sp in S.SPLIT_NAMES}
    for row in manifest.rows:
        sp = row["split"]; i = int(row["row_index"])
        by[sp]["idx"].append(i)
        by[sp]["y"].append(int(row[f"is_ptc__{model}"]))
        by[sp]["subj"].append(row["subject_qid"])
        by[sp]["rel"].append(row["relation_id"])
    out = {}
    for sp, v in by.items():
        out[sp] = (H[np.array(v["idx"])], np.array(v["y"], dtype=int),
                   np.array(v["subj"]), np.array(v["rel"]))
    return out


def op_stats(pred, y):
    """Full confusion-derived stats for binary predictions pred vs labels y."""
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    pos, neg, n = tp + fn, tn + fp, len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / pos if pos else 0.0            # TPR = % PTC gated
    tnr = tn / neg if neg else 0.0
    fpr = fp / neg if neg else 0.0            # % non-conflict gated
    fnr = fn / pos if pos else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    bal = 0.5 * (rec + tnr)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, pos=pos, neg=neg, n=n,
                precision=prec, recall_tpr=rec, specificity_tnr=tnr, fpr=fpr, fnr=fnr,
                f1=f1, balanced_acc=bal,
                pct_gated=(tp + fp) / n if n else 0.0,
                pct_ptc_gated=rec, pct_nonconflict_gated=fpr)


def subject_bootstrap_ci(prob, y, subj, tau, metric_keys, B=BOOT_B, seed=BOOT_SEED):
    """95% CI by resampling SUBJECTS with replacement (records not iid within subj)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(subj)
    # index lists per subject
    by_subj = {s: np.where(subj == s)[0] for s in uniq}
    samples = {k: [] for k in metric_keys}
    for _ in range(B):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_subj[s] for s in chosen])
        pred = (prob[idx] >= tau).astype(int)
        st = op_stats(pred, y[idx])
        for k in metric_keys:
            samples[k].append(st[k])
    ci = {}
    for k in metric_keys:
        a = np.array(samples[k])
        ci[k] = [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return ci


def load_gated(model):
    p = GATED / model / "detector_gated_test.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    out = {}
    dg = d.get("detector_gated_tas", {})
    # dg keyed by 'tau_0.15' etc., each with recovery/pa/fraction_steered fields
    for k, v in (dg.items() if isinstance(dg, dict) else []):
        if isinstance(v, dict) and k.startswith("tau_"):
            tau = round(float(k.split("_", 1)[1]), 2)
            out[tau] = {
                "gated_recovery": v.get("recovery"),
                "gated_pa": v.get("pa"),
                "frac_steered_ptc": v.get("fraction_steered_ptc"),
                "frac_steered_ctrl": v.get("fraction_steered_ctrl"),
            }
    return out


def select_thresholds(cal_val, y_val):
    """Pick thresholds ONLY on validation. Returns dict name->(tau, note)."""
    best_f1, best_f1_tau = -1, 0.15
    best_j, best_j_tau = -2, 0.15
    lowfpr_tau = None
    for t in SWEEP:
        pred = (cal_val >= t).astype(int)
        st = op_stats(pred, y_val)
        if st["f1"] > best_f1:
            best_f1, best_f1_tau = st["f1"], t
        j = st["recall_tpr"] - st["fpr"]
        if j > best_j:
            best_j, best_j_tau = j, t
        if st["fpr"] <= 0.05 and lowfpr_tau is None and st["recall_tpr"] > 0:
            lowfpr_tau = t   # lowest tau achieving val FPR<=5% with nonzero recall
    sel = {
        "primary_tau0.15": (0.15, "current primary (most permissive val threshold)"),
        "val_F1_max": (best_f1_tau, f"maximises validation F1 (={best_f1:.3f})"),
        "val_youdenJ_max": (best_j_tau, f"maximises validation Youden J=TPR-FPR (={best_j:.3f})"),
    }
    if lowfpr_tau is not None:
        sel["val_FPR_le_5pct"] = (lowfpr_tau, "lowest tau with validation FPR<=5% and TPR>0")
    else:
        sel["val_FPR_le_5pct"] = (None, "NOT ATTAINABLE: no tau reaches val FPR<=5% with TPR>0")
    return sel


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = S.load_manifest(REPO / "results" / "splits" /
                               f"subject_disjoint_{S.SPLIT_VERSION}.json")
    all_rows = []
    consistency = []
    for m in MODELS:
        arr = load_split_arrays(m, manifest)
        Xtr, ytr, _, _ = arr["train"]; Xca, yca, _, _ = arr["calibration"]
        Xva, yva, _, _ = arr["validation"]; Xte, yte, subj_te, rel_te = arr["test"]
        det, metrics = fit_eval_splitclean(Xtr, ytr, Xca, yca, Xte, yte,
                                           layer=LAYER[m], X_val=Xva, y_val=yva,
                                           thresholds=SWEEP)
        # per-instance calibrated probabilities (reproduces the pipeline exactly)
        cal_test = det.calibrator.transform(det._raw_prob(Xte))
        cal_val = det.calibrator.transform(det._raw_prob(Xva))
        # consistency vs stored json
        stored = json.loads((OUTDIR / m / "detector_metrics_splitclean.json").read_text())
        consistency.append({
            "model": m,
            "auroc_recomputed": round(metrics["calibrated_test"]["auroc"], 6),
            "auroc_stored": round(stored["calibrated_test"]["auroc"], 6),
            "auprc_recomputed": round(metrics["calibrated_test"]["auprc"], 6),
            "auprc_stored": round(stored["calibrated_test"]["auprc"], 6),
            "match": abs(metrics["calibrated_test"]["auroc"] - stored["calibrated_test"]["auroc"]) < 1e-6
                     and abs(metrics["calibrated_test"]["auprc"] - stored["calibrated_test"]["auprc"]) < 1e-6,
        })
        # dump per-instance test predictions
        with (OUTDIR / f"per_instance_test_{m}.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["subject_qid", "relation_id", "is_ptc", "calibrated_prob"])
            for s, r, yy, pp in zip(subj_te, rel_te, yte, cal_test):
                w.writerow([s, r, int(yy), f"{pp:.6f}"])
        gated = load_gated(m)
        sel = select_thresholds(cal_val, yva)
        # add always-on as a non-detector upper bound
        ops = dict(sel); ops["always_on"] = (0.0, "no detector gate (mechanistic upper bound)")
        ci_keys = ["precision", "recall_tpr", "fpr", "f1", "balanced_acc", "pct_gated"]
        for name, (tau, note) in ops.items():
            if tau is None:
                all_rows.append({"model": m, "operating_point": name, "tau": None, "note": note})
                continue
            pred = (cal_test >= tau).astype(int)
            st = op_stats(pred, yte)
            ci = subject_bootstrap_ci(cal_test, yte, subj_te, tau, ci_keys)
            g = gated.get(round(tau, 2), {})
            row = {"model": m, "operating_point": name, "tau": tau, "note": note,
                   "auroc": round(metrics["calibrated_test"]["auroc"], 4),
                   "auprc_true_base": round(metrics["calibrated_test"]["auprc"], 4),
                   "brier": round(metrics["calibrated_test"]["brier"], 4),
                   "ece": round(metrics["calibrated_test"]["ece"], 4),
                   **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()},
                   "gated_recovery": g.get("gated_recovery"),
                   "gated_pa": g.get("gated_pa"),
                   "ci95": {k: [round(ci[k][0], 4), round(ci[k][1], 4)] for k in ci_keys},
                   "n_test_pos": int(yte.sum()), "n_test_neg": int((yte == 0).sum())}
            all_rows.append(row)
        print(f"{m}: AUROC {metrics['calibrated_test']['auroc']:.4f} "
              f"consistency={'OK' if consistency[-1]['match'] else 'MISMATCH'} "
              f"| val-F1 tau={sel['val_F1_max'][0]} val-J tau={sel['val_youdenJ_max'][0]} "
              f"lowFPR={sel['val_FPR_le_5pct'][0]}")

    out = {"protocol": "subject_disjoint_v1", "bootstrap_B": BOOT_B, "bootstrap_seed": BOOT_SEED,
           "bootstrap_unit": "subject_qid (subject-level resampling)",
           "threshold_selection": "validation split only; test never used for selection",
           "consistency_vs_stored_json": consistency, "operating_points": all_rows}
    (OUTDIR / "operating_points.json").write_text(json.dumps(out, indent=2))
    # CSV (flat)
    flat_cols = ["model", "operating_point", "tau", "auroc", "auprc_true_base", "brier", "ece",
                 "tp", "fp", "tn", "fn", "precision", "recall_tpr", "specificity_tnr", "fpr",
                 "fnr", "f1", "balanced_acc", "pct_gated", "gated_recovery", "gated_pa",
                 "n_test_pos", "n_test_neg", "note"]
    with (OUTDIR / "operating_points.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    # markdown
    with (OUTDIR / "operating_points.md").open("w") as f:
        f.write("# Detector operating points (subject-disjoint, val-selected)\n\n")
        f.write("Bootstrap: %d subject-level resamples, seed %d. Thresholds selected on "
                "VALIDATION only.\n\n" % (BOOT_B, BOOT_SEED))
        f.write("## Consistency vs stored JSON\n\n")
        for c in consistency:
            f.write("- %s: recomputed AUROC %.4f vs stored %.4f -> %s\n" %
                    (c["model"], c["auroc_recomputed"], c["auroc_stored"], "MATCH" if c["match"] else "MISMATCH"))
        f.write("\n## Operating points\n\n")
        f.write("| Model | OP | tau | AUROC | AUPRC | FPR [95% CI] | TPR [95% CI] | %gated | GatedRec | PA |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in all_rows:
            if r.get("tau") is None:
                f.write("| %s | %s | -- | -- | -- | NOT ATTAINABLE | | | | |\n" % (r["model"], r["operating_point"]))
                continue
            ci = r["ci95"]
            f.write("| %s | %s | %s | %.3f | %.3f | %.3f [%.3f,%.3f] | %.3f [%.3f,%.3f] | %.3f | %s | %s |\n" % (
                r["model"], r["operating_point"], r["tau"], r["auroc"], r["auprc_true_base"],
                r["fpr"], ci["fpr"][0], ci["fpr"][1], r["recall_tpr"], ci["recall_tpr"][0], ci["recall_tpr"][1],
                r["pct_gated"], r.get("gated_recovery"), r.get("gated_pa")))
    print("\nWrote results/detector_splitclean/operating_points.{json,csv,md} + per_instance_test_<model>.csv")
    bad = [c for c in consistency if not c["match"]]
    if bad:
        print("WARNING: consistency mismatch for:", [c["model"] for c in bad])


if __name__ == "__main__":
    main()
