#!/usr/bin/env python3
"""Recompute detector metrics under the subject-disjoint protocol (Bucket 4).

Reuses the cached, benchmark-ordered hidden states in
``runs/tas_large/<model>/detector_activations.pt`` -- NO model forward passes.
Fits the probe on train, calibrates on calibration, selects operating
thresholds on validation, and reports on test only. Compares against the old
leaked instance-level metrics (kept as a labeled diagnostic).

    python scripts/recompute_detector_splitclean.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from temporal_conflict import splits as S
from temporal_conflict.steering.detector import fit_eval_splitclean

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "runs" / "tas_large"
OUTDIR = REPO / "results" / "detector_splitclean"
OLD = REPO / "results" / "tas"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]


def load_split_arrays(model: str, manifest: S.SplitManifest):
    """Return {split: (X, y)} using cached H indexed by manifest row_index."""
    d = torch.load(CACHE / model / "detector_activations.pt", weights_only=False)
    H = d["H"].float().numpy()
    n = H.shape[0]
    # Align: manifest row_index -> H row; is_ptc from manifest column.
    by_split = {sp: ([], []) for sp in S.SPLIT_NAMES}
    seen = set()
    for row in manifest.rows:
        i = int(row["row_index"])
        if not (0 <= i < n):
            raise AssertionError(f"{model}: row_index {i} out of range {n}")
        seen.add(i)
        sp = row["split"]
        by_split[sp][0].append(i)
        by_split[sp][1].append(int(row[f"is_ptc__{model}"]))
    if len(seen) != n:
        raise AssertionError(
            f"{model}: manifest covers {len(seen)} of {n} cached rows 1:1")
    out = {}
    for sp, (idx, y) in by_split.items():
        out[sp] = (H[np.array(idx)], np.array(y, dtype=int))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--manifest", type=Path,
                    default=REPO / "results" / "splits" /
                    f"subject_disjoint_{S.SPLIT_VERSION}.json")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = S.load_manifest(args.manifest)
    summary_rows = []
    for m in MODELS:
        arr = load_split_arrays(m, manifest)
        (Xtr, ytr) = arr["train"]
        (Xca, yca) = arr["calibration"]
        (Xte, yte) = arr["test"]
        (Xva, yva) = arr["validation"]
        det, metrics = fit_eval_splitclean(
            Xtr, ytr, Xca, yca, Xte, yte, layer=det_layer(m),
            X_val=Xva, y_val=yva)

        # Old leaked numbers for comparison.
        old_path = OLD / m / "detector_metrics.json"
        old = json.loads(old_path.read_text()) if old_path.exists() else {}
        metrics["comparison_legacy_leaked"] = {
            "note": "legacy instance-level stratify=y split; calibrated & scored "
                    "on the same held-out set; negatives subsampled (base rate "
                    "not deployment-real). AUROC is the base-rate-invariant "
                    "comparison; corrected AUPRC uses the true test base rate.",
            "legacy_auprc": old.get("auprc"),
            "legacy_auroc": old.get("auroc"),
            "legacy_ece": old.get("ece"),
            "legacy_n_test_pos": old.get("n_test_pos"),
            "legacy_n_test_neg": old.get("n_test_neg"),
        }

        mdir = args.outdir / m
        mdir.mkdir(parents=True, exist_ok=True)
        (mdir / "detector_metrics_splitclean.json").write_text(
            json.dumps(metrics, indent=2))

        cal = metrics["calibrated_test"]
        raw = metrics["raw_test"]
        op30 = next((o for o in metrics["operating_points_val_selected"]
                     if o["tau"] == 0.30), {})
        summary_rows.append({
            "model": m,
            "n_test_pos": metrics["n_test_pos"], "n_test_neg": metrics["n_test_neg"],
            "auroc_corrected": round(cal["auroc"], 4),
            "auroc_legacy": old.get("auroc"),
            "auprc_corrected_truebase": round(cal["auprc"], 4),
            "auprc_legacy_subsampled": old.get("auprc"),
            "auprc_raw_corrected": round(raw["auprc"], 4),
            "brier_corrected": round(cal["brier"], 4),
            "ece_corrected": round(cal["ece"], 4),
            "tau0.30_precision": round(op30.get("precision", float("nan")), 4),
            "tau0.30_recall": round(op30.get("recall", float("nan")), 4),
            "tau0.30_fpr": round(op30.get("fpr", float("nan")), 4),
            "tau0.30_frac_steered": round(op30.get("fraction_steered_test",
                                                   float("nan")), 4),
        })
        print(f"{m}: test pos/neg={metrics['n_test_pos']}/{metrics['n_test_neg']} "
              f"AUROC {cal['auroc']:.3f} (legacy {old.get('auroc')})  "
              f"AUPRC(true) {cal['auprc']:.3f} (legacy subsampled {old.get('auprc')})")

    cols = list(summary_rows[0].keys())
    with (args.outdir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(summary_rows)
    (args.outdir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    print(f"\nWrote {args.outdir}/summary.csv and per-model metrics.")


def det_layer(model: str) -> int:
    return int(json.loads((CACHE / model / "afr_profile.json").read_text())["ell_star"])


if __name__ == "__main__":
    main()
