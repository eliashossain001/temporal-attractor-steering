#!/usr/bin/env python3
"""Reproducible tau_rec sensitivity analysis (Bucket 3).

Filter-level sweep recomputed from the cached, benchmark-ordered phase-1
per-instance scores (no model inference), plus a split-aware breakdown and a
fixed-intervention TAS re-filtering that reuses the Bucket-5 held-out test
scores. Does NOT change the PTC definition or the length-normalized scoring.

    python scripts/analyze_tau_rec_sensitivity.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from temporal_conflict import splits as S
from temporal_conflict.analysis.stats import bootstrap_ci

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "large" / "combined_all.jsonl"
OUT = REPO / "results" / "tau_rec_sensitivity"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
GRID = [-2.0, -2.5, -3.0, -3.5, -4.0]
PRIMARY = -3.0
BOOT_B, BOOT_SEED = 10000, 0


def load_phase1(model):
    return [json.loads(l) for l in
            (REPO / "results/phase1" / model / "per_instance.jsonl").open() if l.strip()]


def new_temp_lp(r):
    return r["scores"]["temporal"]["a_new"]["mean_logprob"]


def new_std_lp(r):
    return r["scores"]["standard"]["a_new"]["mean_logprob"]


def filter_sweep(model, rows):
    n = len(rows)
    raw_ptc = sum(r["is_ptc"] for r in rows)
    out = {}
    for tau in GRID:
        kept = [r for r in rows if new_temp_lp(r) >= tau]
        fptc = [r for r in kept if r["is_ptc"]]
        # temporal recovery on filtered PTC is 1.0 by construction (is_ptc
        # already requires temporal new>old); reported for completeness.
        temp_rec = (sum(r["recovers_new_under_temporal"] for r in fptc) / len(fptc)
                    if fptc else 0.0)
        opr = (sum(r["prefers_old_under_standard"] for r in kept) / len(kept)
               if kept else 0.0)
        eg = float(np.mean([new_temp_lp(r) - new_std_lp(r) for r in kept])) if kept else 0.0
        # filtered PTC rate, definition B (among kept) = paper's definition
        indic_B = [1 if r["is_ptc"] else 0 for r in kept]
        rateB, loB, hiB = bootstrap_ci(indic_B, b=BOOT_B, seed=BOOT_SEED)
        out[tau] = {
            "n_benchmark": n, "kept": len(kept), "kept_frac": len(kept) / n,
            "raw_ptc": raw_ptc, "raw_ptc_rate": raw_ptc / n,
            "filtered_ptc": len(fptc),
            "filtered_ptc_rate_among_kept": rateB,      # definition B (paper)
            "fptc_rate_B_ci": [loB, hiB],
            "filtered_ptc_rate_over_benchmark": len(fptc) / n,   # definition A
            "opr_among_kept": opr, "temporal_recovery_on_fptc": temp_rec,
            "elicitation_gap_kept": eg,
            "fptc_ids": set(r["instance_id"] for r in fptc),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bench = [json.loads(l) for l in BENCH.open() if l.strip()]
    manifest = S.load_manifest(REPO / "results/splits/subject_disjoint_v1.json")
    split_of_row = {int(r["row_index"]): r["split"] for r in manifest.rows}

    sweeps = {}
    filt_rows, split_rows, overlap = [], [], {}
    for m in MODELS:
        rows = load_phase1(m)
        assert len(rows) == 8746, f"{m}: {len(rows)} != 8746"
        # benchmark alignment
        assert [r["instance_id"] for r in rows] == [S.canonical_id(b) for b in bench], \
            f"{m}: phase-1 not benchmark-aligned"
        sw = filter_sweep(m, rows)
        sweeps[m] = sw
        for tau in GRID:
            d = sw[tau]
            filt_rows.append({"model": m, "tau_rec": tau,
                              "n_benchmark": d["n_benchmark"], "kept": d["kept"],
                              "kept_frac": round(d["kept_frac"], 4),
                              "raw_ptc": d["raw_ptc"],
                              "filtered_ptc": d["filtered_ptc"],
                              "fptc_rate_among_kept": round(d["filtered_ptc_rate_among_kept"], 4),
                              "fptc_rate_ci_lo": round(d["fptc_rate_B_ci"][0], 4),
                              "fptc_rate_ci_hi": round(d["fptc_rate_B_ci"][1], 4),
                              "fptc_rate_over_benchmark": round(d["filtered_ptc_rate_over_benchmark"], 4),
                              "opr_among_kept": round(d["opr_among_kept"], 4),
                              "elicitation_gap_kept": round(d["elicitation_gap_kept"], 4)})
        # Part C: split-aware filtered-PTC counts (is_ptc AND kept)
        for tau in GRID:
            kept_ptc_by_split = Counter()
            kept_by_split = Counter()
            for i, r in enumerate(rows):
                sp = split_of_row[i]
                if new_temp_lp(r) >= tau:
                    kept_by_split[sp] += 1
                    if r["is_ptc"]:
                        kept_ptc_by_split[sp] += 1
            split_rows.append({
                "model": m, "tau_rec": tau,
                "train_fptc": kept_ptc_by_split["train"],
                "val_fptc": kept_ptc_by_split["validation"],
                "calib_fptc": kept_ptc_by_split["calibration"],
                "test_fptc": kept_ptc_by_split["test"],
                "test_kept": kept_by_split["test"],
                "test_kept_frac": round(kept_by_split["test"] /
                                        sum(1 for i in range(len(rows))
                                            if split_of_row[i] == "test"), 4),
                "small_test_flag": kept_ptc_by_split["test"] < 15})
        # Part 27: PTC-set overlap vs primary
        base = sw[PRIMARY]["fptc_ids"]
        ov = {}
        for tau in GRID:
            s = sw[tau]["fptc_ids"]
            inter = len(base & s); union = len(base | s)
            ov[str(tau)] = {"n": len(s), "retained_from_primary": inter,
                            "added": len(s - base), "removed": len(base - s),
                            "jaccard": round(inter / union, 4) if union else 1.0}
        overlap[m] = ov

    # write filter + split CSVs
    with (OUT / "filter_sweep_all_models.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filt_rows[0].keys())); w.writeheader(); w.writerows(filt_rows)
    with (OUT / "split_counts_by_tau.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(split_rows[0].keys())); w.writeheader(); w.writerows(split_rows)
    (OUT / "ptc_set_overlap.json").write_text(json.dumps(overlap, indent=2))

    # Part D: fixed-intervention TAS re-filtering (reuse Bucket-5 test scores)
    rid_to_row = {r["record_id"]: int(r["row_index"]) for r in manifest.rows}
    fixed_rows = []
    for m in MODELS:
        p1 = load_phase1(m)
        tas_rows = [json.loads(l) for l in
                    (REPO / "results/iti_paired" / m / "paired_ptc_test.jsonl").open()
                    if l.strip()]
        tas_rows = [r for r in tas_rows if r["method"] == "tas"]  # test is_ptc set
        ctrl = [json.loads(l) for l in
                (REPO / "results/iti_paired" / m / "paired_clean_test.jsonl").open()
                if l.strip()]
        pa_fixed = (sum(r["preservation_indicator"] for r in ctrl if r["method"] == "tas")
                    / sum(1 for r in ctrl if r["method"] == "tas"))
        for tau in GRID:
            sub = [r for r in tas_rows
                   if new_temp_lp(p1[rid_to_row[r["record_id"]]]) >= tau]
            rec = [r["recovery_indicator"] for r in sub]
            p, lo, hi = bootstrap_ci(rec) if rec else (0.0, 0.0, 0.0)
            fixed_rows.append({"model": m, "tau_rec": tau,
                               "test_fptc_scored": len(sub),
                               "fixed_tas_recovery": round(p, 4),
                               "rec_ci_lo": round(lo, 4), "rec_ci_hi": round(hi, 4),
                               "pa_unchanged_controls": round(pa_fixed, 4)})
    with (OUT / "fixed_intervention_tas_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fixed_rows[0].keys())); w.writeheader(); w.writerows(fixed_rows)

    # Part E: stability summary
    from scipy.stats import spearmanr
    order_primary = [sweeps[m][PRIMARY]["filtered_ptc_rate_among_kept"] for m in MODELS]
    stability = {"model_order_spearman_vs_primary": {}, "jaccard_vs_primary": {},
                 "ordering_note": ""}
    for tau in GRID:
        order_tau = [sweeps[m][tau]["filtered_ptc_rate_among_kept"] for m in MODELS]
        rho, _ = spearmanr(order_primary, order_tau)
        stability["model_order_spearman_vs_primary"][str(tau)] = round(float(rho), 4)
    for m in MODELS:
        stability["jaccard_vs_primary"][m] = {str(t): overlap[m][str(t)]["jaccard"] for t in GRID}

    summary = {"grid": GRID, "primary": PRIMARY, "bootstrap_B": BOOT_B, "seed": BOOT_SEED,
               "denominator_primary": "B: filtered PTC / kept records (matches paper)",
               "filter_sweep": {m: {str(t): {k: v for k, v in sweeps[m][t].items()
                                             if k != "fptc_ids"} for t in GRID} for m in MODELS},
               "stability": stability, "overlap": overlap}
    (OUT / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2))

    # console
    print("=== Filtered PTC rate (among kept) by tau [reproduces supplement -3 col] ===")
    print("model            " + "  ".join(f"{t:>6}" for t in GRID))
    for m in MODELS:
        print(f"{m:16s} " + "  ".join(f"{sweeps[m][t]['filtered_ptc_rate_among_kept']:.3f}" for t in GRID))
    print("\n=== Kept fraction (%) ===")
    for m in MODELS:
        print(f"{m:16s} " + "  ".join(f"{100*sweeps[m][t]['kept_frac']:5.1f}" for t in GRID))
    print("\n=== model-order Spearman vs primary ===")
    print({t: stability['model_order_spearman_vs_primary'][str(t)] for t in GRID})
    print("\n=== held-out test filtered-PTC count by tau ===")
    for m in MODELS:
        tc = {t: next(r['test_fptc'] for r in split_rows if r['model']==m and r['tau_rec']==t) for t in GRID}
        print(f"  {m}: {tc}")
    print(f"\nWrote artifacts to {OUT}")


if __name__ == "__main__":
    main()
