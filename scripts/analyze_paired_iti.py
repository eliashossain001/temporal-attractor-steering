#!/usr/bin/env python3
"""Paired TAS-vs-ITI statistics on the held-out test split (Bucket 5, Part H).

Reads the aligned per-instance outputs, computes per-model Recovery and PA with
bootstrap CIs, paired differences, discordant counts, exact McNemar p-values,
and Holm-corrected p-values across the four models (separately for Recovery and
PA). Emits machine-readable + LaTeX artifacts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from temporal_conflict.analysis.stats import bootstrap_ci, mcnemar_exact

REPO = Path(__file__).resolve().parents[1]
PAIRED = REPO / "results" / "iti_paired"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni corrected p-values (keyed)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)  # enforce monotonicity
        out[k] = running
    return out


def load_pair(model: str, fname: str, indicator: str):
    """Return aligned (tas_vec, ids_tas, iti_vec, ids_iti) for one subset file."""
    rows = [json.loads(l) for l in (PAIRED / model / fname).open() if l.strip()]
    tas = {r["record_id"]: r[indicator] for r in rows if r["method"] == "tas"}
    iti = {r["record_id"]: r[indicator] for r in rows if r["method"] == "iti"}
    ids = sorted(tas)
    assert ids == sorted(iti), f"{model} {fname}: TAS/ITI id mismatch"
    return [tas[i] for i in ids], [iti[i] for i in ids], ids


def analyze(fname: str, indicator: str, metric: str):
    per_model, raw_p = {}, {}
    for m in MODELS:
        tas, iti, ids = load_pair(m, fname, indicator)
        pt, plo, phi = bootstrap_ci(tas)
        it, ilo, ihi = bootstrap_ci(iti)
        mc = mcnemar_exact(tas, iti)
        per_model[m] = {
            "n": len(ids),
            f"tas_{metric}": pt, f"tas_{metric}_ci": [plo, phi],
            f"iti_{metric}": it, f"iti_{metric}_ci": [ilo, ihi],
            "paired_diff_tas_minus_iti": pt - it,
            "tas_wins_iti_fails": mc["a_wins"], "iti_wins_tas_fails": mc["b_wins"],
            "discordant": mc["discordant"], "mcnemar_p_raw": mc["p_value"],
        }
        raw_p[m] = mc["p_value"]
    corrected = holm(raw_p)
    for m in MODELS:
        per_model[m]["mcnemar_p_holm"] = corrected[m]
    return per_model


def main():
    recovery = analyze("paired_ptc_test.jsonl", "recovery_indicator", "recovery")
    preservation = analyze("paired_clean_test.jsonl", "preservation_indicator", "pa")

    sig = {"recovery": recovery, "preservation": preservation,
           "holm_family_size": len(MODELS),
           "note": "Holm correction applied separately across the 4 models for "
                   "Recovery and for PA. Alpha selected on validation only; test "
                   "evaluated once. Effect sizes and discordant counts reported "
                   "alongside p-values."}
    (PAIRED / "paired_significance.json").write_text(json.dumps(sig, indent=2))

    # CSV
    with (PAIRED / "paired_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "n", "tas", "iti", "diff",
                    "tas_wins", "iti_wins", "discordant", "p_raw", "p_holm"])
        for metric, blk in (("recovery", recovery), ("pa", preservation)):
            for m in MODELS:
                d = blk[m]
                w.writerow([m, metric, d["n"],
                            round(d[f"tas_{metric}"], 4), round(d[f"iti_{metric}"], 4),
                            round(d["paired_diff_tas_minus_iti"], 4),
                            d["tas_wins_iti_fails"], d["iti_wins_tas_fails"],
                            d["discordant"], f"{d['mcnemar_p_raw']:.4g}",
                            f"{d['mcnemar_p_holm']:.4g}"])

    # LaTeX table
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & TAS Rec. & ITI Rec. & $\Delta$ & discord.\ (T/I) & McNemar $p$ (Holm) \\",
             r"\midrule"]
    for m in MODELS:
        d = recovery[m]
        star = "$^{*}$" if d["mcnemar_p_holm"] < 0.05 else ""
        lines.append(
            f"{m} & {d['tas_recovery']:.3f} & {d['iti_recovery']:.3f} & "
            f"{d['paired_diff_tas_minus_iti']:+.3f} & "
            f"{d['tas_wins_iti_fails']}/{d['iti_wins_tas_fails']} & "
            f"{d['mcnemar_p_holm']:.3f}{star} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PAIRED / "paired_significance.tex").write_text("\n".join(lines))

    # console
    print("=== Recovery (held-out test) ===")
    for m in MODELS:
        d = recovery[m]
        print(f"{m:16s} n={d['n']:3d} TAS={d['tas_recovery']:.3f} ITI={d['iti_recovery']:.3f} "
              f"diff={d['paired_diff_tas_minus_iti']:+.3f} "
              f"discord T/I={d['tas_wins_iti_fails']}/{d['iti_wins_tas_fails']} "
              f"p={d['mcnemar_p_raw']:.3g} holm={d['mcnemar_p_holm']:.3g}")
    print("=== Preservation (held-out test) ===")
    for m in MODELS:
        d = preservation[m]
        print(f"{m:16s} n={d['n']:3d} TAS={d['tas_pa']:.3f} ITI={d['iti_pa']:.3f} "
              f"diff={d['paired_diff_tas_minus_iti']:+.3f} "
              f"discord T/I={d['tas_wins_iti_fails']}/{d['iti_wins_tas_fails']} "
              f"p={d['mcnemar_p_raw']:.3g} holm={d['mcnemar_p_holm']:.3g}")
    print(f"\nWrote paired_summary.csv, paired_significance.json, paired_significance.tex")


if __name__ == "__main__":
    main()
