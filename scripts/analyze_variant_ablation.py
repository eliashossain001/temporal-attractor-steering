#!/usr/bin/env python3
"""Analyze the held-out V1/V2/V3 ablation (Bucket 7, Parts F/G)."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from temporal_conflict.analysis.stats import bootstrap_ci, mcnemar_exact

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "variant_ablation"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
NAMES = {"qwen-2.5-1.5b": "Qwen-2.5-1.5B", "qwen-2.5-7b": "Qwen-2.5-7B",
         "mistral-7b-v0.3": "Mistral-7B-v0.3", "llama-3.1-8b": "Llama-3.1-8B"}
VARIANTS = ["v1", "v2", "v3"]


def holm(pv: dict) -> dict:
    items = sorted(pv.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0
    for rank, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - rank) * p)); out[k] = run
    return out


def load(model, variant, subset, indicator):
    rows = [json.loads(l) for l in (OUT / model / f"test_{variant}.jsonl").open()]
    return {r["record_id"]: r[indicator] for r in rows if r["subset"] == subset}


def main():
    summary_rows, support_rows, fallback_rows, relation_rows = [], [], [], []
    paired = {"recovery_v2_vs_v1": {}, "recovery_v2_vs_v3": {},
              "pa_v2_vs_v1": {}, "pa_v2_vs_v3": {}}
    raw_p = {k: {} for k in paired}

    for m in MODELS:
        summ = json.loads((OUT / m / "summary.json").read_text())
        meta = json.loads((OUT / m / "directions_metadata.json").read_text())
        for v in VARIANTS:
            rec = load(m, v, "ptc", "recovery_indicator")
            pa = load(m, v, "control", "preservation_indicator")
            rp, rlo, rhi = bootstrap_ci(list(rec.values()))
            pp, plo, phi = bootstrap_ci(list(pa.values()))
            summary_rows.append({"model": m, "variant": v,
                "alpha": summ["variants"][v]["alpha"], "n_ptc": len(rec),
                "recovery": round(rp, 4), "rec_ci": f"[{rlo:.2f},{rhi:.2f}]",
                "pa": round(pp, 4), "pa_ci": f"[{plo:.2f},{phi:.2f}]",
                "J": round(rp - (1 - pp), 4),
                "test_ptc_fallback": summ["variants"][v]["test_ptc_fallback"]})
        # support + fallback
        support_rows.append({"model": m, **{f"rel_{k}": v for k, v in meta["relation_support"].items()},
                             **{f"dom_{k}": v for k, v in meta["domain_support"].items()}})
        fallback_rows.append({"model": m, "min_support": meta["min_support"],
                              "relations_fallback": [r for r in meta["relation_support"]
                                                     if r not in meta["relation_kept"]],
                              "domains_fallback": [d for d in meta["domain_support"]
                                                   if d not in meta["domain_kept"]],
                              **{f"{v}_test_fallback": summ["variants"][v]["test_ptc_fallback"]
                                 for v in VARIANTS}})

        # paired McNemar V2 vs V1 / V3 (Recovery + PA)
        for tag, (a, b), subset, ind in [
            ("recovery_v2_vs_v1", ("v2", "v1"), "ptc", "recovery_indicator"),
            ("recovery_v2_vs_v3", ("v2", "v3"), "ptc", "recovery_indicator"),
            ("pa_v2_vs_v1", ("v2", "v1"), "control", "preservation_indicator"),
            ("pa_v2_vs_v3", ("v2", "v3"), "control", "preservation_indicator")]:
            da, db = load(m, a, subset, ind), load(m, b, subset, ind)
            ids = sorted(da)
            mc = mcnemar_exact([da[i] for i in ids], [db[i] for i in ids])
            paired[tag][m] = {"n": len(ids), "diff": round(mc["delta_mean"], 4),
                              "v2_wins": mc["a_wins"], "other_wins": mc["b_wins"],
                              "p_raw": mc["p_value"]}
            raw_p[tag][m] = mc["p_value"]

        # per-relation Recovery (V2) where adequate
        rows = [json.loads(l) for l in (OUT / m / "test_v2.jsonl").open()]
        byrel = defaultdict(list)
        for r in rows:
            if r["subset"] == "ptc":
                byrel[r["relation_id"]].append(r["recovery_indicator"])
        for rel, v in sorted(byrel.items()):
            relation_rows.append({"model": m, "relation": rel, "n_test": len(v),
                                  "v2_recovery": round(sum(v) / len(v), 4),
                                  "train_support": meta["relation_support"].get(rel, 0),
                                  "fell_back": int(rel not in meta["relation_kept"])})

    for tag in paired:
        corr = holm(raw_p[tag])
        for m in MODELS:
            paired[tag][m]["p_holm"] = corr[m]

    # write
    def wcsv(name, rows):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    wcsv("variant_summary.csv", summary_rows)
    wcsv("per_relation_summary.csv", relation_rows)
    wcsv("support_counts.csv", support_rows)
    (OUT / "fallback_summary.csv").write_text(
        json.dumps(fallback_rows, indent=2))  # ragged -> json
    (OUT / "paired_variant_tests.json").write_text(json.dumps(paired, indent=2))

    # variant_table.tex (main): per model V1/V2/V3 Recovery, PA, alpha
    L = [r"\begin{tabular}{lccccccc}", r"\toprule",
         r"& \multicolumn{3}{c}{\hdr{Recovery}} & \multicolumn{3}{c}{\hdr{PA}} & \\",
         r"\hdr{Model} & \hdr{V1} & \hdr{V2} & \hdr{V3} & \hdr{V1} & \hdr{V2} & \hdr{V3} & \hdr{$\alpha$ (1/2/3)} \\",
         r"\midrule"]
    S = {(r["model"], r["variant"]): r for r in summary_rows}
    for m in MODELS:
        rec = " & ".join(f"{S[(m,v)]['recovery']:.3f}" for v in VARIANTS)
        pa = " & ".join(f"{S[(m,v)]['pa']:.3f}" for v in VARIANTS)
        al = "/".join(str(int(S[(m,v)]['alpha'])) for v in VARIANTS)
        L.append(f"{NAMES[m]} & {rec} & {pa} & {al} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "variant_table.tex").write_text("\n".join(L))

    # console
    print("=== held-out V1/V2/V3 (Recovery / PA / J) ===")
    for m in MODELS:
        print(f"{NAMES[m]}:")
        for v in VARIANTS:
            r = S[(m, v)]
            print(f"  {v}: Rec={r['recovery']:.3f}{r['rec_ci']} PA={r['pa']:.3f}{r['pa_ci']} "
                  f"J={r['J']:+.3f} a={r['alpha']} fb={r['test_ptc_fallback']}")
    print("\n=== paired McNemar (V2 vs V1 / V3), Holm across models ===")
    for tag, d in paired.items():
        print(f"{tag}:")
        for m in MODELS:
            x = d[m]; print(f"  {NAMES[m]}: diff={x['diff']:+.3f} v2/other={x['v2_wins']}/{x['other_wins']} "
                            f"p={x['p_raw']:.3g} holm={x['p_holm']:.3g}")


if __name__ == "__main__":
    main()
