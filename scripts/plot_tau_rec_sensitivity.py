#!/usr/bin/env python3
"""tau_rec sensitivity: relation-level stability, figure, and LaTeX tables (Bucket 3).

Reads the cached phase-1 scores + the sensitivity artifacts and produces the
relation sensitivity table, a 4-panel figure (kept fraction, filtered PTC rate,
held-out PTC count, fixed-intervention TAS Recovery), and regenerated LaTeX
tables. All values come from files.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from temporal_conflict import splits as S

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "tau_rec_sensitivity"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
NAMES = {"qwen-2.5-1.5b": "Qwen-2.5-1.5B", "qwen-2.5-7b": "Qwen-2.5-7B",
         "mistral-7b-v0.3": "Mistral-7B-v0.3", "llama-3.1-8b": "Llama-3.1-8B"}
GRID = [-2.0, -2.5, -3.0, -3.5, -4.0]
PRIMARY = -3.0
RELS = ["P35", "P169", "P6", "P286", "P488"]


def load_phase1(m):
    return [json.loads(l) for l in
            (REPO / "results/phase1" / m / "per_instance.jsonl").open() if l.strip()]


def relation_sensitivity():
    rows = []
    rank_stability = {}
    for m in MODELS:
        p1 = load_phase1(m)
        per_tau_rate = {}
        for tau in GRID:
            by_rel_kept = defaultdict(int); by_rel_fptc = defaultdict(int)
            for r in p1:
                if r["scores"]["temporal"]["a_new"]["mean_logprob"] >= tau:
                    by_rel_kept[r["relation_pid"]] += 1
                    if r["is_ptc"]:
                        by_rel_fptc[r["relation_pid"]] += 1
            rates = {rel: (by_rel_fptc[rel] / by_rel_kept[rel] if by_rel_kept[rel] else 0.0)
                     for rel in RELS}
            per_tau_rate[tau] = rates
            for rel in RELS:
                rows.append({"model": m, "tau_rec": tau, "relation": rel,
                             "kept": by_rel_kept[rel], "filtered_ptc": by_rel_fptc[rel],
                             "fptc_rate_among_kept": round(rates[rel], 4)})
        # rank of P35/P169 across taus (1 = strongest per-record signal)
        rank_stability[m] = {}
        for tau in GRID:
            order = sorted(RELS, key=lambda rel: -per_tau_rate[tau][rel])
            rank_stability[m][str(tau)] = {"top2": order[:2],
                                           "P35_rank": order.index("P35") + 1,
                                           "P169_rank": order.index("P169") + 1}
    with (OUT / "relation_sensitivity.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (OUT / "relation_rank_stability.json").write_text(json.dumps(rank_stability, indent=2))
    return rank_stability


def figure():
    summ = json.loads((OUT / "sensitivity_summary.json").read_text())["filter_sweep"]
    split = list(csv.DictReader((OUT / "split_counts_by_tau.csv").open()))
    fixed = list(csv.DictReader((OUT / "fixed_intervention_tas_sensitivity.csv").open()))
    colors = {"qwen-2.5-1.5b": "#4477aa", "qwen-2.5-7b": "#66ccee",
              "mistral-7b-v0.3": "#ee6677", "llama-3.1-8b": "#228833"}
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    for m in MODELS:
        c = colors[m]
        kf = [summ[m][str(t)]["kept_frac"] * 100 for t in GRID]
        fr = [summ[m][str(t)]["filtered_ptc_rate_among_kept"] for t in GRID]
        ax[0, 0].plot(GRID, kf, "o-", color=c, label=NAMES[m], ms=4)
        ax[0, 1].plot(GRID, fr, "o-", color=c, ms=4)
        tc = [int(next(r["test_fptc"] for r in split if r["model"] == m
                       and float(r["tau_rec"]) == t)) for t in GRID]
        ax[1, 0].plot(GRID, tc, "o-", color=c, ms=4)
        rec = [float(next(r["fixed_tas_recovery"] for r in fixed if r["model"] == m
                          and float(r["tau_rec"]) == t)) for t in GRID]
        lo = [float(next(r["rec_ci_lo"] for r in fixed if r["model"] == m
                         and float(r["tau_rec"]) == t)) for t in GRID]
        hi = [float(next(r["rec_ci_hi"] for r in fixed if r["model"] == m
                         and float(r["tau_rec"]) == t)) for t in GRID]
        ax[1, 1].plot(GRID, rec, "o-", color=c, ms=4)
        ax[1, 1].fill_between(GRID, lo, hi, color=c, alpha=0.12)

    titles = [("(a) Kept fraction (%)", "kept %"),
              ("(b) Filtered PTC rate (among kept)", "rate"),
              ("(c) Held-out test filtered-PTC count", "count"),
              ("(d) Fixed-intervention TAS Recovery (held-out)", "Recovery")]
    for a, (t, yl) in zip(ax.ravel(), titles):
        a.axvline(PRIMARY, color="#888", ls="--", lw=1)
        a.set_title(t, fontsize=9); a.set_xlabel(r"$\tau_{\mathrm{rec}}$", fontsize=8)
        a.set_ylabel(yl, fontsize=8); a.tick_params(labelsize=7)
        a.set_xticks(GRID)
    ax[0, 1].set_ylim(0, None); ax[1, 1].set_ylim(0, 1)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(OUT / "tau_rec_sensitivity.png", dpi=150, bbox_inches="tight")
    fig.savefig(REPO / "TAS_AAAI27_submission" / "figures" / "tau_rec_sensitivity.png",
                dpi=150, bbox_inches="tight")


def tex_tables():
    summ = json.loads((OUT / "sensitivity_summary.json").read_text())["filter_sweep"]
    split = list(csv.DictReader((OUT / "split_counts_by_tau.csv").open()))
    # filter table (3-threshold main + full grid)
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"\hdr{Model} & \hdr{$-2$} & \hdr{$-2.5$} & \hdr{$-3$} & \hdr{$-3.5$} & \hdr{$-4$} \\",
         r"\midrule",
         r"\multicolumn{6}{l}{\emph{Filtered \PTC{} rate (\PTC{} among retained records)}} \\"]
    for m in MODELS:
        cells = " & ".join((r"\textbf{%.3f}" % summ[m][str(t)]["filtered_ptc_rate_among_kept"]
                            if t == PRIMARY else "%.3f" % summ[m][str(t)]["filtered_ptc_rate_among_kept"])
                           for t in GRID)
        L.append(f"{NAMES[m]} & {cells} \\\\")
    L += [r"\midrule", r"\multicolumn{6}{l}{\emph{Kept fraction (\% of $8{,}746$ passing the filter)}} \\"]
    for m in MODELS:
        cells = " & ".join((r"\textbf{%.1f}" % (summ[m][str(t)]["kept_frac"]*100)
                            if t == PRIMARY else "%.1f" % (summ[m][str(t)]["kept_frac"]*100))
                           for t in GRID)
        L.append(f"{NAMES[m]} & {cells} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tau_rec_table.tex").write_text("\n".join(L))

    # split table (3-threshold)
    L2 = [r"\begin{tabular}{llccccc}", r"\toprule",
          r"\hdr{Model} & \hdr{$\tau_{\mathrm{rec}}$} & \hdr{train} & \hdr{val} & \hdr{calib} & \hdr{test} & \hdr{test kept \%} \\",
          r"\midrule"]
    for m in MODELS:
        for t in (-2.5, -3.0, -3.5):
            r = next(x for x in split if x["model"] == m and float(x["tau_rec"]) == t)
            L2.append(f"{NAMES[m]} & ${t}$ & {r['train_fptc']} & {r['val_fptc']} & "
                      f"{r['calib_fptc']} & {r['test_fptc']} & {float(r['test_kept_frac'])*100:.0f} \\\\")
        L2.append(r"\midrule")
    L2[-1] = r"\bottomrule"
    L2.append(r"\end{tabular}")
    (OUT / "tau_rec_split_table.tex").write_text("\n".join(L2))


def main():
    rank = relation_sensitivity()
    figure()
    tex_tables()
    print("=== relation rank stability (P35/P169) ===")
    for m in MODELS:
        p35 = [rank[m][str(t)]["P35_rank"] for t in GRID]
        p169 = [rank[m][str(t)]["P169_rank"] for t in GRID]
        print(f"  {m}: P35 rank {p35}  P169 rank {p169}")
    print(f"\nWrote relation_sensitivity.csv, tau_rec_sensitivity.png, tau_rec_table.tex, "
          f"tau_rec_split_table.tex")


if __name__ == "__main__":
    main()
