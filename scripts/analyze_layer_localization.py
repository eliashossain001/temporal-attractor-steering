#!/usr/bin/env python3
"""Split-aware activation-patching localization analysis (Bucket 6, Parts A-C,G).

Recomputes per-layer answer-flip rate (AFR) from the cached per-instance
patching outputs, mapped onto the authoritative subject-disjoint split, so the
localization summary can be reported on held-out test subjects instead of the
full-data mixture used by the original Table 1. Also audits the old wrong-layer
control and (when steering results exist) correlates AFR with held-out steering
Recovery.

CPU-only; reuses runs/tas_large/<model>/afr_profile_per_instance.jsonl.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from temporal_conflict import splits as S
from temporal_conflict.analysis.stats import bootstrap_ci

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "runs" / "tas_large"
OUT = REPO / "results" / "layer_localization_splitclean"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
PLATEAU_THRESHOLDS = (0.03, 0.05, 0.10)
MIN_TEST_FOR_PRIMARY = 20   # below this, test AFR is descriptive only


def canon_to_split(manifest: S.SplitManifest) -> dict[str, str]:
    cs: dict[str, set] = defaultdict(set)
    for r in manifest.rows:
        cs[r["canonical_id"]].add(r["split"])
    return {c: next(iter(sp)) for c, sp in cs.items() if len(sp) == 1}


def plateau(afr_by_layer: dict[int, float], peak_layer: int, thr: float):
    """Contiguous layers around peak with AFR within `thr` of peak AFR."""
    peak = afr_by_layer[peak_layer]
    layers = sorted(afr_by_layer)
    lo = hi = peak_layer
    idx = layers.index(peak_layer)
    for j in range(idx - 1, -1, -1):
        if peak - afr_by_layer[layers[j]] <= thr:
            lo = layers[j]
        else:
            break
    for j in range(idx + 1, len(layers)):
        if peak - afr_by_layer[layers[j]] <= thr:
            hi = layers[j]
        else:
            break
    return lo, hi, hi - lo + 1


def analyze_model(model: str, cs: dict[str, str]) -> dict:
    rows = [json.loads(l) for l in
            (CACHE / model / "afr_profile_per_instance.jsonl").open()]
    # group flips by (split, layer)
    by = defaultdict(lambda: defaultdict(list))   # split -> layer -> [flip indicators]
    for r in rows:
        sp = cs.get(r["instance_id"])
        if sp is None:
            continue
        flip = int(bool(r["flipped"]))
        by[sp][r["layer"]].append(flip)
        by["all"][r["layer"]].append(flip)
    # validation+test secondary pool
    for r in rows:
        sp = cs.get(r["instance_id"])
        if sp in ("validation", "test"):
            by["val+test"][r["layer"]].append(int(bool(r["flipped"])))

    layers = sorted({r["layer"] for r in rows})
    ell_star = int(json.loads((CACHE / model / "afr_profile.json").read_text())["ell_star"])

    per_split = {}
    csv_rows = []
    for split, lm in by.items():
        afr = {L: (sum(v) / len(v) if v else 0.0) for L, v in lm.items()}
        peak_layer = max(afr, key=lambda L: afr[L])
        pl = {f"{t}": plateau(afr, peak_layer, t) for t in PLATEAU_THRESHOLDS}
        per_split[split] = {
            "n": len(next(iter(lm.values()))) if lm else 0,
            "peak_layer": peak_layer, "peak_afr": afr[peak_layer],
            "afr_at_ell_star": afr.get(ell_star),
            "plateau": {t: {"lo": pl[t][0], "hi": pl[t][1], "width": pl[t][2]}
                        for t in pl},
        }
        for L in layers:
            v = lm.get(L, [])
            p, lo, hi = bootstrap_ci(v) if v else (0.0, 0.0, 0.0)
            csv_rows.append({"split": split, "layer": L, "n": len(v),
                             "afr": round(p, 4), "ci_lo": round(lo, 4),
                             "ci_hi": round(hi, 4), "n_flips": sum(v)})
    mdir = OUT / model
    mdir.mkdir(parents=True, exist_ok=True)
    with (mdir / "afr_by_layer_splitclean.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "layer", "n", "afr",
                                          "ci_lo", "ci_hi", "n_flips"])
        w.writeheader(); w.writerows(csv_rows)

    # Part C: old wrong-layer control = ell*//8 (baselines/controls default)
    null_layer = max(2, ell_star // 8)
    test_afr = {L: (sum(by["test"].get(L, [])) / len(by["test"][L])
                    if by["test"].get(L) else None) for L in layers}
    all_afr = {L: sum(by["all"][L]) / len(by["all"][L]) for L in layers}
    wl = {"null_layer": null_layer,
          "afr_all": round(all_afr.get(null_layer, 0.0), 4),
          "afr_test": (round(test_afr[null_layer], 4)
                       if test_afr.get(null_layer) is not None else None),
          "inside_plateau": {}}
    for t in PLATEAU_THRESHOLDS:
        lo, hi, _ = per_split["all"]["plateau"][f"{t}"]["lo"], \
            per_split["all"]["plateau"][f"{t}"]["hi"], None
        wl["inside_plateau"][f"{t}"] = bool(lo <= null_layer <= hi)

    return {"model": model, "ell_star": ell_star, "n_layers": len(layers),
            "per_split": per_split, "wrong_layer_control": wl}


def main():
    manifest = S.load_manifest(REPO / "results/splits/subject_disjoint_v1.json")
    cs = canon_to_split(manifest)
    OUT.mkdir(parents=True, exist_ok=True)

    stats = {}
    xrows = []
    for m in MODELS:
        r = analyze_model(m, cs)
        stats[m] = r
        test = r["per_split"].get("test", {})
        allp = r["per_split"]["all"]
        primary = "test" if test.get("n", 0) >= MIN_TEST_FOR_PRIMARY else "val+test"
        pr = r["per_split"][primary]
        xrows.append({
            "model": m, "ell_star": r["ell_star"],
            "primary_split": primary, "primary_n": pr["n"],
            "peak_layer_primary": pr["peak_layer"],
            "peak_afr_primary": round(pr["peak_afr"], 3),
            "afr_at_ell_star_primary": round(pr["afr_at_ell_star"] or 0, 3),
            "plateau05_width_primary": pr["plateau"]["0.05"]["width"],
            "plateau05_range_primary": f"{pr['plateau']['0.05']['lo']}-{pr['plateau']['0.05']['hi']}",
            "peak_afr_alldata": round(allp["peak_afr"], 3),
            "plateau05_width_alldata": allp["plateau"]["0.05"]["width"],
            "wrong_layer": r["wrong_layer_control"]["null_layer"],
            "wrong_layer_afr_all": r["wrong_layer_control"]["afr_all"],
            "wrong_layer_inside_plateau05": r["wrong_layer_control"]["inside_plateau"]["0.05"],
        })
        print(f"{m}: l*={r['ell_star']} primary={primary}(n={pr['n']}) "
              f"peakAFR={pr['peak_afr']:.3f} plateau05={pr['plateau']['0.05']['width']} "
              f"wrong-layer={r['wrong_layer_control']['null_layer']} "
              f"(AFR_all={r['wrong_layer_control']['afr_all']}, "
              f"inside0.05={r['wrong_layer_control']['inside_plateau']['0.05']})")

    (OUT / "localization_statistics.json").write_text(json.dumps(stats, indent=2))
    with (OUT / "cross_model_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(xrows[0].keys()))
        w.writeheader(); w.writerows(xrows)
    print(f"\nWrote {OUT}/localization_statistics.json + cross_model_summary.csv")

    # Plateau-stability conclusion across thresholds (Part 13)
    print("\n=== Plateau width (all-data) by threshold ===")
    for m in MODELS:
        allp = stats[m]["per_split"]["all"]["plateau"]
        print(f"  {m}: 0.03={allp[str(0.03)]['width']} 0.05={allp[str(0.05)]['width']} "
              f"0.10={allp[str(0.10)]['width']}")


if __name__ == "__main__":
    main()
