#!/usr/bin/env python3
"""Held-out V1/V2/V3 Delta-construction ablation (Bucket 7, Parts B-E).

All three directions are built on TRAIN records only, alpha is selected on
VALIDATION, and each variant is evaluated ONCE on the held-out test PTC records
and the identical 200 test controls (the same corrected protocol as Bucket 5).

  V1 global      : one Delta = mean_train_PTC[h_l*(q,c_temp) - h_l*(q)]
  V2 per-relation: one Delta per Wikidata relation (fallback -> global)
  V3 per-domain  : one Delta per domain (fallback -> global)

Fallback: a relation/domain with < MIN_SUPPORT train-PTC records uses the global
direction; the same rule is applied to validation and test (never tuned on test).

    python scripts/run_variant_ablation.py --model qwen-2.5-7b --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from temporal_conflict import splits as S
from temporal_conflict.env import get_hf_token
from temporal_conflict.models import load_model
from temporal_conflict.baselines import paired as P

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "large" / "combined_all.jsonl"
CACHE = REPO / "runs" / "tas_large"
HUB = Path("/shared/models/huggingface/hub")
OUT = REPO / "results" / "variant_ablation"
MODELS = {"qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B", "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
          "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
          "llama-3.1-8b": "meta-llama/Llama-3.1-8B"}
ANSWER_SUFFIX, CAND_PREFIX = "\nAnswer:", " "
N_CONTROL, MIN_SUPPORT, SEED = 200, 10, 0


def snapshot_hash(hf_id):
    d = HUB / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    return sorted(p.name for p in d.iterdir())[0] if d.exists() else "unknown"


def load_phase1(m):
    return [json.loads(l) for l in (REPO / "results/phase1" / m /
            "per_instance.jsonl").open() if l.strip()]


def build_records(bench, scr, mrows, model, want, domain_of):
    recs = []
    for i, b in enumerate(bench):
        row = mrows[i]
        sp, ptc = row["split"], row[f"is_ptc__{model}"]
        ka = row[f"knowledge_absent__{model}"]
        take = ((want == "test_ptc" and sp == "test" and ptc) or
                (want == "val_ptc" and sp == "validation" and ptc) or
                (want == "train_ptc_filt" and sp == "train" and ptc and not ka))
        if take:
            std = scr[i]["scores"]["standard"]
            r = P.Record(i, S.record_id(b), b["subject_qid"], b["relation_pid"],
                         b.get("t_update"), b["prompt_standard"], b["prompt_temporal"],
                         b["a_old_label"], b["a_new_label"],
                         std["a_old"]["mean_logprob"], std["a_new"]["mean_logprob"],
                         bool(ptc), bool(ka))
            recs.append(r)
    return recs


def load_controls(path, model, bench, scr, mrows):
    rows = [r for r in csv.DictReader(Path(path).open()) if r[f"clean__{model}"] == "1"]
    rows.sort(key=lambda r: int(r["rank"]))
    out = []
    for r in rows[:N_CONTROL]:
        i = int(r["row_index"]); b = bench[i]; std = scr[i]["scores"]["standard"]
        out.append(P.Record(i, S.record_id(b), b["subject_qid"], b["relation_pid"],
                            b.get("t_update"), b["prompt_standard"], b["prompt_temporal"],
                            b["a_old_label"], b["a_new_label"],
                            std["a_old"]["mean_logprob"], std["a_new"]["mean_logprob"],
                            False, False))
    return out


def build_directions(loaded, train_ptc, ell, H, domain_of):
    """Return global, per_relation, per_domain (train-only) + support counts."""
    diffs, rels, doms, ids = [], [], [], []
    for rec in train_ptc:
        h_std = H[rec.row_index]
        h_tmp = P.capture_hidden(loaded, rec.prompt_temporal, ell)
        diffs.append(h_tmp - h_std); rels.append(rec.relation_id)
        doms.append(domain_of[rec.row_index]); ids.append(rec.record_id)
    diffs = np.stack(diffs)
    g = torch.tensor(diffs.mean(0), dtype=torch.float32)

    def grouped(keys):
        by = defaultdict(list)
        for k, d in zip(keys, diffs):
            by[k].append(d)
        kept, support = {}, {}
        for k, v in by.items():
            support[k] = len(v)
            if len(v) >= MIN_SUPPORT:
                kept[k] = torch.tensor(np.mean(v, 0), dtype=torch.float32)
        return kept, support

    per_rel, rel_support = grouped(rels)
    per_dom, dom_support = grouped(doms)
    meta = {"n_train": len(ids), "global_norm": float(g.norm()),
            "relation_support": rel_support, "domain_support": dom_support,
            "relation_kept": sorted(per_rel), "domain_kept": sorted(per_dom),
            "min_support": MIN_SUPPORT, "train_record_ids": ids}
    return g, per_rel, per_dom, meta


def run(model, device, resume):
    t0 = time.time()
    mdir = OUT / model; mdir.mkdir(parents=True, exist_ok=True)
    if resume and (mdir / "summary.json").exists():
        print(f"[resume] {model} done"); return
    bench = [json.loads(l) for l in BENCH.open() if l.strip()]
    scr = load_phase1(model)
    assert [S.canonical_id(b) for b in bench] == [r["instance_id"] for r in scr]
    manifest = S.load_manifest(REPO / "results/splits/subject_disjoint_v1.json")
    mrows = {int(r["row_index"]): r for r in manifest.rows}
    domain_of = {i: b["domain"] for i, b in enumerate(bench)}
    ell = int(json.loads((CACHE / model / "afr_profile.json").read_text())["ell_star"])
    H = torch.load(CACHE / model / "detector_activations.pt",
                   weights_only=False)["H"].float().numpy()
    rev = snapshot_hash(MODELS[model])

    train_ptc = build_records(bench, scr, mrows, model, "train_ptc_filt", domain_of)
    val_ptc = build_records(bench, scr, mrows, model, "val_ptc", domain_of)
    test_ptc = build_records(bench, scr, mrows, model, "test_ptc", domain_of)
    val_ctrl = load_controls(REPO / "results/splits/validation_controls_v1.csv",
                             model, bench, scr, mrows)
    test_ctrl = load_controls(REPO / "results/splits/test_controls_v1.csv",
                              model, bench, scr, mrows)
    print(f"[{model}] l*={ell} train_ptc={len(train_ptc)} val_ptc={len(val_ptc)} "
          f"test_ptc={len(test_ptc)}")

    loaded = load_model(name=model, hf_id=MODELS[model], dtype="float16",
                        device_map=device, token=get_hf_token())
    g, per_rel, per_dom, meta = build_directions(loaded, train_ptc, ell, H, domain_of)

    # save directions + metadata
    torch.save({"delta": g, "layer": ell}, mdir / "delta_global.pt")
    torch.save({"per_relation": per_rel, "global": g, "layer": ell},
               mdir / "delta_relation.pt")
    torch.save({"per_domain": per_dom, "global": g, "layer": ell},
               mdir / "delta_domain.pt")
    # assert finiteness + train-only
    for name, dd in [("global", {"g": g}), ("relation", per_rel), ("domain", per_dom)]:
        for k, v in dd.items():
            assert torch.isfinite(v).all(), f"{name}:{k} non-finite"
    manifest.assert_split_only(meta["train_record_ids"], ["train"], "variant-directions")
    # cross-check V1/V2 vs Bucket-5 train-only (same construction/seed)
    b5 = torch.load(REPO / "results/iti_paired" / model /
                    "tas_direction_trainonly.pt", weights_only=False)
    dg = float((g - b5["global_delta"]).norm() / (b5["global_delta"].norm() + 1e-9))
    meta["v1_v2_matches_bucket5_reldiff"] = dg
    (mdir / "directions_metadata.json").write_text(json.dumps({
        "model": model, "model_revision": rev, "layer": ell,
        "split_manifest_sha256": manifest.benchmark_sha256, "seed": SEED,
        "tau_rec": -3.0, "n_domains": len(per_dom),
        "generated_utc": datetime.now(timezone.utc).isoformat(), **meta}, indent=2))

    # direction_fn per variant (with fallback -> global)
    def fn_v1(rec): return g
    def fn_v2(rec): return per_rel.get(rec.relation_id, g)
    def fn_v3(rec): return per_dom.get(domain_of[rec.row_index], g)
    variants = {"v1": fn_v1, "v2": fn_v2, "v3": fn_v3}

    # --- validation alpha sweep ---
    selected = {}
    for name, fn in variants.items():
        sweep = []
        for a in P.ALPHA_GRID:
            r = P.evaluate(loaded, val_ptc, val_ctrl, ell, a, fn, ANSWER_SUFFIX, CAND_PREFIX)
            sweep.append({k: r[k] for k in ("alpha", "recovery", "pa", "J", "n_ptc", "n_control")})
        with (mdir / f"validation_sweep_{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)
        selected[name] = P.select_alpha(sweep)
        print(f"[{model}] {name} alpha*={selected[name]['alpha']} valJ={selected[name]['val_J']:.3f}")
    (mdir / "selected_alpha.json").write_text(json.dumps(selected, indent=2))

    # --- held-out test (once per variant) ---
    def fallback_for(name, rec):
        if name == "v2": return int(rec.relation_id not in per_rel)
        if name == "v3": return int(domain_of[rec.row_index] not in per_dom)
        return 0
    summary = {"model": model, "model_revision": rev, "layer": ell,
               "split_manifest_sha256": manifest.benchmark_sha256,
               "n_test_ptc": len(test_ptc), "n_test_ctrl": len(test_ctrl),
               "min_support": MIN_SUPPORT, "variants": {}}
    per_variant_rows = {}
    for name, fn in variants.items():
        a = selected[name]["alpha"]
        res = P.evaluate(loaded, test_ptc, test_ctrl, ell, a, fn, ANSWER_SUFFIX, CAND_PREFIX)
        for row in res["rows"]:
            i = next(r.row_index for r in (test_ptc + test_ctrl) if r.record_id == row["record_id"])
            rec = next(r for r in (test_ptc + test_ctrl) if r.record_id == row["record_id"])
            row.update({"model": model, "variant": name, "selected_alpha": a,
                        "domain": domain_of[i], "fallback_used": fallback_for(name, rec),
                        "layer": ell, "seed": SEED, "model_revision": rev,
                        "split_manifest_sha256": manifest.benchmark_sha256})
        per_variant_rows[name] = res["rows"]
        fb = sum(fallback_for(name, r) for r in test_ptc)
        summary["variants"][name] = {"alpha": a, "recovery": res["recovery"],
                                     "pa": res["pa"], "J": res["J"],
                                     "test_ptc_fallback": fb}
        with (mdir / f"test_{name}.jsonl").open("w") as f:
            for row in res["rows"]:
                f.write(json.dumps(row) + "\n")
        print(f"[{model}] TEST {name} a={a} Rec={res['recovery']:.3f} PA={res['pa']:.3f} fb={fb}")

    # 1:1 join assert across variants
    for subset in ("ptc", "control"):
        idsets = {n: [r["record_id"] for r in per_variant_rows[n] if r["subset"] == subset]
                  for n in variants}
        base = idsets["v1"]
        for n in variants:
            assert idsets[n] == base, f"{model} {subset} {n} id mismatch"
            assert len(base) == len(set(base)), "dup ids"
    summary["runtime_sec"] = round(time.time() - t0, 1)
    (mdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{model}] done in {summary['runtime_sec']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    for m in (list(MODELS) if a.all_models else [a.model]):
        run(m, a.device, a.resume)


if __name__ == "__main__":
    main()
