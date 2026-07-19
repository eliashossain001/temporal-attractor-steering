#!/usr/bin/env python3
"""Held-out layer-steering diagnostic + layer-specific geometry (Bucket 6, D/F).

Two separate, clearly-labeled analyses, both using TRAIN-only directions and the
Bucket-5 validation-selected alpha, evaluated on held-out TEST records:

  (D) Portability: apply the SAME numeric train-only Delta_{l*} (V2 per-relation)
      at several layers and measure held-out Recovery/PA. This tests whether the
      fixed learned edit remains effective off-peak; it does NOT re-select l*.

  (F) Layer-specific geometry: construct a train-only Delta_l at each layer and
      report, on held-out test PTC, the alignment of each record's temporal
      shift with Delta_l (cosine, projection, norms). Delta_l and Delta_{l*}
      live in different residual spaces and are never compared by cosine.

    python scripts/run_layer_steering_sweep.py --model qwen-2.5-7b --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from temporal_conflict import splits as S
from temporal_conflict.env import get_hf_token
from temporal_conflict.models import load_model
from temporal_conflict.steering.hooks import decoder_layer, make_capture_hook, num_layers
from temporal_conflict.steering.steer import score_with_steering
from temporal_conflict.analysis.stats import bootstrap_ci, mcnemar_exact
from temporal_conflict.baselines import paired as P

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "large" / "combined_all.jsonl"
CACHE = REPO / "runs" / "tas_large"
PAIRED = REPO / "results" / "iti_paired"
OUT = REPO / "results" / "layer_localization_splitclean"
LOCSTATS = OUT / "localization_statistics.json"
MODELS = {"qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B", "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
          "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
          "llama-3.1-8b": "meta-llama/Llama-3.1-8B"}
ANSWER_SUFFIX, CAND_PREFIX = "\nAnswer:", " "


def build_records(model, manifest, want):
    bench = [json.loads(l) for l in BENCH.open() if l.strip()]
    scr = [json.loads(l) for l in (REPO / "results/phase1" / model /
                                   "per_instance.jsonl").open() if l.strip()]
    mrows = {int(r["row_index"]): r for r in manifest.rows}

    def mk(i):
        b, s = bench[i], scr[i]
        std = s["scores"]["standard"]
        return P.Record(i, S.record_id(b), b["subject_qid"], b["relation_pid"],
                        b.get("t_update"), b["prompt_standard"], b["prompt_temporal"],
                        b["a_old_label"], b["a_new_label"],
                        std["a_old"]["mean_logprob"], std["a_new"]["mean_logprob"],
                        bool(mrows[i][f"is_ptc__{model}"]),
                        bool(mrows[i][f"knowledge_absent__{model}"]))
    out = {}
    for i in range(len(bench)):
        sp = mrows[i]["split"]
        ptc = mrows[i][f"is_ptc__{model}"]
        ka = mrows[i][f"knowledge_absent__{model}"]
        if want == "test_ptc" and sp == "test" and ptc:
            out.setdefault("r", []).append(mk(i))
        if want == "train_ptc_filt" and sp == "train" and ptc and not ka:
            out.setdefault("r", []).append(mk(i))
    return out.get("r", [])


def load_test_controls(model, manifest):
    rows = [r for r in csv.DictReader(
        (REPO / "results/splits/test_controls_v1.csv").open())
        if r[f"clean__{model}"] == "1"]
    rows.sort(key=lambda r: int(r["rank"]))
    mrows = {int(r["row_index"]): r for r in manifest.rows}
    bench = [json.loads(l) for l in BENCH.open() if l.strip()]
    scr = [json.loads(l) for l in (REPO / "results/phase1" / model /
                                   "per_instance.jsonl").open() if l.strip()]
    recs = []
    for r in rows[:200]:
        i = int(r["row_index"]); b, s = bench[i], scr[i]
        std = s["scores"]["standard"]
        recs.append(P.Record(i, S.record_id(b), b["subject_qid"], b["relation_pid"],
                             b.get("t_update"), b["prompt_standard"], b["prompt_temporal"],
                             b["a_old_label"], b["a_new_label"],
                             std["a_old"]["mean_logprob"], std["a_new"]["mean_logprob"],
                             False, False))
    return recs


@torch.no_grad()
def capture_all_layers(loaded, prompt, n_layers):
    tok = loaded.tokenizer
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(loaded.device)
    last = ids.shape[1] - 1
    bufs = [[] for _ in range(n_layers)]
    handles = [decoder_layer(loaded.model, L).register_forward_hook(
        make_capture_hook(bufs[L], position=last)) for L in range(n_layers)]
    try:
        loaded.model(ids)
    finally:
        for h in handles:
            h.remove()
    return [b[0].squeeze(0).float().cpu().numpy() for b in bufs]


def eval_layers(model, device):
    t0 = time.time()
    manifest = S.load_manifest(REPO / "results/splits/subject_disjoint_v1.json")
    locstats = json.loads(LOCSTATS.read_text())[model]
    ell = locstats["ell_star"]
    plat = locstats["per_split"]["all"]["plateau"]["0.05"]
    lo, hi = plat["lo"], plat["hi"]

    test_ptc = build_records(model, manifest, "test_ptc")
    train_ptc = build_records(model, manifest, "train_ptc_filt")
    controls = load_test_controls(model, manifest)

    # Bucket-5 train-only TAS direction (V2 per-relation) at l*, + selected alpha.
    tas = torch.load(PAIRED / model / "tas_direction_trainonly.pt", weights_only=False)
    per_rel, global_delta = tas["per_relation"], tas["global_delta"]
    alpha = json.loads((PAIRED / model / "selected_alpha.json").read_text())["tas"]["alpha"]

    def tas_dir(rec):
        return per_rel.get(rec.relation_id, global_delta)

    loaded = load_model(name=model, hf_id=MODELS[model], dtype="float16",
                        device_map=device, token=get_hf_token())
    L = num_layers(loaded.model)

    # eval layers: null, before-plateau, plateau-lo, l*, plateau-hi, final
    cand = sorted({max(2, ell // 8), max(0, lo - 2), lo, ell, hi, L - 1})
    print(f"[{model}] l*={ell} plateau05={lo}-{hi} eval_layers={cand} "
          f"alpha={alpha} n_test_ptc={len(test_ptc)}")

    # --- (D) portability sweep ---
    mdir = OUT / model; mdir.mkdir(parents=True, exist_ok=True)
    per_layer_rows = {}
    sweep = []
    for layer in cand:
        ptc_ind, ctrl_ind = [], []
        rows = []
        for rec in test_ptc:
            d = tas_dir(rec)
            o_s, o_n = score_with_steering(loaded, rec.prompt_standard + ANSWER_SUFFIX,
                                           rec.a_old_label, layer, d, alpha, candidate_prefix=CAND_PREFIX)
            n_s, n_n = score_with_steering(loaded, rec.prompt_standard + ANSWER_SUFFIX,
                                           rec.a_new_label, layer, d, alpha, candidate_prefix=CAND_PREFIX)
            rec_new = int((n_s / max(n_n, 1)) > (o_s / max(o_n, 1)))
            ptc_ind.append(rec_new)
            rows.append({"record_id": rec.record_id, "layer": layer, "subset": "ptc",
                         "recovery_indicator": rec_new})
        for rec in controls:
            d = tas_dir(rec)
            o_s, o_n = score_with_steering(loaded, rec.prompt_standard + ANSWER_SUFFIX,
                                           rec.a_old_label, layer, d, alpha, candidate_prefix=CAND_PREFIX)
            n_s, n_n = score_with_steering(loaded, rec.prompt_standard + ANSWER_SUFFIX,
                                           rec.a_new_label, layer, d, alpha, candidate_prefix=CAND_PREFIX)
            post_new = (n_s / max(n_n, 1)) > (o_s / max(o_n, 1))
            base_new = rec.std_new > rec.std_old
            pres = int(post_new == base_new)
            ctrl_ind.append(pres)
            rows.append({"record_id": rec.record_id, "layer": layer, "subset": "control",
                         "preservation_indicator": pres})
        per_layer_rows[layer] = rows
        rec_p, rlo, rhi = bootstrap_ci(ptc_ind)
        pa_p, plo, phi = bootstrap_ci(ctrl_ind)
        sweep.append({"layer": layer, "is_ell_star": layer == ell,
                      "n_ptc": len(test_ptc), "n_ctrl": len(controls),
                      "recovery": round(rec_p, 4), "rec_ci_lo": round(rlo, 4),
                      "rec_ci_hi": round(rhi, 4), "pa": round(pa_p, 4),
                      "pa_ci_lo": round(plo, 4), "pa_ci_hi": round(phi, 4),
                      "J": round(rec_p - (1 - pa_p), 4)})
        print(f"    layer {layer}: Recovery={rec_p:.3f} PA={pa_p:.3f}")
    with (mdir / "steering_by_layer_test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)

    # paired McNemar vs l* (Recovery + PA)
    paired = {}
    star_rec = {r["record_id"]: r["recovery_indicator"]
                for r in per_layer_rows[ell] if r["subset"] == "ptc"}
    star_pa = {r["record_id"]: r["preservation_indicator"]
               for r in per_layer_rows[ell] if r["subset"] == "control"}
    for layer in cand:
        if layer == ell:
            continue
        lr = {r["record_id"]: r["recovery_indicator"]
              for r in per_layer_rows[layer] if r["subset"] == "ptc"}
        lp = {r["record_id"]: r["preservation_indicator"]
              for r in per_layer_rows[layer] if r["subset"] == "control"}
        ids_r = sorted(star_rec)
        ids_p = sorted(star_pa)
        mc_r = mcnemar_exact([star_rec[i] for i in ids_r], [lr[i] for i in ids_r])
        mc_p = mcnemar_exact([star_pa[i] for i in ids_p], [lp[i] for i in ids_p])
        paired[str(layer)] = {
            "vs_ell_star": ell,
            "recovery_ellstar_minus_layer": round(
                np.mean([star_rec[i] for i in ids_r]) - np.mean([lr[i] for i in ids_r]), 4),
            "rec_discordant_star/layer": f"{mc_r['a_wins']}/{mc_r['b_wins']}",
            "rec_mcnemar_p": mc_r["p_value"],
            "pa_discordant_star/layer": f"{mc_p['a_wins']}/{mc_p['b_wins']}",
            "pa_mcnemar_p": mc_p["p_value"]}
    (mdir / "paired_layer_tests.json").write_text(json.dumps(paired, indent=2))

    # --- (F) layer-specific geometry (train-only Delta_l) ---
    geometry(loaded, model, ell, train_ptc, test_ptc, L, mdir)

    (mdir / "layer_sweep_meta.json").write_text(json.dumps({
        "model": model, "ell_star": ell, "alpha": alpha, "eval_layers": cand,
        "plateau05": [lo, hi], "runtime_sec": round(time.time() - t0, 1),
        "device": device}, indent=2))
    print(f"[{model}] done in {round(time.time()-t0,1)}s")


def geometry(loaded, model, ell, train_ptc, test_ptc, L, mdir):
    # train-only Delta_l per layer
    tr_std = [capture_all_layers(loaded, r.prompt_standard + ANSWER_SUFFIX, L) for r in train_ptc]
    tr_tmp = [capture_all_layers(loaded, r.prompt_temporal + ANSWER_SUFFIX, L) for r in train_ptc]
    delta_l = []
    for layer in range(L):
        diffs = np.stack([tr_tmp[k][layer] - tr_std[k][layer] for k in range(len(train_ptc))])
        delta_l.append(diffs.mean(axis=0))
    # held-out test geometry
    te_std = [capture_all_layers(loaded, r.prompt_standard + ANSWER_SUFFIX, L) for r in test_ptc]
    te_tmp = [capture_all_layers(loaded, r.prompt_temporal + ANSWER_SUFFIX, L) for r in test_ptc]
    rows = []
    for layer in range(L):
        dl = delta_l[layer]; dln = np.linalg.norm(dl) + 1e-9
        coss, projs, shiftnorms = [], [], []
        for k in range(len(test_ptc)):
            s = te_tmp[k][layer] - te_std[k][layer]
            sn = np.linalg.norm(s) + 1e-9
            coss.append(float(np.dot(s, dl) / (sn * dln)))
            projs.append(float(np.dot(s, dl) / dln))
            shiftnorms.append(float(sn))
        rows.append({"layer": layer, "delta_l_norm": round(float(dln), 4),
                     "mean_shift_norm": round(float(np.mean(shiftnorms)), 4),
                     "mean_cosine_shift_vs_deltal": round(float(np.mean(coss)), 4),
                     "mean_projection": round(float(np.mean(projs)), 4),
                     "is_ell_star": layer == ell, "n_test": len(test_ptc)})
    with (mdir / "geometry_by_layer.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    todo = list(MODELS) if args.all_models else [args.model]
    for m in todo:
        if args.resume and (OUT / m / "layer_sweep_meta.json").exists():
            print(f"[resume] {m} done, skip"); continue
        eval_layers(m, args.device)


if __name__ == "__main__":
    main()
