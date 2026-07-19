#!/usr/bin/env python3
"""Paired held-out TAS-vs-ITI runner on the authoritative subject-disjoint split.

Per model: validate protocol -> build Delta_ITI (cached, CPU) and Delta_TAS
(train-only, V2 per-relation) -> validation alpha sweep + independent alpha
selection (J, tie->smaller) -> single held-out test evaluation -> corrected
detector-gated TAS on test. Baseline scores come from the cached screening
file; only steered scores use forward passes. Test-set performance is NEVER
inspected during alpha selection.

    python scripts/run_paired_tas_iti.py --all-models --device cuda:0
    python scripts/run_paired_tas_iti.py --model qwen-2.5-7b --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from temporal_conflict import splits as S
from temporal_conflict.env import get_hf_token
from temporal_conflict.models import load_model
from temporal_conflict.steering.detector import fit_eval_splitclean
from temporal_conflict.baselines.iti import build_iti_direction
from temporal_conflict.baselines import paired as P

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "large" / "combined_all.jsonl"
CACHE = REPO / "runs" / "tas_large"
HUB = Path("/shared/models/huggingface/hub")
MODELS = {
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
    "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
}
ANSWER_SUFFIX, CAND_PREFIX = "\nAnswer:", " "
N_CONTROL = 200
OP_TAUS = (0.15, 0.20, 0.30)


def snapshot_hash(hf_id: str) -> str:
    d = HUB / f"models--{hf_id.replace('/', '--')}" / "snapshots"
    return sorted(p.name for p in d.iterdir())[0] if d.exists() else "unknown"


def load_screening(model: str) -> list[dict]:
    return [json.loads(l) for l in
            (REPO / "results/phase1" / model / "per_instance.jsonl").open() if l.strip()]


def load_controls(path: Path, model: str) -> list[int]:
    rows = [r for r in csv.DictReader(path.open()) if r[f"clean__{model}"] == "1"]
    rows.sort(key=lambda r: int(r["rank"]))
    return [int(r["row_index"]) for r in rows[:N_CONTROL]]


def make_record(i: int, bench: list[dict], scr: list[dict],
                manifest_row: dict, model: str) -> P.Record:
    b, s = bench[i], scr[i]
    std = s["scores"]["standard"]
    return P.Record(
        row_index=i, record_id=S.record_id(b), subject_qid=b["subject_qid"],
        relation_id=b["relation_pid"], update_date=b.get("t_update"),
        prompt_standard=b["prompt_standard"], prompt_temporal=b["prompt_temporal"],
        a_old_label=b["a_old_label"], a_new_label=b["a_new_label"],
        std_old=std["a_old"]["mean_logprob"], std_new=std["a_new"]["mean_logprob"],
        is_ptc=bool(manifest_row[f"is_ptc__{model}"]),
        knowledge_absent=bool(manifest_row[f"knowledge_absent__{model}"]))


def protocol_validation(manifest: S.SplitManifest, outdir: Path) -> dict:
    rep = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "split_manifest_sha256": manifest.benchmark_sha256, "checks": {}}
    subs = {sp: manifest.subjects(sp) for sp in S.SPLIT_NAMES}
    ok = True
    import itertools
    for a, b in itertools.combinations(S.SPLIT_NAMES, 2):
        v = len(subs[a] & subs[b]) == 0
        rep["checks"][f"subjects_disjoint_{a}_{b}"] = v
        ok &= v
    # controls lie in their split
    for split, cpath in (("validation", args_globals["val_controls"]),
                         ("test", args_globals["test_controls"])):
        rows = list(csv.DictReader(Path(cpath).open()))
        in_split = all(r["split"] == split for r in rows)
        rep["checks"][f"{split}_controls_in_split"] = in_split
        ok &= in_split
    rep["PASS"] = ok
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "protocol_validation.json").write_text(json.dumps(rep, indent=2))
    if not ok:
        raise SystemExit("PROTOCOL VALIDATION FAILED; see protocol_validation.json")
    return rep


args_globals = {}


def run_model(model: str, manifest: S.SplitManifest, device: str,
              outroot: Path, dry_run: bool, resume: bool, seed: int) -> dict:
    t0 = time.time()
    outdir = outroot / model
    outdir.mkdir(parents=True, exist_ok=True)
    if resume and (outdir / "test_summary.json").exists():
        print(f"[resume] {model}: test_summary.json exists, skipping")
        return json.loads((outdir / "test_summary.json").read_text())

    bench = [json.loads(l) for l in BENCH.open() if l.strip()]
    scr = load_screening(model)
    assert [S.canonical_id(b) for b in bench] == [r["instance_id"] for r in scr], \
        f"{model}: screening not benchmark-aligned"
    mrows = {int(r["row_index"]): r for r in manifest.rows}

    ell = int(json.loads((CACHE / model / "afr_profile.json").read_text())["ell_star"])
    d = torch.load(CACHE / model / "detector_activations.pt", weights_only=False)
    H = d["H"].float().numpy()
    is_ptc = np.array([bool(mrows[i][f"is_ptc__{model}"]) for i in range(len(bench))])
    ka = np.array([bool(mrows[i][f"knowledge_absent__{model}"]) for i in range(len(bench))])
    prefers_new = np.array([scr[i]["scores"]["standard"]["a_new"]["mean_logprob"]
                            > scr[i]["scores"]["standard"]["a_old"]["mean_logprob"]
                            for i in range(len(bench))])
    rid_of = {i: S.record_id(bench[i]) for i in range(len(bench))}

    def rows_of(split): return [i for i in range(len(bench))
                                if mrows[i]["split"] == split]
    train_rows = rows_of("train")
    # eval PTC subsets = is_ptc (unfiltered, the paper's Recovery denominator)
    val_ptc = [make_record(i, bench, scr, mrows[i], model)
               for i in rows_of("validation") if is_ptc[i]]
    test_ptc = [make_record(i, bench, scr, mrows[i], model)
                for i in rows_of("test") if is_ptc[i]]
    # Delta_TAS train = FILTERED train PTC (is_ptc AND knowledge-present), paper protocol
    train_ptc_filt = [make_record(i, bench, scr, mrows[i], model)
                      for i in train_rows if is_ptc[i] and not ka[i]]
    val_ctrl = [make_record(i, bench, scr, mrows[i], model)
                for i in load_controls(Path(args_globals["val_controls"]), model)]
    test_ctrl = [make_record(i, bench, scr, mrows[i], model)
                 for i in load_controls(Path(args_globals["test_controls"]), model)]

    # --- Delta_ITI (CPU, train-only) ---
    iti = build_iti_direction(H, ell, train_rows, prefers_new, is_ptc, rid_of,
                              cap=500, seed=seed)
    rev = snapshot_hash(MODELS[model])
    torch.save({"delta": torch.tensor(iti.delta), "layer": ell},
               outdir / "iti_direction.pt")
    (outdir / "iti_direction_metadata.json").write_text(json.dumps({
        "model": model, "model_revision": rev, "layer": ell,
        "n_positive": iti.n_positive, "n_negative": iti.n_negative,
        "cap": iti.cap, "seed": iti.seed, "norm": iti.norm,
        "normalization": iti.normalization,
        "split_manifest_sha256": manifest.benchmark_sha256,
        "positive_record_ids": iti.positive_record_ids,
        "negative_record_ids": iti.negative_record_ids}, indent=2))

    counts = {"train_ptc_filtered": len(train_ptc_filt), "val_ptc": len(val_ptc),
              "test_ptc": len(test_ptc), "val_ctrl": len(val_ctrl),
              "test_ctrl": len(test_ctrl), "iti_pos": iti.n_positive,
              "iti_neg": iti.n_negative, "layer": ell, "model_revision": rev}
    print(f"[{model}] {counts}")
    if dry_run:
        (outdir / "dry_run_counts.json").write_text(json.dumps(counts, indent=2))
        return {"model": model, "dry_run": True, "counts": counts}

    # --- Load model, build Delta_TAS train-only ---
    loaded = load_model(name=model, hf_id=MODELS[model], dtype="float16",
                        device_map=device, token=get_hf_token())
    per_rel, global_delta, tas_meta = P.build_tas_v2_trainonly(
        loaded, train_ptc_filt, ell, H)
    torch.save({"per_relation": per_rel, "global_delta": global_delta, "layer": ell},
               outdir / "tas_direction_trainonly.pt")
    (outdir / "tas_direction_metadata.json").write_text(json.dumps({
        "model": model, "model_revision": rev, "layer": ell, "variant": "V2_per_relation",
        "construction": "mean[h_l*(q,c_temp) - h_l*(q)] over TRAIN filtered-PTC",
        "split_manifest_sha256": manifest.benchmark_sha256, **tas_meta}, indent=2))
    # Assert no non-train record contributed.
    manifest.assert_split_only(tas_meta["train_record_ids"], ["train"], "TAS-direction")
    manifest.assert_split_only(iti.positive_record_ids + iti.negative_record_ids,
                               ["train"], "ITI-direction")

    def tas_fn(rec): return P.tas_direction_for(rec, per_rel, global_delta)
    iti_delta = torch.tensor(iti.delta, dtype=torch.float32)
    def iti_fn(rec): return iti_delta

    methods = {"tas": tas_fn, "iti": iti_fn}
    selected = {}
    # --- Validation sweep + alpha selection (per method) ---
    for name, dfn in methods.items():
        sweep = []
        for a in P.ALPHA_GRID:
            r = P.evaluate(loaded, val_ptc, val_ctrl, ell, a, dfn,
                           ANSWER_SUFFIX, CAND_PREFIX)
            sweep.append({k: r[k] for k in ("alpha", "recovery", "pa", "J",
                                            "n_ptc", "n_control")})
        with (outdir / f"validation_sweep_{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            w.writeheader(); w.writerows(sweep)
        selected[name] = P.select_alpha(sweep)
        print(f"[{model}] {name} alpha*={selected[name]['alpha']} "
              f"valJ={selected[name]['val_J']:.3f}")
    (outdir / "selected_alpha.json").write_text(json.dumps(selected, indent=2))

    # --- Held-out test evaluation (once) ---
    test_rows_out = []
    test_summary = {"model": model, "model_revision": rev, "layer": ell,
                    "split_manifest_sha256": manifest.benchmark_sha256,
                    "n_test_ptc": len(test_ptc), "n_test_ctrl": len(test_ctrl),
                    "selected_alpha": {k: selected[k]["alpha"] for k in methods},
                    "methods": {}}
    per_method_rows = {}
    for name, dfn in methods.items():
        a = selected[name]["alpha"]
        res = P.evaluate(loaded, test_ptc, test_ctrl, ell, a, dfn,
                         ANSWER_SUFFIX, CAND_PREFIX)
        for row in res["rows"]:
            row.update({"model": model, "method": name, "split": "test",
                        "selected_alpha": a, "layer": ell, "seed": seed,
                        "split_manifest_sha256": manifest.benchmark_sha256,
                        "model_revision": rev})
        per_method_rows[name] = res["rows"]
        test_summary["methods"][name] = {"alpha": a, "recovery": res["recovery"],
                                         "pa": res["pa"], "J": res["J"]}
        print(f"[{model}] TEST {name} a={a} Recovery={res['recovery']:.3f} "
              f"PA={res['pa']:.3f}")

    # 1:1 join assertions across methods.
    for subset in ("ptc", "control"):
        ids = {name: [r["record_id"] for r in per_method_rows[name]
                      if r["subset"] == subset] for name in methods}
        a_ids, b_ids = ids["tas"], ids["iti"]
        assert a_ids == b_ids, f"{model} {subset}: TAS/ITI record order mismatch"
        assert len(a_ids) == len(set(a_ids)), f"{model} {subset}: duplicate ids"

    # Write aligned per-instance (both methods interleaved by record).
    def write_jsonl(path, subset):
        with path.open("w") as f:
            for name in methods:
                for r in per_method_rows[name]:
                    if r["subset"] == subset:
                        f.write(json.dumps(r) + "\n")
    write_jsonl(outdir / "paired_ptc_test.jsonl", "ptc")
    write_jsonl(outdir / "paired_clean_test.jsonl", "control")

    # --- Part I: corrected detector-gated TAS on test (no extra forward passes) ---
    detector_gated(model, manifest, mrows, H, ell, per_method_rows["tas"],
                   outroot, rev)

    test_summary["runtime_sec"] = round(time.time() - t0, 1)
    test_summary["device"] = device
    (outdir / "test_summary.json").write_text(json.dumps(test_summary, indent=2))
    print(f"[{model}] done in {test_summary['runtime_sec']}s")
    return test_summary


def detector_gated(model, manifest, mrows, H, ell, tas_test_rows, outroot, rev):
    """Corrected detector-gated end-to-end TAS on test, reusing TAS test scores."""
    def split_arr(split):
        idx = [i for i in range(H.shape[0]) if mrows[i]["split"] == split]
        y = np.array([int(mrows[i][f"is_ptc__{model}"]) for i in idx])
        return H[np.array(idx)], y, idx
    Xtr, ytr, _ = split_arr("train")
    Xca, yca, _ = split_arr("calibration")
    Xte, yte, ite = split_arr("test")
    Xva, yva, _ = split_arr("validation")
    det, metrics = fit_eval_splitclean(Xtr, ytr, Xca, yca, Xte, yte, layer=ell,
                                       X_val=Xva, y_val=yva)
    # detector score per test record_id
    score_of = {mrows[i]["record_id"]: float(det.score(H[i])) for i in ite}
    # gated recovery on test PTC (reuse TAS steered winners)
    gated = {}
    for tau in OP_TAUS:
        rec_hits = steered = 0
        ptc_rows = [r for r in tas_test_rows if r["subset"] == "ptc"]
        for r in ptc_rows:
            c = score_of.get(r["record_id"], 0.0)
            if c > tau:
                steered += 1
                rec_hits += int(r["post_edit_winner"] == "new")
            # else no-op: PTC prefers old -> not recovered
        pa_hits = 0
        ctrl_rows = [r for r in tas_test_rows if r["subset"] == "control"]
        steered_ctrl = 0
        for r in ctrl_rows:
            c = score_of.get(r["record_id"], 0.0)
            if c > tau:
                steered_ctrl += 1
                pa_hits += int(r["post_edit_winner"] == r["pre_edit_winner"])
            else:
                pa_hits += 1  # no-op preserves exactly
        gated[f"tau_{tau}"] = {
            "recovery": rec_hits / len(ptc_rows) if ptc_rows else 0.0,
            "pa": pa_hits / len(ctrl_rows) if ctrl_rows else 0.0,
            "fraction_steered_ptc": steered / len(ptc_rows) if ptc_rows else 0.0,
            "fraction_steered_ctrl": steered_ctrl / len(ctrl_rows) if ctrl_rows else 0.0,
        }
    outdir = outroot.parent / "tas_splitclean" / model
    outdir.mkdir(parents=True, exist_ok=True)
    op = {o["tau"]: o for o in metrics["operating_points_val_selected"]}
    (outdir / "detector_gated_test.json").write_text(json.dumps({
        "model": model, "model_revision": rev, "layer": ell,
        "detector_test": {"auprc_true_base": metrics["calibrated_test"]["auprc"],
                          "auroc": metrics["calibrated_test"]["auroc"],
                          "brier": metrics["calibrated_test"]["brier"],
                          "n_test_pos": metrics["n_test_pos"],
                          "n_test_neg": metrics["n_test_neg"]},
        "operating_points": {f"tau_{t}": op.get(t) for t in OP_TAUS},
        "detector_gated_tas": gated}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="accepted; scoring is currently unbatched")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-manifest", default=str(
        REPO / "results/splits/subject_disjoint_v1.json"))
    ap.add_argument("--validation-controls", default=str(
        REPO / "results/splits/validation_controls_v1.csv"))
    ap.add_argument("--test-controls", default=str(
        REPO / "results/splits/test_controls_v1.csv"))
    ap.add_argument("--output-dir", default=str(REPO / "results/iti_paired"))
    args = ap.parse_args()

    args_globals["val_controls"] = args.validation_controls
    args_globals["test_controls"] = args.test_controls
    outroot = Path(args.output_dir)
    manifest = S.load_manifest(args.split_manifest)
    protocol_validation(manifest, outroot)

    todo = list(MODELS) if args.all_models else [args.model]
    if not todo or todo == [None]:
        raise SystemExit("pass --model <name> or --all-models")
    print(f"start {datetime.now(timezone.utc).isoformat()} device={args.device} "
          f"models={todo}")
    for m in todo:
        run_model(m, manifest, args.device, outroot, args.dry_run, args.resume, args.seed)
    print(f"end {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
