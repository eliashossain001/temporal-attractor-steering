#!/usr/bin/env python3
"""Bucket 3: unrestricted free-generation evaluation (complements probability eval).

For every held-out test verified-PTC record (and a control sample), generate a
natural answer with identical greedy decoding under each intervention:
standard / date_prefix / instruction / TAS(always-on) / detector-gated-TAS / ITI.
Steering reuses the SAME raw Delta vectors, layer, and validation-selected alpha as
the probability-based paper evaluation (make_steer_hook, position=-1, applied at
every decode step). Outputs raw generations to CSV; alias-aware scoring is done by
scripts/summarize_generation_errors.py so generation and evaluation are separable.

Runs ONE model at a time (download -> run -> the caller deletes the HF cache before
the next model, since disk is limited).

Usage:
  PYTHONPATH=src python scripts/evaluate_free_generation.py --model qwen-2.5-1.5b --device cuda:0
"""
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
import torch

from temporal_conflict import env as ENV
from temporal_conflict.models import load_model
from temporal_conflict.steering.hooks import decoder_layer, make_steer_hook, make_capture_hook
from temporal_conflict.steering.hooks import _unpack, _repack


def make_prompt_only_steer_hook(direction, alpha):
    """Steer the LAST token, but ONLY on the prompt forward pass (seq_len>1), not on
    subsequent single-token decode steps. This applies TAS/ITI at the query's
    conflict-critical position exactly as in the paper's probability scoring, then
    lets generation proceed from the shifted state (the faithful free-gen analog;
    steering at every decode step over-applies the scoring-selected alpha)."""
    def hook(module, inputs, output):
        hidden, rest = _unpack(output)
        if hidden.shape[1] <= 1:            # decode step (cached) -> no steering
            return output
        new_hidden = hidden.clone()
        delta = (alpha * direction).to(dtype=new_hidden.dtype, device=new_hidden.device)
        new_hidden[:, -1, :] = new_hidden[:, -1, :] + delta
        return _repack(new_hidden, rest)
    return hook

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "large" / "combined_all.jsonl"
SPLIT = REPO / "results" / "splits" / "subject_disjoint_v1.json"
RUNS = REPO / "runs" / "tas_large"
ITI = REPO / "results" / "iti_paired"
OUT = REPO / "results" / "free_generation"
HFID = {"qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B", "qwen-2.5-7b": "Qwen/Qwen2.5-7B",
        "mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3", "llama-3.1-8b": "meta-llama/Llama-3.1-8B"}
MAX_NEW = 25
N_CONTROL = 200


def instance_year(rec):
    for s in (rec.get("t_update"), rec.get("t_new_start")):
        if s:
            return s[:4]
    return ""


def build_prompt(method, rec):
    std = rec["prompt_standard"]
    if method == "standard":
        return std
    if method == "date_prefix":
        return f"In {instance_year(rec)}: {std}"
    if method == "instruction":
        return "Answer with the current, most up-to-date fact. " + std
    return std   # tas / gated / iti steer the standard prompt


def load_records(model):
    bench = [json.loads(l) for l in open(BENCH)]
    man = json.loads(SPLIT.read_text())
    rows = man["rows"] if "rows" in man else man
    test_ptc, controls = [], []
    for row in rows:
        if row["split"] != "test":
            continue
        b = bench[int(row["row_index"])]
        rec = {"record_id": row.get("record_id"), "relation_pid": b["relation_pid"],
               "prompt_standard": b["prompt_standard"], "prompt_temporal": b["prompt_temporal"],
               "t_update": b.get("t_update"), "t_new_start": b.get("t_new_start"),
               "a_old_qid": b["a_old_qid"], "a_old_label": b["a_old_label"],
               "a_new_qid": b["a_new_qid"], "a_new_label": b["a_new_label"]}
        if int(row[f"is_ptc__{model}"]):
            test_ptc.append(rec)
        else:
            controls.append(rec)
    return test_ptc, controls[:N_CONTROL]


@torch.no_grad()
def generate(loaded, prompt, layer, direction, alpha):
    tok, model, device = loaded.tokenizer, loaded.model, loaded.device
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    handle = None
    if direction is not None and alpha:
        h = make_prompt_only_steer_hook(direction.to(device), alpha=float(alpha))
        handle = decoder_layer(model, layer).register_forward_hook(h)
    try:
        out = model.generate(ids, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW,
                             pad_token_id=tok.eos_token_id)
    finally:
        if handle is not None:
            handle.remove()
    gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    # first NON-EMPTY line (base models often emit leading newlines before the answer)
    lines = [ln.strip() for ln in gen.split("\n") if ln.strip()]
    return lines[0] if lines else ""


@torch.no_grad()
def detector_score(loaded, det, prompt, layer):
    """Calibrated conflict score c in [0,1] for the prefix (for gated TAS)."""
    tok, model, device = loaded.tokenizer, loaded.model, loaded.device
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    buf = []
    handle = decoder_layer(model, layer).register_forward_hook(
        make_capture_hook(buf, position=ids.shape[1] - 1))
    try:
        _ = model(ids)
    finally:
        handle.remove()
    h = buf[0].squeeze(0).float().cpu().numpy()
    try:
        return float(det.score(h))
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(HFID))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gated", action="store_true", help="also run detector-gated TAS")
    ap.add_argument("--tau", type=float, default=0.15)
    args = ap.parse_args()
    m = args.model
    OUT.mkdir(parents=True, exist_ok=True)
    ENV.load_env()
    token = ENV.get_hf_token()

    # steering artifacts
    sv = torch.load(RUNS / m / "steering_vectors_v2.pt", weights_only=False)
    per_rel, global_delta, layer = sv["per_relation"], sv["global_delta"], sv["layer"]
    iti = torch.load(ITI / m / "iti_direction.pt", weights_only=False)
    iti_delta = torch.as_tensor(iti["delta"], dtype=torch.float32)
    alphas = json.loads((ITI / m / "selected_alpha.json").read_text())
    a_tas, a_iti = alphas["tas"]["alpha"], alphas["iti"]["alpha"]
    det = None
    if args.gated:
        import pickle
        try:
            det = pickle.load(open(RUNS / m / "detector.pkl", "rb"))
        except Exception as e:
            print("detector load failed, skipping gated:", e)

    def tas_dir(rec):
        return per_rel.get(rec["relation_pid"], global_delta)

    print(f"loading {m} ({HFID[m]})...")
    loaded = load_model(m, HFID[m], dtype="float16", device_map=args.device, token=token)
    snap = ""
    try:
        snap = os.path.basename(os.path.dirname(loaded.model.config._name_or_path)) if False else ""
    except Exception:
        pass

    test_ptc, controls = load_records(m)
    print(f"{m}: test-PTC {len(test_ptc)}, controls {len(controls)}, layer {layer}, "
          f"alpha_tas {a_tas}, alpha_iti {a_iti}")

    methods = ["standard", "date_prefix", "instruction", "tas", "iti"]
    if det is not None:
        methods.append("gated")

    rows_out = []
    # Controls (for PA) only need steering methods; prompt methods don't apply.
    control_methods = [m2 for m2 in methods if m2 in ("standard", "tas", "iti", "gated")]

    def run_set(recs, is_control):
        for j, rec in enumerate(recs):
            for method in (control_methods if is_control else methods):
                prompt = build_prompt(method, rec)
                direction, alpha = None, 0.0
                if method == "tas":
                    direction, alpha = tas_dir(rec), a_tas
                elif method == "iti":
                    direction, alpha = iti_delta, a_iti
                elif method == "gated":
                    c = detector_score(loaded, det, prompt, layer)
                    if c > args.tau:
                        direction, alpha = tas_dir(rec), a_tas * c
                gen = generate(loaded, prompt, layer, direction, alpha)
                rows_out.append({
                    "model": m, "method": method, "is_control": int(is_control),
                    "record_id": rec["record_id"], "relation_pid": rec["relation_pid"],
                    "a_old_qid": rec["a_old_qid"], "a_old_label": rec["a_old_label"],
                    "a_new_qid": rec["a_new_qid"], "a_new_label": rec["a_new_label"],
                    "prompt": prompt, "generated": gen})
            if (j + 1) % 20 == 0:
                print(f"  {'ctrl' if is_control else 'ptc'} {j+1}/{len(recs)}")
    run_set(test_ptc, False)
    run_set(controls, True)

    # append to a per-model CSV (so a failed later model doesn't lose earlier ones)
    path = OUT / f"generated_outputs_{m}.csv"
    cols = ["model", "method", "is_control", "record_id", "relation_pid",
            "a_old_qid", "a_old_label", "a_new_qid", "a_new_label", "prompt", "generated"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows_out)
    (OUT / f"gen_meta_{m}.json").write_text(json.dumps({
        "model": m, "hf_id": HFID[m], "layer": layer, "alpha_tas": a_tas, "alpha_iti": a_iti,
        "max_new_tokens": MAX_NEW, "decoding": "greedy (do_sample=False, num_beams=1)",
        "n_test_ptc": len(test_ptc), "n_control": len(controls), "methods": methods}, indent=2))
    print(f"wrote {path} ({len(rows_out)} generations)")


if __name__ == "__main__":
    main()
