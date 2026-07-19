#!/usr/bin/env python3
"""Build the authoritative subject-disjoint split manifest + control manifests.

Deterministic, versioned, reused by the detector, TAS, ITI, and alpha-selection.
Per-model PTC / knowledge-absent labels are read from the cached
``runs/tas_large/<model>/detector_activations.pt`` tensors (which are in exact
benchmark order), so no model forward passes are needed.

    python scripts/build_subject_disjoint_splits.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from temporal_conflict import splits as S

REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / "data" / "large" / "combined_all.jsonl"
CACHE = REPO / "runs" / "tas_large"
OUTDIR = REPO / "results" / "splits"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]
N_CONTROL = 200  # standardized clean-control count (replaces old 200/500 mix)


def git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def load_benchmark() -> list[dict]:
    return [json.loads(l) for l in BENCHMARK.open() if l.strip()]


def load_model_labels(model: str, n: int) -> dict[str, torch.Tensor]:
    """Per-record is_ptc / knowledge_absent from the cached (benchmark-ordered)
    activation tensor. Validated for length + order alignment by the caller."""
    d = torch.load(CACHE / model / "detector_activations.pt", weights_only=False)
    if len(d["instance_ids"]) != n:
        raise AssertionError(
            f"{model}: cached rows {len(d['instance_ids'])} != benchmark {n}")
    return {"instance_ids": d["instance_ids"],
            "is_ptc": d["is_ptc"], "knowledge_absent": d["knowledge_absent"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--seed", type=int, default=S.SPLIT_SEED)
    ap.add_argument("--n-control", type=int, default=N_CONTROL)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing manifest (a timestamped backup is kept)")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    bench = load_benchmark()
    n = len(bench)
    print(f"Loaded {n} benchmark records")
    bench_hash = S.sha256_file(BENCHMARK)

    # Canonical benchmark-order ids for positional cache alignment.
    bench_canon = [S.canonical_id(r) for r in bench]

    # Per-model labels, aligned by exact benchmark order.
    labels = {}
    for m in MODELS:
        lb = load_model_labels(m, n)
        if lb["instance_ids"] != bench_canon:
            raise AssertionError(f"{m}: cached instance_ids not in benchmark order")
        labels[m] = lb
    print("Cached activations align 1:1 with benchmark (order verified) for all models")

    # Build subject-disjoint split.
    subj_split = S.build_subject_splits(bench, seed=args.seed)

    # Assemble manifest rows.
    rows = []
    for i, r in enumerate(bench):
        rid = S.record_id(r)
        row = {
            "record_id": rid,
            "canonical_id": bench_canon[i],
            "row_index": i,
            "subject_qid": r["subject_qid"],
            "relation_id": r["relation_pid"],
            "update_date": r.get("t_update"),
            "split": subj_split[r["subject_qid"]],
            "grouping_key": r["subject_qid"],
            "seed": args.seed,
            "split_version": S.SPLIT_VERSION,
        }
        for m in MODELS:
            row[f"is_ptc__{m}"] = int(bool(labels[m]["is_ptc"][i]))
            row[f"knowledge_absent__{m}"] = int(bool(labels[m]["knowledge_absent"][i]))
        rows.append(row)

    # Hard structural assertions.
    S.assert_valid_partition(rows, n_expected=n)

    # Per-model PTC-positive floor in every held-out split.
    ptc_by = {m: defaultdict(int) for m in MODELS}
    for row in rows:
        for m in MODELS:
            if row[f"is_ptc__{m}"]:
                ptc_by[m][row["split"]] += 1
    for m in MODELS:
        for sp in S.HELDOUT_SPLITS:
            c = ptc_by[m][sp]
            if c < S.MIN_POS_PER_HELDOUT:
                raise AssertionError(
                    f"{m}: only {c} PTC positives in '{sp}' "
                    f"(< {S.MIN_POS_PER_HELDOUT}); common split infeasible, "
                    f"switch to model-specific manifests.")
    print("Per-model PTC-positive floor satisfied in validation/calibration/test")

    # ------------------------------------------------------------------ #
    # Write manifest (CSV + JSON) with backup-on-overwrite.
    # ------------------------------------------------------------------ #
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = args.outdir / f"subject_disjoint_{S.SPLIT_VERSION}.csv"
    json_path = args.outdir / f"subject_disjoint_{S.SPLIT_VERSION}.json"
    meta_path = args.outdir / f"subject_disjoint_{S.SPLIT_VERSION}_metadata.json"
    for p in (csv_path, json_path, meta_path):
        if p.exists() and not args.force:
            raise SystemExit(f"{p} exists; pass --force (a backup will be kept).")
        if p.exists():
            bak = p.with_suffix(p.suffix + f".bak.{ts}")
            p.rename(bak)
            print(f"Backed up {p.name} -> {bak.name}")

    fields = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(json.dumps(
        {"version": S.SPLIT_VERSION, "seed": args.seed,
         "benchmark_sha256": bench_hash, "rows": rows}, indent=2))

    # Counts for metadata.
    split_counts = Counter(r["split"] for r in rows)
    subj_by_split = defaultdict(set)
    rel_by_split = defaultdict(Counter)
    for r in rows:
        subj_by_split[r["split"]].add(r["subject_qid"])
        rel_by_split[r["split"]][r["relation_id"]] += 1

    meta = {
        "purpose": "Authoritative subject-disjoint split for detector, TAS, ITI, "
                   "alpha-selection, and paired testing (Bucket 4).",
        "split_version": S.SPLIT_VERSION,
        "algorithm": "group-by-subject_qid; stratify subjects by rarest relation; "
                     "per-stratum seeded shuffle; largest-remainder allocation",
        "grouping_unit": "subject_qid",
        "proportions": dict(S.PROPORTIONS),
        "seed": args.seed,
        "benchmark_path": str(BENCHMARK),
        "benchmark_sha256": bench_hash,
        "benchmark_record_count": n,
        "git_commit": git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "record_id_scheme": "relation:subject:old->new@t_update",
        "roles": {
            "train": "detector fit; TAS & ITI direction construction",
            "validation": "alpha / operating-point selection",
            "calibration": "isotonic calibration; detector threshold selection",
            "test": "final detector, TAS, and paired TAS-vs-ITI metrics",
        },
        "records_by_split": dict(split_counts),
        "unique_subjects_by_split": {k: len(v) for k, v in subj_by_split.items()},
        "relation_counts_by_split": {k: dict(v) for k, v in rel_by_split.items()},
        "ptc_counts_by_model_split": {
            m: {sp: ptc_by[m][sp] for sp in S.SPLIT_NAMES} for m in MODELS},
        "ptc_rate_by_model_split": {
            m: {sp: round(ptc_by[m][sp] / max(split_counts[sp], 1), 4)
                for sp in S.SPLIT_NAMES} for m in MODELS},
        "min_pos_per_heldout_required": S.MIN_POS_PER_HELDOUT,
        "n_control_standard": args.n_control,
        "control_rationale": (
            "Standardized clean-control count = 200 for both validation and "
            "test, replacing the OracleTASStage(200)/TASEvaluationStage(500) "
            "inconsistency. 200 is feasible for every model (tightest is "
            "qwen-2.5-1.5b with >=341 clean controls in test) and is the "
            "smaller of the two legacy values, avoiding overstated precision. "
            "Controls are non-PTC AND knowledge-present, drawn only from "
            "validation/test, identical for TAS and ITI within a model."),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # ------------------------------------------------------------------ #
    # Control manifests (validation + test): clean, per-model eligibility,
    # deterministic rank so TAS and ITI load identical controls.
    # ------------------------------------------------------------------ #
    for split in ("validation", "test"):
        ctrl_rows = []
        for r_i, row in enumerate(rows):
            if row["split"] != split:
                continue
            elig = {m: (not row[f"is_ptc__{m}"]) and (not row[f"knowledge_absent__{m}"])
                    for m in MODELS}
            if not any(elig.values()):
                continue
            ctrl_rows.append({
                "record_id": row["record_id"], "row_index": row["row_index"],
                "subject_qid": row["subject_qid"], "relation_id": row["relation_id"],
                "split": split,
                **{f"clean__{m}": int(elig[m]) for m in MODELS},
            })
        # deterministic rank within split (stable by record_id)
        ctrl_rows.sort(key=lambda x: x["record_id"])
        for k, cr in enumerate(ctrl_rows):
            cr["rank"] = k
        cpath = args.outdir / f"{split}_controls_{S.SPLIT_VERSION}.csv"
        if cpath.exists():
            cpath.rename(cpath.with_suffix(cpath.suffix + f".bak.{ts}"))
        with cpath.open("w", newline="") as f:
            cols = ["rank", "record_id", "row_index", "subject_qid",
                    "relation_id", "split"] + [f"clean__{m}" for m in MODELS]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(ctrl_rows)
        avail = {m: sum(cr[f"clean__{m}"] for cr in ctrl_rows) for m in MODELS}
        print(f"{split} controls: {len(ctrl_rows)} candidates; "
              f"clean-per-model {avail}")
        for m in MODELS:
            if avail[m] < args.n_control:
                print(f"  [warn] {m} has {avail[m]} < N_control={args.n_control} "
                      f"clean controls in {split}")

    print(f"\nWrote:\n  {csv_path}\n  {json_path}\n  {meta_path}")
    print(f"\nRecords by split: {dict(split_counts)}")
    print(f"Subjects by split: {{k:len(v) for k,v}} = "
          f"{ {k: len(v) for k, v in subj_by_split.items()} }")


if __name__ == "__main__":
    main()
