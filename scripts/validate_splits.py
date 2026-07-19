#!/usr/bin/env python3
"""End-to-end validation of the subject-disjoint split package (Bucket 4, Part I).

Runs hard assertions and exits non-zero on any failure:
  - deterministic rerun (identical assignments)
  - exact benchmark coverage, unique record ids, valid labels, test non-empty
  - zero subject overlap across splits
  - train / calibration / test separation (control + detector reuse checks)
  - control-manifest records lie only in their split (subject-disjoint)
  - one-to-one cached-activation alignment for every model

    python scripts/validate_splits.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

from temporal_conflict import splits as S

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "large" / "combined_all.jsonl"
SPLITDIR = REPO / "results" / "splits"
CACHE = REPO / "runs" / "tas_large"
MODELS = ["qwen-2.5-1.5b", "qwen-2.5-7b", "mistral-7b-v0.3", "llama-3.1-8b"]

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def main() -> None:
    bench = [json.loads(l) for l in BENCH.open() if l.strip()]
    n = len(bench)

    # 1. Determinism.
    a = S.build_subject_splits(bench)
    b = S.build_subject_splits(bench)
    check("deterministic rerun", a == b)

    # 2. Manifest structure.
    manifest = S.load_manifest(SPLITDIR / f"subject_disjoint_{S.SPLIT_VERSION}.json")
    try:
        S.assert_valid_partition(manifest.rows, n_expected=n)
        check("exact coverage / unique ids / valid labels / test non-empty", True)
    except AssertionError as e:
        check("exact coverage / unique ids / valid labels / test non-empty", False, str(e))

    # 3. Zero subject overlap (independent recompute).
    subj_splits: dict[str, set] = {}
    for r in manifest.rows:
        subj_splits.setdefault(r["subject_qid"], set()).add(r["split"])
    crossing = [s for s, sp in subj_splits.items() if len(sp) > 1]
    check("zero subject overlap across splits", not crossing,
          f"{len(crossing)} crossing" if crossing else "")

    # 4. Manifest matches freshly-built assignment (manifest not hand-edited).
    fresh = S.build_subject_splits(bench)
    mism = [r for r in manifest.rows if fresh[r["subject_qid"]] != r["split"]]
    check("manifest == fresh deterministic build", not mism,
          f"{len(mism)} mismatched rows" if mism else "")

    # 5. Control manifests: records lie only in their declared split; subject-disjoint
    #    from train/calibration.
    train_subj = manifest.subjects("train") | manifest.subjects("calibration")
    for split in ("validation", "test"):
        path = SPLITDIR / f"{split}_controls_{S.SPLIT_VERSION}.csv"
        rows = list(csv.DictReader(path.open()))
        bad_split = [r for r in rows if manifest.by_record.get(r["record_id"]) != split]
        # subject of each control must not be in train/calibration
        bad_subj = [r for r in rows if r["subject_qid"] in train_subj]
        dup = len({r["record_id"] for r in rows}) != len(rows)
        check(f"{split} controls in-split & subject-disjoint & unique",
              not bad_split and not bad_subj and not dup,
              f"{len(bad_split)} off-split, {len(bad_subj)} train-subject, dup={dup}")

    # 6. One-to-one cached-activation alignment per model.
    bench_canon = [S.canonical_id(r) for r in bench]
    for m in MODELS:
        d = torch.load(CACHE / m / "detector_activations.pt", weights_only=False)
        ok = (len(d["instance_ids"]) == n and d["instance_ids"] == bench_canon
              and d["H"].shape[0] == n)
        check(f"cached activations 1:1 aligned ({m})", ok)

    # 7. Detector protocol reuse guard: fit(train)/calibrate(calib)/report(test)
    #    use pairwise-disjoint record sets (by construction from manifest).
    tr, ca, te = (manifest.record_ids("train"), manifest.record_ids("calibration"),
                  manifest.record_ids("test"))
    check("train/calibration/test pairwise disjoint",
          not (tr & ca) and not (tr & te) and not (ca & te))

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    print(f"\n{len(checks)-n_fail}/{len(checks)} checks passed.")
    if n_fail:
        sys.exit(f"{n_fail} VALIDATION FAILURE(S)")
    print("ALL VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
