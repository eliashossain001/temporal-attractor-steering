#!/usr/bin/env python3
"""Leakage audit for the subject-disjoint split (Bucket 4, Part C).

Checks pairwise overlap across train/validation/calibration/test for identity
keys that MUST be disjoint (records, subjects, subject labels, subject-relation
pairs, prompts), and reports answer-entity overlap separately as descriptive
information (not prohibited: the same office-holder may validly recur across
unrelated subjects).

    python scripts/audit_split_leakage.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from temporal_conflict import splits as S

REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / "data" / "large" / "combined_all.jsonl"
OUTDIR = REPO / "results" / "splits"

# Keys that must never cross splits (subject leakage).
PROHIBITED = [
    "record_id", "subject_qid", "subject_label_norm",
    "subject_relation", "prompt_standard", "prompt_norm",
]
# Descriptive only (answer entities may validly recur).
DESCRIPTIVE = ["old_answer_qid", "new_answer_qid"]


def norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main() -> None:
    manifest = S.load_manifest(OUTDIR / f"subject_disjoint_{S.SPLIT_VERSION}.json")
    bench = {S.record_id(r): r for r in
             (json.loads(l) for l in BENCHMARK.open() if l.strip())}

    # Build per-split key sets.
    keysets: dict[str, dict[str, set]] = {
        k: defaultdict(set) for k in [*PROHIBITED, *DESCRIPTIVE]}
    for rid, split in manifest.by_record.items():
        r = bench[rid]
        vals = {
            "record_id": rid,
            "subject_qid": r["subject_qid"],
            "subject_label_norm": norm(r.get("subject_label")),
            "subject_relation": f"{r['subject_qid']}|{r['relation_pid']}",
            "prompt_standard": r.get("prompt_standard"),
            "prompt_norm": norm(r.get("prompt_standard")),
            "old_answer_qid": r.get("a_old_qid"),
            "new_answer_qid": r.get("a_new_qid"),
        }
        for k, v in vals.items():
            keysets[k][split].add(v)

    report = {"prohibited": {}, "descriptive": {}, "violations": []}
    for key in PROHIBITED:
        report["prohibited"][key] = {}
        for a, b in combinations(S.SPLIT_NAMES, 2):
            ov = len(keysets[key][a] & keysets[key][b])
            report["prohibited"][key][f"{a}|{b}"] = ov
            if ov > 0:
                report["violations"].append(
                    {"key": key, "pair": f"{a}|{b}", "overlap": ov})
    for key in DESCRIPTIVE:
        report["descriptive"][key] = {
            f"{a}|{b}": len(keysets[key][a] & keysets[key][b])
            for a, b in combinations(S.SPLIT_NAMES, 2)}

    # Adjacent-transition check: does any subject have transitions in >1 split?
    subj_splits = defaultdict(set)
    for rid, split in manifest.by_record.items():
        subj_splits[bench[rid]["subject_qid"]].add(split)
    crossing_subjects = {s: sorted(v) for s, v in subj_splits.items() if len(v) > 1}
    report["adjacent_transitions_same_subject_crossing"] = len(crossing_subjects)

    report["PASS"] = (not report["violations"]) and (not crossing_subjects)

    json_path = OUTDIR / "leakage_report.json"
    json_path.write_text(json.dumps(report, indent=2))

    # Human-readable summary.
    lines = ["# Split Leakage Audit\n",
             f"Split: subject_disjoint_{S.SPLIT_VERSION} (seed {manifest.seed})",
             f"Benchmark sha256: {manifest.benchmark_sha256[:16]}...\n",
             "## Prohibited-key overlaps (must all be 0)"]
    for key in PROHIBITED:
        mx = max(report["prohibited"][key].values())
        lines.append(f"- {key}: max pairwise overlap = {mx}  "
                     f"{'OK' if mx == 0 else 'VIOLATION'}")
    lines.append(f"- subjects crossing splits: {len(crossing_subjects)} "
                 f"{'OK' if not crossing_subjects else 'VIOLATION'}")
    lines.append("\n## Descriptive overlaps (allowed; answer entities may recur)")
    for key in DESCRIPTIVE:
        mx = max(report["descriptive"][key].values())
        lines.append(f"- {key}: max pairwise overlap = {mx} (informational)")
    lines.append(f"\n## RESULT: {'PASS' if report['PASS'] else 'FAIL'}")
    md_path = OUTDIR / "leakage_report.md"
    md_path.write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWrote {json_path}\n      {md_path}")
    if not report["PASS"]:
        raise SystemExit("LEAKAGE AUDIT FAILED")


if __name__ == "__main__":
    main()
