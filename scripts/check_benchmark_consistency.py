#!/usr/bin/env python3
"""Bucket 2: benchmark consistency checker.

Verifies that the released benchmark, the verification manifest, the manual-audit
sample, and the manuscript counts all agree. No network; reads local artifacts.

Checks:
  - total accepted == 8746
  - per-relation counts match the manuscript figures
  - no duplicate record IDs / no duplicate (subject,relation,transition) keys
  - old_qid != new_qid for every record
  - t_update years within [2010, 2024]
  - every record passes every deterministic rule (verification_manifest)
  - every audit-sample record_id exists in the final benchmark
  - manifest accepted count matches the benchmark

Usage:  python scripts/check_benchmark_consistency.py     (exit 0 == all pass)
"""
import csv, json, sys, collections
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data" / "large" / "combined_all.jsonl"
MANIFEST = REPO / "results" / "benchmark_verification" / "verification_manifest.csv"
AUDIT = REPO / "results" / "benchmark_audit" / "audit_sample.csv"

EXPECTED_TOTAL = 8746
EXPECTED_RELATION = {"P6": 3583, "P35": 183, "P169": 208, "P286": 2391, "P488": 2381}
YEAR_RANGE = (2010, 2024)


def rid(r):
    return f'{r["relation_pid"]}:{r["subject_qid"]}:{r["a_old_qid"]}->{r["a_new_qid"]}@{r.get("t_update")}'


def main():
    recs = [json.loads(l) for l in open(BENCH)]
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, ok, detail))

    chk("total_accepted==8746", len(recs) == EXPECTED_TOTAL, f"got {len(recs)}")
    relc = collections.Counter(r["relation_pid"] for r in recs)
    chk("relation_counts_match_manuscript", dict(relc) == EXPECTED_RELATION,
        f"got {dict(relc)}")
    ids = [rid(r) for r in recs]
    chk("no_duplicate_record_ids", len(set(ids)) == len(ids),
        f"{len(ids)-len(set(ids))} dups")
    keys = [(r["relation_pid"], r["subject_qid"], r["a_old_qid"], r["a_new_qid"], r.get("t_update")) for r in recs]
    chk("no_duplicate_transition_keys", len(set(keys)) == len(keys),
        f"{len(keys)-len(set(keys))} dups")
    chk("old_qid!=new_qid_all", all(r["a_old_qid"] != r["a_new_qid"] for r in recs))
    yrs_ok = all(YEAR_RANGE[0] <= int(r["t_update"][:4]) <= YEAR_RANGE[1] for r in recs if r.get("t_update"))
    chk("t_update_in_2010_2024", yrs_ok)

    # verification manifest agreement
    if MANIFEST.exists():
        mrows = list(csv.DictReader(open(MANIFEST)))
        chk("manifest_covers_all", len(mrows) == len(recs), f"manifest {len(mrows)}")
        all_pass = all(r["accepted"] == "True" for r in mrows)
        chk("all_records_pass_all_rules", all_pass,
            f"{sum(1 for r in mrows if r['accepted']!='True')} fail")
        rejected_have_reason = all(r["rejection_reason"] for r in mrows if r["accepted"] != "True")
        chk("rejected_have_reason", rejected_have_reason)
    else:
        chk("verification_manifest_present", False, "run build_verification_manifest.py")

    # audit sample ids subset of benchmark
    if AUDIT.exists():
        arows = list(csv.DictReader(open(AUDIT)))
        bench_ids = set(ids)
        # audit uses record_id scheme relation:subject:old->new@t
        missing = [a["record_id"] for a in arows if a.get("record_id") not in bench_ids]
        chk("audit_sample_ids_in_benchmark", len(missing) == 0,
            f"{len(missing)} audit ids not found (id-scheme mismatch tolerated if 0)")
    else:
        chk("audit_sample_present", False)

    print("# Benchmark consistency check\n")
    n_fail = 0
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
        if not ok:
            n_fail += 1
    print(f"\n{len(checks)-n_fail}/{len(checks)} checks pass.")
    sys.exit(n_fail)


if __name__ == "__main__":
    main()
