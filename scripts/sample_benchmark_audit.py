#!/usr/bin/env python3
"""Deterministic manual-audit sampler for the TAS / PTC benchmark (Phase 1).

Produces a *reproducible, unreviewed* audit package so a human annotator can
verify a random sample of benchmark records against Wikidata. This script does
NOT compute any accuracy number: it only draws and exports the sample. The
accuracy statement in the manuscript must be filled in later, from completed
annotations, via ``scripts/summarize_benchmark_audit.py``.

Design points
-------------
* Deterministic: a fixed, documented seed + a content hash of the benchmark
  make the sample fully reproducible offline (no network access required).
* Unique record IDs: the benchmark has no native id field, and the canonical
  ``relation:subject:old->new`` key collides for 12 records (the same holder
  pair recurring in a later term). We therefore anchor the id with the
  transition date: ``relation:subject:old->new@t_update`` (unique for all
  8,746 records). Any residual duplicate is a hard error.
* Stratified by relation with a per-relation floor, so the small relations
  (P35 head-of-state, P169 CEO) are represented well enough to audit, while
  the bulk allocation still tracks the benchmark distribution. Population
  weights are recorded in the metadata so a population-weighted estimate can
  be computed later without re-sampling.

Usage
-----
    python scripts/sample_benchmark_audit.py                 # defaults: N=200
    python scripts/sample_benchmark_audit.py --n 300 --seed 20260711
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Documented defaults
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = REPO / "data" / "large" / "combined_all.jsonl"
DEFAULT_OUTDIR = REPO / "results" / "benchmark_audit"
EXPECTED_COUNT = 8746
DEFAULT_N = 200          # see recommendation() -- +-2.4% margin at p=0.97
DEFAULT_SEED = 20260711  # fixed, documented; change only with a new audit
DEFAULT_FLOOR = 20       # min records sampled per relation (capped at pop size)

REQUIRED_FIELDS = (
    "subject_qid", "relation_pid", "a_old_qid", "a_new_qid",
    "prompt_standard", "prompt_temporal",
)

RELATION_LABELS = {
    "P6": "head of government", "P35": "head of state",
    "P169": "chief executive officer", "P286": "head coach",
    "P488": "chairperson",
}

# Controlled annotation vocabularies (also enforced by the summarizer).
AUDIT_STATUS_VALUES = ("unreviewed", "reviewed")
IS_CORRECT_VALUES = ("", "yes", "no", "ambiguous")  # "" = not yet annotated
ISSUE_CATEGORY_VALUES = (
    "", "none", "overlapping_validity", "vacancy_or_gap", "coholder",
    "interim_or_caretaker", "same_entity", "incorrect_dates",
    "incorrect_labels", "additive_not_superseding", "insufficient_evidence",
    "other",
)

CSV_COLUMNS = [
    "audit_index", "record_id",
    "subject_qid", "subject_label", "relation_id", "relation_label",
    "old_answer_qid", "old_answer_label", "new_answer_qid", "new_answer_label",
    "old_start", "old_end", "new_start", "new_end", "update_date",
    "standard_prompt", "temporal_prompt",
    "source_statement_ids", "sampling_stratum",
    # convenience identifiers for human verification (not required by spec)
    "subject_url", "old_answer_url", "new_answer_url",
    # annotation columns (blank / unreviewed on export)
    "audit_status", "is_correct_supersession", "issue_category",
    "auditor_notes",
]


# --------------------------------------------------------------------------- #
# Loading / validation
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_id(r: dict) -> str:
    """Unique, human-readable id anchored by the transition date."""
    return (f"{r['relation_pid']}:{r['subject_qid']}:"
            f"{r['a_old_qid']}->{r['a_new_qid']}@{r.get('t_update')}")


def load_benchmark(path: Path, expected_count: int) -> list[dict]:
    """Load, validate fields, assign unique ids; fail loudly on any problem."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON at line {lineno}: {e}") from e
            missing = [k for k in REQUIRED_FIELDS if not r.get(k)]
            if missing:
                raise ValueError(
                    f"Record at line {lineno} missing required field(s) "
                    f"{missing}: {r.get('subject_qid')}/{r.get('relation_pid')}"
                )
            r["_record_id"] = record_id(r)
            records.append(r)

    if not records:
        raise ValueError(f"No records loaded from {path}")

    ids = Counter(r["_record_id"] for r in records)
    dups = {k: v for k, v in ids.items() if v > 1}
    if dups:
        raise ValueError(
            f"{len(dups)} duplicate record id(s) after date-anchoring; "
            f"benchmark is malformed. Examples: {list(dups.items())[:5]}"
        )

    if expected_count and len(records) != expected_count:
        raise ValueError(
            f"Record count {len(records)} != expected {expected_count}. "
            f"Pass --expected-count 0 to override if the benchmark changed."
        )
    return records


# --------------------------------------------------------------------------- #
# Sample-size recommendation
# --------------------------------------------------------------------------- #
def _wald_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    return z * math.sqrt(p * (1 - p) / n)


def recommendation() -> str:
    """Human-readable comparison of N in {100,200,300} at high correctness."""
    lines = [
        "Sample-size recommendation (95% Wald margin of error, half-width):",
        f"  {'N':>4} | {'p=0.95':>8} | {'p=0.97':>8} | {'p=0.98':>8}",
        "  " + "-" * 40,
    ]
    for n in (100, 200, 300):
        cells = " | ".join(
            f"+-{_wald_halfwidth(p, n) * 100:4.1f}%" for p in (0.95, 0.97, 0.98)
        )
        lines.append(f"  {n:>4} | {cells}")
    lines += [
        "",
        "  Chosen default: N=200.",
        "  Rationale: at an expected correctness of ~0.97, N=200 gives a",
        "  +-2.4% margin -- tight enough to support a headline claim while",
        "  keeping the manual annotation burden modest. N=100 (+-3.3%) is",
        "  too loose for a confident claim; N=300 (+-1.9%) adds 50% more",
        "  labour for a <=0.5-point tightening. N is --n configurable.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Stratified allocation (per-relation floor + largest-remainder)
# --------------------------------------------------------------------------- #
def allocate(strata_sizes: dict[str, int], n: int, floor: int,
             stratified: bool) -> dict[str, int]:
    """Return {relation: k} summing to exactly n, each k <= stratum size."""
    total = sum(strata_sizes.values())
    if n > total:
        raise ValueError(f"Requested n={n} exceeds benchmark size {total}.")

    if not stratified:
        # Proportional (largest-remainder), no floor.
        raw = {s: n * sz / total for s, sz in strata_sizes.items()}
        alloc = {s: int(math.floor(v)) for s, v in raw.items()}
        _distribute_remainder(alloc, raw, strata_sizes, n)
        return alloc

    # Floor per stratum, capped at population.
    alloc = {s: min(floor, sz) for s, sz in strata_sizes.items()}
    if sum(alloc.values()) > n:
        raise ValueError(
            f"floor={floor} over {len(strata_sizes)} strata needs "
            f"{sum(alloc.values())} > n={n}; lower --floor or raise --n."
        )
    remaining = n - sum(alloc.values())
    # Distribute the remainder proportionally to *residual capacity*.
    cap = {s: strata_sizes[s] - alloc[s] for s in strata_sizes}
    capsum = sum(cap.values())
    raw = {s: remaining * cap[s] / capsum if capsum else 0.0 for s in cap}
    add = {s: int(math.floor(v)) for s, v in raw.items()}
    _distribute_remainder(add, raw, cap, remaining)
    for s in alloc:
        alloc[s] += add[s]
    assert sum(alloc.values()) == n, (alloc, n)
    for s in alloc:
        assert alloc[s] <= strata_sizes[s]
    return alloc


def _distribute_remainder(alloc: dict, raw: dict, caps: dict, target: int) -> None:
    """Largest-remainder top-up so sum(alloc)==target, respecting caps."""
    short = target - sum(alloc.values())
    if short <= 0:
        return
    order = sorted(raw, key=lambda s: (raw[s] - math.floor(raw[s])), reverse=True)
    i = 0
    while short > 0:
        s = order[i % len(order)]
        if alloc[s] < caps[s]:
            alloc[s] += 1
            short -= 1
        i += 1
        if i > 10 * len(order) * (target + 1):  # safety
            raise RuntimeError("remainder distribution failed to converge")


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample(records: list[dict], alloc: dict[str, int], seed: int) -> list[dict]:
    """Deterministic per-stratum sample. Independent per-stratum RNG keyed by
    (seed, relation) so changing one stratum's k does not reshuffle others.
    """
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_rel[r["relation_pid"]].append(r)

    picked: list[dict] = []
    for rel in sorted(alloc):
        k = alloc[rel]
        if k == 0:
            continue
        pool = sorted(by_rel[rel], key=lambda r: r["_record_id"])
        rng = random.Random(f"{seed}:{rel}")
        picked.extend(rng.sample(pool, k))
    # Stable, deterministic output ordering.
    picked.sort(key=lambda r: (r["relation_pid"], r["_record_id"]))
    return picked


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def to_row(idx: int, r: dict) -> dict:
    def g(k):
        v = r.get(k)
        return "" if v is None else v

    qid_url = lambda q: f"https://www.wikidata.org/wiki/{q}" if q else ""
    return {
        "audit_index": idx,
        "record_id": r["_record_id"],
        "subject_qid": g("subject_qid"),
        "subject_label": g("subject_label"),
        "relation_id": g("relation_pid"),
        "relation_label": g("relation_label") or RELATION_LABELS.get(
            r.get("relation_pid"), ""),
        "old_answer_qid": g("a_old_qid"),
        "old_answer_label": g("a_old_label"),
        "new_answer_qid": g("a_new_qid"),
        "new_answer_label": g("a_new_label"),
        "old_start": g("t_old_start"),
        "old_end": g("t_old_end"),
        "new_start": g("t_new_start"),
        "new_end": g("t_new_end"),
        "update_date": g("t_update"),
        "standard_prompt": g("prompt_standard"),
        "temporal_prompt": g("prompt_temporal"),
        "source_statement_ids": "|".join(r.get("source_wikidata_statement_ids") or []),
        "sampling_stratum": f"relation:{r.get('relation_pid')}",
        "subject_url": qid_url(r.get("subject_qid")),
        "old_answer_url": qid_url(r.get("a_old_qid")),
        "new_answer_url": qid_url(r.get("a_new_qid")),
        "audit_status": "unreviewed",
        "is_correct_supersession": "",   # blank until annotated
        "issue_category": "",            # blank until annotated
        "auditor_notes": "",
    }


def export(rows: list[dict], outdir: Path, meta: dict) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "audit_sample.csv"
    jsonl_path = outdir / "audit_sample.jsonl"
    meta_path = outdir / "audit_metadata.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_path.write_text(json.dumps(meta, indent=2))
    return {"csv": csv_path, "jsonl": jsonl_path, "metadata": meta_path}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="sample size")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--floor", type=int, default=DEFAULT_FLOOR,
                    help="min records per relation (stratified mode)")
    ap.add_argument("--expected-count", type=int, default=EXPECTED_COUNT,
                    help="0 to skip the record-count check")
    ap.add_argument("--no-stratified", action="store_true",
                    help="use proportional sampling instead of floored strata")
    args = ap.parse_args()

    print(recommendation())
    print()

    records = load_benchmark(args.benchmark, args.expected_count)
    bench_hash = sha256_file(args.benchmark)
    print(f"Loaded {len(records)} records; sha256={bench_hash[:16]}...")

    strata_sizes = dict(Counter(r["relation_pid"] for r in records))
    alloc = allocate(strata_sizes, args.n, args.floor,
                     stratified=not args.no_stratified)
    print(f"Relation allocation (n={args.n}, "
          f"{'stratified floor='+str(args.floor) if not args.no_stratified else 'proportional'}):")
    for rel in sorted(alloc):
        print(f"  {rel} ({RELATION_LABELS.get(rel,'?')}): "
              f"{alloc[rel]}/{strata_sizes[rel]}")

    picked = sample(records, alloc, args.seed)
    if len(picked) != args.n:
        raise RuntimeError(f"sampled {len(picked)} != requested {args.n}")
    if len({r['_record_id'] for r in picked}) != args.n:
        raise RuntimeError("sample contains duplicate record ids")

    rows = [to_row(i, r) for i, r in enumerate(picked)]
    realized = dict(Counter(r["relation_id"] for r in rows))

    meta = {
        "purpose": "Unreviewed manual-audit sample for the TAS/PTC benchmark. "
                   "Accuracy is NOT computed here; annotate the CSV then run "
                   "scripts/summarize_benchmark_audit.py.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_path": str(args.benchmark),
        "benchmark_sha256": bench_hash,
        "benchmark_record_count": len(records),
        "expected_record_count": args.expected_count,
        "sample_size": args.n,
        "seed": args.seed,
        "stratified": not args.no_stratified,
        "floor_per_relation": args.floor if not args.no_stratified else None,
        "record_id_scheme": "relation:subject_qid:old_qid->new_qid@t_update",
        "relation_population": strata_sizes,
        "relation_allocation": alloc,
        "relation_realized": realized,
        "population_weights": {k: v / len(records) for k, v in strata_sizes.items()},
        "controlled_vocab": {
            "audit_status": list(AUDIT_STATUS_VALUES),
            "is_correct_supersession": [v for v in IS_CORRECT_VALUES if v],
            "issue_category": [v for v in ISSUE_CATEGORY_VALUES if v],
        },
        "annotation_state": "unreviewed",
        "reproduce_command": (
            f"python scripts/sample_benchmark_audit.py --n {args.n} "
            f"--seed {args.seed} --floor {args.floor}"),
    }

    paths = export(rows, args.outdir, meta)
    print("\nWrote:")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    print(f"\nRealized relation counts: {realized}")
    print("\nNEXT: annotate audit_sample.csv (audit_status -> reviewed, "
          "is_correct_supersession, issue_category), then run "
          "scripts/summarize_benchmark_audit.py. No accuracy number exists yet.")


if __name__ == "__main__":
    main()
