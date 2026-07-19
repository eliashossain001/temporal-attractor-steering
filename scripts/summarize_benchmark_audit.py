#!/usr/bin/env python3
"""Summarize a *completed* benchmark manual audit into accuracy statistics.

Reads the annotated ``audit_sample.csv`` produced by
``scripts/sample_benchmark_audit.py`` and reports correctness rates with
confidence intervals, an error breakdown, and a LaTeX-ready sentence.

Guardrails
----------
* Refuses to run if ANY row is still ``unreviewed`` (no partial claims).
* Validates controlled-vocabulary values; fails loudly on anything invalid.
* Never silently discards ``ambiguous`` rows: they stay in the strict
  denominator and are reported explicitly; ``resolved`` accuracy excludes
  them but the count excluded is always shown.
* Computes an unweighted rate (yes/total, as requested) AND a
  population-weighted estimate (using the per-relation weights recorded in
  audit_metadata.json), because the sample is stratified.

This script does not touch the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "results" / "benchmark_audit"

VALID_STATUS = {"unreviewed", "reviewed"}
VALID_CORRECT = {"yes", "no", "ambiguous"}
VALID_ISSUE = {
    "none", "overlapping_validity", "vacancy_or_gap", "coholder",
    "interim_or_caretaker", "same_entity", "incorrect_dates",
    "incorrect_labels", "additive_not_superseding", "insufficient_evidence",
    "other",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def bootstrap_ci(indicators: list[int], b: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 0/1 list (deterministic)."""
    n = len(indicators)
    if n == 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(b):
        s = sum(indicators[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * b)]
    hi = means[int((1 - alpha / 2) * b) - 1]
    return lo, hi


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    return rows


def validate(rows: list[dict]) -> None:
    """Fail loudly on unreviewed rows or invalid controlled values."""
    problems = []
    unreviewed = [r for r in rows if r.get("audit_status", "").strip() != "reviewed"]
    if unreviewed:
        idxs = [r.get("audit_index", "?") for r in unreviewed]
        raise SystemExit(
            f"REFUSING TO SUMMARIZE: {len(unreviewed)} row(s) not 'reviewed' "
            f"(audit_index {idxs[:10]}{'...' if len(idxs) > 10 else ''}). "
            f"Complete the annotation first.")
    for r in rows:
        st = r.get("audit_status", "").strip()
        cor = r.get("is_correct_supersession", "").strip()
        iss = r.get("issue_category", "").strip()
        idx = r.get("audit_index", "?")
        if st not in VALID_STATUS:
            problems.append(f"row {idx}: bad audit_status {st!r}")
        if cor not in VALID_CORRECT:
            problems.append(f"row {idx}: bad is_correct_supersession {cor!r} "
                            f"(must be one of {sorted(VALID_CORRECT)})")
        if iss not in VALID_ISSUE:
            problems.append(f"row {idx}: bad issue_category {iss!r}")
        # Consistency: a 'yes' should carry issue none (or blank->none).
        if cor == "yes" and iss not in {"none"}:
            problems.append(f"row {idx}: is_correct=yes but issue_category={iss!r} "
                            f"(a correct supersession should have issue 'none')")
        if cor in {"no", "ambiguous"} and iss == "none":
            problems.append(f"row {idx}: is_correct={cor} but issue_category='none' "
                            f"(record an issue category for non-'yes' rows)")
    if problems:
        raise SystemExit("Annotation validation failed:\n  " +
                         "\n  ".join(problems))


def summarize(rows: list[dict], meta: dict | None) -> dict:
    labels = [r["is_correct_supersession"].strip() for r in rows]
    counts = Counter(labels)
    n_total = len(rows)
    n_yes = counts.get("yes", 0)
    n_no = counts.get("no", 0)
    n_amb = counts.get("ambiguous", 0)

    # Strict: ambiguous counts against correctness (stays in denominator).
    strict_p, strict_lo, strict_hi = wilson(n_yes, n_total)
    strict_boot = bootstrap_ci([1 if l == "yes" else 0 for l in labels])
    # Resolved: exclude ambiguous, but report how many were excluded.
    n_resolved = n_yes + n_no
    resolved_p, resolved_lo, resolved_hi = wilson(n_yes, n_resolved)

    # Error breakdowns (never drop ambiguous).
    by_relation = defaultdict(lambda: Counter())
    by_issue = Counter()
    for r in rows:
        rel = r.get("relation_id", "?")
        cor = r["is_correct_supersession"].strip()
        by_relation[rel][cor] += 1
        if cor in {"no", "ambiguous"}:
            by_issue[r.get("issue_category", "").strip() or "none"] += 1

    # Population-weighted strict estimate (stratified sample correction).
    weighted = None
    if meta and meta.get("population_weights"):
        w = meta["population_weights"]
        rel_yes = {rel: c.get("yes", 0) for rel, c in by_relation.items()}
        rel_tot = {rel: sum(c.values()) for rel, c in by_relation.items()}
        acc = 0.0
        wsum = 0.0
        for rel in rel_tot:
            if rel in w and rel_tot[rel] > 0:
                acc += w[rel] * (rel_yes[rel] / rel_tot[rel])
                wsum += w[rel]
        weighted = acc / wsum if wsum else None

    return {
        "n_total": n_total, "n_yes": n_yes, "n_no": n_no, "n_ambiguous": n_amb,
        "n_resolved": n_resolved,
        "strict_accuracy": strict_p,
        "strict_wilson95": [strict_lo, strict_hi],
        "strict_bootstrap95": list(strict_boot),
        "resolved_accuracy": resolved_p,
        "resolved_wilson95": [resolved_lo, resolved_hi],
        "population_weighted_strict": weighted,
        "by_relation": {rel: dict(c) for rel, c in by_relation.items()},
        "errors_by_issue_category": dict(by_issue),
    }


def latex_sentence(s: dict, meta: dict | None) -> str:
    n = s["n_total"]
    pct = 100 * s["strict_accuracy"]
    lo, hi = (100 * x for x in s["strict_wilson95"])
    strat = ""
    if meta and meta.get("stratified"):
        strat = "stratified "
    amb = s["n_ambiguous"]
    amb_clause = (f" ({amb} ambiguous case{'s' if amb != 1 else ''} counted as "
                  f"not-confirmed)") if amb else ""
    return (
        f"On a {strat}random sample of $N{{=}}{n}$ benchmark records spanning "
        f"all five relations, ${pct:.1f}\\%$ were confirmed correct "
        f"supersessions by manual audit against Wikidata "
        f"(Wilson $95\\%$ CI $[{lo:.1f}\\%, {hi:.1f}\\%]$){amb_clause}."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="write summary JSON here (default <dir>/audit_summary.json)")
    args = ap.parse_args()

    csv_path = args.csv or (args.dir / "audit_sample.csv")
    meta_path = args.dir / "audit_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None

    rows = load_rows(csv_path)
    validate(rows)
    s = summarize(rows, meta)
    sentence = latex_sentence(s, meta)

    print(f"Audited: {s['n_total']}  "
          f"yes={s['n_yes']}  no={s['n_no']}  ambiguous={s['n_ambiguous']}")
    print(f"Strict accuracy    = yes/total       = {s['strict_accuracy']:.4f} "
          f"Wilson95 {tuple(round(x,4) for x in s['strict_wilson95'])} "
          f"boot95 {tuple(round(x,4) for x in s['strict_bootstrap95'])}")
    print(f"Resolved accuracy  = yes/(yes+no)     = {s['resolved_accuracy']:.4f} "
          f"Wilson95 {tuple(round(x,4) for x in s['resolved_wilson95'])} "
          f"(excludes {s['n_ambiguous']} ambiguous)")
    if s["population_weighted_strict"] is not None:
        print(f"Population-weighted strict (stratified correction) = "
              f"{s['population_weighted_strict']:.4f}")
    print("\nBy relation (yes/no/ambiguous):")
    for rel, c in sorted(s["by_relation"].items()):
        print(f"  {rel}: {c}")
    print("\nErrors by issue_category (no + ambiguous):")
    for k, v in sorted(s["errors_by_issue_category"].items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print("\nLaTeX-ready sentence:\n  " + sentence)

    out = args.out or (args.dir / "audit_summary.json")
    payload = {**s, "latex_sentence": sentence,
               "source_csv": str(csv_path),
               "benchmark_sha256": (meta or {}).get("benchmark_sha256")}
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
