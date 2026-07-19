#!/usr/bin/env python3
"""Bucket 2: emit the runnable SPARQL extraction queries + README.

Writes one concrete, runnable query per relation to data/benchmark_queries/ plus a
parameterized template and a README. Each query extracts RAW position-holder
statements (holder, start P580, end P582, rank) with English labels; the
consecutive-transition construction and the deterministic admission rules
(abutment |gap|<=60d, tenure>=180d, distinct holders, single-holder relations,
canonical English labels, year range) are applied downstream by
scripts/build_verification_manifest.py. Deprecated-rank statements are excluded;
preferred and normal ranks are both retained.

Usage:  python scripts/export_sparql_queries.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "benchmark_queries"

RELATIONS = {
    "P35":  ("head_of_state", "head of state"),
    "P169": ("ceo", "chief executive officer"),
    "P286": ("head_coach", "head coach"),
    "P488": ("chairperson", "chairperson"),
    "P6":   ("head_of_government", "head of government"),
}

TEMPLATE = """# Position-holder extraction for {pid} ({label}).
# Extracts every holder statement (value + start P580 + end P582 + rank) with
# English labels. Deprecated-rank statements are excluded; preferred and normal
# ranks are BOTH returned (rank is emitted so downstream logic can prefer the
# highest rank per (subject, time) if needed). Missing P580/P582 are returned as
# unbound and are rejected downstream (MISSING_TIME_BOUNDARY). Consecutive
# (old->new) transitions and the deterministic admission rules are built in
# scripts/build_verification_manifest.py.
SELECT ?subject ?subjectLabel ?holder ?holderLabel ?start ?end ?rank
WHERE {{
  ?subject p:{pid} ?stmt .
  ?stmt ps:{pid} ?holder .
  ?stmt wikibase:rank ?rank .
  FILTER(?rank != wikibase:DeprecatedRank)
  OPTIONAL {{ ?stmt pq:P580 ?start . }}   # statement start time
  OPTIONAL {{ ?stmt pq:P582 ?end . }}     # statement end time
  # English labels only; items without an English label for the holder are
  # rejected downstream (MISSING_ENGLISH_LABEL).
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
ORDER BY ?subject ?start
"""

README = """# Benchmark SPARQL extraction queries

These queries reproduce the RAW candidate extraction behind the released
`data/large/combined_all.jsonl` benchmark (8,746 accepted records). One query per
Wikidata position-holder relation; a parameterized template is also provided.

## Files
- `P35_head_of_state.rq`, `P169_ceo.rq`, `P286_head_coach.rq`,
  `P488_chairperson.rq`, `P6_head_of_government.rq`
- `_template.rq` (parameterized; substitute the P-number)

## Endpoint & invocation
- Endpoint: `https://query.wikidata.org/sparql`
- Retrieval convention: HTTP GET/POST with `Accept: application/sparql-results+json`
  and a descriptive `User-Agent`. Example:
  ```
  curl -s -H "Accept: application/sparql-results+json" \\
       -H "User-Agent: tas-benchmark-audit/1.0 (research)" \\
       --data-urlencode query@data/benchmark_queries/P6_head_of_government.rq \\
       https://query.wikidata.org/sparql > raw_P6.json
  ```
- Retrieval date of the released benchmark: recorded in
  `results/benchmark_audit/audit_metadata.json` (`generated_utc`); Wikidata is a
  living resource, so re-extraction on a later date may differ slightly.
- Pagination/batching: the large relations (P6, P286, P488) exceed comfortable
  single-request size; batch with `LIMIT/OFFSET` ordered by `?subject ?start`, or
  split by subject country/type. The small relations (P35, P169) return in one
  request.
- Raw output format: SPARQL JSON results (one row per holder statement).
- Cache location: raw extraction JSON is not committed (large, living data);
  regenerate with the commands above.

## Rank & timestamp handling
- Deprecated-rank statements are excluded in the query.
- Preferred and normal ranks are both returned; `?rank` is emitted so downstream
  code can keep the highest-rank statement per (subject, interval).
- `P580` (start) / `P582` (end) may be unbound; unbound boundaries are rejected
  downstream as `MISSING_TIME_BOUNDARY`.

## Downstream deterministic admission (see scripts/build_verification_manifest.py)
1. Build consecutive (a_old -> a_new) transitions per subject, ordered by start.
2. Admit a transition only if ALL hold:
   - abutting windows: `|t_old_end - t_new_start| <= 60` days;
   - minimum old tenure: `t_old_end - t_old_start >= 180` days;
   - distinct holders: `a_old != a_new` (rejects re-elections / SAME_HOLDER);
   - single-holder relation (P6/P35/P169/P286/P488);
   - both answers have canonical English labels (no bare QIDs);
   - `t_update = t_new_start` in `[2010, 2024]`.
3. Rejection codes: MISSING_ENGLISH_LABEL, MISSING_TIME_BOUNDARY,
   OVERLAPPING_HOLDERS (|gap|>60 overlap), LONG_VACANCY (gap>60),
   SHORT_OLD_TENURE, SAME_HOLDER, COHOLDER_CASE, OUTSIDE_YEAR_RANGE,
   DUPLICATE_RECORD, ADDITIVE_NOT_SUPERSEDING, AMBIGUOUS_WIKIDATA_STATEMENT.

## Expected counts (accepted, after deterministic filtering)
| Relation | PID | Accepted |
|---|---|---|
| head of government | P6 | 3583 |
| head coach | P286 | 2391 |
| chairperson | P488 | 2381 |
| CEO | P169 | 208 |
| head of state | P35 | 183 |
| **Total** | | **8746** |

Pre-filter candidate counts depend on the Wikidata snapshot and are reproduced by
running the queries above; the released repo retains only the accepted set.
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_template.rq").write_text(TEMPLATE.format(pid="{PID}", label="{RELATION LABEL}"))
    for pid, (slug, label) in RELATIONS.items():
        (OUT / f"{pid}_{slug}.rq").write_text(TEMPLATE.format(pid=pid, label=label))
    (OUT / "README.md").write_text(README)
    print("Wrote", len(RELATIONS) + 1, "query files + README to", OUT)


if __name__ == "__main__":
    main()
