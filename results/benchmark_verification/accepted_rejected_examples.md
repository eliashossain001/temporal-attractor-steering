# Accepted and rejected examples (Bucket 2)

Accepted examples are drawn from the released benchmark
(`data/large/combined_all.jsonl`). Rejected examples are real Wikidata cases
retrieved live from `https://query.wikidata.org/sparql` (QIDs given for
verification); each illustrates a distinct deterministic rejection reason and
shows why the corresponding admission rule is needed. These are illustrative real
cases, not a claim about the pre-filter candidate distribution (the upstream raw
candidate set is regenerable from the released queries but is not retained).

| Status | Relation | Subject | Old answer | New answer | Time evidence | Decision reason |
|---|---|---|---|---|---|---|
| ACCEPTED | head of state (P35) | France (Q142) | Nicolas Sarkozy (Q329) | François Hollande (Q157) | old 2007-05-16 → 2012-05-15; new from 2012-05-15 | abutting (gap 0 d), tenure 1826 d ≥180, distinct holders, English labels → admitted |
| ACCEPTED | CEO (P169) | Deutsche Telekom (Q9396) | René Obermann (Q1572591) | Timotheus Höttges (Q1386366) | old 2006-11-13 → 2013-12-31; new from 2014-01-01 | abutting (gap 1 d ≤60), tenure 2604 d ≥180, distinct holders → admitted |
| REJECTED | head of government (P6) | 28th Canadian Ministry (Q220542) | Stephen Harper (Q206) | Stephen Harper (Q206) | three consecutive terms 2006→2008→2011→2015, same holder | **SAME_HOLDER**: consecutive transition is Harper→Harper (a re-appointment, not a supersession) |
| REJECTED | CEO (P169) | Google Nest (Q2119882) | Tony Fadell (Q92879) | — | statement has end 2016-06-03 but no start (P580) | **MISSING_TIME_BOUNDARY**: the transition boundary is undefined, so no valid old/new interval can be formed |
| REJECTED | CEO (P169) | GNOME Foundation (Q1056660) | Stormy Peters (Q...) | Richard Littauer (Q...) | two "current" CEO statements (since 2008 and since 2024), neither with an end date | **OVERLAPPING_HOLDERS / AMBIGUOUS**: intervals cannot be ordered into a single clean hand-off; a naive consecutive-transition builder would fabricate a spurious conflict |

Rejection reasons illustrated: SAME_HOLDER (re-election), MISSING_TIME_BOUNDARY
(missing/ambiguous timestamp), OVERLAPPING_HOLDERS (co-holder / overlapping
validity). Additional codes (LONG_VACANCY, SHORT_OLD_TENURE, OUTSIDE_YEAR_RANGE,
MISSING_ENGLISH_LABEL, COHOLDER_CASE, DUPLICATE_RECORD, ADDITIVE_NOT_SUPERSEDING)
are defined in `scripts/build_verification_manifest.py` and applied by the same
rule logic.
