# Split Leakage Audit

Split: subject_disjoint_v1 (seed 20260712)
Benchmark sha256: ad75be4ed3cfefc4...

## Prohibited-key overlaps (must all be 0)
- record_id: max pairwise overlap = 0  OK
- subject_qid: max pairwise overlap = 0  OK
- subject_label_norm: max pairwise overlap = 0  OK
- subject_relation: max pairwise overlap = 0  OK
- prompt_standard: max pairwise overlap = 0  OK
- prompt_norm: max pairwise overlap = 0  OK
- subjects crossing splits: 0 OK

## Descriptive overlaps (allowed; answer entities may recur)
- old_answer_qid: max pairwise overlap = 122 (informational)
- new_answer_qid: max pairwise overlap = 89 (informational)

## RESULT: PASS