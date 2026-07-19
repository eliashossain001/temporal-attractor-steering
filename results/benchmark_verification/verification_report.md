# Benchmark deterministic verification report

- released records: 8746
- pass ALL deterministic rules: 8746
- fail >=1 rule: 0

- gap_days (t_new_start - t_old_end): min -60, max 60 (rule: 0..60)
- old_tenure_days: min 180, max 6209 (rule: >= 180)
- same-holder (a_old==a_new): 0
- missing English label: 0

NOTE: the released benchmark contains only ACCEPTED records; the upstream raw
candidate set / rejection log is not retained in this repo. Re-running the released
SPARQL (data/benchmark_queries/) through this rule logic reproduces the full
accept/reject decision and rejection breakdown for any candidate set.
