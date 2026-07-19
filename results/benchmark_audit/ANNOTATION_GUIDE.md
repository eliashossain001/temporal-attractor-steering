# Benchmark Manual-Audit Annotation Guide

This guide governs the manual verification of a random sample of the TAS / PTC
benchmark (`data/large/combined_all.jsonl`, 8,746 records). The goal is to
measure what fraction of sampled records are **correct superseding temporal
transitions** when checked against Wikidata.

**Nothing here computes an accuracy number.** You annotate `audit_sample.csv`;
`scripts/summarize_benchmark_audit.py` then computes the statistics. Do not
write any accuracy figure into the paper until every row is `reviewed`.

---

## What each record claims

Each record asserts a single-valued position handed off from an **old holder**
to a **new holder** at an update date:

- `subject_label` (`subject_qid`) held a `relation_label` (`relation_id`) role.
- `old_answer_label` was the holder over `[old_start, old_end]`.
- `new_answer_label` became the holder over `[new_start, new_end]`.
- `update_date` is the claimed transition (the new holder's start).

Use the Wikidata links in the CSV (`subject_url`, `old_answer_url`,
`new_answer_url`) and, when needed, the `source_statement_ids` to check the
underlying statements directly.

---

## The five relations (all single-valued at a time)

| ID | Meaning |
|----|---------|
| P6 | head of government |
| P35 | head of state |
| P169 | chief executive officer |
| P286 | head coach |
| P488 | chairperson |

Each is expected to have **one** holder at any instant. A record that implies
two simultaneous holders, or a return of the same holder, needs scrutiny.

---

## Decision: is this a correct supersession?

Set `is_correct_supersession` to **`yes`**, **`no`**, or **`ambiguous`**, and
set `audit_status` to `reviewed` for every row you finish.

### Mark `yes` (correct) only if ALL of the following hold

1. **Genuine precedence** — the old holder demonstrably held the role
   *immediately before* the new holder (no other holder in between).
2. **Replacement, not addition** — the new fact *supersedes* the old one; it is
   not an additional, co-existing value (e.g. a second co-chair, a deputy, a
   different office). The role is single-valued and the holder changed.
3. **Non-overlapping validity** — `[old_start, old_end]` and
   `[new_start, new_end]` do not *materially* overlap. A shared boundary date
   (old_end == new_start) is fine; a multi-month overlap is not.
4. **Correct dates and labels** — the transition date and both entity labels
   match Wikidata (correct people/orgs, correct spelling/entity, correct dates).
5. **Appropriately single-valued** — the relation genuinely admits one holder
   at a time for this subject during this window.
6. **Clean hand-off** — not a caretaker/interim gap, not a vacancy, not the
   same entity on both sides (see below).

If every box is checked, `is_correct_supersession = yes` and
`issue_category = none`.

### Mark `no` (incorrect) if ANY disqualifying condition holds

Record the specific reason in `issue_category`:

| `issue_category` | Use when |
|------------------|----------|
| `overlapping_validity` | old and new validity windows materially overlap |
| `vacancy_or_gap` | a real gap/vacancy separates the two holders (not a direct hand-off) |
| `coholder` | two holders held the role simultaneously (not single-valued here) |
| `interim_or_caretaker` | the "new" (or old) holder is an interim/acting/caretaker placeholder |
| `same_entity` | old and new refer to the same person/org (a re-election or relabel, not a change) |
| `incorrect_dates` | the transition/validity dates are wrong or implausible |
| `incorrect_labels` | wrong entity, wrong name, or bare-QID/mislabeled answer |
| `additive_not_superseding` | the new fact adds to rather than replaces the old (not a real supersession) |
| `other` | a real problem not covered above (explain in `auditor_notes`) |

Any `no` must have a non-`none` `issue_category`.

### Mark `ambiguous` if you genuinely cannot decide

Use `ambiguous` when the evidence is insufficient or conflicting and a
confident `yes`/`no` is not warranted — for example, Wikidata is internally
inconsistent, sources disagree, or the dates are too vague to judge overlap.
Set `issue_category` to the closest applicable reason or `insufficient_evidence`,
and explain in `auditor_notes`.

**Ambiguous rows are never discarded.** In the strict accuracy
(`yes / total`) they count as *not confirmed*; the resolved accuracy
(`yes / (yes + no)`) excludes them but the count is always reported. Prefer a
sparing, well-justified use of `ambiguous`.

---

## Consistency rules enforced by the summarizer

- Every row must have `audit_status = reviewed` (else the summarizer refuses).
- `is_correct_supersession` ∈ {`yes`, `no`, `ambiguous`}.
- `issue_category` ∈ the controlled list above.
- `yes` ⇒ `issue_category = none`.
- `no` / `ambiguous` ⇒ `issue_category ≠ none`.

Violations cause a loud failure, not a silent drop.

---

## Worked mini-examples

- **`yes`**: France / head of state, Sarkozy (…–2012-05-15) → Hollande
  (2012-05-15–…). Single office, adjacent windows, correct dates and people.
- **`no` / `same_entity`**: old and new answer are the same QID (a re-election).
- **`no` / `interim_or_caretaker`**: the "new" holder is an acting minister for
  three weeks before the permanent appointment.
- **`no` / `vacancy_or_gap`**: old holder ends 2015-03, new holder starts
  2016-01, with the seat vacant in between.
- **`ambiguous`**: two Wikidata statements give conflicting end dates and it is
  impossible to tell whether the windows overlap.

---

## Workflow

1. Open `audit_sample.csv`.
2. For each row, verify against Wikidata using the provided URLs / statement IDs.
3. Fill `is_correct_supersession`, `issue_category`, `auditor_notes`; set
   `audit_status = reviewed`.
4. When all rows are reviewed, run:
   ```
   python scripts/summarize_benchmark_audit.py --dir results/benchmark_audit
   ```
5. Copy the emitted LaTeX sentence into the manuscript, replacing the
   `[N]` / `[X%]` placeholders. Do this only after annotation is complete.
