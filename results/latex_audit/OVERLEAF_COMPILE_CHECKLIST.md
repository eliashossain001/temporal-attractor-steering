# Overleaf Compile Checklist (Bucket 7)

Source-level reference audit is **clean** (0 broken internal refs, 0 duplicate
labels, 0 missing citation keys — see `reference_audit.md`). Overleaf must
confirm the visual render, since no local TeX engine is available.

## Upload / sync
- [ ] Sync all modified files: `main.tex`, `supplement.tex`, `ref.bib`,
      and everything under `figures/` (11 PNGs; `tas_overview_architecture.pdf`
      is no longer referenced and may be removed).
- [ ] Confirm `aaai2027.sty` and `aaai2027.bst` are present.
- [ ] `main.tex` and `supplement.tex` are **separate documents** (two PDFs);
      set the correct main document per compile.

## Compile sequence (each document)
- [ ] **Recompile from scratch** (clear cached aux) — required so cleveref +
      the `.aux` refs resolve after recent edits.
- [ ] Run: pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX (≥3 passes so all
      `\cref`/`\cite` resolve).

## Inspect logs for
- [ ] `LaTeX Warning: There were undefined references` → none expected.
- [ ] `LaTeX Warning: Citation ... undefined` → none expected (all 58 keys
      present in `ref.bib`).
- [ ] `LaTeX Warning: There were multiply-defined labels` → none expected.
- [ ] `Missing figure file` → none expected (all 11 figures verified present).
- [ ] Overfull \hbox on wide tables → the wide 7-column layer-localization
      table is wrapped in `\resizebox`; others render at natural `\small`.

## Visual confirmation
- [ ] **No rendered `?`** anywhere. The prior `?`s came from three `\cref` to
      *unnumbered* subsections (AAAI `secnumdepth=0` in main); these are now
      prose. Section cross-refs in main must stay prose (sections are unnumbered);
      table/figure/equation/theorem/definition refs use `\cref` (numbered → OK).
- [ ] Theorem 1 (TAS Correctness) renders in the Theoretical Motivation section.
- [ ] The held-out V1/V2/V3 table (supplement) renders; the old pilot table is
      gone (no duplicate/conflicting variant table).
- [ ] `[N]` / `[X%]` in the benchmark-verification paragraph are **known
      pending placeholders** (manual-audit bucket), not `?` — fill from
      `scripts/summarize_benchmark_audit.py` before submission.

## Do NOT
- [ ] Do not add `\cref` to a label defined only in the *other* document
      (separate PDFs cannot cross-reference without `xr`/`externaldocument`,
      which is not configured). Use prose ("see the supplementary material").
