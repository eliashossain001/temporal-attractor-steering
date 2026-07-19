# Overleaf Final Compilation Checklist (Bucket 9, Part L)

No local TeX engine was available during Bucket 9, so **source-level** checks are green
(`scripts/run_final_audits.py` → LaTeX audit CLEAN, numerical audit 0 stale / 0 missing,
no `[N]`/`[X%]` brackets). Visual/PDF confirmation MUST be done in Overleaf.

## 1. Upload / sync
- [ ] Sync changed files: `main.tex`, `supplement.tex`, `ref.bib`.
- [ ] Sync everything under `figures/` (PNGs). `aaai2027.sty` + `aaai2027.bst` present.
- [ ] `main.tex` and `supplement.tex` are **separate documents** → two PDFs; set the
      correct "Main document" per compile (Menu → Main document).

## 2. Compile sequence (each document, from scratch)
- [ ] Recompile from scratch (clear cached `.aux`) — required for `cleveref` + `\cref`.
- [ ] pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX (≥3 passes so all `\cref`/`\cite` resolve).
- [ ] `\bibliography` uses `ref.bib`; style `aaai2027.bst`. BibTeX runs clean (34 entries).

## 3. Inspect logs
- [ ] "There were undefined references" → none expected.
- [ ] "Citation ... undefined" → none expected (58 cite-occurrences, all keys in ref.bib).
- [ ] "multiply-defined labels" → none expected (28 main / 41 supp labels, no dupes).
- [ ] "Missing figure file" → none expected; confirm every `\includegraphics` resolves.
- [ ] Overfull \hbox on wide tables → check `tab:layer-localization`, `tab:tas-vs-iti`,
      `tab:tau-rec-split`, `tab:tau-rec-sweep`, `tab:phase1-extended-full` (wide, `\small`).

## 4. Visual confirmation (both PDFs)
- [ ] **Search rendered PDF for `?`** (unresolved refs) — zero expected.
- [ ] **Search rendered PDF for `[N]`, `[X%]`, `TODO`, `FIXME`, `placeholder`** — zero.
- [ ] Theorem/definition/assumption/proposition environments render correctly.
- [ ] Every table fits its column; no content bleeds into the margin or the other column.
- [ ] Figures readable at print size (axis labels, legends).
- [ ] Anonymity: no author names, institution, or de-anonymizing repo URL/path.

## 5. Page count (Part M)
- [ ] Confirm main paper is within the AAAI-27 page limit (technical content).
- [ ] If OVER, reduce in this order WITHOUT touching protected content
      (detector limitation, held-out TAS-vs-ITI, theorem statement, benchmark definition,
      primary patching result, prompt-baseline comparison, random-direction control):
      1. Move detailed V1/V2/V3 to supplement (keep one main-text sentence).
      2. Move detailed τ_rec counts to supplement.
      3. Compress localization interpretation prose.
      4. Shorten Related Work.
      5. Tighten theorem *interpretation* (not the statement).
      6. Trim implementation details already duplicated in supplement.
- [ ] Do NOT use font/margin/spacing hacks or alter the AAAI style.

## 6. Known items to eyeball specifically
- [ ] `fig:detector-pr` (supplement): caption now labels the PR curves as the LEGACY
      leaked-split (overstated) view and points to `tab:detector-corrected`. Confirm the
      **figure image itself** is acceptable as a "legacy/superseded" illustration, or
      regenerate on the corrected fold / remove the figure. (See final report — HIGH.)
- [ ] Benchmark-verification sentence (main, Benchmark section) is now non-quantitative
      (no `[N]`/`[X%]`); confirm it reads cleanly.
- [ ] Abstract + Conclusion TAS range now reads `29–62%` (was stale `29–57%`).
