# Manual audit of the free-generation classifier / alias matcher (Bucket 3, Part 4)

**Sample.** 128 generations, stratified across all 4 models and all 5 output
categories (up to 8 per model x category). ~90 were inspected in full (untruncated
generation text vs. gold new/old labels), spanning every model, method, and category.

**Automatic-label accuracy.**
- NEW vs not-NEW (the primary metric): ~100% agreement. No new false positives; the
  new false-negatives found were fixed (see corrections). Alias handling verified on
  hard cases: name reordering ("Abe Shinzo" -> Shinzo Abe), diacritics
  ("Milanovic"), titles ("President Maithripala Sirisena"), "Sir Richard Lambert".
- OLD: ~100% agreement (matches the outdated entity or its aliases).
- Full 5-way: ~90% agreement. The ~10% disagreements are ALL
  other_entity <-> definitional boundary cases: generations that describe an
  organisation/office or repeat the question ("The President of the Republic of
  Croatia (Croatian: ...)", "Athletic Club is a Spanish football club...",
  "Who is the chairperson of KDU-CSL?") are sometimes labelled other_entity when
  they contain no person answer. Both categories are non-answers, so this does not
  affect the new/old/answer-bearing metrics.

**Corrections made (now in the pipeline).**
1. Leading-newline parser bug: base models (esp. Mistral, Llama under steering) emit
   leading newlines; the parser now takes the first NON-EMPTY line. Mistral went from
   0/955 -> 955/955 non-empty; the affected models were fully re-run.
2. Empty-Wikidata-label false negatives: some gold entities have an empty current
   English Wikidata label (Wikidata drift; e.g. Gus Malzahn Q5620734). Matching now
   also uses the benchmark's stored gold label, recovering these (e.g. Qwen-7B old
   0.077->0.103, Llama new 0.034->0.046).

**Final unresolved rate.** The genuinely-unresolvable-to-an-entity outputs are the
`definitional` + `empty` categories (base-model non-answers), which are correctly
identified as non-answers rather than parser failures. Residual classifier ambiguity
(other_entity vs definitional) is ~10% of inspected cases and does not touch the
new-answer rate.

**Conclusion.** Alias-aware new/old detection is reliable for these outputs; the
other_entity/definitional split is approximate and is reported as such.
