# Related-Work Verification (Bucket 9, Part F)

Method: each entry web-verified for title / authors / venue / identifier and relevance.
"Predates deadline?" is relative to an AAAI-27 submission (deadline ~Aug 2026); Jan–Jun
2026 papers qualify. No citation was fabricated; nothing was added to the manuscript body
in this pass (page budget unconfirmed) — verified reviewer suggestions are listed below
with ready-to-paste BibTeX and suggested placement for the author to add if space allows.

## A. 2026 entries already in `ref.bib` — all VERIFIED REAL

| key | title | venue | id | cited? | verdict |
|---|---|---|---|---|---|
| `kang2026model` | Model Whisper: Steering Vectors Unlock LLMs' Potential in Test-time | AAAI-26 | arXiv:2512.04748 | yes | REAL, metadata OK, relevant (test-time steering) |
| `li2026costom` | CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment | ACL 2026 Main (Best Paper) | arXiv:2604.10031 | yes | REAL, metadata OK, relevant (causal activation steering) |
| `dey2026temporal` | Temporal Fact Conflicts in LLMs: Reproducibility Insights (DYNAMICQA+MULAN) | ECIR 2026 | arXiv:2603.15892 | yes | REAL, highly relevant (temporal fact conflict). Optionally upgrade to ECIR `@inproceedings`. |
| `pham2026knowledge` | Where Knowledge Collides: Mechanistic Study of Intra-Memory Knowledge Conflict | NeuSymBridge@AAAI-26 (workshop) | arXiv:2601.09445 | yes | REAL, highly relevant (mechanistic intra-memory conflict) |

No fabricated 2026 citations found. All four cited and defined.

## B. Reviewer-suggested papers — verified

| suggestion | found? | canonical paper | venue/id | recommend |
|---|---|---|---|---|
| Xie et al. (knowledge conflicts) | YES | Adaptive Chameleon or Stubborn Sloth | ICLR 2024, arXiv:2305.13300 | ADD (Related Work: Knowledge Conflicts) — foundational behavioral KC study; concise contrast: they study context-vs-memory receptivity, we study intra-memory temporal conflict. |
| Kortukov et al. (context-memory, real docs) | YES | Studying LLM Behaviors Under Context-Memory Conflicts With Real Documents | COLM 2024, arXiv:2404.16032 | OPTIONAL (future work / RAG framing) — context-memory (external doc) conflict; PTC is parametric-only, so not a direct baseline. |
| Wallat et al. (temporal conflict resolution) | YES | When Facts Change: Temporal Knowledge Conflict Resolution in LLMs | ACL 2026 Findings | ADD (Intro / Temporal QA) — closest temporal-conflict-resolution work; contrast: they resolve, we localize+causally attribute. |
| Heyman et al. (prompt-mimicking steering) | YES | Steer Like the LLM: Activation Steering that Mimics Prompting | ICML 2026, arXiv:2605.03907 | ADD (Activation Steering) — directly relevant: prompt-vs-steering gap; supports our finding that a date-prefix prompt beats TAS. |
| "Momentum Steering" | NO canonical match | (search surfaced AUSteer / activation-momentum variants, no single "Momentum Steering" paper) | — | EXCLUDE (not found / metadata uncertain). |
| ContextFocus | NO | not located | — | EXCLUDE (not found). |
| Chen et al. (RAG/context compliance) | AMBIGUOUS | multiple candidates; no unique match to "Chen et al. RAG compliance" | — | EXCLUDE (metadata uncertain; would be title-similarity only). |
| adversarial robustness of activation steering | TOPIC | e.g. "Understanding (Un)Reliability of Steering Vectors" arXiv:2505.22637 | — | OPTIONAL (Limitations) — not a specific reviewer paper; cite only if discussing steering robustness. |

## C. Existing-bib hygiene
- Unused entries in `ref.bib` (defined, never `\cite`d): `kuhn2023semantic`, `mahaut2024factual`.
  Harmless (BibTeX drops uncited entries), but some venue checkers flag them; remove or cite.

## D. Ready-to-paste BibTeX for recommended additions (verified metadata)

```bibtex
@inproceedings{xie2024adaptive,
  title={Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts},
  author={Xie, Jian and Zhang, Kai and Chen, Jiangjie and Lou, Renze and Su, Yu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
@inproceedings{wallat2026facts,
  title={When Facts Change: Temporal Knowledge Conflict Resolution in Large Language Models},
  author={Wallat, Jonas and Nejdl, Wolfgang and Sikdar, Sandipan},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  year={2026}
}
@inproceedings{heyman2026steer,
  title={Steer Like the LLM: Activation Steering that Mimics Prompting},
  author={Heyman, Geert and Vandeputte, Frederik},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
@inproceedings{kortukov2024studying,
  title={Studying Large Language Model Behaviors Under Context-Memory Conflicts With Real Documents},
  author={Kortukov, Evgenii and Rubinstein, Alexander and Nguyen, Elisa and Oh, Seong Joon},
  booktitle={Conference on Language Modeling (COLM)},
  year={2024}
}
```
> Author must confirm the ICLR/ACL/COLM author lists above against the source page before pasting (venue metadata verified; full author strings should be double-checked).
