# Results

JSON metrics from the full 8,746-record benchmark, four models. Heavy
binaries (activation tensors `*.pt`, detector pickles `*.pkl`) are
gitignored under `runs/`; the JSONs here are sufficient to reproduce
every number reported in the paper.

## Layout

```
results/
├── phase1/                              Phase 1 PTC screening
│   ├── <model>/summary.json             Aggregate OPR / Recovery / PTC rate
│   ├── <model>/per_instance.jsonl       Per-record scores (4 log-probs)
│   └── _compare/compare.{md,csv}        Cross-model comparison table
└── tas/                                 Phase 2 Temporal Attractor Steering
    └── <model>/
        ├── afr_profile.json             Per-layer AFR sweep; ell* chosen
        ├── oracle_tas_relation.json     Phase 2D alpha sweep, alpha* chosen
        ├── detector_metrics.json        AUPRC / AUROC / ECE / threshold sweep
        └── tas_eval_tau{0.15,0.20,0.30}.json
                                          End-to-end Recovery / PA at each tau
```

## Models

| Key                | HF id                       |
|--------------------|-----------------------------|
| `qwen-2.5-1.5b`    | `Qwen/Qwen2.5-1.5B`         |
| `qwen-2.5-7b`      | `Qwen/Qwen2.5-7B`           |
| `mistral-7b-v0.3`  | `mistralai/Mistral-7B-v0.3` |
| `llama-3.1-8b`     | `meta-llama/Llama-3.1-8B`   |

## Headline numbers

See [`phase1/_compare/compare.md`](phase1/_compare/compare.md) for the
Phase 1 row-per-model summary. For Phase 2, the headline operating
point per model is the best `(Recovery, PA)` pair from the three
`tas_eval_tau*.json` files, judged by `J = Recovery - (1 - PA)`.

## Regenerating

Every file here is produced by `python -m temporal_conflict.cli tas
--model <key> --data data/large --runs-root runs/main`. The CLI is
idempotent: deleting a single JSON triggers only the relevant stage to
re-run.
