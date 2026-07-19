# Detector operating points (subject-disjoint, val-selected)

Bootstrap: 10000 subject-level resamples, seed 0. Thresholds selected on VALIDATION only.

## Consistency vs stored JSON

- qwen-2.5-1.5b: recomputed AUROC 0.4665 vs stored 0.4665 -> MATCH
- qwen-2.5-7b: recomputed AUROC 0.5714 vs stored 0.5714 -> MATCH
- mistral-7b-v0.3: recomputed AUROC 0.6574 vs stored 0.6574 -> MATCH
- llama-3.1-8b: recomputed AUROC 0.5984 vs stored 0.5984 -> MATCH

## Operating points

| Model | OP | tau | AUROC | AUPRC | FPR [95% CI] | TPR [95% CI] | %gated | GatedRec | PA |
|---|---|---|---|---|---|---|---|---|---|
| qwen-2.5-1.5b | primary_tau0.15 | 0.15 | 0.467 | 0.028 | 0.005 [0.002,0.009] | 0.031 [0.000,0.107] | 0.005 | 0.03125 | 1.0 |
| qwen-2.5-1.5b | val_F1_max | 0.28 | 0.467 | 0.028 | 0.003 [0.001,0.006] | 0.000 [0.000,0.000] | 0.003 | None | None |
| qwen-2.5-1.5b | val_youdenJ_max | 0.28 | 0.467 | 0.028 | 0.003 [0.001,0.006] | 0.000 [0.000,0.000] | 0.003 | None | None |
| qwen-2.5-1.5b | val_FPR_le_5pct | 0.03 | 0.467 | 0.028 | 0.006 [0.002,0.011] | 0.031 [0.000,0.107] | 0.007 | None | None |
| qwen-2.5-1.5b | always_on | 0.0 | 0.467 | 0.028 | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 | None | None |
| qwen-2.5-7b | primary_tau0.15 | 0.15 | 0.571 | 0.056 | 0.025 [0.014,0.040] | 0.128 [0.029,0.250] | 0.028 | 0.07692307692307693 | 0.97 |
| qwen-2.5-7b | val_F1_max | 0.04 | 0.571 | 0.056 | 0.125 [0.099,0.153] | 0.333 [0.179,0.500] | 0.131 | None | None |
| qwen-2.5-7b | val_youdenJ_max | 0.04 | 0.571 | 0.056 | 0.125 [0.099,0.153] | 0.333 [0.179,0.500] | 0.131 | None | None |
| qwen-2.5-7b | val_FPR_le_5pct | 0.07 | 0.571 | 0.056 | 0.045 [0.030,0.064] | 0.256 [0.114,0.414] | 0.052 | None | None |
| qwen-2.5-7b | always_on | 0.0 | 0.571 | 0.056 | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 | None | None |
| mistral-7b-v0.3 | primary_tau0.15 | 0.15 | 0.657 | 0.091 | 0.103 [0.074,0.135] | 0.225 [0.113,0.346] | 0.109 | 0.056338028169014086 | 1.0 |
| mistral-7b-v0.3 | val_F1_max | 0.1 | 0.657 | 0.091 | 0.103 [0.074,0.135] | 0.225 [0.113,0.346] | 0.109 | None | None |
| mistral-7b-v0.3 | val_youdenJ_max | 0.1 | 0.657 | 0.091 | 0.103 [0.074,0.135] | 0.225 [0.113,0.346] | 0.109 | None | None |
| mistral-7b-v0.3 | val_FPR_le_5pct | 0.2 | 0.657 | 0.091 | 0.005 [0.002,0.009] | 0.028 [0.000,0.094] | 0.006 | 0.014084507042253521 | 1.0 |
| mistral-7b-v0.3 | always_on | 0.0 | 0.657 | 0.091 | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 | None | None |
| llama-3.1-8b | primary_tau0.15 | 0.15 | 0.598 | 0.097 | 0.058 [0.040,0.076] | 0.149 [0.069,0.238] | 0.064 | 0.06896551724137931 | 0.98 |
| llama-3.1-8b | val_F1_max | 0.13 | 0.598 | 0.097 | 0.093 [0.068,0.120] | 0.207 [0.117,0.306] | 0.100 | None | None |
| llama-3.1-8b | val_youdenJ_max | 0.11 | 0.598 | 0.097 | 0.121 [0.093,0.150] | 0.264 [0.169,0.367] | 0.130 | None | None |
| llama-3.1-8b | val_FPR_le_5pct | 0.24 | 0.598 | 0.097 | 0.013 [0.006,0.022] | 0.035 [0.000,0.091] | 0.014 | None | None |
| llama-3.1-8b | always_on | 0.0 | 0.598 | 0.097 | 1.000 [1.000,1.000] | 1.000 [1.000,1.000] | 1.000 | None | None |
