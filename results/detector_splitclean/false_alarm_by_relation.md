# False-alarm (false-positive) breakdown at tau=0.15 (Bucket 1, Part C.5)

False positives = non-conflict test records gated by the detector at tau=0.15,
by relation. FP rate = FP / (non-conflict records of that relation).


## qwen-2.5-1.5b: total FP 6/1305 non-conflict (FPR 0.005)
| relation | FP | non-conflict n | FP rate |
|---|---|---|---|
| head of gov (P6) | 3 | 518 | 0.006 |
| head coach (P286) | 2 | 351 | 0.006 |
| chairperson (P488) | 1 | 382 | 0.003 |
| head of state (P35) | 0 | 22 | 0.000 |
| CEO (P169) | 0 | 32 | 0.000 |

## qwen-2.5-7b: total FP 33/1298 non-conflict (FPR 0.025)
| relation | FP | non-conflict n | FP rate |
|---|---|---|---|
| head of gov (P6) | 14 | 512 | 0.027 |
| head coach (P286) | 10 | 348 | 0.029 |
| head of state (P35) | 4 | 21 | 0.190 |
| CEO (P169) | 3 | 32 | 0.094 |
| chairperson (P488) | 2 | 385 | 0.005 |

## mistral-7b-v0.3: total FP 130/1266 non-conflict (FPR 0.103)
| relation | FP | non-conflict n | FP rate |
|---|---|---|---|
| chairperson (P488) | 50 | 385 | 0.130 |
| head of gov (P6) | 48 | 508 | 0.094 |
| head coach (P286) | 19 | 325 | 0.058 |
| CEO (P169) | 8 | 31 | 0.258 |
| head of state (P35) | 5 | 17 | 0.294 |

## llama-3.1-8b: total FP 72/1250 non-conflict (FPR 0.058)
| relation | FP | non-conflict n | FP rate |
|---|---|---|---|
| head of gov (P6) | 24 | 499 | 0.048 |
| head coach (P286) | 21 | 333 | 0.063 |
| chairperson (P488) | 19 | 367 | 0.052 |
| head of state (P35) | 6 | 22 | 0.273 |
| CEO (P169) | 2 | 29 | 0.069 |
