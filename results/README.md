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

---

## Final reproduction & provenance (Bucket 9)

**Environment.** `pip install -e .` (Python 3.10+, PyTorch 2.11, Transformers 5.5, fp16);
2× TITAN RTX (24 GB) used for the model passes. Benchmark and split are content-addressed.

**Authoritative hashes / seeds.**
- Benchmark: `data/large/combined_all.jsonl` — sha256 `ad75be4ed3cfefc474e4cf743814f8dd31fee1088531e3fc57c26ab8698fcf0b`, 8,746 records.
- Split manifest: `results/splits/subject_disjoint_v1.json` — sha256 `b263c324dc3a1d44af9775a6ed1bb6734f151f7bf72b00d0e9a411cbd3c10015`.
  (NOTE: the `split_manifest_sha256` field stored inside some `test_summary.json` files
  currently holds the *benchmark* hash, not this file's hash — a provenance bug to fix.)
- Split seed `20260712`; benchmark-audit sample seed `20260711`; bootstrap/control seed `0`.
- Model revisions: qwen-2.5-1.5b `8faed761…`, qwen-2.5-7b `d1497293…`,
  mistral-7b-v0.3 `caa1feb0…`, llama-3.1-8b `d04e592b…` (full hashes in `results/final_audit/result_source_manifest.json`).

**Reproduction order.**
1. build/verify benchmark → `data/large/combined_all.jsonl`
2. build subject-disjoint split → `scripts/build_subject_disjoint_splits.py`
3. leakage audit → `scripts/audit_split_leakage.py`
4. detector recompute (corrected) → `scripts/recompute_detector_splitclean.py`
5. paired TAS/ITI → `scripts/run_paired_tas_iti.py` → `scripts/analyze_paired_iti.py`
6. localization → `scripts/analyze_layer_localization.py` (+ `plot_layer_localization.py`)
7. τ_rec sensitivity → `scripts/analyze_tau_rec_sensitivity.py`
8. V1/V2/V3 ablation → `scripts/run_variant_ablation.py` → `scripts/analyze_variant_ablation.py`
9. baselines/controls → `scripts/run_baselines_controls.py` → `scripts/summarize_baselines.py`
10. final audits (no model runs) → `python scripts/run_final_audits.py`

**Final-audit commands (cheap, deterministic, no model runs):**
```
python scripts/audit_manuscript_numbers.py --repo .   # manifest + numerical audit
python scripts/audit_latex_references.py              # LaTeX/reference audit
python scripts/run_final_audits.py --repo .           # orchestrates all of the above
```
Outputs land in `results/final_audit/`.

**Benchmark manual audit.** The 200-record stratified sample
(`results/benchmark_audit/audit_sample.csv`, seed 20260711) is currently **unreviewed**
(`is_correct_supersession` blank for all rows). The paper's benchmark-verification sentence
is intentionally **non-quantitative** until annotation is complete; then run
`scripts/summarize_benchmark_audit.py` to produce the accuracy + Wilson/bootstrap CI.
