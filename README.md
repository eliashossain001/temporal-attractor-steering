# Temporal Attractor Steering (TAS) for Parametric Temporal Conflict

Code accompanying an anonymous EMNLP submission on *Parametric Temporal
Conflict (PTC)*, a failure mode in which open-weight language models
default to outdated answers under standard prompting even though the
newer answer is recoverable from parametric memory under a temporal
cue. This repository contains:

- the PTC screening pipeline (Phase 1),
- the Temporal Attractor Steering pipeline (Phase 2: locate / build
  steering vector / train detector / evaluate),
- a single CLI that orchestrates both,
- the extended evaluations reported in the supplement: prompt/steering
  baselines and controls, an ITI baseline with a paired held-out
  TAS-vs-ITI comparison, V1/V2/V3 steering-vector ablations, the
  τ_rec sensitivity sweep, patch-vs-edit layer localization, and
  free-generation evaluation,
- a subject-disjoint split protocol with a leakage audit and the
  corrected split-clean detector,
- deterministic benchmark-construction, verification, and manuscript
  /LaTeX audits,
- a 500-record subset for fast reviewer reproduction, and
- a curated [`results/`](results/) tree with the JSON metrics from the
  full 8,746-record benchmark on four open-weight LMs.

## Repository layout

```
.
├── configs/models.yaml          Model registry (HF ids, dtype, etc.)
├── data/
│   ├── subset/                   500-record reviewer subset
│   └── large/                    8,746-record full benchmark
├── results/                      Curated JSON metrics (4 models, 8,746 records)
│                                 incl. splits/, baselines/, iti_paired/,
│                                 variant_ablation/, tau_rec_sensitivity/,
│                                 detector_splitclean/, free_generation/,
│                                 benchmark_verification/, final_audit/
├── runs/                         Re-created by the pipeline; gitignored
├── scripts/                      Reproduction + audit scripts (see below)
│   └── run_all_models.sh         Batch runner over every model
├── src/temporal_conflict/        Python package
│   ├── cli.py                    Single CLI entry point
│   ├── pipeline.py               Orchestrator
│   ├── config.py                 Typed config + RunPaths
│   ├── stages.py                 OOP wrappers per pipeline stage
│   ├── splits.py                 Authoritative subject-disjoint split protocol
│   ├── env.py, io.py, models.py, metrics.py, schema.py, verify.py
│   ├── steering/                 Activations, locator, steer, detector, TAS
│   ├── baselines/                Prompt/steering baselines, ITI, paired TAS-vs-ITI
│   └── analysis/                 Model-free post-hoc stats (bootstrap CIs, McNemar)
├── pyproject.toml
├── requirements.txt
└── .env.example                  Copy to .env and fill in HF_TOKEN
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # uses pyproject.toml
# or, for an editable install without packaging metadata:
pip install -r requirements.txt
```

Set up the Hugging Face token (gated models like Llama and Mistral
require an approved access request on the Hub):

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=hf_...
```

## Reviewer quickstart (500-record subset, one model, no GPU required)

The subset takes about ten minutes on a single mid-range GPU and a few
hours on CPU. Pick the smallest model (`qwen-2.5-1.5b`) for the
fastest path.

```bash
python -m temporal_conflict.cli tas \
    --model qwen-2.5-1.5b \
    --data data/subset \
    --runs-root runs/reviewer
```

This runs all seven stages, writing per-stage JSON artifacts under
`runs/reviewer/qwen-2.5-1.5b/`. Each stage is idempotent: re-running
only re-does the stages whose outputs are missing. Pass `--force` to
rebuild from scratch.

## Full benchmark (all four models)

```bash
DATA=data/large RUNS=runs/full bash scripts/run_all_models.sh
```

This iterates `qwen-2.5-1.5b`, `qwen-2.5-7b`, `mistral-7b-v0.3`,
`llama-3.1-8b`. The full pipeline produces the JSONs in
[`results/`](results/).

## CLI reference

```
python -m temporal_conflict.cli screen    --model <key> [--data ...] [--runs-root ...]
python -m temporal_conflict.cli locate    --model <key> [...]
python -m temporal_conflict.cli tas       --model <key> [...]
python -m temporal_conflict.cli eval      --model <key> --tau 0.20 [--alpha 2.0]
python -m temporal_conflict.cli baselines --model <key> --screening <per_instance.jsonl> [...]
python -m temporal_conflict.cli stats     --results-tas <dir> [--out ...] [--table ...]
```

- `screen` runs Phase 1 only (per-instance scoring + summary).
- `locate` runs through the layer locator (Phase 2B), producing the
  AFR profile and ell*.
- `tas` runs the full pipeline through Phase 2F (detector-gated TAS
  evaluation at the default `tau` grid).
- `eval` runs Phase 2F at a single `(tau, alpha)` against a trained
  detector and a chosen alpha (defaults to alpha* from the oracle
  stage).
- `baselines` computes the WP-A prompt baselines (standard / temporal /
  RAG / contrastive decoding) and WP-B steering controls on the same
  verified-PTC subset used by the TAS evaluation.
- `stats` is model-free: it reads the Phase 2F `tas_eval_tau*.json`
  outputs and emits bootstrap CIs and paired McNemar tests.

## Extended evaluations and audits (paper supplement)

The supplement's tables and figures are reproduced by the scripts in
[`scripts/`](scripts/), all reading from the committed benchmark and the
one versioned subject-disjoint split. They run in this order:

```bash
# 1. Authoritative subject-disjoint split + leakage audit
python scripts/build_subject_disjoint_splits.py
python scripts/audit_split_leakage.py
python scripts/validate_splits.py

# 2. Corrected (split-clean) detector: recompute, operating points, calibration
python scripts/recompute_detector_splitclean.py
python scripts/detector_operating_points.py
python scripts/plot_detector_calibration.py

# 3. Paired held-out TAS-vs-ITI, layer localization, τ_rec sweep, V1/V2/V3 ablation
python scripts/run_paired_tas_iti.py         && python scripts/analyze_paired_iti.py
python scripts/analyze_layer_localization.py && python scripts/plot_layer_localization.py
python scripts/analyze_tau_rec_sensitivity.py
python scripts/run_variant_ablation.py       && python scripts/analyze_variant_ablation.py

# 4. Prompt/steering baselines and controls, free-generation evaluation
python scripts/run_baselines_controls.py     && python scripts/summarize_baselines.py
python scripts/evaluate_free_generation.py   && python scripts/summarize_generation_errors.py

# 5. Benchmark construction/verification + deterministic final audits (no model runs)
python scripts/export_sparql_queries.py
python scripts/build_verification_manifest.py
python scripts/audit_manuscript_numbers.py --repo .
python scripts/audit_latex_references.py
python scripts/run_final_audits.py --repo .        # orchestrates the audits
```

The final-audit scripts (step 5) are cheap, deterministic, and run no
models; their outputs land in `results/final_audit/`. See
[`results/README.md`](results/README.md) for the content-addressed split
and result hashes and the manual benchmark-audit protocol.

## Pipeline architecture

The pipeline is structured as a set of OOP stages composed by a single
`Pipeline` orchestrator.

```python
from temporal_conflict.pipeline import Pipeline, PipelineOptions

opts = PipelineOptions(
    data_dir="data/subset",
    runs_root="runs/reviewer",
    taus=[0.20],                         # single threshold for a fast pass
)
Pipeline("qwen-2.5-1.5b", options=opts).run()
```

Stage classes (`src/temporal_conflict/stages.py`):

| Stage                       | Phase | Output JSON / binary                     |
|-----------------------------|-------|------------------------------------------|
| `ScreeningStage`            | 1     | `per_instance.jsonl`, `summary.json`      |
| `ActivationCacheStage`      | 2A    | `activations.pt`                         |
| `LayerLocatorStage`         | 2B    | `afr_profile.json`                        |
| `SteeringVectorBuilderStage`| 2C    | `steering_vectors_v2.pt`                  |
| `OracleTASStage`            | 2D    | `oracle_tas_relation.json`                |
| `DetectorStage`             | 2E    | `detector.pkl`, `detector_metrics.json`   |
| `TASEvaluationStage`        | 2F    | `tas_eval_tau<tau>.json`                  |

Each subclass of `Stage` defines `outputs()` and `_run()`; the base
class handles the idempotency check and timing log.

## Data

- `data/subset/merged_sample.jsonl` (~464 KB, 500 records): a small
  sample drawn from the full benchmark, with all five Wikidata
  relations represented. Suitable for end-to-end reviewer reproduction
  on a single GPU in under fifteen minutes.
- `data/large/combined_all.jsonl` (~17 MB, 8,746 records): the
  benchmark used for the headline numbers in the paper.

Records are JSONL with one PTC quadruple `(q, a_old, a_new, t_update)`
per line. See [`src/temporal_conflict/schema.py`](src/temporal_conflict/schema.py)
for the full field list.

## Results

Pre-computed JSON metrics for the four open-weight models on the full
benchmark are in [`results/`](results/). The directory includes Phase
1 per-instance scores, AFR profiles, oracle alpha sweeps, detector
metrics, and end-to-end TAS evaluations at three threshold values.
Heavy binaries (`*.pt`, `*.pkl`) are excluded from version control;
they regenerate from the same CLI call.

## License

MIT. See LICENSE.
