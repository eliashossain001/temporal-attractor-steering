# Result Source Manifest (auto-generated)

Split sha256: `b263c324dc3a1d44af9775a6ed1bb6734f151f7bf72b00d0e9a411cbd3c10015`

Benchmark: `/home/elias/elias_projects/tas_framework/data/large/combined_all.jsonl` sha256 `ad75be4ed3cfefc474e4cf743814f8dd31fee1088531e3fc57c26ab8698fcf0b` (8746 records)

Model revisions:

- qwen-2.5-1.5b: `8faed761d45a263340a0528343f099c05c9a4323`
- qwen-2.5-7b: `d149729398750b98c0af14eb82c78cfe92750796`
- mistral-7b-v0.3: `caa1feb0e54d415e2df31207e5f4e273e33509b1`
- llama-3.1-8b: `d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`

## Results -> artifacts

### filtered_ptc_at_tau_-3.0

- status: diagnostic (full benchmark)
- source: results/tau_rec_sensitivity/filter_sweep_all_models.csv
- script: scripts/analyze_tau_rec_sensitivity.py
- value: `{"qwen-2.5-1.5b": {"filtered_ptc": "101", "kept": "2466", "kept_frac": "0.282", "fptc_rate_among_kept": "0.041", "raw_ptc": "209"}, "qwen-2.5-7b": {"filtered_ptc": "248", "kept": "3484", "kept_frac": "0.3984", "fptc_rate_among_kept": "0.0712", "raw_ptc": "318"}, "mistral-7b-v0.3": {"filtered_ptc": "481", "kept": "5649", "kept_frac": "0.6459", "fptc_rate_among_kept": "0.0851", "raw_ptc": "523"}, "llama-3.1-8b": {"filtered_ptc": "597", "kept": "5781", "kept_frac": "0.661", "fptc_rate_among_kept": "0.1033", "raw_ptc": "641"}}`

### date_prefix_recovery

- status: diagnostic (full verified-PTC set)
- source: results/baselines/<model>/prompt_baselines.json
- script: scripts/run_baselines_controls.py
- value: `{"qwen-2.5-1.5b": {"date_prefix_recovery": 0.6138613861386139, "n_ptc": 101}, "qwen-2.5-7b": {"date_prefix_recovery": 0.782258064516129, "n_ptc": 248}, "mistral-7b-v0.3": {"date_prefix_recovery": 0.7941787941787942, "n_ptc": 481}, "llama-3.1-8b": {"date_prefix_recovery": 0.8053691275167785, "n_ptc": 596}}`

### tas_oracle_recovery_fulldata

- status: diagnostic (full verified-PTC set)
- source: results/tas/<model>/oracle_tas_relation.json (by_alpha[alpha_star].ptc.recovery_after)
- note: abstract 29-62% range
- value: `{"llama-3.1-8b": {"alpha_star": 6.0, "n": 597, "recovery": 0.38190954773869346, "source": "results/tas/llama-3.1-8b/oracle_tas_relation.json"}, "mistral-7b-v0.3": {"alpha_star": 6.0, "n": 481, "recovery": 0.29313929313929316, "source": "results/tas/mistral-7b-v0.3/oracle_tas_relation.json"}, "qwen-2.5-1.5b": {"alpha_star": 2.0, "n": 101, "recovery": 0.5148514851485149, "source": "results/tas/qwen-2.5-1.5b/oracle_tas_relation.json"}, "qwen-2.5-7b": {"alpha_star": 2.0, "n": 248, "recovery": 0.6169354838709677, "source": "results/tas/qwen-2.5-7b/oracle_tas_relation.json"}}`

### random_direction_control

- status: diagnostic (full verified-PTC set)
- source: results/baselines/<model>/steering_controls.json
- value: `{"qwen-2.5-1.5b": {"recovery": 0.25742574257425743, "pa": null}, "qwen-2.5-7b": {"recovery": 0.1774193548387097, "pa": null}, "mistral-7b-v0.3": {"recovery": 0.24116424116424118, "pa": null}, "llama-3.1-8b": {"recovery": 0.31208053691275167, "pa": null}}`

### tas_vs_iti_heldout

- status: HELD-OUT subject-disjoint test
- source: results/iti_paired/paired_significance.json
- script: scripts/analyze_paired_iti.py
- note: Recovery: no Holm-significant difference on any model. PA: only qwen-2.5-1.5b Holm-significant (ITI>TAS, p=0.006).

### detector_corrected

- status: HELD-OUT subject-disjoint (true base rate)
- source: results/detector_splitclean/summary.{json,csv}
- script: scripts/recompute_detector_splitclean.py
- note: corrected AUROC 0.47-0.66, AUPRC 0.028-0.097; qwen-2.5-1.5b AUROC 0.467 (below chance). LEGACY leaked AUPRC 0.258-0.525 must NOT be cited as current.

### benchmark_manual_audit

- status: unreviewed
- source: results/benchmark_audit/audit_sample.csv
- note: UNREVIEWED as of this audit; quantitative accuracy removed from paper (non-quantitative description substituted).

