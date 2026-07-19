# Detector calibration/validation sufficiency (Bucket 1, Part 4)

Per-model PTC-positive / negative counts by split (subject-disjoint v1).
Direction Delta and probe are fit on TRAIN only; isotonic calibration on CALIBRATION only;
thresholds on VALIDATION only; TEST untouched.

| Model | train +/- | val +/- | calib +/- | test +/- |
|---|---|---|---|---|
| qwen-2.5-1.5b | 123/5117 | 35/1271 | 19/844 | 32/1305 |
| qwen-2.5-7b | 189/5051 | 58/1248 | 32/831 | 39/1298 |
| mistral-7b-v0.3 | 311/4929 | 83/1223 | 59/804 | 71/1266 |
| llama-3.1-8b | 399/4841 | 91/1215 | 65/798 | 87/1250 |

## Isotonic-calibration stability
- qwen-2.5-1.5b: calibration positives = 19 -> DATA-LIMITED (isotonic on <25 positives is high-variance; report descriptively); validation positives = 35; train positives = 123.
- qwen-2.5-7b: calibration positives = 32 -> adequate; validation positives = 58; train positives = 189.
- mistral-7b-v0.3: calibration positives = 59 -> adequate; validation positives = 83; train positives = 311.
- llama-3.1-8b: calibration positives = 65 -> adequate; validation positives = 91; train positives = 399.

## Enlarging validation/calibration without contamination
- Test is subject-disjoint and untouched; it is NOT combined with validation or calibration.
- The steering direction Delta and the linear probe are fit on TRAIN ONLY, so calibration/validation can be
  enlarged by reallocating subjects from TRAIN (not from test) without contaminating direction construction.
- Even so, positives are the binding constraint: total PTC positives are small (esp. Qwen-2.5-1.5B),
  so reallocation trades train positives (which the probe needs) for calibration positives.
- Recommendation: keep the untouched test split; for detector diagnostics prefer repeated subject-disjoint
  cross-validation over a single small calibration fold. Where positives remain <25 (Qwen-2.5-1.5B),
  calibration is data-limited and estimates are reported as descriptive, not inferential.
- We do NOT silently merge validation and test.
