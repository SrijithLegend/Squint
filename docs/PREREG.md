# Pre-registration

Written before any experiment is run (CLAUDE.md section 12, S0). Changes after results
exist must be logged in `DECISIONS.md` with a reason.

## Question

Does INT8 quantization hurt a small detector's robustness to image corruptions more than
it hurts its clean-image accuracy?

## Hypotheses

- **H1:** The INT8 PTQ model's accuracy loss (mAP and counting error) grows with corruption
  severity, i.e. the FP32 minus INT8 gap is larger under corruption than on clean images.
- **H2:** QAT recovers part of that extra gap.
- **H0:** The gap is the same or negligible. This is an equally acceptable result.

## Data

COCO 2017, `person` class only. Splits by image id, seeded, disjoint, committed in
`data/splits/`. `train` from train2017, `dev` held out from train2017, `test` = val2017
images with at least one non-crowd person. Test is touched only for final runs.
Counts are computed from the files, not stated here.

## Models

`fp32` (fine-tuned), `ptq_int8`, `qat_int8` (only if the chosen track supports it;
otherwise H2 is dropped and stated). INT8 numbers must come from a real INT8 runtime.

## Conditions

`clean` plus 4 corruptions x 4 severities = 17. Initial parameters (tuned on dev only,
then frozen in `configs/eval.yaml` before any test run):

| Corruption | Severity 1..4 |
|---|---|
| gaussian_blur (sigma) | 1, 2, 3, 4 |
| gaussian_noise (sigma on [0,1]) | 0.04, 0.08, 0.12, 0.18 |
| low_light (brightness) | 0.6, 0.4, 0.25, 0.15 (+ gamma > 1, noise after darkening) |
| jpeg (quality) | 50, 30, 15, 8 |

Tuning rule: choose levels so FP32 mAP on dev falls roughly evenly across levels 1..4.

## Metrics

- mAP@[.5:.95] and AP50 (pycocotools), person class.
- Counting: MAE, RMSE, mean signed error vs non-crowd GT count. Primary threshold
  t = 0.5; secondary t chosen on dev to minimize FP32 count-MAE, applied unchanged to
  all variants.
- Model size (MB); latency (batch 1, fixed threads, 20 warm-up, median and p95 of >= 100).

## Seeds and statistics

- 3 training seeds; QAT paired to its FP32 seed; PTQ uses 3 calibration-subset seeds.
- Paired image-level bootstrap (resample test images, same indices for every model),
  >= 1,000 resamples for count metrics, >= 200 for mAP. Seeded.
- Per-seed mean +/- std reported alongside pooled bootstrap CIs.

## Key quantity and decision rule

```
gap(c, s) = [mAP_FP32(c, s) - mAP_INT8(c, s)] - [mAP_FP32(clean) - mAP_INT8(clean)]
```

Same for count-MAE with sign flipped.

- **H1 supported** if the paired-bootstrap 95% CI of `gap` lies entirely above 0 at
  severities 3 and 4 for at least one corruption family, for mAP or count-MAE.
- **Null (H0)** if the 95% CI of `gap` includes 0 at all severities. Reported as such.
- **H2 supported** if QAT's gap is smaller than PTQ's with a paired-bootstrap 95% CI of
  the difference above 0.
- No significance claim from overlapping or single-seed differences.
- No corruptions are added after seeing results.
