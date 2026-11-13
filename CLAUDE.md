# CLAUDE.md: Squint

Project instructions for Claude Code. Put this file at the repo root.
Work stage by stage (section 12). **Stop at every gate and report before continuing.**

---

## 1. Project in one paragraph

Fine-tune a small object detector, compress it to INT8, and measure how much **accuracy and
people-counting error** degrade under image corruptions (blur, noise, low light, JPEG)
compared with clean images. The deliverable is a small, rigorous, reproducible empirical
study with confidence intervals, an honest write-up, and a clean public repo. It is meant to
support a Google Student Researcher (vision) application and a possible Namma AI hackathon
submission (edge object detection and counting, Raspberry Pi).

The engineering (export, quantize, run) is a means. **The evaluation is the product.**

## 2. Research question and hypotheses

> Does INT8 quantization hurt a small detector's robustness to image corruptions more than
> it hurts its clean-image accuracy?

- **H1:** The INT8 PTQ model's accuracy loss (mAP and counting error) grows with corruption
  severity, i.e. the FP32 minus INT8 gap is larger under corruption than on clean images.
- **H2:** QAT recovers part of that extra gap.
- **H0 (equally acceptable result):** the gap is the same or negligible. Report it honestly.

Write these into `docs/PREREG.md` **before running any experiment** (section 12, S0).

## 3. Non-negotiable rules

1. **Never invent or estimate a number.** Anything unmeasured is written as "not measured".
2. **Never tune on the test set.** Hyperparameters, corruption severities, score thresholds
   and calibration data come from the dev split only.
3. **Calibration data is never test data.** PTQ calibration images come from the dev split.
4. **INT8 numbers must come from a real INT8 execution path**, not FP32 fake-quant. Record
   which runtime produced them in `run_meta.json`. If you can only get fake-quant, say so
   everywhere the number appears.
5. **Report negative and null results.** Do not drop a configuration because it looks bad.
6. **Fixed seeds, fixed splits.** Split files are committed. Every result file records
   git commit, seeds, library versions, device and runtime.
7. Flag anything you could not verify (an API, a version, a published number) with
   `UNVERIFIED:` in code comments and docs. Check the current docs before relying on
   remembered APIs. Quantization and ExecuTorch APIs change between releases.
8. No secrets in the repo. Do not push to any remote unless I ask. Small, descriptive commits.
9. Before any download larger than ~1 GB, tell me the size and where it will go.
10. If a gate fails or an assumption breaks, **stop and ask me**. Don't silently switch scope.

## 4. Environment and constraints

- Target compute: **free Colab/Kaggle GPU (about T4)** for training; CPU for INT8 eval.
  Sessions can disconnect, so training must **checkpoint every epoch and resume**.
- Python 3.10+, PyTorch 2.x. Pin everything in `requirements.txt` after the first
  working run (`pip freeze`) and keep a short `requirements-min.txt`.
- Raspberry Pi: **optional**. If I don't have one, measure latency on laptop CPU with a
  fixed thread count and label it "laptop CPU", never "edge device".
- Each experiment is a config file plus one command. No notebooks as the source of truth
  (notebooks may call the scripts).

## 5. Repo layout

```
quant-robust-det/
  CLAUDE.md
  README.md                 # filled at the end (section 14)
  docs/
    PREREG.md               # hypotheses, metrics, analysis plan (written first)
    DECISIONS.md            # every gate decision + reason + date
    RESULTS.md              # generated tables + error analysis
  configs/
    data.yaml  train.yaml  quant.yaml  eval.yaml
  data/                     # gitignored; splits/*.json committed
    splits/{train,dev,test}_ids.json
  src/qrd/
    data.py                 # COCO person subset, split, loaders
    models.py               # builds detector, Detector protocol
    train.py                # fine-tune (resume-able)
    quantize.py             # PTQ and QAT (track-specific backends)
    corruptions.py          # deterministic corruptions
    metrics.py              # mAP (pycocotools), counting metrics
    evaluate.py             # runs a model x condition grid -> JSON
    stats.py                # bootstrap CIs, paired gap analysis
    bench_latency.py        # latency, size
    report.py               # JSON -> tables + plots
  scripts/                  # thin CLIs; one per stage
  tests/
  results/                  # raw JSON, committed (small); big files gitignored
  Makefile                  # make data|train|quant|eval|report|test
```

## 6. Decision gate G1: choose the compression track (days 1-2)

The harness must be **backend-agnostic**. Define a `Detector` protocol:
`predict(images: list[np.ndarray]) -> list[{boxes_xyxy, scores, labels}]`. The evaluator
only talks to this interface, so the compression track can change without touching the study.

Try tracks in order. A track **passes** only if it produces correct boxes on 20 real images
from the INT8 model, with person-class AP50 within a few points of its own FP32 version.

| Track | Stack | Notes |
|---|---|---|
| **A** | torchvision `ssdlite320_mobilenet_v3_large` (COCO weights) + PT2E quantization (`torch.export` -> `prepare_pt2e`/`convert_pt2e` with an XNNPACK quantizer), optionally exported to ExecuTorch (`.pte`) | `UNVERIFIED:` check the current ExecuTorch XNNPACK quantization tutorial for exact APIs. Detection post-processing (NMS, box decode) may not export cleanly, so export the raw backbone and heads and do decode/NMS in Python. This matches the hackathon stack if it works. |
| **B** | Same torchvision model -> ONNX (raw head outputs, NMS outside) -> ONNX Runtime static INT8 (QDQ) with a calibration data reader | Lower API risk, well-documented. Real INT8 kernels on CPU. |
| **C** | Ultralytics YOLO (nano size) -> built-in INT8 export (ONNX/TFLite/OpenVINO) | Fastest to a working INT8 model. Note the AGPL license in `README.md`. |

**Gate G1 passes when:** one track yields a real-INT8 model whose outputs I can score with
my own `metrics.py`. Record the chosen track and the evidence in `docs/DECISIONS.md`.
If A and B both fail after about 2 working days, fall back to C and tell me.

QAT needs training-time fake-quant. For Track A use PT2E QAT (`prepare_qat_pt2e`,
`UNVERIFIED`); for B/C, if QAT is not feasible, **report PTQ only** and say so. Do not fake it.

## 7. Data

- **Dataset:** COCO 2017, **`person` class only** (counting = number of person instances).
  `UNVERIFIED:` confirm annotation license/terms and the exact image counts from the files;
  never hardcode counts from memory. Compute and print them.
- **Splits (by image id, seeded, disjoint, saved to `data/splits/*.json`):**
  - `train`: subset of train2017 images containing persons (cap to a size that trains in
    a Colab session; log the number)
  - `dev`: held-out part of train2017 (corruption severity tuning, PTQ calibration,
    score-threshold choice)
  - `test`: val2017 images containing at least one non-crowd person. **Touched only for
    final runs.**
- Ground-truth count per image = number of non-`iscrowd` person boxes. Report how crowd
  regions are handled (they are ignored in AP via the standard COCO handling).
- Caveat to state in the write-up: the COCO-pretrained weights have already seen train2017.
  Test uses val2017 only. Say this explicitly.
- Stretch only (not required): VisDrone for small/dense objects.

## 8. Corruptions (`corruptions.py`)

Deterministic given `(image_id, corruption, severity, seed)`. Operate on uint8 RGB.

| Name | Parameter by severity 1..4 (initial; tune on **dev** only) |
|---|---|
| `gaussian_blur` | sigma = 1, 2, 3, 4 |
| `gaussian_noise` | sigma = 0.04, 0.08, 0.12, 0.18 (on [0,1]) |
| `low_light` | brightness factor 0.6, 0.4, 0.25, 0.15, gamma > 1, plus shot/read noise applied **after** darkening |
| `jpeg` | quality = 50, 30, 15, 8 |

Rule for tuning severities on dev: choose levels so the **FP32** model's mAP falls roughly
evenly across levels 1..4. Freeze them, commit them to `configs/eval.yaml`, and never
change them after test runs begin. Ground-truth boxes are unchanged by these corruptions.

## 9. Metrics (`metrics.py`)

All metrics are computed **per model x condition** and saved as JSON.

- **mAP@[.5:.95]** and **AP50**, person class, via `pycocotools`.
- **Counting:** for score threshold `t`, predicted count = detections with score >= `t`
  after NMS. Report **MAE**, **RMSE**, and **mean signed error** (bias) vs GT count.
  - Primary `t = 0.5`. Also report `t` chosen on **dev** to maximize count-MAE for the
    FP32 model, applied unchanged to all variants. (Quantization shifts score
    distributions, so show both and discuss.)
- **Size:** model file size in MB.
- **Latency:** batch 1, fixed thread count, 20 warm-up runs, then median and p95 of >=100
  timed runs. Record device name, CPU model, thread count, runtime and version.
  Latency is measured on the real runtime that produced the INT8 numbers.

## 10. Statistics (`stats.py`)

- **Seeds:** 3 training seeds. QAT seed is paired to its FP32 seed. PTQ uses 3
  calibration-subset seeds. Report mean +/- std across seeds **and** pooled bootstrap CIs.
- **Image-level bootstrap:** resample test **images** (not boxes) with replacement,
  **paired** across models so the same resampled images score every model.
  Use at least 1,000 resamples for count metrics. For mAP, use at least 200 resamples
  (it is expensive; document the number used). Seeded.
- **Key quantity, the robustness gap:**

  `gap(c, s) = [mAP_FP32(c, s) - mAP_INT8(c, s)] - [mAP_FP32(clean) - mAP_INT8(clean)]`

  for corruption `c` at severity `s`. Same definition for counting MAE with sign flipped.
  H1 is supported if the paired-bootstrap 95% CI of `gap` is above 0 at the higher severities.
  State the decision rule in `PREREG.md` before looking at results.
- Do not claim significance from overlapping or single-seed differences.

## 11. Experiment matrix

- Models: `fp32` (fine-tuned), `ptq_int8`, `qat_int8` (if feasible)
- Conditions: `clean` + 4 corruptions x 4 severities = 17
- Seeds: 3
- Total: up to 3 models x 17 conditions x 3 seeds = 153 evaluations on the test set.
- **Cost check first:** time a 50-image run for each model/runtime, extrapolate, and
  show me the estimate before launching the full grid. Use a fixed dev subset
  (e.g. 500 images) for all iteration; the full test set is for final runs only.
- Optional ablations (only after the core grid is done): PTQ calibration size
  (100 / 500 images), per-tensor vs per-channel quantization.

## 12. Stages (each ends with a gate and a report to me)

### S0: Scaffold and pre-registration (day 1)
- [ ] Create repo layout, `Makefile`, `pyproject`/requirements, CI-free local tests
- [ ] Write `docs/PREREG.md`: question, H1/H2/H0, metrics, severities (initial), seeds,
      decision rule for `gap`, what counts as null
- [ ] Commit the `Detector` protocol and a stub evaluator
- **Done when:** `make test` runs (even if few tests) and PREREG is committed.

### S1: Gate G1, feasibility (days 1-2)
- [ ] Load pretrained detector, run on 5 COCO images, draw boxes, save to `results/sanity/`
- [ ] Try Track A, then B, then C (section 6). For each: INT8 file produced? real INT8
      path? boxes sane? Log in `docs/DECISIONS.md`.
- [ ] Sanity: FP32 AP on a small val2017 subset is in the right range vs the published
      number for the model (`UNVERIFIED:` look it up in the torchvision/Ultralytics docs;
      class-restricted AP will differ, so compare qualitatively and say so).
- **Done when:** a track passes G1, the choice is recorded, and I have approved it.

### S2: Data and baseline (week 1)
- [ ] `data.py`: person subset, seeded disjoint splits saved to `data/splits/`
- [ ] `train.py`: fine-tune FP32 for N epochs (N chosen to fit a Colab session),
      checkpoint every epoch, resume from checkpoint, mixed precision, seeded
- [ ] `evaluate.py`: clean-image evaluation producing mAP and counting metrics JSON
- [ ] `corruptions.py` with tests, plus a contact sheet of every corruption x severity
      saved to `results/corruption_samples.png`
- [ ] Tune severities on **dev**, freeze in `configs/eval.yaml`
- **Gate G2:** FP32 baseline numbers on dev/test-subset, corruption sheet. Report to me.

### S3: Compression (week 2)
- [ ] `quantize.py`: PTQ with calibration from **dev**; QAT if the track supports it
- [ ] Verify real INT8 execution (log op types or runtime used)
- [ ] Clean-image eval for `ptq_int8` and `qat_int8` vs FP32
- [ ] `bench_latency.py`: size and latency for all variants, on the target device
- **Gate G3:** a table of clean-image mAP, count MAE, size, latency per variant. Report to me.

### S4: The study (week 3)
- [ ] Run the full grid (section 11) for all seeds; write one JSON per
      (model, seed, condition) into `results/raw/`
- [ ] `stats.py`: seed aggregation, paired image-bootstrap CIs, `gap(c, s)` with CIs
- [ ] `report.py` plots: (1) mAP vs severity per corruption, FP32 vs INT8 lines with CI
      bands; (2) count-MAE vs severity; (3) `gap` vs severity with CIs; (4) size/latency
      bar chart
- [ ] Failure gallery: the 12 images with the largest INT8-vs-FP32 count disagreement
      under the hardest low-light level, with boxes drawn
- **Gate G4:** all figures generated from raw JSON by one command. Report to me.

### S5: Write-up and release (week 4)
- [ ] `docs/RESULTS.md` auto-generated tables + hand-written analysis section
- [ ] `README.md` (section 14) with a one-command reproduction path
- [ ] Limitations section (section 14), a license note, a data-terms note
- [ ] Re-run from a clean clone on a fresh environment to confirm reproducibility
- **Done when:** `make all` reproduces every table and figure from the committed splits
  and configs, and every number in the README traces to a file in `results/`.

## 13. Tests (must pass before each gate)

- Corruptions: same `(image_id, name, severity, seed)` gives byte-identical output; severity
  1 differs less from the original than severity 4 (e.g. by PSNR).
- Splits: train/dev/test are disjoint by image id; every test image has >=1 non-crowd person.
- Counting metric: a hand-made 3-image fixture with known predicted/GT counts gives the
  exact expected MAE, RMSE, bias.
- Metrics: a perfect-prediction fixture gives AP = 1.0; an empty-prediction fixture gives AP = 0.
- Bootstrap: seeded runs are reproducible; paired resampling uses identical image indices
  across models.
- Box format conversions (xyxy <-> xywh) round-trip.

## 14. README and reporting requirements

README sections: Question - Headline result (one table, one figure) - Method (model,
data splits, compression track, runtime) - Reproduce (exact commands) - Results with
CIs - Failure analysis - **Limitations** - Licenses and data terms - Citation of
datasets and methods used.

**Limitations that must appear (edit to reflect what actually happened):** one detector
architecture; one dataset (COCO person); pretrained weights saw train2017; synthetic
corruptions, not real sensor noise; limited seeds (3); latency on [device]; PTQ/QAT
recipe not exhaustively tuned; INT8 runtime = [name/version].

## 15. Resume bullet template (fill only from measured values)

```
quant-robust-det: robustness of INT8-quantized object detectors
- Measured how INT8 post-training quantization changes a MobileNet-based detector's
  person-counting accuracy under blur, noise, low light and JPEG corruption on COCO
  (N=[test images] images, 3 seeds); count MAE rose from [A] to [B] (95% CI [..]) at
  the hardest low-light level vs [C] to [D] on clean images, [Z]x smaller and [L]x faster
  on [device].
- Built a reproducible evaluation harness (paired image-bootstrap CIs, pre-registered
  hypotheses, backend-agnostic detector interface) and released code and raw results.
```

If the result is null, say so in the bullet: "found no significant robustness penalty
from INT8 (gap CI includes 0)". A clean null is still a valid study.

## 16. Risks and fallbacks

| Risk | Fallback |
|---|---|
| ExecuTorch/PT2E can't export the detector | Track B, then C (section 6). Keep ExecuTorch for a later stretch. |
| QAT not available on the chosen track | PTQ-only study; state it in the README and drop H2 |
| Colab session limits | Checkpoint/resume; smaller train subset; fewer epochs; log what was used |
| INT8 eval too slow on CPU | Evaluate on a fixed, documented random subset of test (>=1,000 images), same subset for every model |
| No Raspberry Pi | Laptop CPU latency, clearly labeled |
| Effect is tiny or absent | Report it, with CIs. Do not add more corruptions after seeing results to "find" an effect. |
| Time overruns | Cut order: ablations -> second corruption family -> QAT. Never cut the CIs or the pre-registration. |

## 17. When to stop and ask me

- Any gate (G1-G4) reached
- A track fails or an API doesn't exist as remembered
- A number looks too good (e.g. INT8 beats FP32 by a lot), then suspect a bug before celebrating
- Before launching anything expected to run longer than ~30 minutes or download > 1 GB
- Before changing a frozen config (severities, thresholds, splits)

## 18. Definition of done

- [ ] `docs/PREREG.md` committed before any result files exist (check git history)
- [ ] All results regenerate from `results/raw/` with one command
- [ ] Every number in `README.md` and the resume bullet is traceable to a file
- [ ] Tests pass; fresh-clone reproduction succeeded
- [ ] Limitations are honest and specific
- [ ] No secrets, no dataset files, no large checkpoints in git