# Results

<!-- AUTO:START -->
Test split: n = 1000 images, 3 training seeds (PTQ calibration seed = training seed). ± is std across seeds; [..] is the pooled paired image-bootstrap 95% CI (200 resamples for mAP/AP50, 1000 for counts).

### Clean images

| model | mAP@[.5:.95] | AP50 | count MAE @0.5 | count bias @0.5 | count MAE @0.25 (dev-chosen) |
|---|---|---|---|---|---|
| FP32 | 0.267 ± 0.002 [0.252, 0.284] | 0.472 ± 0.007 [0.447, 0.498] | 2.506 ± 0.056 [2.278, 2.729] | -2.443 ± 0.057 [-2.669, -2.213] | 1.883 ± 0.079 [1.713, 2.051] |
| INT8 PTQ | 0.244 ± 0.006 [0.230, 0.261] | 0.445 ± 0.017 [0.422, 0.471] | 2.617 ± 0.072 [2.382, 2.846] | -2.544 ± 0.068 [-2.777, -2.308] | 1.938 ± 0.145 [1.773, 2.096] |

### Size and latency (laptop CPU (not an edge device): 13th Gen Intel(R) Core(TM) i5-13420H, 4 threads, batch 1, torch 2.14.1+cu130, onnxruntime 1.27.0)

| variant | size MB | network median / p95 ms | end-to-end median / p95 ms |
|---|---|---|---|
| fp32_torch | 14.05 | 59.38 / 74.13 | 147.70 / 457.09 |
| fp32_ort | 13.82 | 28.39 / 98.15 | 68.32 / 74.82 |
| int8_ort | 3.91 | 17.72 / 21.51 | 67.91 / 72.70 |

INT8 path: onnxruntime 1.27.0 CPUExecutionProvider, executed INT8 ops {'QLinearConv': 98, 'QLinearAdd': 10, 'QLinearGlobalAveragePool': 8, 'QLinearMul': 8, 'QLinearConcat': 2} (seed 0 model); calibration = 100 dev images.

### mAP@[.5:.95] per condition

| condition | FP32 | INT8 PTQ |
|---|---|---|
| clean | 0.267 ± 0.002 [0.252, 0.284] | 0.244 ± 0.006 [0.230, 0.261] |
| gaussian_blur:1 | 0.215 ± 0.003 [0.200, 0.230] | 0.182 ± 0.010 [0.170, 0.197] |
| gaussian_blur:2 | 0.152 ± 0.007 [0.142, 0.166] | 0.123 ± 0.009 [0.114, 0.136] |
| gaussian_blur:3 | 0.087 ± 0.014 [0.079, 0.096] | 0.072 ± 0.010 [0.065, 0.080] |
| gaussian_blur:4 | 0.039 ± 0.009 [0.034, 0.044] | 0.037 ± 0.007 [0.033, 0.043] |
| gaussian_noise:1 | 0.211 ± 0.005 [0.198, 0.227] | 0.189 ± 0.005 [0.177, 0.204] |
| gaussian_noise:2 | 0.153 ± 0.008 [0.141, 0.166] | 0.135 ± 0.003 [0.124, 0.147] |
| gaussian_noise:3 | 0.093 ± 0.009 [0.084, 0.103] | 0.080 ± 0.001 [0.072, 0.089] |
| gaussian_noise:4 | 0.040 ± 0.006 [0.035, 0.045] | 0.036 ± 0.006 [0.032, 0.041] |
| low_light:1 | 0.209 ± 0.004 [0.197, 0.226] | 0.166 ± 0.015 [0.154, 0.179] |
| low_light:2 | 0.148 ± 0.007 [0.136, 0.161] | 0.082 ± 0.029 [0.074, 0.090] |
| low_light:3 | 0.093 ± 0.011 [0.084, 0.102] | 0.034 ± 0.019 [0.031, 0.039] |
| low_light:4 | 0.040 ± 0.006 [0.035, 0.045] | 0.014 ± 0.006 [0.010, 0.016] |
| jpeg:1 | 0.218 ± 0.008 [0.205, 0.233] | 0.200 ± 0.006 [0.188, 0.215] |
| jpeg:2 | 0.162 ± 0.010 [0.150, 0.175] | 0.149 ± 0.007 [0.137, 0.163] |
| jpeg:3 | 0.101 ± 0.015 [0.092, 0.111] | 0.094 ± 0.006 [0.084, 0.103] |
| jpeg:4 | 0.043 ± 0.010 [0.037, 0.049] | 0.042 ± 0.004 [0.037, 0.048] |

### count MAE @ t=0.5 per condition

| condition | FP32 | INT8 PTQ |
|---|---|---|
| clean | 2.506 ± 0.056 [2.278, 2.729] | 2.617 ± 0.072 [2.382, 2.846] |
| gaussian_blur:1 | 2.772 ± 0.050 [2.533, 3.002] | 2.898 ± 0.074 [2.653, 3.123] |
| gaussian_blur:2 | 3.079 ± 0.045 [2.839, 3.318] | 3.118 ± 0.115 [2.876, 3.361] |
| gaussian_blur:3 | 3.408 ± 0.112 [3.163, 3.656] | 3.306 ± 0.149 [3.065, 3.551] |
| gaussian_blur:4 | 3.669 ± 0.119 [3.422, 3.912] | 3.408 ± 0.233 [3.166, 3.656] |
| gaussian_noise:1 | 2.840 ± 0.053 [2.599, 3.072] | 2.942 ± 0.021 [2.702, 3.173] |
| gaussian_noise:2 | 3.215 ± 0.046 [2.973, 3.446] | 3.292 ± 0.026 [3.049, 3.531] |
| gaussian_noise:3 | 3.550 ± 0.057 [3.307, 3.787] | 3.611 ± 0.031 [3.367, 3.849] |
| gaussian_noise:4 | 3.861 ± 0.025 [3.620, 4.098] | 3.876 ± 0.037 [3.637, 4.114] |
| low_light:1 | 2.802 ± 0.030 [2.560, 3.028] | 3.024 ± 0.060 [2.780, 3.258] |
| low_light:2 | 3.132 ± 0.025 [2.892, 3.362] | 3.331 ± 0.157 [3.088, 3.572] |
| low_light:3 | 3.453 ± 0.087 [3.205, 3.696] | 3.544 ± 0.464 [3.303, 3.781] |
| low_light:4 | 3.750 ± 0.179 [3.504, 3.991] | 3.624 ± 0.569 [3.385, 3.867] |
| jpeg:1 | 2.782 ± 0.058 [2.547, 3.014] | 2.871 ± 0.069 [2.630, 3.099] |
| jpeg:2 | 3.069 ± 0.052 [2.831, 3.300] | 3.124 ± 0.031 [2.884, 3.353] |
| jpeg:3 | 3.402 ± 0.074 [3.167, 3.629] | 3.432 ± 0.036 [3.191, 3.665] |
| jpeg:4 | 3.735 ± 0.055 [3.492, 3.967] | 3.728 ± 0.033 [3.486, 3.962] |

### Robustness gap, mAP

gap(c,s) = INT8's loss under corruption minus its loss on clean (positive: INT8 hurt more by corruption). **Bold** = pooled 95% CI entirely above 0; *italic* = entirely below 0.

| condition | gap | 95% CI | per-seed |
|---|---|---|---|
| gaussian_blur:1 | **0.0101** | [0.0064, 0.0135] | 0.0105, 0.0119, 0.0079 |
| gaussian_blur:2 | **0.0062** | [0.0020, 0.0105] | 0.0097, 0.0119, -0.0031 |
| gaussian_blur:3 | *-0.0085* | [-0.0126, -0.0046] | 0.0043, -0.0069, -0.0230 |
| gaussian_blur:4 | *-0.0213* | [-0.0248, -0.0185] | -0.0123, -0.0216, -0.0299 |
| gaussian_noise:1 | -0.0015 | [-0.0042, 0.0018] | 0.0003, -0.0070, 0.0022 |
| gaussian_noise:2 | *-0.0051* | [-0.0087, -0.0012] | -0.0021, -0.0138, 0.0007 |
| gaussian_noise:3 | *-0.0104* | [-0.0138, -0.0062] | -0.0118, -0.0223, 0.0030 |
| gaussian_noise:4 | *-0.0190* | [-0.0222, -0.0151] | -0.0269, -0.0286, -0.0015 |
| low_light:1 | **0.0205** | [0.0167, 0.0245] | 0.0341, 0.0183, 0.0089 |
| low_light:2 | **0.0426** | [0.0367, 0.0484] | 0.0789, 0.0303, 0.0186 |
| low_light:3 | **0.0349** | [0.0286, 0.0410] | 0.0609, 0.0294, 0.0145 |
| low_light:4 | 0.0025 | [-0.0020, 0.0095] | 0.0059, 0.0021, -0.0005 |
| jpeg:1 | *-0.0062* | [-0.0092, -0.0030] | -0.0050, -0.0108, -0.0028 |
| jpeg:2 | *-0.0108* | [-0.0138, -0.0072] | -0.0121, -0.0173, -0.0031 |
| jpeg:3 | *-0.0161* | [-0.0190, -0.0123] | -0.0117, -0.0274, -0.0091 |
| jpeg:4 | *-0.0230* | [-0.0258, -0.0194] | -0.0193, -0.0315, -0.0182 |

### Robustness gap, mae@0.5

gap(c,s) = INT8's loss under corruption minus its loss on clean (positive: INT8 hurt more by corruption). **Bold** = pooled 95% CI entirely above 0; *italic* = entirely below 0.

| condition | gap | 95% CI | per-seed |
|---|---|---|---|
| gaussian_blur:1 | 0.0153 | [-0.0177, 0.0480] | -0.0550, 0.0840, 0.0170 |
| gaussian_blur:2 | *-0.0723* | [-0.1057, -0.0377] | -0.1350, 0.0080, -0.0900 |
| gaussian_blur:3 | *-0.2133* | [-0.2463, -0.1817] | -0.2800, -0.2140, -0.1460 |
| gaussian_blur:4 | *-0.3713* | [-0.4037, -0.3397] | -0.5370, -0.3590, -0.2180 |
| gaussian_noise:1 | -0.0083 | [-0.0393, 0.0180] | -0.0360, -0.0360, 0.0470 |
| gaussian_noise:2 | *-0.0343* | [-0.0630, -0.0063] | -0.0250, -0.1140, 0.0360 |
| gaussian_noise:3 | *-0.0497* | [-0.0777, -0.0213] | -0.0740, -0.1360, 0.0610 |
| gaussian_noise:4 | *-0.0957* | [-0.1217, -0.0723] | -0.1240, -0.1490, -0.0140 |
| low_light:1 | **0.1110** | [0.0783, 0.1453] | 0.1250, 0.1120, 0.0960 |
| low_light:2 | **0.0883** | [0.0473, 0.1283] | -0.0780, 0.1500, 0.1930 |
| low_light:3 | -0.0200 | [-0.0603, 0.0217] | -0.4690, 0.2220, 0.1870 |
| low_light:4 | *-0.2373* | [-0.2660, -0.2090] | -0.7000, -0.0040, -0.0080 |
| jpeg:1 | -0.0220 | [-0.0520, 0.0073] | -0.0160, -0.0310, -0.0190 |
| jpeg:2 | *-0.0567* | [-0.0853, -0.0283] | -0.0440, -0.0960, -0.0300 |
| jpeg:3 | *-0.0813* | [-0.1093, -0.0533] | -0.0560, -0.1290, -0.0590 |
| jpeg:4 | *-0.1180* | [-0.1460, -0.0920] | -0.1050, -0.1480, -0.1010 |

### Robustness gap, mae@0.25

gap(c,s) = INT8's loss under corruption minus its loss on clean (positive: INT8 hurt more by corruption). **Bold** = pooled 95% CI entirely above 0; *italic* = entirely below 0.

| condition | gap | 95% CI | per-seed |
|---|---|---|---|
| gaussian_blur:1 | 0.0433 | [-0.0237, 0.1070] | 0.0110, 0.0210, 0.0980 |
| gaussian_blur:2 | -0.0563 | [-0.1264, 0.0193] | -0.0650, -0.0720, -0.0320 |
| gaussian_blur:3 | *-0.2280* | [-0.3013, -0.1570] | -0.2290, -0.2850, -0.1700 |
| gaussian_blur:4 | *-0.4133* | [-0.4844, -0.3460] | -0.5070, -0.5090, -0.2240 |
| gaussian_noise:1 | -0.0390 | [-0.0993, 0.0157] | -0.0620, -0.0930, 0.0380 |
| gaussian_noise:2 | *-0.0637* | [-0.1237, -0.0043] | -0.1210, -0.1910, 0.1210 |
| gaussian_noise:3 | *-0.0650* | [-0.1237, -0.0139] | -0.1710, -0.2250, 0.2010 |
| gaussian_noise:4 | *-0.1020* | [-0.1560, -0.0533] | -0.2630, -0.2010, 0.1580 |
| low_light:1 | **0.0667** | [0.0047, 0.1307] | 0.0730, -0.0910, 0.2180 |
| low_light:2 | **0.1910** | [0.1160, 0.2610] | 0.0640, 0.1770, 0.3320 |
| low_light:3 | **0.1653** | [0.0980, 0.2273] | -0.2000, 0.3550, 0.3410 |
| low_light:4 | 0.0213 | [-0.0350, 0.0760] | -0.2960, 0.1910, 0.1690 |
| jpeg:1 | *-0.0973* | [-0.1527, -0.0430] | -0.1290, -0.1130, -0.0500 |
| jpeg:2 | *-0.1613* | [-0.2280, -0.0976] | -0.2120, -0.2260, -0.0460 |
| jpeg:3 | *-0.2050* | [-0.2730, -0.1343] | -0.2670, -0.2760, -0.0720 |
| jpeg:4 | *-0.2890* | [-0.3593, -0.2173] | -0.3900, -0.2690, -0.2080 |

### Pre-registered decision (docs/PREREG.md)

**mixed: some gap CIs exclude 0 but H1's rule (CI above 0 at severities 3 and 4 of one family) is not met**. Conditions meeting the H1 rule: none.
<!-- AUTO:END -->

## Analysis

All numbers above come from `results/stats.json` (test split, n = 1000 images, 3 training seeds, PTQ INT8 on ONNX Runtime CPU).

**Pre-registered decision: H1 is not supported.** The rule (gap CI entirely above 0 at severities 3 and 4 for at least one family) is met by no family. H2 (QAT) was not tested: QAT was not implemented, only PTQ.

**The result is mixed, not a clean null.** Several gap CIs exclude 0, but mostly in the direction opposite to H1:
- *Clean cost of INT8:* mAP 0.267 -> 0.244 (about 2.3 points); count MAE @0.5 2.51 -> 2.62.
- *Low light is the one place INT8 loses extra mAP:* the mAP gap is positive at severities 1-3 (+0.021, +0.043, +0.035, all CIs above 0) and indistinguishable from 0 at severity 4 (+0.003, CI [-0.002, +0.010]). At severity 4 both models are near the floor (FP32 0.040, INT8 0.014), so the gap shrinks because there is little left to lose. This is a mid-severity effect, not one that "grows with severity", which is why the rule is not met.
- *Blur, noise, JPEG:* the mAP gap is negative at severities 3-4 (CI below 0). INT8 loses *less* than it did on clean images, because both models converge toward the same floor. Read this as a floor/ceiling effect, not as INT8 being more robust.
- *Counting:* count-MAE gaps at t = 0.5 are negative for blur, noise, JPEG and low light sev 4 (INT8 degrades less than FP32 on top of its clean loss). Low light sev 1-2 is positive (+0.111, +0.088). No count gap supports H1 at severities 3 and 4.

**Failure mode at low_light:4 (what the gallery shows).** In 10 of the 12 largest count disagreements (seed 0, t = 0.5), FP32 outputs no person and INT8 outputs one (once two) box that covers almost the whole image (red boxes in `results/figures/failure_gallery.png`). In the other two, FP32 finds 3 people and INT8 returns one large box. This degenerate whole-image box with a score of at least 0.5 is a concrete INT8 artifact in very dark inputs. It adds a false positive per image, and that plausibly accounts for part of the low-light mAP loss. UNVERIFIED: the cause. I did not test whether it comes from activation-range clipping at calibration or from the head's score scale; no ablation was run. This is also why count MAE can look no worse for INT8 while mAP is much worse: one spurious box can offset misses in the count while still hurting ranking-based AP. That explanation is a hypothesis, not tested.

**Limitations specific to this analysis.**
- The fine-tuned FP32 model is weaker than the COCO-pretrained torchvision weights on dev (0.261 vs 0.290 mAP); I did not retune it. The study measures the fine-tuned model.
- The COCO-pretrained weights saw train2017; dev is drawn from train2017, so its pretrained numbers are inflated. Test is val2017 only.
- Only 1000 test images; 3 training seeds; 200 bootstrap resamples for mAP (CIs are Monte Carlo-noisy), 1000 for counts.
- 48 gap CIs are reported without multiplicity correction; the pre-registered rule is the decision, and the other CIs are descriptive.
- Severities were tuned on dev so FP32 mAP falls evenly (about 20% per level) and frozen before test runs. Blur severity 4 and the other level-4 settings drive mAP close to zero, so level-4 comparisons are floor-limited.
- Latency is laptop CPU (i5-13420H, 4 threads), not an edge device. INT8 is 1.60x faster on the raw network (28.4 -> 17.7 ms median vs FP32 ORT) but end-to-end is 68.3 vs 67.9 ms (1.01x), because decode/NMS in Python dominates.
- Not measured: QAT, per-channel and calibration-size ablations, Raspberry Pi.
