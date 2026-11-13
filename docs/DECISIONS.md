# Decisions

Every gate decision, with reason and date.

| Date | Gate | Decision | Reason |
|---|---|---|---|
| 2026-11-13 | S0 | Repo root is the `Squint` repo itself, not a nested `quant-robust-det/` folder | Avoids a redundant directory level; package is still `src/qrd` |
| 2026-11-13 | G1 | **Track B chosen** (torchvision ssdlite320_mobilenet_v3_large -> ONNX raw heads, NMS in Python -> ONNX Runtime 1.27.0 static INT8, QDQ, CPU). Track A (PT2E/ExecuTorch) not attempted; C not needed. | Evidence (`results/sanity/g1_compare.txt`, first 100 val2017 person images, gate check only, nothing tuned): person mAP / AP50 fp32 torch 0.2841 / 0.5084, fp32 ORT identical, **int8 ORT 0.2618 / 0.4756** (AP50 about 3.3 points below FP32). Real INT8 path: ORT-optimized graph runs 98 `QLinearConv`, 10 `QLinearAdd`, etc. Size 13.8 MB -> 3.9 MB. Boxes visually sane on 5 images (`results/sanity/`). Class-restricted person AP is not comparable to published all-class COCO mAP; qualitative only. You said to continue without stopping, so G1 is treated as approved. |
| 2026-11-13 | G1 | QAT: PTQ-only unless a PyTorch QAT -> ONNX QDQ -> ORT path works in S3 | Rule in CLAUDE.md section 6; H2 dropped if not feasible |
