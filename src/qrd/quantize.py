"""Track B PTQ: fine-tuned FP32 checkpoint -> ONNX (raw heads) -> ONNX Runtime static INT8 (QDQ).

Calibration images come from the DEV split only (never test). Calibration subset is seeded.
QAT is not implemented on this track (CLAUDE.md section 6: report PTQ only, H2 dropped).

python -m src.qrd.quantize --seed 0 [--n-cal 100] [--per-channel]
"""
import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from . import data, ssd

MODELS = Path("data/models")


def export_fp32(seed):
    """ONNX of the fine-tuned seed-`seed` model."""
    m = ssd.build()
    m.load_state_dict(torch.load(f"data/ckpt/fp32_seed{seed}/final.pt", map_location="cpu"))
    path = MODELS / f"fp32_s{seed}.onnx"
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(ssd.RawSSD(m.eval()), torch.zeros(1, 3, 320, 320), str(path), input_names=["x"],
                      output_names=["cls", "box"], dynamic_axes={"x": {0: "b"}, "cls": {0: "b"}, "box": {0: "b"}},
                      opset_version=17, dynamo=False)
    return path


class _Reader(CalibrationDataReader):
    def __init__(self, imgs):
        self.it = iter(imgs)

    def get_next(self):
        im = next(self.it, None)
        return None if im is None else {"x": ssd.preprocess([im])[0]}


def calibration_ids(cal_seed, n):
    ids = data.load_split("dev")
    return sorted(int(i) for i in np.random.default_rng(1000 + cal_seed).choice(ids, n, replace=False))


def ptq(seed, n_cal=100, per_channel=False, tag="ptq"):
    fp32 = export_fp32(seed)
    ids = calibration_ids(seed, n_cal)
    imgs = [data.load_image("dev", i) for i in ids]
    out = MODELS / f"{tag}_s{seed}.onnx"
    quantize_static(str(fp32), str(out), _Reader(imgs), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=per_channel)
    # evidence of a real INT8 execution path: op types of the graph ORT actually runs
    so = ort.SessionOptions(); so.optimized_model_filepath = str(MODELS / f"{tag}_s{seed}_opt.onnx")
    ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    ops = collections.Counter(n.op_type for n in onnx.load(so.optimized_model_filepath).graph.node)
    meta = {"seed": seed, "calibration_split": "dev", "calibration_ids": ids, "per_channel": per_channel,
            "quant_format": "QDQ", "activation": "QUInt8", "weight": "QInt8", "runtime": f"onnxruntime {ort.__version__} CPUExecutionProvider",
            "executed_int8_ops": {k: v for k, v in ops.items() if k.startswith("QLinear") or k in ("ConvInteger", "MatMulInteger")},
            "size_mb_fp32": os.path.getsize(fp32) / 1e6, "size_mb_int8": os.path.getsize(out) / 1e6}
    Path("results/quant").mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(f"results/quant/{tag}_s{seed}.json", "w"), indent=1)
    assert meta["executed_int8_ops"], "no INT8 ops in the executed graph: this would be fake-quant"
    return meta


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--seed", type=int, default=0); a.add_argument("--n-cal", type=int, default=100)
    a.add_argument("--per-channel", action="store_true"); a.add_argument("--tag", default="ptq")
    a = a.parse_args()
    m = ptq(a.seed, a.n_cal, a.per_channel, a.tag)
    print({k: v for k, v in m.items() if k != "calibration_ids"})
