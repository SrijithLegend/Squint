"""Track B: export raw SSD -> ONNX -> ORT static INT8 (QDQ), compare boxes & AP50 vs FP32.
Calibration images come from the train2017 split used as 'dev' (never test)."""
import json, sys, urllib.request
from pathlib import Path

import cv2, numpy as np, torch

sys.path.insert(0, "src")
from qrd import ssd
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

W = Path("results/sanity"); W.mkdir(parents=True, exist_ok=True)
Path("data/models").mkdir(parents=True, exist_ok=True)
m = ssd.build()
fp32 = "data/models/ssdlite_fp32.onnx"
torch.onnx.export(ssd.RawSSD(m), torch.zeros(1, 3, 320, 320), fp32, input_names=["x"],
                  output_names=["cls", "box"], dynamic_axes={"x": {0: "b"}, "cls": {0: "b"}, "box": {0: "b"}},
                  opset_version=17, dynamo=False)

# calibration: 20 train2017 person images (dev-style, not val/test)
root = Path("data/coco"); (root / "train2017").mkdir(exist_ok=True)
ann = json.load(open(root / "annotations/instances_train2017.json"))
ids = sorted({a["image_id"] for a in ann["annotations"] if a["category_id"] == 1 and not a["iscrowd"]})[:20]
cal = []
for i in ids:
    p = root / "train2017" / f"{i:012d}.jpg"
    if not p.exists():
        urllib.request.urlretrieve(f"http://images.cocodataset.org/train2017/{i:012d}.jpg", p)
    cal.append(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
class R(CalibrationDataReader):
    def __init__(s): s.it = iter(cal)
    def get_next(s):
        im = next(s.it, None)
        return None if im is None else {"x": ssd.preprocess([im])[0]}
int8 = "data/models/ssdlite_int8.onnx"
quantize_static(fp32, int8, R(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
import onnx, collections
ops = collections.Counter(n.op_type for n in onnx.load(int8).graph.node)
print("op types:", dict(ops))
import os; print("MB fp32/int8:", os.path.getsize(fp32)/1e6, os.path.getsize(int8)/1e6)
