"""G1 check: FP32 vs ORT-INT8 person AP on a small val2017 subset (gate check only, no tuning)."""
import json, sys, urllib.request
from pathlib import Path
import cv2, numpy as np, onnx, collections
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
sys.path.insert(0, "src")
from qrd import ssd
from qrd.models import xyxy_to_xywh

N = 100
root = Path("data/coco"); gt = COCO(str(root / "annotations/instances_val2017.json"))
ids = sorted(gt.getImgIds(catIds=[1]))
ids = [i for i in ids if any(not a["iscrowd"] for a in gt.loadAnns(gt.getAnnIds(imgIds=i, catIds=[1])))][:N]
m = ssd.build()
dets = {"fp32_torch": ssd.SSDDetector(m, ssd.torch_raw_fn(m)),
        "fp32_ort": ssd.SSDDetector(m, ssd.ort_raw_fn("data/models/ssdlite_fp32.onnx")),
        "int8_ort": ssd.SSDDetector(m, ssd.ort_raw_fn("data/models/ssdlite_int8.onnx"))}
# real INT8 evidence: ORT-optimized graph op types
so = ort.SessionOptions(); so.optimized_model_filepath = "data/models/int8_opt.onnx"
ort.InferenceSession("data/models/ssdlite_int8.onnx", so, providers=["CPUExecutionProvider"])
ops = collections.Counter(n.op_type for n in onnx.load("data/models/int8_opt.onnx").graph.node)
print("ORT executed graph:", {k: v for k, v in ops.items() if "Q" in k or "Conv" in k}, "| ort", ort.__version__)
imgs = {}
for i in ids:
    p = root / "val2017" / f"{i:012d}.jpg"
    if not p.exists(): urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{i:012d}.jpg", p)
    imgs[i] = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
for name, d in dets.items():
    res = []
    for i in ids:
        r = d.predict([imgs[i]])[0]
        for b, s in zip(xyxy_to_xywh(r["boxes_xyxy"]), r["scores"]):
            res.append({"image_id": i, "category_id": 1, "bbox": [float(x) for x in b], "score": float(s)})
    E = COCOeval(gt, gt.loadRes(res), "bbox"); E.params.imgIds = ids; E.params.catIds = [1]
    E.evaluate(); E.accumulate(); E.summarize()
    print(f"RESULT {name}: n_img={len(ids)} mAP={E.stats[0]:.4f} AP50={E.stats[1]:.4f}")
