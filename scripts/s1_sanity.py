"""S1: FP32 sanity on 5 val2017 images -> results/sanity/*.jpg"""
import json, sys, urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "src")
from qrd import ssd

root = Path("data/coco"); (root / "val2017").mkdir(exist_ok=True)
ann = json.load(open(root / "annotations/instances_val2017.json"))
pid = sorted({a["image_id"] for a in ann["annotations"] if a["category_id"] == 1 and not a["iscrowd"]})[:5]
m = ssd.build(); det = ssd.SSDDetector(m, ssd.torch_raw_fn(m))
Path("results/sanity").mkdir(parents=True, exist_ok=True)
for i in pid:
    p = root / "val2017" / f"{i:012d}.jpg"
    if not p.exists():
        urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{i:012d}.jpg", p)
    im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
    d = det.predict([im])[0]
    out = im.copy()
    for b, s in zip(d["boxes_xyxy"], d["scores"]):
        if s >= 0.5:
            cv2.rectangle(out, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 255, 0), 2)
    gt = sum(1 for a in ann["annotations"] if a["image_id"] == i and a["category_id"] == 1 and not a["iscrowd"])
    print(i, "GT persons", gt, "pred>=0.5", int((d["scores"] >= 0.5).sum()))
    cv2.imwrite(f"results/sanity/{i:012d}.jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
