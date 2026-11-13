"""SSDLite320 MobileNetV3: raw backbone+heads (exportable) with decode/NMS in Python.

Anchors are constant for a fixed 320x320 input, so they are computed once.
Any backend (torch FP32, ONNX Runtime FP32/INT8) only has to supply raw head outputs.
"""
from collections import OrderedDict

import numpy as np
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.image_list import ImageList

PERSON = 1  # torchvision COCO category index for person
SIZE = 320


def build(weights="DEFAULT"):
    return ssdlite320_mobilenet_v3_large(weights=weights).eval()


class RawSSD(torch.nn.Module):
    """(B,3,320,320) normalized -> (cls_logits (B,A,91), bbox_regression (B,A,4))."""

    def __init__(self, m):
        super().__init__()
        self.backbone, self.head = m.backbone, m.head

    def forward(self, x):
        f = self.backbone(x)
        f = list(f.values()) if not isinstance(f, torch.Tensor) else [f]
        o = self.head(f)
        return o["cls_logits"], o["bbox_regression"]


def anchors(m):
    x = torch.zeros(1, 3, SIZE, SIZE)
    f = m.backbone(x)
    f = list(f.values()) if not isinstance(f, torch.Tensor) else [f]
    return m.anchor_generator(ImageList(x, [(SIZE, SIZE)]), f)


def preprocess(images):
    """uint8 RGB HxWx3 list -> (B,3,320,320) float32 normalized, orig sizes."""
    import cv2
    out, sizes = [], []
    for im in images:
        sizes.append(im.shape[:2])
        r = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        out.append(((r - 0.5) / 0.5).transpose(2, 0, 1))
    return np.stack(out), sizes


class SSDDetector:
    """Detector protocol impl. `raw_fn(np (B,3,320,320)) -> (cls, box) numpy`."""

    def __init__(self, m, raw_fn, score_thresh=0.001):
        self.m, self.raw_fn = m, raw_fn
        self.anchors = anchors(m)
        m.score_thresh = score_thresh

    def predict(self, images):
        x, sizes = preprocess(images)
        cls, box = self.raw_fn(x)
        head = OrderedDict(cls_logits=torch.as_tensor(cls), bbox_regression=torch.as_tensor(box))
        anc = self.anchors * len(images)
        with torch.no_grad():
            dets = self.m.postprocess_detections(head, anc, [(SIZE, SIZE)] * len(images))
        res = []
        for d, (h, w) in zip(dets, sizes):
            k = d["labels"] == PERSON
            b = d["boxes"][k].numpy().copy()
            b[:, [0, 2]] *= w / SIZE
            b[:, [1, 3]] *= h / SIZE
            res.append({"boxes_xyxy": b, "scores": d["scores"][k].numpy(), "labels": d["labels"][k].numpy()})
        return res


def torch_raw_fn(m):
    raw = RawSSD(m).eval()

    def fn(x):
        with torch.no_grad():
            c, b = raw(torch.from_numpy(x))
        return c.numpy(), b.numpy()
    return fn


def ort_raw_fn(path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    return lambda x: tuple(s.run(None, {n: x}))
