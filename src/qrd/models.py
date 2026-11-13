"""Backend-agnostic detector interface. The evaluator only talks to this."""
from typing import Protocol, TypedDict

import numpy as np


class Detections(TypedDict):
    boxes_xyxy: np.ndarray  # (N, 4) float, pixel coords
    scores: np.ndarray      # (N,)
    labels: np.ndarray      # (N,) int; this study uses person only


class Detector(Protocol):
    def predict(self, images: list[np.ndarray]) -> list[Detections]:
        """images: uint8 RGB HxWx3 arrays. Returns post-NMS detections per image."""
        ...


def xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    return np.stack([b[..., 0], b[..., 1], b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], -1)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    return np.stack([b[..., 0], b[..., 1], b[..., 0] + b[..., 2], b[..., 1] + b[..., 3]], -1)
