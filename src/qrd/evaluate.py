"""Stub evaluator: runs a Detector over images. Metrics/grid arrive in S2."""
import numpy as np

from .models import Detections, Detector


def run(detector: Detector, images: list[np.ndarray]) -> list[Detections]:
    return detector.predict(images)
