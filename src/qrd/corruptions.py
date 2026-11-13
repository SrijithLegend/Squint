"""Deterministic corruptions on uint8 RGB. Same (image_id, name, severity, seed) -> same bytes."""
import zlib

import cv2
import numpy as np
import yaml

NAMES = ["gaussian_blur", "gaussian_noise", "low_light", "jpeg"]
EVAL_CFG = "configs/eval.yaml"


def params(cfg=EVAL_CFG):
    return yaml.safe_load(open(cfg))["corruptions"]


def _rng(image_id, name, severity, seed):
    return np.random.default_rng(zlib.crc32(f"{image_id}|{name}|{severity}|{seed}".encode()))


def corrupt(img, image_id, name, severity, seed=0, cfg=None):
    """severity 1..4. name 'clean' returns the image unchanged."""
    if name == "clean":
        return img
    p = (cfg or params())[name][severity - 1]
    rng = _rng(image_id, name, severity, seed)
    if name == "gaussian_blur":
        return cv2.GaussianBlur(img, (0, 0), sigmaX=p["sigma"])
    x = img.astype(np.float32) / 255.0
    if name == "gaussian_noise":
        x = x + rng.normal(0, p["sigma"], x.shape)
    elif name == "low_light":
        x = (x * p["brightness"]) ** p["gamma"]
        x = rng.poisson(np.clip(x, 0, 1) * p["peak"]) / p["peak"]  # shot noise, after darkening
        x = x + rng.normal(0, p["read_sigma"], x.shape)            # read noise
    elif name == "jpeg":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, p["quality"]])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(name)
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
