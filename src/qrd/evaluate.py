"""Evaluate a Detector over a (split, condition) grid -> one JSON per (model, seed, condition).

python -m src.qrd.evaluate --model fp32 --seed 0 --split dev --n 500 [--conditions clean,jpeg:3]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import yaml

from . import corruptions, data, metrics, ssd
from .models import Detector

THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


def run(detector: Detector, images: list[np.ndarray]):  # kept: stub API from S0
    return detector.predict(images)


def all_conditions():
    return ["clean"] + [f"{c}:{s}" for c in corruptions.NAMES for s in (1, 2, 3, 4)]


def evaluate_condition(det: Detector, split, ids, cond, ann, seed=0, cfg=None, bs=8):
    """Returns (metrics dict, per-image preds {id: (boxes, scores)})."""
    name, sev = (cond.split(":") + [0])[:2] if cond != "clean" else ("clean", 0)
    preds = {}
    for k in range(0, len(ids), bs):
        chunk = ids[k:k + bs]
        imgs = [corruptions.corrupt(data.load_image(split, i), i, name, int(sev), seed, cfg) for i in chunk]
        for i, r in zip(chunk, det.predict(imgs)):
            preds[i] = (r["boxes_xyxy"], r["scores"])
    gt = {i: ann.boxes_xyxy(i) for i in ids}
    crowd = {i: ann.crowd_xyxy(i) for i in ids}
    ap, ap50 = metrics.coco_ap(gt, preds, crowd)
    gt_n = [len(gt[i]) for i in ids]
    out = {"condition": cond, "n_images": len(ids), "mAP": ap, "AP50": ap50,
           "count_by_threshold": {str(t): metrics.count_metrics([metrics.pred_count(preds[i][1], t) for i in ids], gt_n)
                                  for t in THRESHOLDS}}
    return out, preds


def save_dets(path, ids, preds):
    off = np.cumsum([0] + [len(preds[i][1]) for i in ids])
    np.savez_compressed(path, ids=np.array(ids), off=off,
                        boxes=np.concatenate([preds[i][0] for i in ids]).astype(np.float32).reshape(-1, 4),
                        scores=np.concatenate([preds[i][1] for i in ids]).astype(np.float32))


def run_meta(**kw):
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    return {"git_commit": sha, "git_dirty": dirty, "torch": torch.__version__, "onnxruntime": onnxruntime.__version__,
            "device": "cpu", **kw}


def load_detector(model, seed):
    m = ssd.build()
    if model == "fp32":
        m.load_state_dict(torch.load(f"data/ckpt/fp32_seed{seed}/final.pt", map_location="cpu"))
        return ssd.SSDDetector(m, ssd.torch_raw_fn(m)), "torch-fp32"
    raise ValueError(f"model {model!r} not implemented yet (S3)")  # ponytail: int8 variants arrive with quantize.py


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="fp32"); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--split", default="dev"); a.add_argument("--n", type=int, default=0)
    a.add_argument("--conditions", default="all"); a.add_argument("--out", default="results/raw")
    a.add_argument("--threads", type=int, default=4); a.add_argument("--tag", default="")
    a = a.parse_args()
    torch.set_num_threads(a.threads)
    cfg_all = yaml.safe_load(open(corruptions.EVAL_CFG))
    ids = data.load_split(a.split)
    if a.n:
        ids = ids[:a.n]  # ids are seeded-sampled and sorted; first-n is a fixed, documented subset
    ann = data.Annotations(a.split)
    det, runtime = load_detector(a.model, a.seed)
    conds = all_conditions() if a.conditions == "all" else a.conditions.split(",")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for cond in conds:
        t0 = time.time()
        res, preds = evaluate_condition(det, a.split, ids, cond, ann, a.seed, cfg_all["corruptions"])
        res.update(model=a.model, seed=a.seed, split=a.split, runtime=runtime, sec=round(time.time() - t0, 1),
                   meta=run_meta(threads=a.threads, severities_frozen=cfg_all["frozen"]))
        stem = f"{a.tag}{a.split}_{a.model}_s{a.seed}_{cond.replace(':', '_')}"
        json.dump(res, open(out / f"{stem}.json", "w"), indent=1)
        Path("data/dets").mkdir(parents=True, exist_ok=True)
        save_dets(f"data/dets/{stem}.npz", ids, preds)
        c = res["count_by_threshold"]["0.5"]
        print(f"{cond:16s} mAP {res['mAP']:.4f} AP50 {res['AP50']:.4f} count-MAE@0.5 {c['mae']:.3f} bias {c['bias']:+.3f} ({res['sec']}s)", flush=True)


if __name__ == "__main__":
    main()
