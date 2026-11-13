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


def evaluate_condition(dets, split, ids, cond, ann, seed=0, cfg=None, bs=8):
    """dets: {name: Detector}. Each image is corrupted ONCE and fed to every detector.
    Returns {name: (metrics dict, per-image preds {id: (boxes, scores)})}."""
    name, sev = (cond.split(":") + [0])[:2] if cond != "clean" else ("clean", 0)
    preds = {k: {} for k in dets}
    for k in range(0, len(ids), bs):
        chunk = ids[k:k + bs]
        imgs = [corruptions.corrupt(data.load_image(split, i), i, name, int(sev), seed, cfg) for i in chunk]
        for dn, d in dets.items():
            for i, r in zip(chunk, d.predict(imgs)):
                preds[dn][i] = (r["boxes_xyxy"], r["scores"])
    gt = {i: ann.boxes_xyxy(i) for i in ids}
    crowd = {i: ann.crowd_xyxy(i) for i in ids}
    gt_n = [len(gt[i]) for i in ids]
    out = {}
    for dn in dets:
        ap, ap50 = metrics.coco_ap(gt, preds[dn], crowd)
        out[dn] = ({"condition": cond, "n_images": len(ids), "mAP": ap, "AP50": ap50,
                    "count_by_threshold": {str(t): metrics.count_metrics(
                        [metrics.pred_count(preds[dn][i][1], t) for i in ids], gt_n) for t in THRESHOLDS}}, preds[dn])
    return out


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


ONNX_TAG = {"ptq_int8": "ptq", "ptq_pc": "ptq_pc", "ptq_cal25": "ptq_cal25", "ptq_cal500": "ptq_cal500"}


def load_detector(model, seed, threads=4):
    m = ssd.build()
    if model == "pretrained":
        return ssd.SSDDetector(m, ssd.torch_raw_fn(m)), "torch-fp32 (COCO-pretrained, not fine-tuned)"
    if model == "fp32":
        m.load_state_dict(torch.load(f"data/ckpt/fp32_seed{seed}/final.pt", map_location="cpu"))
        return ssd.SSDDetector(m, ssd.torch_raw_fn(m)), f"torch {torch.__version__} fp32 cpu"
    if model in ONNX_TAG:  # real INT8: ONNX Runtime QLinearConv etc. (see results/quant/*.json)
        return (ssd.SSDDetector(m, ssd.ort_raw_fn(f"data/models/{ONNX_TAG[model]}_s{seed}.onnx", threads)),
                f"onnxruntime {onnxruntime.__version__} CPUExecutionProvider int8 (QDQ)")
    raise ValueError(model)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--models", default="fp32"); a.add_argument("--seeds", default="0")
    a.add_argument("--split", default="dev"); a.add_argument("--n", type=int, default=0)
    a.add_argument("--conditions", default="all"); a.add_argument("--out", default="results/raw")
    a.add_argument("--threads", type=int, default=4)
    a = a.parse_args()
    torch.set_num_threads(a.threads)
    cfg_all = yaml.safe_load(open(corruptions.EVAL_CFG))
    ids = data.load_split(a.split)
    if a.n:
        ids = ids[:a.n]  # ids are seeded-sampled and sorted; first-n is a fixed, documented subset
    ann = data.Annotations(a.split)
    pairs = [(m, int(s)) for m in a.models.split(",") for s in a.seeds.split(",")]
    conds = all_conditions() if a.conditions == "all" else a.conditions.split(",")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); Path("data/dets").mkdir(parents=True, exist_ok=True)
    stem = lambda m, s, c: f"{a.split}_{m}_s{s}_{c.replace(':', '_')}"
    loaded = {}
    for cond in conds:
        todo = [(m, s) for m, s in pairs if not (out / f"{stem(m, s, cond)}.json").exists()]
        if not todo:
            continue
        for m, s in todo:
            if (m, s) not in loaded:
                loaded[(m, s)] = load_detector(m, s, a.threads)
        t0 = time.time()
        res = evaluate_condition({f"{m}|{s}": loaded[(m, s)][0] for m, s in todo}, a.split, ids, cond, ann, 0, cfg_all["corruptions"])
        for m, s in todo:
            r, preds = res[f"{m}|{s}"]
            r.update(model=m, seed=s, split=a.split, runtime=loaded[(m, s)][1], sec_shared=round(time.time() - t0, 1),
                     meta=run_meta(threads=a.threads, severities_frozen=cfg_all["frozen"], corruption_seed=0))
            json.dump(r, open(out / f"{stem(m, s, cond)}.json", "w"), indent=1)
            save_dets(f"data/dets/{stem(m, s, cond)}.npz", ids, preds)
            c = r["count_by_threshold"]["0.5"]
            print(f"{m:10s} s{s} {cond:16s} mAP {r['mAP']:.4f} AP50 {r['AP50']:.4f} count-MAE@0.5 {c['mae']:.3f} bias {c['bias']:+.3f}", flush=True)
        print(f"  [{cond}] {time.time() - t0:.0f}s for {len(todo)} models", flush=True)


if __name__ == "__main__":
    main()
