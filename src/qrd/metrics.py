"""mAP (pycocotools, person) and counting metrics. Pure functions of detections + GT."""
import contextlib
import io

import numpy as np

from .models import xyxy_to_xywh


def coco_ap(gt_xyxy, preds, crowd=None):
    """gt_xyxy: {img_id: (N,4) non-crowd boxes}; preds: {img_id: (boxes_xyxy, scores)};
    crowd: {img_id: (M,4) crowd boxes in xyxy} (ignored by standard COCO handling).
    Returns (mAP@[.5:.95], AP50). Empty-GT-only input -> (nan, nan)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    crowd = crowd or {}
    ids = sorted(gt_xyxy)
    images = [{"id": i, "width": 10**5, "height": 10**5} for i in ids]
    anns, aid = [], 1
    for i in ids:
        for b, c in [(b, 0) for b in xyxy_to_xywh(gt_xyxy[i])] + [(b, 1) for b in xyxy_to_xywh(np.asarray(crowd.get(i, np.zeros((0, 4)))))]:
            anns.append({"id": aid, "image_id": i, "category_id": 1, "bbox": [float(x) for x in b],
                         "area": float(b[2] * b[3]), "iscrowd": c})
            aid += 1
    res = [{"image_id": i, "category_id": 1, "bbox": [float(x) for x in b], "score": float(s)}
           for i in ids for b, s in zip(xyxy_to_xywh(preds[i][0]), preds[i][1])]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
        gt.createIndex()
        if not res:
            return 0.0, 0.0
        E = COCOeval(gt, gt.loadRes(res), "bbox")
        E.params.imgIds = ids
        E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0]), float(E.stats[1])


def count_metrics(pred_counts, gt_counts):
    e = np.asarray(pred_counts, float) - np.asarray(gt_counts, float)
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e**2).mean())), "bias": float(e.mean())}


def pred_count(scores, t):
    return int((np.asarray(scores) >= t).sum())
