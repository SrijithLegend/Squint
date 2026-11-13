"""Seed aggregation, paired image-level bootstrap, and the robustness gap.

mAP bootstrap is done by weighting images (multiplicity from the resample) over ONE global score ordering of
all detections: equal to duplicating images, but no per-resample COCOeval. Validated against pycocotools in tests.
"""
import contextlib
import json
import io

import numpy as np

from . import data, metrics
from .models import xyxy_to_xywh

REC = np.linspace(0, 1, 101)


def load_dets(path):
    z = np.load(path)
    ids, off = z["ids"].tolist(), z["off"]
    return ids, [(z["boxes"][off[k]:off[k + 1]], z["scores"][off[k]:off[k + 1]]) for k in range(len(ids))]


class ApBoot:
    """Per-image COCO matching done once; `ap(weights)` -> (mAP@[.5:.95], AP50) for integer image weights."""

    def __init__(self, ids, preds, gt, crowd):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        images = [{"id": i, "width": 10**5, "height": 10**5} for i in ids]
        anns, aid = [], 1
        for i in ids:
            for b, c in [(b, 0) for b in xyxy_to_xywh(gt[i])] + [(b, 1) for b in xyxy_to_xywh(crowd[i])]:
                anns.append({"id": aid, "image_id": i, "category_id": 1, "bbox": [float(x) for x in b],
                             "area": float(b[2] * b[3]), "iscrowd": c}); aid += 1
        res = [{"image_id": i, "category_id": 1, "bbox": [float(x) for x in b], "score": float(s)}
               for i, (bb, ss) in zip(ids, preds) for b, s in zip(xyxy_to_xywh(bb), ss)]
        self.n = len(ids)
        with contextlib.redirect_stdout(io.StringIO()):
            g = COCO(); g.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}; g.createIndex()
            E = COCOeval(g, g.loadRes(res), "bbox") if res else None
            if E is None:
                self.empty = True; return
            E.params.imgIds = ids; E.evaluate(); E.accumulate(); E.summarize()
        self.empty, self.ref = False, (float(E.stats[0]), float(E.stats[1]))
        I = len(ids); sc, dm, di, im, ngt = [], [], [], [], np.zeros(I)
        for k in range(I):  # evalImgs layout with 1 category: [area_idx * I + image_idx]; area 0 = 'all'
            e = E.evalImgs[k]
            if e is None:
                continue
            ngt[k] = int((np.asarray(e["gtIgnore"]) == 0).sum())
            sc.append(np.asarray(e["dtScores"])); dm.append(np.asarray(e["dtMatches"])); di.append(np.asarray(e["dtIgnore"]))
            im.append(np.full(len(e["dtScores"]), k))
        sc, im = np.concatenate(sc), np.concatenate(im)
        o = np.argsort(-sc, kind="mergesort")
        dm, di = np.concatenate(dm, 1)[:, o], np.concatenate(di, 1)[:, o]
        self.img = im[o]; self.ngt = ngt
        self.tp = ((dm != 0) & ~di.astype(bool)).astype(np.float64)
        self.fp = ((dm == 0) & ~di.astype(bool)).astype(np.float64)

    def ap(self, w=None):
        if self.empty:
            return 0.0, 0.0
        w = np.ones(self.n) if w is None else w
        wd = w[self.img]
        npig = float((w * self.ngt).sum())
        if npig == 0:
            return float("nan"), float("nan")
        tp, fp = np.cumsum(self.tp * wd, 1), np.cumsum(self.fp * wd, 1)
        aps = []
        for t in range(tp.shape[0]):
            den = tp[t] + fp[t]
            pr = np.divide(tp[t], den, out=np.zeros_like(den), where=den > 0)
            rc = tp[t] / npig
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            inds = np.searchsorted(rc, REC, side="left")
            q = np.zeros(101); ok = inds < len(pr); q[ok] = pr[inds[ok]]
            aps.append(q.mean())
        return float(np.mean(aps)), float(aps[0])


def resample_weights(n, B, seed):
    """(B, n) integer multiplicities from image-level resampling with replacement. Same matrix for every model."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    return np.stack([np.bincount(r, minlength=n) for r in idx]).astype(np.float64)


def boot_count_mae(abs_err, W):
    """abs_err (n,), W (B,n) -> (B,) MAE of each resample."""
    return (W @ abs_err) / W.sum(1)


def ci(x, level=0.95):
    a = (1 - level) / 2
    return [float(np.quantile(x, a)), float(np.quantile(x, 1 - a))]


def gap_curve(fp_vals, int_vals, sign):
    """fp_vals/int_vals: {cond: array over resamples (already averaged over seeds)}.
    gap(c) = sign*[fp(c) - int(c)] - sign*[fp(clean) - int(clean)]; sign=+1 for mAP, -1 for count-MAE."""
    base = sign * (fp_vals["clean"] - int_vals["clean"])
    return {c: sign * (fp_vals[c] - int_vals[c]) - base for c in fp_vals}


# ---------------------------------------------------------------- full analysis (reads results/raw + data/dets)
def analyze(split="test", seeds=(0, 1, 2), out="results/stats.json", b_map=200, b_cnt=1000, boot_seed=12345,
            models=("fp32", "ptq_int8")):
    import pickle
    from pathlib import Path

    import yaml

    from . import evaluate
    ids = data.load_split(split); ann = data.Annotations(split); n = len(ids)
    gt = {i: ann.boxes_xyxy(i) for i in ids}; crowd = {i: ann.crowd_xyxy(i) for i in ids}
    gtn = np.array([len(gt[i]) for i in ids], float)
    cfg = yaml.safe_load(open("configs/eval.yaml"))
    ts = [0.5] + ([cfg["count_threshold_dev"]] if cfg.get("count_threshold_dev") else [])
    W = resample_weights(n, b_cnt, boot_seed)  # paired: one matrix for every model/seed/condition
    conds = evaluate.all_conditions()
    cache = Path("data/cache") / f"boot_{split}.pkl"
    store = pickle.load(open(cache, "rb")) if cache.exists() else {}
    for m in models:
        for s in seeds:
            for c in conds:
                key = (m, s, c, tuple(ts), b_map, b_cnt, boot_seed)
                if key in store:
                    continue
                pids, preds = load_dets(f"data/dets/{split}_{m}_s{s}_{c.replace(':', '_')}.npz")
                assert pids == ids, "detections were made on a different image list"
                ab = ApBoot(ids, preds, gt, crowd)
                (mp, a50) = ab.ap(None)
                r = {"mAP": mp, "AP50": a50, "mAP_b": np.empty(b_map), "AP50_b": np.empty(b_map)}
                for b in range(b_map):
                    r["mAP_b"][b], r["AP50_b"][b] = ab.ap(W[b])
                for t in ts:
                    e = np.array([metrics.pred_count(p[1], t) for p in preds], float) - gtn
                    r[f"mae@{t}"], r[f"rmse@{t}"], r[f"bias@{t}"] = np.abs(e).mean(), np.sqrt((e**2).mean()), e.mean()
                    r[f"mae_b@{t}"] = boot_count_mae(np.abs(e), W)
                    r[f"bias_b@{t}"] = (W @ e) / W.sum(1)
                store[key] = r
                print("boot", m, s, c, flush=True)
        pickle.dump(store, open(cache, "wb"))
    g = lambda m, s, c: store[(m, s, c, tuple(ts), b_map, b_cnt, boot_seed)]
    metrics_ = [("mAP", b_map), ("AP50", b_map)] + [(f"mae@{t}", b_cnt) for t in ts] + [(f"bias@{t}", b_cnt)]
    res = {"split": split, "n_images": n, "seeds": list(seeds), "b_map": b_map, "b_count": b_cnt, "boot_seed": boot_seed,
           "thresholds": ts, "conditions": {}, "gap": {}}
    for c in conds:
        res["conditions"][c] = {}
        for m in models:
            d = {}
            for k, B in metrics_:
                pt = np.array([g(m, s, c)[k] for s in seeds])
                bk = k.replace("mAP", "mAP_b").replace("AP50", "AP50_b") if k in ("mAP", "AP50") else k.replace("@", "_b@", 1)
                pooled = np.mean([g(m, s, c)[bk][:B] for s in seeds], 0)
                d[k] = {"mean": float(pt.mean()), "std": float(pt.std(ddof=1)) if len(pt) > 1 else None,
                        "ci95_pooled_bootstrap": ci(pooled), "per_seed": pt.tolist()}
            res["conditions"][c][m] = d
    # robustness gap: INT8 loss under corruption minus INT8 loss on clean. Positive = INT8 degrades MORE under corruption.
    for k, B, sign in [("mAP", b_map, +1)] + [(f"mae@{t}", b_cnt, -1) for t in ts]:
        bk = "mAP_b" if k == "mAP" else k.replace("@", "_b@", 1)
        fpb = {c: np.mean([g("fp32", s, c)[bk][:B] for s in seeds], 0) for c in conds}
        inb = {c: np.mean([g("ptq_int8", s, c)[bk][:B] for s in seeds], 0) for c in conds}
        gb = gap_curve(fpb, inb, sign)
        pt = {}
        for s in seeds:
            base = sign * (g("fp32", s, "clean")[k] - g("ptq_int8", s, "clean")[k])
            pt[s] = {c: sign * (g("fp32", s, c)[k] - g("ptq_int8", s, c)[k]) - base for c in conds}
        res["gap"][k] = {c: {"point_mean_over_seeds": float(np.mean([pt[s][c] for s in seeds])),
                             "ci95_pooled_bootstrap": ci(gb[c]), "per_seed": [pt[s][c] for s in seeds]}
                         for c in conds if c != "clean"}
    # PREREG decision rule: H1 supported if gap CI lies entirely above 0 at severities 3 AND 4 for >=1 family
    sup = []
    for k, tab in res["gap"].items():
        for fam in ("gaussian_blur", "gaussian_noise", "low_light", "jpeg"):
            if all(tab[f"{fam}:{sv}"]["ci95_pooled_bootstrap"][0] > 0 for sv in (3, 4)):
                sup.append({"metric": k, "family": fam})
    res["h1_supported_by"] = sup
    all_incl0 = all(v["ci95_pooled_bootstrap"][0] <= 0 <= v["ci95_pooled_bootstrap"][1] for tab in res["gap"].values() for v in tab.values())
    res["h1_decision"] = ("H1 supported" if sup else "null (H0): gap CI includes 0 at every condition" if all_incl0 else
                          "mixed: some gap CIs exclude 0 but H1's rule (CI above 0 at severities 3 and 4 of one family) is not met")
    Path(out).parent.mkdir(exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    r = analyze()
    print(r["h1_decision"], r["h1_supported_by"])
