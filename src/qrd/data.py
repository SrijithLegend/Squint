"""COCO 2017 person subset: seeded disjoint splits, on-demand image download, loaders."""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("data/coco")
SPLITS = Path("data/splits")
SEED = 0


def _ann(year_split):
    return json.load(open(ROOT / f"annotations/instances_{year_split}2017.json"))


def person_ids(ann):
    """image ids with >=1 non-crowd person."""
    return sorted({a["image_id"] for a in ann["annotations"] if a["category_id"] == 1 and not a["iscrowd"]})


def build_splits(n_train, n_dev, n_test, seed=SEED):
    rng = np.random.default_rng(seed)
    tr = person_ids(_ann("train"))
    pick = rng.permutation(tr)
    out = {"train": sorted(map(int, pick[:n_train])),
           "dev": sorted(map(int, pick[n_train:n_train + n_dev])),
           "test": sorted(map(int, rng.permutation(person_ids(_ann("val")))[:n_test]))}
    SPLITS.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        json.dump(v, open(SPLITS / f"{k}_ids.json", "w"))
    return out


def load_split(name):
    return json.load(open(SPLITS / f"{name}_ids.json"))


def folder(split):  # test lives in val2017, the others in train2017
    return "val2017" if split == "test" else "train2017"


def img_path(split, i):
    return ROOT / folder(split) / f"{i:012d}.jpg"


def ensure_images(split, ids, workers=16):
    (ROOT / folder(split)).mkdir(parents=True, exist_ok=True)

    def get(i):
        p = img_path(split, i)
        for _ in range(4):  # timeout + retry; a hung socket otherwise stalls the pool
            if p.exists():
                return
            try:
                with urllib.request.urlopen(f"http://images.cocodataset.org/{folder(split)}/{i:012d}.jpg", timeout=20) as r:
                    b = r.read()
                p.with_suffix(".tmp").write_bytes(b)
                p.with_suffix(".tmp").replace(p)
            except Exception:
                pass
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, ids))


def load_image(split, i):
    return cv2.cvtColor(cv2.imread(str(img_path(split, i))), cv2.COLOR_BGR2RGB)


class Annotations:
    """Person annotations for one split's source json, indexed by image id."""

    def __init__(self, split):
        # Cache only this split's person annotations: the full COCO json costs GBs of RAM when parsed.
        cache = Path("data/cache") / f"ann_{split}.json"
        if cache.exists():
            rows = json.load(open(cache))
        else:
            ids = set(load_split(split))
            a = _ann("val" if split == "test" else "train")
            rows = [x for x in a["annotations"] if x["category_id"] == 1 and x["image_id"] in ids]
            del a
            cache.parent.mkdir(parents=True, exist_ok=True)
            json.dump(rows, open(cache, "w"))
        self.by_img = {}
        for x in rows:
            self.by_img.setdefault(x["image_id"], []).append(x)

    def boxes_xyxy(self, i):
        b = [x["bbox"] for x in self.by_img.get(i, []) if not x["iscrowd"]]
        b = np.array(b, dtype=np.float32).reshape(-1, 4)
        b[:, 2:] += b[:, :2]
        return b

    def crowd_xyxy(self, i):
        b = np.array([x["bbox"] for x in self.by_img.get(i, []) if x["iscrowd"]], dtype=np.float32).reshape(-1, 4)
        b[:, 2:] += b[:, :2]
        return b

    def count(self, i):
        return sum(1 for x in self.by_img.get(i, []) if not x["iscrowd"])
