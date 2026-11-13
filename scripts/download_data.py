import sys; sys.path.insert(0, "src")
from qrd import data
for k in ("dev", "test", "train"):
    ids = data.load_split(k); data.ensure_images(k, ids)
    print(k, len(ids), sum(data.img_path(k, i).exists() for i in ids), flush=True)
