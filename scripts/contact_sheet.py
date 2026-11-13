"""Contact sheet: every corruption x severity on one fixed dev image -> results/corruption_samples.png"""
import sys; sys.path.insert(0, "src")
import cv2, numpy as np
from qrd import corruptions, data

ann = data.Annotations("dev")
ids = data.load_split("dev")
iid = next(i for i in ids if ann.count(i) >= 3)  # first dev image with >=3 people (deterministic)
img = data.load_image("dev", iid)
img = cv2.resize(img, (320, int(img.shape[0] * 320 / img.shape[1])))
cfg = corruptions.params()
rows = []
for name in corruptions.NAMES:
    cells = [img] + [corruptions.corrupt(img, iid, name, s, 0, cfg) for s in (1, 2, 3, 4)]
    cells = [cv2.putText(c.copy(), lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
             for c, lab in zip(cells, [f"{name} clean"] + [f"sev {s}" for s in (1, 2, 3, 4)])]
    rows.append(np.hstack(cells))
cv2.imwrite("results/corruption_samples.png", cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
print("dev image id", iid)
