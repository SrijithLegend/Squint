"""Fine-tune SSDLite on COCO person (persons only as targets). Checkpoints every epoch; resumable."""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from . import data, ssd


class PersonSet(torch.utils.data.Dataset):
    def __init__(self, split, seed):
        self.split, self.ids, self.ann, self.rng = split, data.load_split(split), data.Annotations(split), random.Random(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        im = data.load_image(self.split, i)
        b = self.ann.boxes_xyxy(i)
        if self.rng.random() < 0.5:  # horizontal flip
            im = im[:, ::-1].copy()
            w = im.shape[1]
            b = np.stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]], 1) if len(b) else b
        x = torch.from_numpy(im).permute(2, 0, 1).float() / 255
        keep = (b[:, 2] - b[:, 0] > 1) & (b[:, 3] - b[:, 1] > 1)
        b = b[keep]
        return x, {"boxes": torch.from_numpy(b).float().reshape(-1, 4), "labels": torch.ones(len(b), dtype=torch.int64)}


def _collate(b):
    return tuple(zip(*b))


def run(cfg_path="configs/train.yaml", seed=0):
    cfg = yaml.safe_load(open(cfg_path))
    out = Path(cfg["out_dir"]) / f"fp32_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = ssd.build().to(dev)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    start = 0
    ck = sorted(out.glob("ep*.pt"))
    if ck:  # resume
        s = torch.load(ck[-1], map_location=dev)
        m.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        start = s["epoch"] + 1
        print(f"resumed from {ck[-1]}", flush=True)
    g = torch.Generator().manual_seed(seed)
    dl = torch.utils.data.DataLoader(PersonSet("train", seed), batch_size=cfg["batch"], shuffle=True, generator=g,
                                     num_workers=cfg["workers"], collate_fn=_collate, drop_last=True)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    log = []
    for ep in range(start, cfg["epochs"]):
        m.train()
        for mod in m.modules():  # freeze BN statistics (small-batch fine-tune)
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.eval()
        t, tot, n = time.time(), 0.0, 0
        for xs, ts in dl:
            xs = [x.to(dev) for x in xs]; ts = [{k: v.to(dev) for k, v in d.items()} for d in ts]
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = sum(m(xs, ts).values())
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item(); n += 1
            if n % 20 == 0:
                print(f"ep{ep} it{n} loss {tot/n:.4f} {time.time()-t:.0f}s", flush=True)
        sched.step()
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep}, out / f"ep{ep}.pt")
        log.append({"epoch": ep, "train_loss": tot / max(n, 1), "sec": time.time() - t})
        print(log[-1], flush=True)
    meta = {"seed": seed, "device": dev, "torch": torch.__version__, "cfg": cfg, "log": log}
    json.dump(meta, open(out / "train_meta.json", "w"), indent=1)
    torch.save(m.state_dict(), out / "final.pt")


if __name__ == "__main__":
    run(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
