"""Tune corruption severities on DEV only (FP32 seed 0).

RULE (fixed before running): for each corruption, sweep one parameter on the first 250 dev images, then
interpolate (linear in mAP, along the sweep) the parameter whose mAP = 80/60/40/20 % of clean dev mAP.
Those 4 parameters become severities 1..4. `verify` re-evaluates them on all 500 dev images.
  python scripts/tune_severities.py sweep <name>   |   choose   |   verify <name>
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, torch, yaml
from qrd import corruptions, data, evaluate

SWEEP = {  # (param key, fixed extras, candidates in order of increasing harshness)
    "gaussian_blur": ("sigma", {}, [1, 2, 3, 4, 6, 8, 12, 16]),
    "gaussian_noise": ("sigma", {}, [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5]),
    "low_light": ("brightness", {"gamma": 1.5, "peak": 60.0, "read_sigma": 0.01}, [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05]),  # 0.25..0.05 added after the first sweep never reached 60% of clean
    "jpeg": ("quality", {}, [75, 50, 30, 15, 8, 4, 2, 1]),
}
TARGETS = [0.8, 0.6, 0.4, 0.2]
OUT = Path("results/tuning"); OUT.mkdir(parents=True, exist_ok=True)


def ev(det, ann, ids, name, p):
    cfg = {name: [p]}
    r = evaluate.evaluate_condition({"m": det}, "dev", ids, f"{name}:1", ann, 0, cfg)["m"][0]
    return r["mAP"], r["count_by_threshold"]["0.5"]["mae"]


def main():
    cmd = sys.argv[1]
    if cmd == "choose":
        clean = json.load(open(OUT / "clean.json"))["mAP"]
        chosen = {}
        for name, (key, extra, _) in SWEEP.items():
            s = json.load(open(OUT / f"sweep_{name}.json"))["points"]
            xs, ys = [q["value"] for q in s], [q["mAP"] for q in s]
            vals = []
            for t in TARGETS:
                tgt = t * clean
                k = next((i for i in range(1, len(ys)) if ys[i] <= tgt), None)
                if k is None:
                    print(f"WARNING {name}: sweep never reaches {t:.0%} of clean (min mAP {min(ys):.4f}); using harshest value")
                    vals.append(xs[-1]); continue
                f = (ys[k - 1] - tgt) / max(ys[k - 1] - ys[k], 1e-9)
                vals.append(xs[k - 1] + f * (xs[k] - xs[k - 1]))
            vals = [round(v, 3) if key != "quality" else int(round(v)) for v in vals]
            chosen[name] = [{key: v, **extra} for v in vals]
            print(name, vals)
        json.dump({"clean_dev_mAP_250": clean, "chosen": chosen}, open(OUT / "chosen.json", "w"), indent=1)
        return
    torch.set_num_threads(8)
    ann = data.Annotations("dev"); alls = data.load_split("dev")
    det, _ = evaluate.load_detector("fp32", 0)
    if cmd == "clean":
        r = evaluate.evaluate_condition({"m": det}, "dev", alls[:250], "clean", ann)["m"][0]
        json.dump({"mAP": r["mAP"], "n": 250}, open(OUT / "clean.json", "w")); print("clean", r["mAP"])
    elif cmd == "sweep":
        name = sys.argv[2]; key, extra, cands = SWEEP[name]
        f = OUT / f"sweep_{name}.json"; pts = json.load(open(f))["points"] if f.exists() else []  # reuse finished points
        for v in [c for c in cands if c not in {q["value"] for q in pts}]:
            m, c = ev(det, ann, alls[:250], name, {key: v, **extra}); pts.append({"value": v, "mAP": m, "count_mae": c}); pts.sort(key=lambda q: cands.index(q["value"]))
            print(name, v, round(m, 4), flush=True)
        json.dump({"n": 250, "points": pts}, open(OUT / f"sweep_{name}.json", "w"), indent=1)
    elif cmd == "verify":
        name = sys.argv[2]; chosen = json.load(open(OUT / "chosen.json"))["chosen"][name]; pts = []
        for p in chosen:
            m, c = ev(det, ann, alls, name, p); pts.append({"params": p, "mAP": m, "count_mae": c}); print(name, p, round(m, 4), flush=True)
        json.dump({"n": 500, "points": pts}, open(OUT / f"verify_{name}.json", "w"), indent=1)


main()
