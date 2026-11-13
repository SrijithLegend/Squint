"""Model size + latency (batch 1, fixed threads, 20 warm-up, median/p95 of >=100 timed runs).
Measured on the runtime that produced the INT8 numbers (ONNX Runtime) plus torch fp32 as the eval reference.
Label: laptop CPU, never edge device.   python -m src.qrd.bench_latency [--seed 0] [--threads 4] [--runs 200]
"""
import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from . import data, ssd


def cpu_name():
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or platform.processor()
    except Exception:
        return platform.processor()


def timeit(fn, warmup=20, runs=200):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95)), "runs": runs, "warmup": warmup}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--seed", type=int, default=0); a.add_argument("--threads", type=int, default=4)
    a.add_argument("--runs", type=int, default=200)
    a = a.parse_args()
    torch.set_num_threads(a.threads)
    m = ssd.build()
    m.load_state_dict(torch.load(f"data/ckpt/fp32_seed{a.seed}/final.pt", map_location="cpu"))
    x = np.random.default_rng(0).standard_normal((1, 3, 320, 320)).astype(np.float32)
    img = data.load_image("test", data.load_split("test")[0])  # fixed real image, for end-to-end timing
    variants = {"fp32_torch": (ssd.torch_raw_fn(m), f"data/ckpt/fp32_seed{a.seed}/final.pt"),
                "fp32_ort": (ssd.ort_raw_fn(f"data/models/fp32_s{a.seed}.onnx", a.threads), f"data/models/fp32_s{a.seed}.onnx"),
                "int8_ort": (ssd.ort_raw_fn(f"data/models/ptq_s{a.seed}.onnx", a.threads), f"data/models/ptq_s{a.seed}.onnx")}
    out = {"device": "laptop CPU (not an edge device)", "cpu": cpu_name(), "threads": a.threads, "batch": 1,
           "torch": torch.__version__, "onnxruntime": ort.__version__, "variants": {}}
    for k, (fn, path) in variants.items():
        det = ssd.SSDDetector(m, fn)
        out["variants"][k] = {"size_mb": os.path.getsize(path) / 1e6, "raw_network": timeit(lambda: fn(x), runs=a.runs),
                              "end_to_end": timeit(lambda: det.predict([img]), runs=a.runs)}
        print(k, out["variants"][k], flush=True)
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/latency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
