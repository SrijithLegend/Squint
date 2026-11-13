"""results/stats.json (+ latency.json, dets) -> figures in results/figures and the AUTO block of docs/RESULTS.md.
python -m src.qrd.report"""
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import corruptions, data, stats

FIG = Path("results/figures")
COL = {"fp32": "#1f77b4", "ptq_int8": "#d62728"}
LAB = {"fp32": "FP32", "ptq_int8": "INT8 PTQ"}
FAM = corruptions.NAMES


def _curve(S, fam, model, metric):
    cs = ["clean"] + [f"{fam}:{s}" for s in (1, 2, 3, 4)]
    d = [S["conditions"][c][model][metric] for c in cs]
    return np.array([x["mean"] for x in d]), np.array([x["ci95_pooled_bootstrap"] for x in d])


def fig_metric(S, metric, ylabel, fname):
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for a, fam in zip(ax, FAM):
        for m in ("fp32", "ptq_int8"):
            y, c = _curve(S, fam, m, metric)
            a.plot(range(5), y, "-o", color=COL[m], label=LAB[m])
            a.fill_between(range(5), c[:, 0], c[:, 1], color=COL[m], alpha=.2)
        a.set_title(fam)
        a.set_xlabel("severity (0 = clean)")
        a.set_xticks(range(5))
    ax[0].set_ylabel(ylabel)
    ax[0].legend()
    fig.suptitle(f"{ylabel}: mean over {len(S['seeds'])} seeds, band = pooled paired image-bootstrap 95% CI "
                 f"(test, n={S['n_images']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def fig_gap(S):
    keys = list(S["gap"])
    fig, ax = plt.subplots(len(keys), 4, figsize=(15, 3.0 * len(keys)), sharey="row", squeeze=False)
    for r, k in enumerate(keys):
        for a, fam in zip(ax[r], FAM):
            v = [S["gap"][k][f"{fam}:{s}"] for s in (1, 2, 3, 4)]
            y = np.array([x["point_mean_over_seeds"] for x in v])
            c = np.array([x["ci95_pooled_bootstrap"] for x in v])
            a.errorbar(range(1, 5), y, yerr=[y - c[:, 0], c[:, 1] - y], fmt="o-", capsize=3, color="#444")
            a.axhline(0, color="k", lw=.8)
            a.set_title(f"{fam}  [{k}]", fontsize=9)
            a.set_xticks(range(1, 5))
            a.set_xlabel("severity")
        ax[r][0].set_ylabel("gap (>0: INT8 hurt more)")
    fig.tight_layout()
    fig.savefig(FIG / "gap_vs_severity.png", dpi=130)
    plt.close(fig)


def fig_latency(L):
    v = L["variants"]
    names = list(v)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].bar(names, [v[n]["size_mb"] for n in names])
    ax[0].set_title("file size (MB)")
    for a, k, t in ((ax[1], "raw_network", "network only (ms)"), (ax[2], "end_to_end", "end-to-end incl. decode/NMS (ms)")):
        med = np.array([v[n][k]["median_ms"] for n in names])
        p95 = np.array([v[n][k]["p95_ms"] for n in names])
        a.bar(names, med, yerr=[np.zeros(len(med)), p95 - med], capsize=3)
        a.set_title(f"median, p95 whisker: {t}", fontsize=9)
    fig.suptitle(f"{L['device']}: {L['cpu']}, {L['threads']} threads, batch 1", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "size_latency.png", dpi=130)
    plt.close(fig)


def failure_gallery(S, cond="low_light:4", k=12, seed=0, t=0.5):
    split = S["split"]
    ids = data.load_split(split)
    ann = data.Annotations(split)
    fam, sev = cond.split(":")

    def d(m):
        return stats.load_dets(f"data/dets/{split}_{m}_s{seed}_{cond.replace(':', '_')}.npz")[1]

    pf, pi = d("fp32"), d("ptq_int8")
    cf = np.array([(p[1] >= t).sum() for p in pf])
    ci = np.array([(p[1] >= t).sum() for p in pi])
    top = np.argsort(-np.abs(cf - ci), kind="stable")[:k]
    cfg = corruptions.params()
    tiles = []
    for j in top:
        im = corruptions.corrupt(data.load_image(split, ids[j]), ids[j], fam, int(sev), 0, cfg)
        im = np.ascontiguousarray(im)
        sc = 400 / im.shape[1]
        im = cv2.resize(im, (400, int(im.shape[0] * sc)))
        for p, col in ((pf[j], (0, 200, 0)), (pi[j], (255, 40, 40))):
            for b, s in zip(p[0], p[1]):
                if s >= t:
                    x0, y0, x1, y1 = (b * sc).astype(int)
                    cv2.rectangle(im, (x0, y0), (x1, y1), col, 1)
        cv2.putText(im, f"fp32 {cf[j]} / int8 {ci[j]} / gt {ann.count(ids[j])}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
        tile = np.zeros((300, 400, 3), np.uint8)
        tile[:min(300, im.shape[0])] = im[:300]
        tiles.append(tile)
    rows = [np.hstack(tiles[r * 4:r * 4 + 4]) for r in range(3)]
    cv2.imwrite(str(FIG / "failure_gallery.png"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    return [{"image_id": ids[j], "fp32_count": int(cf[j]), "int8_count": int(ci[j]), "gt": ann.count(ids[j])} for j in top]


def f(x, n=3):
    return f"{x:.{n}f}"


def _cell(r):
    lo, hi = r["ci95_pooled_bootstrap"]
    sd = "n/a" if r["std"] is None else f(r["std"])
    return f"{f(r['mean'])} ± {sd} [{f(lo)}, {f(hi)}]"


def tables(S, L, Q):
    ts = S["thresholds"]
    cl = S["conditions"]["clean"]
    out = [f"Test split: n = {S['n_images']} images, {len(S['seeds'])} training seeds (PTQ calibration seed = training seed). "
           f"± is std across seeds; [..] is the pooled paired image-bootstrap 95% CI ({S['b_map']} resamples for mAP/AP50, "
           f"{S['b_count']} for counts).", "",
           "### Clean images", "",
           "| model | mAP@[.5:.95] | AP50 | count MAE @0.5 | count bias @0.5 |" + "".join(f" count MAE @{t} (dev-chosen) |" for t in ts[1:]),
           "|---|---|---|---|---|" + "---|" * (len(ts) - 1)]
    for m in ("fp32", "ptq_int8"):
        r = cl[m]
        out.append(f"| {LAB[m]} | {_cell(r['mAP'])} | {_cell(r['AP50'])} | {_cell(r['mae@0.5'])} | {_cell(r['bias@0.5'])} |"
                   + "".join(f" {_cell(r[f'mae@{t}'])} |" for t in ts[1:]))
    if L:
        out += ["", f"### Size and latency ({L['device']}: {L['cpu']}, {L['threads']} threads, batch 1, torch {L['torch']}, "
                    f"onnxruntime {L['onnxruntime']})", "",
                "| variant | size MB | network median / p95 ms | end-to-end median / p95 ms |", "|---|---|---|---|"]
        for n, v in L["variants"].items():
            out.append(f"| {n} | {f(v['size_mb'], 2)} | {f(v['raw_network']['median_ms'], 2)} / {f(v['raw_network']['p95_ms'], 2)} | "
                       f"{f(v['end_to_end']['median_ms'], 2)} / {f(v['end_to_end']['p95_ms'], 2)} |")
    if Q:
        out += ["", f"INT8 path: {Q['runtime']}, executed INT8 ops {Q['executed_int8_ops']} (seed {Q['seed']} model); "
                    f"calibration = {len(Q['calibration_ids'])} dev images."]
    for metric, title in (("mAP", "mAP@[.5:.95]"), ("mae@0.5", "count MAE @ t=0.5")):
        out += ["", f"### {title} per condition", "", "| condition | FP32 | INT8 PTQ |", "|---|---|---|"]
        for c in S["conditions"]:
            out.append(f"| {c} | {_cell(S['conditions'][c]['fp32'][metric])} | {_cell(S['conditions'][c]['ptq_int8'][metric])} |")
    for k, tab in S["gap"].items():
        out += ["", f"### Robustness gap, {k}", "",
                "gap(c,s) = INT8's loss under corruption minus its loss on clean (positive: INT8 hurt more by corruption). "
                "**Bold** = pooled 95% CI entirely above 0; *italic* = entirely below 0.", "",
                "| condition | gap | 95% CI | per-seed |", "|---|---|---|---|"]
        for c, v in tab.items():
            lo, hi = v["ci95_pooled_bootstrap"]
            g = f(v["point_mean_over_seeds"], 4)
            g = f"**{g}**" if lo > 0 else f"*{g}*" if hi < 0 else g
            out.append(f"| {c} | {g} | [{f(lo, 4)}, {f(hi, 4)}] | {', '.join(f(x, 4) for x in v['per_seed'])} |")
    out += ["", "### Pre-registered decision (docs/PREREG.md)", "",
            f"**{S['h1_decision']}**. Conditions meeting the H1 rule: {S['h1_supported_by'] or 'none'}."]
    return "\n".join(out)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    S = json.load(open("results/stats.json"))
    L = json.load(open("results/latency.json")) if Path("results/latency.json").exists() else None
    q = sorted(Path("results/quant").glob("ptq_s0.json"))
    Q = json.load(open(q[0])) if q else None
    fig_metric(S, "mAP", "mAP@[.5:.95]", "map_vs_severity.png")
    fig_metric(S, "mae@0.5", "count MAE (t=0.5)", "count_mae_vs_severity.png")
    fig_gap(S)
    if L:
        fig_latency(L)
    json.dump(failure_gallery(S), open("results/failure_gallery.json", "w"), indent=1)
    p = Path("docs/RESULTS.md")
    txt = p.read_text(encoding="utf-8") if p.exists() else \
        "# Results\n\n<!-- AUTO:START -->\n<!-- AUTO:END -->\n\n## Analysis\n\n(not written yet)\n"
    a, b = txt.index("<!-- AUTO:START -->"), txt.index("<!-- AUTO:END -->")
    p.write_text(txt[:a] + "<!-- AUTO:START -->\n" + tables(S, L, Q) + "\n" + txt[b:], encoding="utf-8")
    print("report written")


if __name__ == "__main__":
    main()
