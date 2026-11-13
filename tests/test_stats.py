import numpy as np

from qrd import metrics, stats


def _toy(n=30, seed=0, id0=0):
    r = np.random.default_rng(seed)
    gt, crowd, preds = {}, {}, []
    for k in range(n):
        m = r.integers(1, 5)
        xy = r.uniform(0, 200, (m, 2)); wh = r.uniform(20, 80, (m, 2))
        g = np.hstack([xy, xy + wh])
        gt[id0 + k] = g; crowd[id0 + k] = np.zeros((0, 4))
        p = np.vstack([g + r.normal(0, 6, g.shape), r.uniform(0, 250, (3, 2)).repeat(2, 1) + [0, 0, 30, 30]])
        preds.append((p, r.uniform(0.01, 1, len(p))))
    return gt, crowd, preds


def test_fast_ap_matches_pycocotools():
    gt, crowd, preds = _toy()
    ids = sorted(gt)
    ab = stats.ApBoot(ids, preds, gt, crowd)
    ref = metrics.coco_ap(gt, {i: p for i, p in zip(ids, preds)}, crowd)
    np.testing.assert_allclose(ab.ap(None), ref, atol=1e-9)


def test_weights_equal_duplicated_images():
    gt, crowd, preds = _toy(20)
    ids = sorted(gt)
    w = np.random.default_rng(1).integers(0, 3, len(ids)).astype(float)
    gt2, crowd2, preds2, k = {}, {}, [], 0
    for i, p, wi in zip(ids, preds, w):
        for _ in range(int(wi)):
            gt2[k], crowd2[k] = gt[i], crowd[i]; preds2.append(p); k += 1
    a = stats.ApBoot(ids, preds, gt, crowd).ap(w)
    b = stats.ApBoot(sorted(gt2), preds2, gt2, crowd2).ap(None)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_bootstrap_seeded_and_paired():
    W1, W2 = stats.resample_weights(50, 100, 7), stats.resample_weights(50, 100, 7)
    assert (W1 == W2).all() and (W1.sum(1) == 50).all()
    assert not (stats.resample_weights(50, 100, 8) == W1).all()


def test_boot_count_mae_and_gap():
    e = np.array([1., 2, 3, 4]); W = np.array([[1., 1, 1, 1], [4, 0, 0, 0]])
    np.testing.assert_allclose(stats.boot_count_mae(e, W), [2.5, 1.0])
    g = stats.gap_curve({"clean": np.array([.5]), "c": np.array([.4])}, {"clean": np.array([.45]), "c": np.array([.2])}, +1)
    np.testing.assert_allclose(g["c"], (.4 - .2) - (.5 - .45))
