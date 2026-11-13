import numpy as np

from qrd.metrics import coco_ap, count_metrics, pred_count


def test_counting_fixture():
    # errors: +1, -2, 0 -> MAE 1, RMSE sqrt(5/3), bias -1/3
    m = count_metrics([3, 1, 2], [2, 3, 2])
    assert m["mae"] == 1.0 and abs(m["rmse"] - (5 / 3) ** 0.5) < 1e-12 and abs(m["bias"] + 1 / 3) < 1e-12


def test_pred_count_threshold():
    assert pred_count([0.9, 0.5, 0.49], 0.5) == 2


GT = {1: np.array([[10, 10, 50, 90]], float), 2: np.array([[0, 0, 30, 30], [40, 40, 90, 90]], float)}


def test_perfect_ap_is_one():
    p = {i: (b.copy(), np.ones(len(b))) for i, b in GT.items()}
    assert coco_ap(GT, p) == (1.0, 1.0)


def test_empty_ap_is_zero():
    p = {i: (np.zeros((0, 4)), np.zeros(0)) for i in GT}
    assert coco_ap(GT, p) == (0.0, 0.0)
