import numpy as np

from qrd.evaluate import run
from qrd.models import xywh_to_xyxy, xyxy_to_xywh


def test_box_roundtrip():
    b = np.random.default_rng(0).uniform(0, 100, (10, 4))
    b[:, 2:] += b[:, :2]  # make valid xyxy
    assert np.allclose(xywh_to_xyxy(xyxy_to_xywh(b)), b)


class _Dummy:
    def predict(self, images):
        return [{"boxes_xyxy": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, int)} for _ in images]


def test_stub_evaluator_uses_protocol():
    out = run(_Dummy(), [np.zeros((8, 8, 3), np.uint8)] * 2)
    assert len(out) == 2 and out[0]["boxes_xyxy"].shape == (0, 4)
