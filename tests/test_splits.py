import pytest

from qrd import data

pytest.importorskip("pycocotools")
S = {k: set(data.load_split(k)) for k in ("train", "dev", "test")}


def test_disjoint():
    assert not (S["train"] & S["dev"]) and not (S["train"] & S["test"]) and not (S["dev"] & S["test"])


def test_test_has_noncrowd_person():
    a = data.Annotations("test")
    assert all(a.count(i) >= 1 for i in S["test"])
