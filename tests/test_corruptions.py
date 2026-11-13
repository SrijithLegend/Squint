import numpy as np
import pytest

from qrd.corruptions import NAMES, corrupt, params


def _img():
    return np.random.default_rng(1).integers(0, 256, (96, 128, 3), dtype=np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255**2 / mse)


@pytest.mark.parametrize("name", NAMES)
def test_deterministic(name):
    a = corrupt(_img(), 7, name, 3, seed=0)
    assert a.tobytes() == corrupt(_img(), 7, name, 3, seed=0).tobytes()
    assert a.dtype == np.uint8 and a.shape == (96, 128, 3)


@pytest.mark.parametrize("name", NAMES)
def test_severity_monotone_psnr(name):
    im = _img()
    assert psnr(im, corrupt(im, 7, name, 1)) > psnr(im, corrupt(im, 7, name, 4))


def test_noise_depends_on_seed():
    im = _img()
    assert corrupt(im, 7, "gaussian_noise", 2, seed=0).tobytes() != corrupt(im, 7, "gaussian_noise", 2, seed=1).tobytes()
