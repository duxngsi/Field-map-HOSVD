"""Tests for hosvd.metrics and hosvd.datasets."""
import numpy as np
import pytest

from hosvd.datasets import make_synthetic_field, add_noise, DEFAULT_SHAPE
from hosvd.metrics import (
    frobenius_norm,
    relative_error,
    rmse,
    snr_db,
    psnr_db,
    compression_ratio,
)


def test_relative_error_zero_for_identical():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5, 6))
    assert relative_error(x, x) == 0.0
    assert rmse(x, x) == 0.0


def test_relative_error_value():
    x = np.ones((10,))
    y = np.zeros((10,))
    assert relative_error(x, y) == pytest.approx(1.0)


def test_relative_error_handles_zero_reference():
    z = np.zeros((5,))
    assert relative_error(z, z) == 0.0


def test_frobenius_norm():
    x = np.array([3.0, 4.0])
    assert frobenius_norm(x) == pytest.approx(5.0)


def test_snr_infinite_for_perfect():
    x = np.arange(10.0)
    assert snr_db(x, x) == float("inf")


def test_snr_increases_as_noise_drops():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50,))
    near = x + 0.01 * rng.normal(size=x.shape)
    far = x + 0.5 * rng.normal(size=x.shape)
    assert snr_db(x, near) > snr_db(x, far)


def test_psnr_infinite_for_perfect():
    x = np.linspace(0, 1, 16)
    assert psnr_db(x, x) == float("inf")


def test_compression_ratio_matches_definition():
    ratio = compression_ratio(
        original_shape=(20, 60, 15, 15, 3),
        core_shape=(3, 8, 4, 4, 3),
        factor_shapes=[(20, 3), None, (15, 4), (15, 4), None],
    )
    stored = 3 * 8 * 4 * 4 * 3 + 20 * 3 + 15 * 4 + 15 * 4
    assert ratio == pytest.approx(810000 / stored)


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
def test_synthetic_field_shape_and_determinism():
    a = make_synthetic_field((6, 8, 5, 5, 3), rank=3, seed=0)
    b = make_synthetic_field((6, 8, 5, 5, 3), rank=3, seed=0)
    assert a.shape == (6, 8, 5, 5, 3)
    assert np.array_equal(a, b)            # reproducible
    c = make_synthetic_field((6, 8, 5, 5, 3), rank=3, seed=1)
    assert not np.array_equal(a, c)        # seed changes output


def test_default_shape():
    field = make_synthetic_field()
    assert field.shape == DEFAULT_SHAPE


def test_add_noise_level():
    clean = make_synthetic_field((8, 10, 6, 6, 3), rank=3, seed=2)
    noisy = add_noise(clean, level=0.1, seed=3)
    # measured relative noise is in the right ballpark of the requested level
    assert 0.05 < relative_error(clean, noisy) < 0.2
    # zero level is a no-op copy
    assert np.array_equal(add_noise(clean, level=0.0), clean)
