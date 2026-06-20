"""Tests for automatic rank selection (energy threshold and target error)."""
import numpy as np
import pytest

from hosvd import HOSVDCompressor, select_ranks
from hosvd.core import _rank_for_energy, _rank_for_error_budget
from hosvd.datasets import make_synthetic_field, add_noise
from hosvd.metrics import relative_error


# --------------------------------------------------------------------------- #
# Low-level rank pickers
# --------------------------------------------------------------------------- #
def test_rank_for_energy_basic():
    spectrum = np.array([10.0, 1.0, 0.1, 0.01])      # energies 100, 1, .01, .0001
    # ~0.99 of the energy is in the first component
    assert _rank_for_energy(spectrum, 0.98) == 1
    assert _rank_for_energy(spectrum, 0.999) == 2
    assert _rank_for_energy(spectrum, 1.0) == 4      # need everything


def test_rank_for_error_budget_basic():
    spectrum = np.array([10.0, 1.0, 0.1, 0.01])
    total = float((spectrum ** 2).sum())
    # budget large enough to drop the last two components
    assert _rank_for_error_budget(spectrum, budget_sq=0.02) == 2
    # zero budget keeps everything
    assert _rank_for_error_budget(spectrum, budget_sq=0.0) == 4
    # budget >= total still keeps at least one
    assert _rank_for_error_budget(spectrum, budget_sq=2 * total) == 1


def test_rank_for_error_budget_zero_keeps_tiny_nonzero_tail():
    # Regression: a tiny-but-nonzero component must NOT be dropped at budget 0
    # (an absolute round-off tolerance used to discard it, breaking losslessness).
    spectrum = np.array([1.0, np.sqrt(0.5e-9)])
    assert _rank_for_error_budget(spectrum, budget_sq=0.0) == 2


def test_rel_error_zero_lossless_under_extreme_dynamic_range():
    # Regression for the same leak at the fit level: one slice dominates the
    # energy by 10 orders of magnitude, so a relative round-off tolerance would
    # have truncated the small modes. rel_error=0 must stay exactly lossless.
    rng = np.random.default_rng(0)
    tensor = rng.standard_normal((6, 6, 6))
    tensor[0] *= 1e5
    model = HOSVDCompressor(rel_error=0.0).fit(tensor)
    assert model.reconstruction_error(tensor) < 1e-12


# --------------------------------------------------------------------------- #
# energy_threshold
# --------------------------------------------------------------------------- #
def test_energy_threshold_recovers_low_rank():
    field = make_synthetic_field((10, 20, 8, 8, 3), rank=3, seed=0)
    model = HOSVDCompressor(energy_threshold=0.9999).fit(field)
    # a clean rank-3 field needs only ~3 components in the truncatable modes
    assert model.effective_ranks[0] <= 4
    assert model.reconstruction_error(field) < 1e-3
    assert model.compression_ratio > 1.0


def test_energy_threshold_one_is_lossless():
    field = make_synthetic_field((6, 10, 5, 5, 3), rank=4, seed=1)
    model = HOSVDCompressor(energy_threshold=1.0).fit(field)
    assert model.reconstruction_error(field) < 1e-10


def test_energy_threshold_monotonic():
    field = add_noise(make_synthetic_field((8, 16, 6, 6, 3), rank=4, seed=2), 0.05, seed=3)
    low = select_ranks(field, energy_threshold=0.90)
    high = select_ranks(field, energy_threshold=0.999)
    assert all(a <= b for a, b in zip(low, high))


def test_energy_threshold_denoises_at_elbow():
    clean = make_synthetic_field((10, 24, 8, 8, 3), rank=4, seed=4)
    noisy = add_noise(clean, level=0.08, seed=5)
    # cutting just below the noise energy finds the signal rank and denoises
    model = HOSVDCompressor(energy_threshold=0.99).fit(noisy)
    assert relative_error(clean, model.reconstruct()) < relative_error(clean, noisy)


# --------------------------------------------------------------------------- #
# rel_error (global error-bounded selection)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [0.02, 0.05, 0.10, 0.20])
def test_rel_error_bound_is_respected(target):
    field = add_noise(make_synthetic_field((10, 30, 8, 8, 3), rank=4, seed=6), 0.05, seed=7)
    model = HOSVDCompressor(rel_error=target).fit(field)
    assert model.reconstruction_error(field) <= target + 1e-9


def test_rel_error_smaller_target_keeps_more():
    field = make_synthetic_field((10, 30, 8, 8, 3), rank=5, seed=8)
    loose = select_ranks(field, rel_error=0.20)
    tight = select_ranks(field, rel_error=0.01)
    assert all(a <= b for a, b in zip(loose, tight))


def test_rel_error_zero_is_lossless():
    field = make_synthetic_field((6, 10, 5, 5, 3), rank=4, seed=9)
    model = HOSVDCompressor(rel_error=0.0).fit(field)
    assert model.reconstruction_error(field) < 1e-10


def test_rel_error_high_target_compresses_hard():
    field = make_synthetic_field((10, 30, 8, 8, 3), rank=3, seed=10)
    model = HOSVDCompressor(rel_error=0.10).fit(field)
    assert model.compression_ratio > 5.0
    assert model.reconstruction_error(field) <= 0.10 + 1e-9


# --------------------------------------------------------------------------- #
# helper + validation
# --------------------------------------------------------------------------- #
def test_select_ranks_matches_model():
    field = make_synthetic_field((8, 12, 6, 6, 3), rank=3, seed=11)
    ranks = select_ranks(field, energy_threshold=0.99)
    model = HOSVDCompressor(energy_threshold=0.99).fit(field)
    assert ranks == model.effective_ranks


def test_mutually_exclusive_selection_raises():
    with pytest.raises(ValueError):
        HOSVDCompressor(ranks=3, energy_threshold=0.99)
    with pytest.raises(ValueError):
        HOSVDCompressor(energy_threshold=0.99, rel_error=0.01)


@pytest.mark.parametrize("bad", [-0.1, 0.0, 1.5])
def test_energy_threshold_range_validation(bad):
    with pytest.raises(ValueError):
        HOSVDCompressor(energy_threshold=bad)


def test_rel_error_negative_validation():
    with pytest.raises(ValueError):
        HOSVDCompressor(rel_error=-0.5)
