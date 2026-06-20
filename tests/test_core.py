"""Tests for hosvd.core: tensor algebra, compression and reconstruction."""
import numpy as np
import pytest

from hosvd import HOSVDCompressor, hosvd, DataSVDCompress
from hosvd.core import mode_unfold, mode_fold, mode_n_product
from hosvd.datasets import make_synthetic_field, add_noise
from hosvd.metrics import relative_error


# --------------------------------------------------------------------------- #
# Low-level tensor algebra
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_unfold_fold_roundtrip(mode):
    rng = np.random.default_rng(0)
    tensor = rng.normal(size=(3, 4, 5, 2))
    unfolded = mode_unfold(tensor, mode)
    assert unfolded.shape[0] == tensor.shape[mode]
    folded = mode_fold(unfolded, mode, tensor.shape)
    assert np.allclose(folded, tensor)


def test_mode_n_product_shapes():
    rng = np.random.default_rng(1)
    tensor = rng.normal(size=(3, 4, 5))
    matrix = rng.normal(size=(7, 4))  # (J, I_mode)
    out = mode_n_product(tensor, matrix, mode=1)
    assert out.shape == (3, 7, 5)


def test_mode_n_product_matches_unfolding():
    rng = np.random.default_rng(2)
    tensor = rng.normal(size=(3, 4, 5))
    matrix = rng.normal(size=(6, 4))
    out = mode_n_product(tensor, matrix, mode=1)
    expected = mode_fold(matrix @ mode_unfold(tensor, 1), 1, out.shape)
    assert np.allclose(out, expected)


# --------------------------------------------------------------------------- #
# Compression correctness
# --------------------------------------------------------------------------- #
def test_full_rank_reconstruction_is_exact():
    field = make_synthetic_field((6, 10, 5, 5, 3), rank=4, seed=0)
    model = HOSVDCompressor().fit(field)          # ranks=None -> full rank
    assert model.reconstruction_error(field) < 1e-10
    assert model.compression_ratio == pytest.approx(1.0)


def test_low_rank_field_compresses_losslessly():
    # a rank-3 field is captured exactly when every kept rank >= 3
    field = make_synthetic_field((8, 12, 6, 6, 3), rank=3, seed=1)
    model = HOSVDCompressor(ranks=(3, 8, 4, 4, 3)).fit(field)
    assert model.reconstruction_error(field) < 1e-8
    assert model.compression_ratio > 5.0


def test_error_decreases_monotonically_with_rank():
    field = make_synthetic_field((8, 12, 6, 6, 3), rank=5, seed=2)
    errors = [HOSVDCompressor(ranks=r).fit(field).reconstruction_error(field)
              for r in (1, 2, 3, 4, 5)]
    for a, b in zip(errors, errors[1:]):
        assert b <= a + 1e-12


@pytest.mark.parametrize("method", ["gram", "svd"])
def test_methods_agree(method):
    field = make_synthetic_field((6, 10, 5, 5, 3), rank=4, seed=3)
    ref = HOSVDCompressor(ranks=(4, 4, 4, 4, 3), method="gram").fit(field).reconstruct()
    out = HOSVDCompressor(ranks=(4, 4, 4, 4, 3), method=method).fit(field).reconstruct()
    assert relative_error(ref, out) < 1e-10


def test_gram_and_svd_report_consistent_spectrum_lengths():
    # mode-0 unfolding is (12, 9): wide-and-short, where the gram path used to
    # return 12 singular values vs the svd path's 9. They should now match.
    field = make_synthetic_field((12, 3, 3), rank=2, seed=31)
    gram = HOSVDCompressor(method="gram").fit(field)
    svd = HOSVDCompressor(method="svd").fit(field)
    for sg, ss in zip(gram.singular_values, svd.singular_values):
        assert sg.shape == ss.shape


def test_sequential_and_classic_close():
    field = make_synthetic_field((6, 10, 5, 5, 3), rank=4, seed=4)
    seq = HOSVDCompressor(ranks=(4, 5, 4, 4, 3), sequential=True).fit(field).reconstruct()
    cls = HOSVDCompressor(ranks=(4, 5, 4, 4, 3), sequential=False).fit(field).reconstruct()
    # both are valid HOSVD truncations; for a near-low-rank field they agree well
    assert relative_error(field, seq) < 1e-6
    assert relative_error(field, cls) < 1e-6


def test_denoising_recovers_clean_signal():
    clean = make_synthetic_field((10, 20, 8, 8, 3), rank=3, seed=5)
    noisy = add_noise(clean, level=0.10, seed=6)
    recovered = HOSVDCompressor(ranks=(3, 6, 3, 3, 3)).fit(noisy).reconstruct()
    assert relative_error(clean, recovered) < relative_error(clean, noisy)
    assert relative_error(clean, recovered) < 0.03   # << 10% input noise


# --------------------------------------------------------------------------- #
# Factor properties and ranks
# --------------------------------------------------------------------------- #
def test_factors_have_orthonormal_columns():
    field = make_synthetic_field((6, 10, 5, 5, 3), rank=4, seed=7)
    model = HOSVDCompressor(ranks=(3, 4, 4, 4, 3)).fit(field)
    for factor in model.factors:
        if factor is not None:
            gram = factor.T @ factor
            assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-10)


def test_full_rank_mode_keeps_no_factor():
    field = make_synthetic_field((4, 6, 5, 5, 3), rank=3, seed=8)
    model = HOSVDCompressor(ranks=(2, 6, 4, 4, 3)).fit(field)
    # mode 1 (size 6, rank 6) and mode 4 (size 3, rank 3) are kept full -> None
    assert model.factors[1] is None
    assert model.factors[4] is None
    assert model.factors[0] is not None


def test_ranks_are_clipped_to_shape():
    field = make_synthetic_field((4, 6, 5, 5, 3), rank=3, seed=9)
    model = HOSVDCompressor(ranks=(99, 99, 99, 99, 99)).fit(field)
    assert model.effective_ranks == [4, 6, 5, 5, 3]
    assert model.reconstruction_error(field) < 1e-10


def test_scalar_rank_applies_to_all_modes():
    field = make_synthetic_field((6, 8, 5, 5, 3), rank=2, seed=10)
    model = HOSVDCompressor(ranks=2).fit(field)
    assert model.effective_ranks == [2, 2, 2, 2, 2]


def test_wrong_rank_length_raises():
    field = make_synthetic_field((4, 6, 5, 5, 3), rank=2, seed=11)
    with pytest.raises(ValueError):
        HOSVDCompressor(ranks=(2, 2, 2)).fit(field)


def test_unknown_method_raises():
    field = make_synthetic_field((4, 6, 5, 5, 3), rank=2, seed=12)
    with pytest.raises(ValueError):
        HOSVDCompressor(ranks=2, method="bogus").fit(field)


# --------------------------------------------------------------------------- #
# Functional API and backward compatibility
# --------------------------------------------------------------------------- #
def test_functional_hosvd():
    field = make_synthetic_field((6, 8, 5, 5, 3), rank=3, seed=13)
    core, factors = hosvd(field, (3, 4, 4, 4, 3))
    assert core.shape == (3, 4, 4, 4, 3)
    assert len(factors) == field.ndim


def test_backward_compatible_class():
    field = make_synthetic_field((8, 12, 6, 6, 3), rank=3, seed=14)
    d = DataSVDCompress(field, (3, 8, 4, 4, 3))
    d.compress_ndm()
    d.recover()
    assert relative_error(field, d.recovered_data) < 1e-8
    # historical attributes exist and v holds row-vector bases (rank x I)
    assert np.shape(d.v[0]) == (3, 8)
    assert d.s[0].ndim == 1


def test_works_on_2d_matrix():
    rng = np.random.default_rng(15)
    mat = rng.normal(size=(20, 12))
    model = HOSVDCompressor(ranks=(5, 5)).fit(mat)
    assert model.core.shape == (5, 5)
    # reconstruction is a valid rank-5 approximation
    assert model.reconstruction_error(mat) < 1.0


def test_reconstruct_before_fit_raises():
    with pytest.raises(RuntimeError):
        HOSVDCompressor().reconstruct()
