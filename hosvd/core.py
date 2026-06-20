"""
hosvd.core
==========

Memory-efficient Higher-Order Singular Value Decomposition (HOSVD) for
compressing and denoising multi-dimensional field-map data.

The implementation uses the *sequentially truncated* HOSVD (ST-HOSVD)
scheme (Vannieuwenhoven, Vandebril & Meerbergen, 2012): the factor matrices
are estimated one mode at a time from the *partially* truncated tensor,
which is both faster and at least as accurate as the classic
De Lathauwer HOSVD for the same target ranks.

For each mode ``n`` the factor matrix ``U^(n)`` holds the leading left
singular vectors of the mode-``n`` unfolding ``X_(n)``.  Rather than taking
the SVD of the (potentially huge) unfolding directly, the leading vectors
can be obtained from the small ``I_n x I_n`` Gram matrix
``X_(n) X_(n)^T`` (``method="gram"``, the default) -- this is what makes
the routine usable on large 5-D field maps where the unfolding has only a
few hundred rows but millions of columns.  A direct-SVD path
(``method="svd"``) is also provided for reference and for better
conditioning on near-degenerate spectra.

中文摘要
--------
本模块用高阶奇异值分解 (HOSVD) 对多维场图数据做压缩与去噪。采用顺序截断
ST-HOSVD：逐个维度从“已部分压缩”的张量上估计因子矩阵。默认通过 mode-n
的 Gram 矩阵 ``X_(n) X_(n)^T`` 求主奇异向量，避免对超大展开矩阵直接做 SVD，
因而能处理大型 5 维场图。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "mode_unfold",
    "mode_fold",
    "mode_n_product",
    "hosvd",
    "HOSVDCompressor",
    "DataSVDCompress",
]


# --------------------------------------------------------------------------- #
# Low-level tensor algebra
# --------------------------------------------------------------------------- #
def mode_unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """Return the mode-``mode`` unfolding (matricization) of ``tensor``.

    The result has shape ``(I_mode, prod(other dims))`` with the ``mode``
    axis moved to the front (Kolda & Bader convention; column ordering
    follows NumPy C order, which :func:`mode_fold` inverts exactly).
    """
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def mode_fold(matrix: np.ndarray, mode: int, shape: Sequence[int]) -> np.ndarray:
    """Inverse of :func:`mode_unfold`: fold ``matrix`` back into ``shape``."""
    shape = tuple(shape)
    moved = (shape[mode],) + tuple(s for i, s in enumerate(shape) if i != mode)
    return np.moveaxis(matrix.reshape(moved), 0, mode)


def mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """Mode-``mode`` product ``tensor x_mode matrix``.

    ``matrix`` has shape ``(J, I_mode)`` and contracts the ``mode`` axis of
    ``tensor`` (size ``I_mode``), producing a new size ``J`` along that mode.
    """
    return np.moveaxis(np.tensordot(matrix, tensor, axes=(1, mode)), 0, mode)


def _mode_factor(
    tensor: np.ndarray, mode: int, rank: int, method: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Leading ``rank`` left singular vectors of the mode-``mode`` unfolding.

    Returns ``(factor, singular_values)`` where ``factor`` has shape
    ``(I_mode, rank)`` with orthonormal columns and ``singular_values`` is the
    full (descending) mode-``mode`` singular spectrum.
    """
    unfolding = mode_unfold(tensor, mode)
    dim = unfolding.shape[0]
    rank = int(min(rank, dim))

    if method == "gram":
        # X_(n) X_(n)^T is small (I_mode x I_mode) and symmetric PSD; its
        # eigenvectors are exactly the left singular vectors of X_(n).
        gram = unfolding @ unfolding.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        singular_values = np.sqrt(np.clip(eigvals, 0.0, None))
        factor = np.ascontiguousarray(eigvecs[:, :rank])
    elif method == "svd":
        left, singular_values, _ = np.linalg.svd(unfolding, full_matrices=False)
        factor = np.ascontiguousarray(left[:, :rank])
    else:
        raise ValueError(f"unknown method {method!r}; use 'gram' or 'svd'")

    return factor, singular_values


# --------------------------------------------------------------------------- #
# High-level compressor
# --------------------------------------------------------------------------- #
class HOSVDCompressor:
    """Truncated HOSVD compressor for dense N-dimensional arrays.

    Parameters
    ----------
    ranks : int, sequence of int, or None
        Target multilinear rank kept per mode.  ``None`` (default) keeps the
        full rank of every mode (lossless apart from round-off).  An int
        applies the same rank to every mode.  A sequence gives one rank per
        mode; values are clipped to each mode's size.
    method : {"gram", "svd"}
        How factor matrices are computed.  ``"gram"`` (default) eigendecomposes
        the small mode-``n`` Gram matrix -- fast and memory-light for tall-thin
        unfoldings.  ``"svd"`` takes the SVD of the unfolding directly and is
        more robust for near-degenerate spectra.
    sequential : bool
        If ``True`` (default) use sequentially truncated HOSVD: each factor is
        estimated from the already partially compressed tensor.  If ``False``
        every factor is estimated from the original tensor (classic HOSVD).

    Attributes
    ----------
    core : ndarray
        The compressed core tensor after :meth:`fit`.
    factors : list[ndarray | None]
        Per-mode factor matrices (``I_mode x rank``).  ``None`` marks a mode
        that was kept at full rank and therefore left untouched.
    singular_values : list[ndarray]
        Per-mode singular spectra (useful for choosing ranks / scree plots).
    """

    def __init__(
        self,
        ranks: "Optional[int | Sequence[int]]" = None,
        *,
        method: str = "gram",
        sequential: bool = True,
    ) -> None:
        self.ranks = ranks
        self.method = method
        self.sequential = sequential

        self.original_shape: Optional[Tuple[int, ...]] = None
        self.effective_ranks: Optional[List[int]] = None
        self.core: Optional[np.ndarray] = None
        self.factors: Optional[List[Optional[np.ndarray]]] = None
        self.singular_values: Optional[List[np.ndarray]] = None

    # -- internal ----------------------------------------------------------- #
    def _resolve_ranks(self, shape: Sequence[int]) -> List[int]:
        if self.ranks is None:
            return list(shape)
        if np.isscalar(self.ranks):
            return [min(int(self.ranks), s) for s in shape]
        ranks = list(self.ranks)
        if len(ranks) != len(shape):
            raise ValueError(
                f"ranks has length {len(ranks)} but tensor has {len(shape)} modes"
            )
        return [min(int(r), s) for r, s in zip(ranks, shape)]

    # -- public API --------------------------------------------------------- #
    def fit(self, tensor: np.ndarray) -> "HOSVDCompressor":
        """Compress ``tensor`` in place of the model state and return ``self``."""
        tensor = np.asarray(tensor, dtype=float)
        self.original_shape = tensor.shape
        ranks = self._resolve_ranks(tensor.shape)
        self.effective_ranks = ranks

        factors: List[Optional[np.ndarray]] = [None] * tensor.ndim
        singular_values: List[Optional[np.ndarray]] = [None] * tensor.ndim
        core = tensor

        for mode in range(tensor.ndim):
            source = core if self.sequential else tensor
            factor, spectrum = _mode_factor(source, mode, ranks[mode], self.method)
            singular_values[mode] = spectrum
            if ranks[mode] >= tensor.shape[mode]:
                # Mode kept at full rank: leave it untouched (lossless).
                continue
            factors[mode] = factor
            core = mode_n_product(core, factor.T, mode)

        self.factors = factors
        self.singular_values = singular_values
        self.core = core
        return self

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the (approximate) original tensor from core and factors."""
        if self.core is None or self.factors is None:
            raise RuntimeError("call fit() before reconstruct()")
        tensor = self.core
        for mode, factor in enumerate(self.factors):
            if factor is not None:
                tensor = mode_n_product(tensor, factor, mode)
        return tensor

    def fit_reconstruct(self, tensor: np.ndarray) -> np.ndarray:
        """Convenience: :meth:`fit` then :meth:`reconstruct`."""
        return self.fit(tensor).reconstruct()

    # -- diagnostics -------------------------------------------------------- #
    @property
    def n_stored_values(self) -> int:
        """Number of scalars needed to store the compressed representation."""
        if self.core is None:
            raise RuntimeError("call fit() first")
        total = int(np.prod(self.core.shape))
        for factor in self.factors or []:
            if factor is not None:
                total += int(factor.size)
        return total

    @property
    def compression_ratio(self) -> float:
        """Original element count divided by stored element count (>1 == smaller)."""
        if self.original_shape is None:
            raise RuntimeError("call fit() first")
        return int(np.prod(self.original_shape)) / self.n_stored_values

    def reconstruction_error(self, original: np.ndarray) -> float:
        """Relative Frobenius reconstruction error ``||X - X_hat|| / ||X||``."""
        original = np.asarray(original, dtype=float)
        approx = self.reconstruct()
        denom = float(np.linalg.norm(original))
        if denom == 0.0:
            return 0.0
        return float(np.linalg.norm(original - approx) / denom)


def hosvd(
    tensor: np.ndarray,
    ranks: "Optional[int | Sequence[int]]" = None,
    *,
    method: str = "gram",
    sequential: bool = True,
) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    """Functional one-shot HOSVD.

    Returns ``(core, factors)`` -- see :class:`HOSVDCompressor` for the
    meaning of the arguments and the structure of ``factors``.
    """
    model = HOSVDCompressor(ranks, method=method, sequential=sequential).fit(tensor)
    return model.core, model.factors


# --------------------------------------------------------------------------- #
# Backward-compatible wrapper (pre-2018 API)
# --------------------------------------------------------------------------- #
class DataSVDCompress:
    """Drop-in replacement for the original 2018 class.

    Preserves the historical attributes/methods so legacy scripts keep
    working while delegating to :class:`HOSVDCompressor`::

        d = DataSVDCompress(field, keeps=(3, 8, 4, 4, 3))
        d.compress_ndm()
        d.recover()
        d.recovered_data, d.compressed_data, d.v, d.s

    New code should prefer :class:`HOSVDCompressor` directly.
    """

    def __init__(self, data: np.ndarray, keeps: Sequence[int]) -> None:
        self.original_data = np.asarray(data, dtype=float)
        self.original_shape = self.original_data.shape
        self.keeps = list(keeps)
        self._model = HOSVDCompressor(self.keeps)
        self.compressed_data = self.original_data
        self.recovered_data = self.original_data
        # ``v`` historically held row-vector bases (rank x I); 1 == untouched.
        self.v: List = [1] * self.original_data.ndim
        self.s: List = [None] * self.original_data.ndim

    def _sync_from_model(self) -> None:
        self.compressed_data = self._model.core
        self.s = list(self._model.singular_values)
        self.v = [
            1 if factor is None else factor.T for factor in self._model.factors
        ]

    def compress_ndm(self) -> "DataSVDCompress":
        """Compress to the requested ``keeps`` ranks."""
        self._model = HOSVDCompressor(self.keeps)
        self._model.fit(self.original_data)
        self._sync_from_model()
        return self

    def HOSVD(self) -> "DataSVDCompress":
        """Full (lossless) HOSVD -- keep every mode at full rank."""
        self._model = HOSVDCompressor(list(self.original_shape))
        self._model.fit(self.original_data)
        self._sync_from_model()
        return self

    def recover(self) -> np.ndarray:
        """Reconstruct ``recovered_data`` from the compressed representation."""
        self.recovered_data = self._model.reconstruct()
        return self.recovered_data
