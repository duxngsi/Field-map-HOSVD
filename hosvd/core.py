"""
hosvd.core
==========

Memory-efficient Higher-Order Singular Value Decomposition (HOSVD) for
compressing and denoising multi-dimensional field-map data.

This is the open-source implementation behind

    X. Du and L. Groening, "Compression and noise reduction of field maps,"
    Phys. Rev. Accel. Beams 21, 084601 (2018).
    https://doi.org/10.1103/PhysRevAccelBeams.21.084601

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
the routine usable on large field maps where the unfolding has only a few
hundred rows but millions of columns.  A direct-SVD path (``method="svd"``)
is also provided for reference and for better conditioning on near-degenerate
spectra.

Ranks can be fixed explicitly, or chosen automatically from the data via a
per-mode energy threshold (``energy_threshold``) or a global target relative
error (``rel_error``).

中文摘要
--------
本模块用高阶奇异值分解 (HOSVD) 对多维场图数据做压缩与去噪，是论文
PRAB 21, 084601 (2018) 的开源实现。采用顺序截断 ST-HOSVD：逐个维度从
“已部分压缩”的张量上估计因子矩阵；默认通过 mode-n 的 Gram 矩阵
``X_(n) X_(n)^T`` 求主奇异向量，避免对超大展开矩阵直接做 SVD。每个维度保留
的秩可显式给定，也可由能量阈值 ``energy_threshold`` 或目标相对误差
``rel_error`` 自动选取。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "mode_unfold",
    "mode_fold",
    "mode_n_product",
    "hosvd",
    "select_ranks",
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


def _mode_basis(
    tensor: np.ndarray, mode: int, method: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Full left-singular basis and singular spectrum of the mode-``mode`` unfolding.

    Returns ``(basis, singular_values)`` where ``basis`` has shape
    ``(I_mode, K)`` with orthonormal columns ordered by descending singular
    value (``K`` is the number of available singular vectors) and
    ``singular_values`` is the matching 1-D descending spectrum.
    """
    unfolding = mode_unfold(tensor, mode)
    if method == "gram":
        # X_(n) X_(n)^T is small (I_mode x I_mode) and symmetric PSD; its
        # eigenvectors are exactly the left singular vectors of X_(n).
        gram = unfolding @ unfolding.T
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1]
        # Keep at most as many vectors as the unfolding can support (its rank is
        # bounded by the column count) so the "gram" and "svd" paths report
        # consistent spectrum lengths even when the unfolding is wide-and-short.
        order = order[: min(unfolding.shape)]
        singular_values = np.sqrt(np.clip(eigvals[order], 0.0, None))
        basis = np.ascontiguousarray(eigvecs[:, order])
    elif method == "svd":
        basis, singular_values, _ = np.linalg.svd(unfolding, full_matrices=False)
        basis = np.ascontiguousarray(basis)
    else:
        raise ValueError(f"unknown method {method!r}; use 'gram' or 'svd'")
    return basis, singular_values


def _rank_for_energy(spectrum: np.ndarray, threshold: float) -> int:
    """Smallest rank whose retained energy fraction reaches ``threshold``.

    Energy is measured as the sum of squared singular values.
    """
    energy = np.asarray(spectrum, dtype=float) ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return 1
    cumulative = np.cumsum(energy) / total
    rank = int(np.searchsorted(cumulative, threshold, side="left")) + 1
    return max(1, min(rank, spectrum.size))


def _rank_for_error_budget(spectrum: np.ndarray, budget_sq: float) -> int:
    """Smallest rank whose *discarded* energy is within ``budget_sq``.

    ``discarded(R) = sum(spectrum[R:]**2)`` decreases with ``R``; this returns
    the smallest ``R`` with ``discarded(R) <= budget_sq``.  The round-off slack
    is proportional to ``budget_sq`` so that it vanishes when ``budget_sq == 0``
    -- which keeps the ``rel_error=0`` path exactly lossless instead of letting
    an absolute tolerance discard small-but-nonzero components.
    """
    energy = np.asarray(spectrum, dtype=float) ** 2
    n = energy.size
    # Tail sums accumulated from the small end avoid catastrophic cancellation;
    # discarded[n] = 0 (keeping every component discards nothing).
    discarded = np.empty(n + 1, dtype=float)
    discarded[:n] = np.cumsum(energy[::-1])[::-1]       # discarded[R] = sum(energy[R:])
    discarded[n] = 0.0
    tolerance = budget_sq * (1.0 + 1e-9)               # budget-relative round-off slack
    rank = int(np.argmax(discarded <= tolerance))      # first R satisfying the budget
    return max(1, min(rank, n))


# --------------------------------------------------------------------------- #
# High-level compressor
# --------------------------------------------------------------------------- #
class HOSVDCompressor:
    """Truncated HOSVD compressor for dense N-dimensional arrays.

    The kept multilinear rank can be specified in three mutually exclusive
    ways (give at most one; otherwise every mode is kept at full rank):

    * ``ranks`` -- explicit rank per mode (``int`` for all modes, or a sequence).
    * ``energy_threshold`` -- per mode, keep the fewest leading components whose
      cumulative energy (sum of squared singular values) reaches this fraction,
      e.g. ``0.999``.
    * ``rel_error`` -- choose ranks so the overall relative Frobenius
      reconstruction error is guaranteed not to exceed this value, e.g. ``0.01``.

    Parameters
    ----------
    ranks : int, sequence of int, or None
        Explicit target rank per mode.  Values are clipped to each mode's size.
    energy_threshold : float or None
        Per-mode retained-energy fraction in ``(0, 1]``.
    rel_error : float or None
        Target overall relative Frobenius error (``>= 0``).
    method : {"gram", "svd"}
        How factor matrices are computed.  ``"gram"`` (default) eigendecomposes
        the small mode-``n`` Gram matrix -- fast and memory-light for tall-thin
        unfoldings.  ``"svd"`` takes the SVD of the unfolding directly.
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
        kept at full rank and therefore left untouched.
    effective_ranks : list[int]
        The rank actually kept per mode (after explicit/auto selection).
    singular_values : list[ndarray]
        Per-mode singular spectra (useful for choosing ranks / scree plots).
    """

    def __init__(
        self,
        ranks: "Optional[int | Sequence[int]]" = None,
        *,
        energy_threshold: Optional[float] = None,
        rel_error: Optional[float] = None,
        method: str = "gram",
        sequential: bool = True,
    ) -> None:
        self.ranks = ranks
        self.energy_threshold = energy_threshold
        self.rel_error = rel_error
        self.method = method
        self.sequential = sequential
        self._validate_selection()

        self.original_shape: Optional[Tuple[int, ...]] = None
        self.effective_ranks: Optional[List[int]] = None
        self.core: Optional[np.ndarray] = None
        self.factors: Optional[List[Optional[np.ndarray]]] = None
        self.singular_values: Optional[List[np.ndarray]] = None

    # -- internal ----------------------------------------------------------- #
    def _validate_selection(self) -> None:
        chosen = [
            self.ranks is not None,
            self.energy_threshold is not None,
            self.rel_error is not None,
        ]
        if sum(chosen) > 1:
            raise ValueError(
                "specify at most one of ranks, energy_threshold, rel_error"
            )
        if self.energy_threshold is not None and not 0.0 < self.energy_threshold <= 1.0:
            raise ValueError("energy_threshold must be in (0, 1]")
        if self.rel_error is not None and self.rel_error < 0.0:
            raise ValueError("rel_error must be >= 0")

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

    def _rank_for_mode(
        self, mode: int, spectrum: np.ndarray, explicit: Optional[List[int]],
        full_size: int, budget_sq: Optional[float],
    ) -> int:
        if explicit is not None:
            rank = explicit[mode]
        elif self.energy_threshold is not None:
            rank = _rank_for_energy(spectrum, self.energy_threshold)
        elif budget_sq is not None:
            rank = _rank_for_error_budget(spectrum, budget_sq)
        else:
            rank = full_size
        return int(max(1, min(rank, full_size, spectrum.size)))

    # -- public API --------------------------------------------------------- #
    def fit(self, tensor: np.ndarray) -> "HOSVDCompressor":
        """Compress ``tensor`` into the model state and return ``self``."""
        tensor = np.asarray(tensor, dtype=float)
        self.original_shape = tensor.shape
        explicit = self._resolve_ranks(tensor.shape) if self.ranks is not None else None

        # rel_error guarantee: ||X - X_hat||_F^2 <= sum_n (energy discarded at mode n).
        # Spreading the squared-error budget evenly over the modes so that each mode
        # discards at most budget_sq gives total error <= ndim * budget_sq =
        # rel_error^2 ||X||^2, i.e. relative error <= rel_error -- for both the
        # sequential (ST-HOSVD) and the classic HOSVD paths.
        budget_sq: Optional[float] = None
        if self.rel_error is not None:
            total_energy = float(np.vdot(tensor.ravel(), tensor.ravel()).real)
            budget_sq = self.rel_error ** 2 * total_energy / tensor.ndim

        factors: List[Optional[np.ndarray]] = [None] * tensor.ndim
        singular_values: List[Optional[np.ndarray]] = [None] * tensor.ndim
        effective_ranks: List[int] = [0] * tensor.ndim
        core = tensor

        for mode in range(tensor.ndim):
            source = core if self.sequential else tensor
            basis, spectrum = _mode_basis(source, mode, self.method)
            singular_values[mode] = spectrum
            rank = self._rank_for_mode(
                mode, spectrum, explicit, tensor.shape[mode], budget_sq
            )
            effective_ranks[mode] = rank
            if rank >= tensor.shape[mode]:
                # Mode kept at full rank: leave it untouched (lossless).
                continue
            factor = np.ascontiguousarray(basis[:, :rank])
            factors[mode] = factor
            core = mode_n_product(core, factor.T, mode)

        self.effective_ranks = effective_ranks
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
    energy_threshold: Optional[float] = None,
    rel_error: Optional[float] = None,
    method: str = "gram",
    sequential: bool = True,
) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    """Functional one-shot HOSVD.

    Returns ``(core, factors)`` -- see :class:`HOSVDCompressor` for the
    meaning of the arguments and the structure of ``factors``.
    """
    model = HOSVDCompressor(
        ranks,
        energy_threshold=energy_threshold,
        rel_error=rel_error,
        method=method,
        sequential=sequential,
    ).fit(tensor)
    return model.core, model.factors


def select_ranks(
    tensor: np.ndarray,
    *,
    energy_threshold: Optional[float] = None,
    rel_error: Optional[float] = None,
    method: str = "gram",
    sequential: bool = True,
) -> List[int]:
    """Return the per-mode ranks an adaptive criterion would choose.

    Convenience wrapper that fits a compressor and reports the ranks it picked,
    so you can inspect / reuse them without holding on to the compression.
    """
    model = HOSVDCompressor(
        energy_threshold=energy_threshold,
        rel_error=rel_error,
        method=method,
        sequential=sequential,
    ).fit(tensor)
    return list(model.effective_ranks)


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
