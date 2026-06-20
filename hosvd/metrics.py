"""
hosvd.metrics
=============

Small quality / size metrics for evaluating a HOSVD compression.

误差与压缩率度量：相对 Frobenius 误差、信噪比、压缩率等。
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "frobenius_norm",
    "relative_error",
    "rmse",
    "snr_db",
    "psnr_db",
    "compression_ratio",
]


def frobenius_norm(tensor: np.ndarray) -> float:
    """Frobenius norm of an arbitrary-dimensional array."""
    return float(np.linalg.norm(np.asarray(tensor, dtype=float).ravel()))


def relative_error(original: np.ndarray, approx: np.ndarray) -> float:
    """Relative Frobenius error ``||original - approx|| / ||original||``."""
    original = np.asarray(original, dtype=float)
    approx = np.asarray(approx, dtype=float)
    denom = frobenius_norm(original)
    if denom == 0.0:
        return 0.0
    return frobenius_norm(original - approx) / denom


def rmse(original: np.ndarray, approx: np.ndarray) -> float:
    """Root-mean-square error between two equally shaped arrays."""
    original = np.asarray(original, dtype=float)
    approx = np.asarray(approx, dtype=float)
    return float(np.sqrt(np.mean((original - approx) ** 2)))


def snr_db(original: np.ndarray, approx: np.ndarray) -> float:
    """Signal-to-noise ratio in decibels, treating ``original - approx`` as noise."""
    original = np.asarray(original, dtype=float)
    approx = np.asarray(approx, dtype=float)
    noise = frobenius_norm(original - approx)
    if noise == 0.0:
        return float("inf")
    return float(20.0 * np.log10(frobenius_norm(original) / noise))


def psnr_db(original: np.ndarray, approx: np.ndarray, peak: Optional[float] = None) -> float:
    """Peak signal-to-noise ratio in decibels.

    ``peak`` defaults to the dynamic range ``max - min`` of ``original``.
    """
    original = np.asarray(original, dtype=float)
    approx = np.asarray(approx, dtype=float)
    if peak is None:
        peak = float(original.max() - original.min())
    mse = float(np.mean((original - approx) ** 2))
    if mse == 0.0:
        return float("inf")
    if peak == 0.0:
        return 0.0
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(mse))


def compression_ratio(
    original_shape: Sequence[int],
    core_shape: Sequence[int],
    factor_shapes: Iterable[Optional[Sequence[int]]] = (),
) -> float:
    """Compression ratio from raw shapes (original elements / stored elements)."""
    stored = int(np.prod(core_shape))
    for shape in factor_shapes:
        if shape is not None:
            stored += int(np.prod(shape))
    if stored == 0:
        return float("inf")
    return int(np.prod(original_shape)) / stored
