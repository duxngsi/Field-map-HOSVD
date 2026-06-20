"""
hosvd.datasets
==============

Reproducible synthetic field-map data so the demo and the tests can run
without the original (unpublished) ``Field_5D.npy`` measurement file.

``make_synthetic_field`` builds a smooth, low-multilinear-rank 5-D vector
field on a structured grid -- exactly the kind of data HOSVD compresses
well -- and ``add_noise`` lets you contaminate it so the denoising behaviour
can be demonstrated.

合成数据：生成平滑、低多重秩的 5 维矢量场，用于在缺少原始 ``Field_5D.npy``
的情况下运行示例与测试；``add_noise`` 用于加噪以演示去噪。

Layout convention (mirrors the original project)::

    axis 0 : x grid
    axis 1 : z grid (longitudinal, the "many steps" axis)
    axis 2 : y grid
    axis 3 : parameter / time sample
    axis 4 : field component (Ex, Ey, Ez)
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

__all__ = ["make_synthetic_field", "add_noise", "DEFAULT_SHAPE"]

DEFAULT_SHAPE: Tuple[int, ...] = (20, 60, 15, 15, 3)


def make_synthetic_field(
    shape: Sequence[int] = DEFAULT_SHAPE,
    rank: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Return a smooth, separable, low-rank field of the requested ``shape``.

    The field is a sum of ``rank`` separable terms, each a product of smooth
    low-frequency 1-D profiles along every axis.  This guarantees a small
    multilinear rank, so truncated HOSVD reconstructs it to high accuracy and
    achieves a large compression ratio.

    Parameters
    ----------
    shape : sequence of int
        Output tensor shape.  Any number of dimensions is supported.
    rank : int
        Number of separable terms summed together (controls effective rank).
    seed : int
        Seed for the (frequencies / phases / amplitudes) random generator.
    """
    shape = tuple(int(s) for s in shape)
    rng = np.random.default_rng(seed)
    grids = [np.linspace(0.0, 1.0, n) for n in shape]

    field = np.zeros(shape, dtype=float)
    for k in range(rank):
        amplitude = float(rng.uniform(0.5, 1.5)) / (k + 1)
        term = np.full(shape, amplitude, dtype=float)
        for axis, n in enumerate(shape):
            freq = float(rng.integers(1, 3)) + k * 0.5
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            profile = np.cos(2.0 * np.pi * freq * grids[axis] + phase)
            # broadcast the 1-D profile along ``axis``
            broadcast_shape = [1] * len(shape)
            broadcast_shape[axis] = n
            term = term * profile.reshape(broadcast_shape)
        field += term

    return field


def add_noise(
    field: np.ndarray,
    level: float = 0.05,
    seed: int = 1,
) -> np.ndarray:
    """Add zero-mean Gaussian noise scaled to ``level`` times the field std.

    ``level=0.05`` corresponds to roughly 5% relative noise.
    """
    field = np.asarray(field, dtype=float)
    if level <= 0.0:
        return field.copy()
    rng = np.random.default_rng(seed)
    sigma = level * float(field.std())
    return field + rng.normal(scale=sigma, size=field.shape)
