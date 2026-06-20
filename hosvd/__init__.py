"""
Field-map HOSVD
===============

Memory-efficient Higher-Order Singular Value Decomposition (HOSVD) for
compressing and denoising multi-dimensional field-map data.

Quickstart
----------
>>> import numpy as np
>>> from hosvd import HOSVDCompressor, make_synthetic_field
>>> field = make_synthetic_field()                 # smooth 5-D vector field
>>> model = HOSVDCompressor(ranks=(3, 8, 4, 4, 3)).fit(field)
>>> recovered = model.reconstruct()
>>> round(model.compression_ratio, 1) > 1
True

See :class:`hosvd.core.HOSVDCompressor` for the full API.
"""
from __future__ import annotations

from .core import (
    DataSVDCompress,
    HOSVDCompressor,
    hosvd,
    mode_fold,
    mode_n_product,
    mode_unfold,
)
from .datasets import DEFAULT_SHAPE, add_noise, make_synthetic_field
from .metrics import (
    compression_ratio,
    frobenius_norm,
    psnr_db,
    relative_error,
    rmse,
    snr_db,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # core
    "HOSVDCompressor",
    "hosvd",
    "DataSVDCompress",
    "mode_unfold",
    "mode_fold",
    "mode_n_product",
    # datasets
    "make_synthetic_field",
    "add_noise",
    "DEFAULT_SHAPE",
    # metrics
    "relative_error",
    "rmse",
    "snr_db",
    "psnr_db",
    "compression_ratio",
    "frobenius_norm",
]
