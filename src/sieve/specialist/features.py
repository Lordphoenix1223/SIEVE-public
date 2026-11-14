"""Frozen per-curve features; no population fit and no labels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FEATURE_VERSION = "kepler-rrlyrae-fft-v1"
SERIES_LENGTH = 4767
DOMAIN = "KeplerLightCurves-11218015/interpolated-equal-length-v1"


def features(curves: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize each curve independently; summarize shape and the power spectrum.

    Frequencies are cycles/sample, not physical periods. No timestamps are invented.
    Absolute scale is deliberately excluded: the archive preprocessing is not a
    transferable photometric calibration. Callers enforce the declared domain.
    """
    x = np.asarray(curves, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != SERIES_LENGTH or not np.isfinite(x).all():
        raise ValueError(f"expected finite curves of length {SERIES_LENGTH}")
    scale = x.std(axis=1)
    if np.any(scale <= np.maximum(np.abs(x).max(axis=1), 1.0) * 1e-12):
        raise ValueError("constant or numerically degenerate curve")
    z = (x - x.mean(axis=1, keepdims=True)) / scale[:, None]
    time = np.linspace(-1, 1, SERIES_LENGTH)
    slopes = z @ time / (time @ time)
    detrended = z - slopes[:, None] * time
    power = np.abs(np.fft.rfft(detrended, axis=1)) ** 2
    power[:, 0] = 0
    power /= power.sum(axis=1, keepdims=True)
    top = np.argsort(-power, axis=1, kind="stable")[:, :8]
    top_power = np.take_along_axis(power, top, axis=1)
    q = np.quantile(z, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99], axis=1).T
    differences = np.diff(z, axis=1)
    shape = np.column_stack(
        [
            slopes,
            (z**3).mean(axis=1),
            (z**4).mean(axis=1),
            np.abs(differences).mean(axis=1),
            (differences**2).mean(axis=1),
            (z > 0).mean(axis=1),
            (differences > 0).mean(axis=1),
            np.sum(np.diff(np.signbit(z), axis=1), axis=1) / SERIES_LENGTH,
        ]
    )
    lags = np.column_stack(
        [(z[:, :-lag] * z[:, lag:]).mean(axis=1) for lag in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
    )
    edges = np.unique(np.geomspace(1, power.shape[1], 33).astype(int))
    bands = np.column_stack(
        [power[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    )
    entropy = -(power * np.log(np.maximum(power, 1e-300))).sum(axis=1)
    output = np.column_stack(
        [q, shape, lags, top / SERIES_LENGTH, top_power, np.log1p(1000 * bands), entropy]
    )
    if not np.isfinite(output).all():
        raise ValueError("nonfinite derived features")
    return output
