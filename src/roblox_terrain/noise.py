"""Small value-noise helpers for micro-detail.

The macro shape of a real scene comes from the DEM. These functions only add
furrows and a few decimeters of high-frequency relief.
"""

from __future__ import annotations

import numpy as np


def value_noise(rows: int, cols: int, period: float, rng: np.random.Generator) -> np.ndarray:
    """Bilinear value noise in 0..1. ``period`` is the cell size of the coarse grid."""
    period = max(2.0, float(period))
    gh = max(2, int(round(rows / period)) + 2)
    gw = max(2, int(round(cols / period)) + 2)
    coarse = rng.random((gh, gw))
    yy, xx = np.meshgrid(
        np.linspace(0, gh - 1, rows),
        np.linspace(0, gw - 1, cols),
        indexing="ij",
    )
    y0 = np.floor(yy).astype(np.int32)
    x0 = np.floor(xx).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    fy = yy - y0
    fx = xx - x0
    v00 = coarse[y0, x0]
    v01 = coarse[y0, x1]
    v10 = coarse[y1, x0]
    v11 = coarse[y1, x1]
    return (v00 * (1 - fy) * (1 - fx) + v01 * (1 - fy) * fx + v10 * fy * (1 - fx) + v11 * fy * fx).astype(
        np.float64
    )


def fbm(rows: int, cols: int, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """Fractional value noise in about 0..1. Not a terrain by itself."""
    total = np.zeros((rows, cols), dtype=np.float64)
    weight = 0.0
    amplitude = 1.0
    period = max(8.0, min(rows, cols) / 6)
    for _ in range(octaves):
        total += amplitude * value_noise(rows, cols, period, rng)
        weight += amplitude
        amplitude *= 0.5
        period = max(2.0, period * 0.5)
    return total / weight


def blur(values: np.ndarray) -> np.ndarray:
    """One binomial 3x3 smooth. Edges are repeated."""
    pad = np.pad(values, 1, mode="edge")
    return (
        pad[:-2, :-2]
        + 2 * pad[:-2, 1:-1]
        + pad[:-2, 2:]
        + 2 * pad[1:-1, :-2]
        + 4 * pad[1:-1, 1:-1]
        + 2 * pad[1:-1, 2:]
        + pad[2:, :-2]
        + 2 * pad[2:, 1:-1]
        + pad[2:, 2:]
    ) / 16.0
