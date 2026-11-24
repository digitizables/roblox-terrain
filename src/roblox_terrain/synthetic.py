"""A procedural valley used when the caller asks for ``synthetic-valley``.

The shape is a valley, two hills, a pit lake, and a terraced shelf. It is a
stand-in so the hydrology and field passes can be tested without a download.
It is not a reconstruction of the coordinates stored on the preset.
"""

from __future__ import annotations

import numpy as np

from roblox_terrain.cover import BARE, CROPLAND, FOREST, ORCHARD, PASTURE, URBAN, WETLAND


def synthetic_landscape(rows: int, cols: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    yn = yy / max(rows - 1, 1)
    xn = xx / max(cols - 1, 1)
    # A channel that wanders from the north edge toward the south-east.
    center = 0.32 + 0.28 * yn + 0.06 * np.sin(yn * 9.0)
    valley = np.exp(-((xn - center) ** 2) / 0.012)
    west_hill = np.exp(-(((xn - 0.14) ** 2) / 0.02 + ((yn - 0.72) ** 2) / 0.03)) * 36.0
    east_hill = np.exp(-(((xn - 0.82) ** 2) / 0.018 + ((yn - 0.22) ** 2) / 0.02)) * 28.0
    shelf = (1.0 - xn) * 16.0 + yn * 5.0
    heights = 140.0 + shelf + west_hill + east_hill - valley * 16.0
    # Terraces on the gentle eastern ground, the way old fields step a slope.
    terraces = xn > 0.58
    heights = np.where(terraces, np.floor(heights / 2.2) * 2.2, heights)
    # A closed pit that should become a lake rather than a stream.
    lake_r = slice(max(0, rows // 5 - 2), rows // 5 + 3)
    lake_c = slice(max(0, cols // 5), cols // 5 + 5)
    heights[lake_r, lake_c] -= 5.5
    # A little roughness that must not become the landform.
    heights += (rng.random((rows, cols)) - 0.5) * 0.35
    cover = np.full((rows, cols), PASTURE, dtype=np.uint8)
    cover[xn > 0.48] = CROPLAND
    cover[(xn < 0.28) & (yn > 0.35)] = FOREST
    cover[(xn > 0.62) & (yn < 0.28) & (xn < 0.78)] = ORCHARD
    cover[(xn > 0.88) & (yn > 0.55)] = URBAN
    cover[(xn > 0.7) & (yn > 0.8)] = WETLAND
    cover[(xn < 0.12) & (yn < 0.2)] = BARE
    return heights.astype(np.float32), cover
