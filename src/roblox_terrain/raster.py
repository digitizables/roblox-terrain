"""Rasterize rings and polylines onto the north-up grid. No GDAL required."""

from __future__ import annotations

import numpy as np

from roblox_terrain.grid import GeoGrid


def rasterize_ring(grid: GeoGrid, ring: np.ndarray) -> np.ndarray:
    """Even-odd fill of one lon/lat ring. ``ring`` is (N, 2) as lon, lat."""
    mask = np.zeros((grid.rows, grid.cols), dtype=bool)
    if len(ring) < 3:
        return mask
    lats = np.asarray(grid.row_lat(np.arange(grid.rows)), dtype=np.float64)
    lons = np.asarray(grid.col_lon(np.arange(grid.cols)), dtype=np.float64)
    coords = np.asarray(ring, dtype=np.float64)
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    xs = coords[:, 0]
    ys = coords[:, 1]
    for row, lat in enumerate(lats):
        hits: list[float] = []
        for index in range(len(coords) - 1):
            y0 = ys[index]
            y1 = ys[index + 1]
            if (y0 <= lat < y1) or (y1 <= lat < y0):
                x0 = xs[index]
                x1 = xs[index + 1]
                denom = y1 - y0
                if denom == 0:
                    continue
                hits.append(x0 + (lat - y0) / denom * (x1 - x0))
        hits.sort()
        for left, right in zip(hits[0::2], hits[1::2]):
            if right < left:
                left, right = right, left
            start = int(np.searchsorted(lons, left, side="left"))
            stop = int(np.searchsorted(lons, right, side="right"))
            if stop > start:
                mask[row, start:stop] = True
    return mask


def rasterize_polygon(grid: GeoGrid, rings: list[np.ndarray]) -> np.ndarray:
    """First ring is the shell. Later rings are holes."""
    if not rings:
        return np.zeros((grid.rows, grid.cols), dtype=bool)
    mask = rasterize_ring(grid, rings[0])
    for hole in rings[1:]:
        mask &= ~rasterize_ring(grid, hole)
    return mask


def paint_priority(cover: np.ndarray, mask: np.ndarray, code: int, priority: dict[int, int]) -> None:
    """Write ``code`` where the mask is set and the new class outranks the old one."""
    if not np.any(mask):
        return
    current = cover[mask]
    incoming = priority.get(code, 0)
    ranks = np.fromiter((priority.get(int(item), 0) for item in current), dtype=np.int16, count=current.size)
    cover[mask] = np.where(incoming >= ranks, code, current)


def rasterize_polyline(grid: GeoGrid, coords: np.ndarray, width: int = 1) -> np.ndarray:
    """Bresenham cells under a lon/lat polyline, dilated by ``width`` cells."""
    mask = np.zeros((grid.rows, grid.cols), dtype=bool)
    if len(coords) < 2:
        return mask
    for start, end in zip(coords[:-1], coords[1:]):
        r0, c0 = _cell(grid, float(start[0]), float(start[1]))
        r1, c1 = _cell(grid, float(end[0]), float(end[1]))
        for row, col in _bresenham(r0, c0, r1, c1):
            if 0 <= row < grid.rows and 0 <= col < grid.cols:
                mask[row, col] = True
    if width > 1:
        mask = _dilate(mask, width - 1)
    return mask


def _cell(grid: GeoGrid, lon: float, lat: float) -> tuple[int, int]:
    row, col = grid.lonlat_to_rowcol(np.float64(lon), np.float64(lat))
    return int(round(float(row))), int(round(float(col)))


def _bresenham(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    row, col = r0, c0
    while True:
        points.append((row, col))
        if row == r1 and col == c1:
            break
        doubled = 2 * err
        if doubled > -dc:
            err -= dc
            row += sr
        if doubled < dr:
            err += dr
            col += sc
        if len(points) > 100000:
            break
    return points


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    rows, cols = mask.shape
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        r0 = max(0, y - radius)
        r1 = min(rows, y + radius + 1)
        c0 = max(0, x - radius)
        c1 = min(cols, x + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def sample_grid(source: np.ndarray, src: GeoGrid, dst: GeoGrid, *, nearest: bool = False) -> np.ndarray:
    """Resample a north-up grid onto another north-up grid. Bilinear, or nearest."""
    rows = np.arange(dst.rows)
    cols = np.arange(dst.cols)
    lat = np.asarray(dst.row_lat(rows), dtype=np.float64)
    lon = np.asarray(dst.col_lon(cols), dtype=np.float64)
    src_row, src_col = src.lonlat_to_rowcol(lon[None, :], lat[:, None])
    if nearest:
        r = np.clip(np.rint(src_row), 0, src.rows - 1).astype(np.int32)
        c = np.clip(np.rint(src_col), 0, src.cols - 1).astype(np.int32)
        return source[r, c]
    return _bilinear(source, src_row, src_col)


def _bilinear(source: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    r0 = np.floor(row).astype(np.int32)
    c0 = np.floor(col).astype(np.int32)
    r0 = np.clip(r0, 0, source.shape[0] - 1)
    c0 = np.clip(c0, 0, source.shape[1] - 1)
    r1 = np.clip(r0 + 1, 0, source.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, source.shape[1] - 1)
    fr = np.clip(row - r0, 0, 1)
    fc = np.clip(col - c0, 0, 1)
    v00 = source[r0, c0]
    v01 = source[r0, c1]
    v10 = source[r1, c0]
    v11 = source[r1, c1]
    return (v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc + v10 * fr * (1 - fc) + v11 * fr * fc).astype(
        source.dtype if source.dtype.kind == "f" else np.float64
    )
