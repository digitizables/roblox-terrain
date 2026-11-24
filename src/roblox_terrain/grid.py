"""Geographic grid. Row 0 is north. Column 0 is west.

Roblox is Y-up. This plugin maps east to +X and south to +Z, so geographic
north is -Z and a camera looking down -Z looks north. The origin of a scene
is the center of the window, plus any tile offset the caller asked for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

METERS_PER_DEGREE_LAT = 111_320.0


@dataclass(frozen=True)
class GeoGrid:
    west: float
    south: float
    east: float
    north: float
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows < 2 or self.cols < 2:
            raise ValueError("A grid needs at least 2 by 2 cells")
        if self.east <= self.west or self.north <= self.south:
            raise ValueError("Grid bounds are empty")

    @property
    def center_lat(self) -> float:
        return (self.south + self.north) / 2

    @property
    def center_lon(self) -> float:
        return (self.west + self.east) / 2

    @property
    def meters_per_degree_lon(self) -> float:
        return METERS_PER_DEGREE_LAT * math.cos(math.radians(self.center_lat))

    @property
    def extent_north_m(self) -> float:
        return (self.north - self.south) * METERS_PER_DEGREE_LAT

    @property
    def extent_east_m(self) -> float:
        return (self.east - self.west) * self.meters_per_degree_lon

    @property
    def cell_height_m(self) -> float:
        return self.extent_north_m / self.rows

    @property
    def cell_width_m(self) -> float:
        return self.extent_east_m / self.cols

    def row_lat(self, row: np.ndarray | float) -> np.ndarray | float:
        span = self.north - self.south
        return self.north - (np.asarray(row) + 0.5) * span / self.rows

    def col_lon(self, col: np.ndarray | float) -> np.ndarray | float:
        span = self.east - self.west
        return self.west + (np.asarray(col) + 0.5) * span / self.cols

    def local_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """East and north meters from the window center, shape (rows, cols)."""
        rows = np.arange(self.rows)
        cols = np.arange(self.cols)
        lat = np.asarray(self.row_lat(rows), dtype=np.float64)
        lon = np.asarray(self.col_lon(cols), dtype=np.float64)
        north = (lat - self.center_lat) * METERS_PER_DEGREE_LAT
        east = (lon - self.center_lon) * self.meters_per_degree_lon
        east_grid, north_grid = np.meshgrid(east, north)
        return east_grid, north_grid

    def lonlat_to_rowcol(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fractional row and column. Row 0 is the northern edge."""
        row = (self.north - lat) / (self.north - self.south) * self.rows - 0.5
        col = (lon - self.west) / (self.east - self.west) * self.cols - 0.5
        return row, col


def grid_for_bounds(west: float, south: float, east: float, north: float, grid_size: int) -> GeoGrid:
    """``grid_size`` is the longer side. The other side follows the ground aspect."""
    mid_lat = (south + north) / 2
    east_m = (east - west) * METERS_PER_DEGREE_LAT * math.cos(math.radians(mid_lat))
    north_m = (north - south) * METERS_PER_DEGREE_LAT
    aspect = east_m / max(north_m, 1e-6)
    if aspect >= 1:
        cols = grid_size
        rows = max(16, int(round(grid_size / aspect)))
    else:
        rows = grid_size
        cols = max(16, int(round(grid_size * aspect)))
    return GeoGrid(west=west, south=south, east=east, north=north, rows=rows, cols=cols)


def window_bounds(
    grid: GeoGrid,
    *,
    extent_m: float,
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> tuple[float, float, float, float]:
    """Lon/lat bounds of a cell window. ``row`` 0 is the northernmost cell."""
    north = grid.north - row * (grid.north - grid.south) / grid.rows
    south = grid.north - (row + rows) * (grid.north - grid.south) / grid.rows
    west = grid.west + col * (grid.east - grid.west) / grid.cols
    east = grid.west + (col + cols) * (grid.east - grid.west) / grid.cols
    del extent_m
    return west, south, east, north


def center_window(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    focus_lat: float,
    focus_lon: float,
    extent_m: float,
) -> tuple[float, float, float, float]:
    """A square window in meters, clipped so it stays inside the source bounds."""
    mid_lat = focus_lat
    m_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(mid_lat))
    half_lat = (extent_m / 2) / METERS_PER_DEGREE_LAT
    half_lon = (extent_m / 2) / m_lon
    # Shift the focus if the window would leave the source.
    focus_lat = min(max(focus_lat, south + half_lat), north - half_lat)
    focus_lon = min(max(focus_lon, west + half_lon), east - half_lon)
    return focus_lon - half_lon, focus_lat - half_lat, focus_lon + half_lon, focus_lat + half_lat
