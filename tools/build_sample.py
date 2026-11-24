"""Crop the bundled Cotswolds extract from tiles and an OSM map already on disk.

The raw tiles and the OSM document are not part of the package. This writes
the arrays and the GeoJSON that ``cotswolds-windrush`` loads offline.

Expected inputs, from a previous fetch:

  %TEMP%\\ter_14_8110_5417.png
  %TEMP%\\ter_14_8110_5418.png
  %TEMP%\\ter_14_8111_5417.png
  %TEMP%\\ter_14_8111_5418.png
  %TEMP%\\cotswolds.osm

Zoom 14, tile X 8110-8111, tile Y 5417-5418. Row 0 of each PNG is north.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roblox_terrain.acquire import BUNDLE_DEM, BUNDLE_LAND, _clip_vectors, _stitch, dump_geojson
from roblox_terrain.grid import METERS_PER_DEGREE_LAT, GeoGrid
from roblox_terrain.pngutil import read_rgb_png, terrarium_elevation
from roblox_terrain.presets import get_preset
from roblox_terrain.raster import sample_grid
from roblox_terrain.vectors import parse_osm_xml

ZOOM = 14
TILES = ((8110, 5417), (8110, 5418), (8111, 5417), (8111, 5418))


def main() -> None:
    temp = Path(os.environ.get("TEMP") or os.environ.get("TMP") or ".")
    tiles = {}
    for x, y in TILES:
        path = temp / f"ter_{ZOOM}_{x}_{y}.png"
        if not path.is_file():
            raise SystemExit(f"Missing Terrarium tile {path}")
        tiles[(x, y)] = terrarium_elevation(read_rgb_png(path))
    mosaic, bounds = _stitch(tiles, ZOOM)
    preset = get_preset("cotswolds-windrush")
    if (
        preset.west < bounds[0]
        or preset.east > bounds[2]
        or preset.south < bounds[1]
        or preset.north > bounds[3]
    ):
        raise SystemExit(f"Preset bbox is outside the mosaic {bounds}")
    source = GeoGrid(
        west=bounds[0],
        south=bounds[1],
        east=bounds[2],
        north=bounds[3],
        rows=int(mosaic.shape[0]),
        cols=int(mosaic.shape[1]),
    )
    mid = (preset.south + preset.north) / 2
    east_m = (preset.east - preset.west) * METERS_PER_DEGREE_LAT * np.cos(np.radians(mid))
    north_m = (preset.north - preset.south) * METERS_PER_DEGREE_LAT
    rows = max(2, int(round(north_m / source.cell_height_m)))
    cols = max(2, int(round(east_m / source.cell_width_m)))
    target = GeoGrid(
        west=preset.west,
        south=preset.south,
        east=preset.east,
        north=preset.north,
        rows=rows,
        cols=cols,
    )
    cropped = sample_grid(mosaic, source, target).astype(np.float32)
    if not np.isfinite(cropped).all():
        raise SystemExit("Cropped elevation has a non-finite value")
    osm_path = temp / "cotswolds.osm"
    if not osm_path.is_file():
        raise SystemExit(f"Missing OSM extract {osm_path}")
    vectors = parse_osm_xml(osm_path.read_bytes())
    clipped = _clip_vectors(vectors, target)
    clipped.attribution = "© OpenStreetMap contributors"
    BUNDLE_DEM.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        BUNDLE_DEM,
        elevation=cropped,
        west=np.float64(preset.west),
        south=np.float64(preset.south),
        east=np.float64(preset.east),
        north=np.float64(preset.north),
    )
    dump_geojson(clipped, BUNDLE_LAND)
    covers: dict[str, int] = {}
    for poly in clipped.polygons:
        covers[poly.cover] = covers.get(poly.cover, 0) + 1
    print(f"elevation {cropped.shape} min {float(cropped.min()):.1f} max {float(cropped.max()):.1f}")
    print(f"relief {float(cropped.max() - cropped.min()):.1f} m  cell {target.cell_width_m:.2f} by {target.cell_height_m:.2f} m")
    print(f"polygons {len(clipped.polygons)} lines {len(clipped.lines)} covers {covers}")
    print(f"wrote {BUNDLE_DEM}")
    print(f"wrote {BUNDLE_LAND}")


if __name__ == "__main__":
    main()
