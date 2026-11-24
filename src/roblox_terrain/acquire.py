"""Load elevation and land cover from a bundle, a file, or a public download.

Network access is off unless the request sets ``allow_network``. Downloads are
cached under the plugin home. The bundled Cotswolds extract is the path the
example uses, so a fresh checkout can generate that region with no account
and no API key.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from roblox_terrain.cover import BY_NAME, PRIORITY, UNKNOWN, WORLDCOVER
from roblox_terrain.errors import DataUnavailable, ValidationError
from roblox_terrain.grid import GeoGrid, center_window, grid_for_bounds, window_bounds
from roblox_terrain.pngutil import read_rgb_png, terrarium_elevation
from roblox_terrain.raster import paint_priority, rasterize_polyline, sample_grid
from roblox_terrain.sources import COPERNICUS, HYDRO_DERIVED, HYDROSHEDS, OSM, SRTM, TERRARIUM, WORLDCOVER as WORLDCOVER_SOURCE
from roblox_terrain.vectors import VectorSet, load_vectors, parse_osm_xml

USER_AGENT = "roblox-terrain-plugin/0.1 (educational; local generator)"
DATA_DIR = Path(__file__).resolve().parent / "data"
BUNDLE_DEM = DATA_DIR / "cotswolds_windrush_dem.npz"
BUNDLE_LAND = DATA_DIR / "cotswolds_windrush_land.json"


@dataclass
class Inputs:
    grid: GeoGrid
    heights: np.ndarray
    cover: np.ndarray
    vectors: VectorSet
    sources: list[dict]
    notes: list[str] = field(default_factory=list)
    stream_mask: np.ndarray | None = None
    lake_mask: np.ndarray | None = None


def bundled_available() -> bool:
    return BUNDLE_DEM.is_file() and BUNDLE_LAND.is_file()


def load_dem_npz(path: Path) -> tuple[np.ndarray, GeoGrid]:
    blob = np.load(path)
    elevation = np.asarray(blob["elevation"], dtype=np.float32)
    if elevation.ndim != 2:
        raise ValidationError(f"{path} elevation array must be 2D")
    grid = GeoGrid(
        west=float(blob["west"]),
        south=float(blob["south"]),
        east=float(blob["east"]),
        north=float(blob["north"]),
        rows=int(elevation.shape[0]),
        cols=int(elevation.shape[1]),
    )
    return elevation, grid


def resolve_real_inputs(
    *,
    preset_id: str,
    west: float,
    south: float,
    east: float,
    north: float,
    play_extent_m: float,
    grid_size: int,
    focus_lat: float | None,
    focus_lon: float | None,
    allow_network: bool,
    cache_dir: Path | None,
    dem_path: str,
    landcover_path: str,
    vectors_path: str,
    hydro_path: str,
    bundled: bool,
) -> Inputs:
    notes: list[str] = []
    sources: list[dict] = [HYDRO_DERIVED]
    if dem_path:
        heights, dem_grid, dem_source = _read_dem_file(Path(dem_path))
        sources.append(dem_source)
    elif bundled and bundled_available() and preset_id == "cotswolds-windrush":
        heights, dem_grid = load_dem_npz(BUNDLE_DEM)
        sources.append(dict(TERRARIUM))
        notes.append("Elevation is the bundled AWS Terrain Tiles extract, not a live download.")
    elif allow_network:
        if cache_dir is None:
            raise DataUnavailable("allow_network needs a cache directory.")
        heights, dem_grid = fetch_terrarium(west, south, east, north, cache_dir)
        sources.append(dict(TERRARIUM))
        notes.append("Elevation was downloaded from AWS Terrain Tiles and cached.")
    else:
        raise DataUnavailable(
            "No elevation is available for this region. Use preset cotswolds-windrush (bundled), "
            "pass dem_path to a GeoTIFF or npz, or set allow_network to download AWS Terrain Tiles. "
            "Nothing was fetched."
        )
    if vectors_path:
        vectors = load_vectors(Path(vectors_path))
        sources.append(dict(OSM))
    elif bundled and BUNDLE_LAND.is_file() and preset_id == "cotswolds-windrush" and not vectors_path:
        vectors = load_vectors(BUNDLE_LAND)
        sources.append(dict(OSM))
        notes.append("Field polygons and waterways are the bundled OpenStreetMap extract.")
    elif allow_network and not vectors_path:
        if cache_dir is None:
            raise DataUnavailable("allow_network needs a cache directory.")
        vectors = fetch_osm(west, south, east, north, cache_dir)
        sources.append(dict(OSM))
        notes.append("Land cover was downloaded from the OpenStreetMap map API and cached.")
    else:
        vectors = VectorSet()
        notes.append(
            "No field polygons were loaded. Cropland will be subdivided procedurally. "
            "Pass vectors_path or allow_network to use OpenStreetMap."
        )
    cover_grid, cover_codes, cover_source = _optional_landcover(landcover_path, dem_grid)
    if cover_source:
        sources.append(cover_source)
    hydro_grid = _optional_hydro(hydro_path)
    if hydro_grid is not None:
        sources.append(dict(HYDROSHEDS))
    heights, dem_grid = _clip_dem_to_request(heights, dem_grid, west, south, east, north)
    window, framing = _frame(dem_grid, heights, vectors, play_extent_m, focus_lat, focus_lon)
    if framing != "full":
        notes.append(
            f"Framed a {play_extent_m:.0f} m window inside the source ({framing}). "
            "Raise play_extent_m to keep more of the bbox, or use plan_tiles to lay the rest beside it."
        )
    grid = grid_for_bounds(*window, grid_size)
    sampled = sample_grid(heights, dem_grid, grid).astype(np.float32)
    cover = np.full(sampled.shape, UNKNOWN, dtype=np.uint8)
    if cover_grid is not None and cover_codes is not None:
        resampled = sample_grid(cover_codes, cover_grid, grid, nearest=True).astype(np.int32)
        for code, mapped in WORLDCOVER.items():
            cover[resampled == code] = mapped
        # Arrays already in our class ids (values 0..10) are copied through.
        native = (resampled >= 0) & (resampled <= 10) & (resampled != 0)
        # Only treat small ids as native when the file was not WorldCover (max <= 10).
        if int(cover_codes.max()) <= 10:
            cover = resampled.astype(np.uint8)
    from roblox_terrain.fields import paint_cover

    cover = paint_cover(cover, _clip_vectors(vectors, grid), grid, PRIORITY)
    stream_mask, lake_mask = _burn_water(grid, vectors)
    if hydro_grid is not None:
        accumulation, hydro_bounds = hydro_grid
        hydro_sampled = sample_grid(accumulation, hydro_bounds, grid)
        threshold = max(50.0, float(np.percentile(hydro_sampled, 92)))
        stream_mask = stream_mask | (hydro_sampled >= threshold)
    return Inputs(
        grid=grid,
        heights=sampled,
        cover=cover,
        vectors=_clip_vectors(vectors, grid),
        sources=sources,
        notes=notes,
        stream_mask=stream_mask,
        lake_mask=lake_mask,
    )


def fetch_terrarium(west: float, south: float, east: float, north: float, cache_dir: Path, zoom: int | None = None) -> tuple[np.ndarray, GeoGrid]:
    """Download the Terrarium tiles that cover a bbox and crop them to it."""
    mid_lat = (south + north) / 2
    zoom = zoom if zoom is not None else _zoom_for(mid_lat, 12.0)
    x0, x1, y0, y1 = _tile_range(west, south, east, north, zoom)
    tiles: dict[tuple[int, int], np.ndarray] = {}
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            png = _cached_tile(cache_dir, zoom, x, y)
            tiles[(x, y)] = terrarium_elevation(read_rgb_png(png))
    mosaic, bounds = _stitch(tiles, zoom)
    grid = GeoGrid(west=bounds[0], south=bounds[1], east=bounds[2], north=bounds[3], rows=mosaic.shape[0], cols=mosaic.shape[1])
    window = GeoGrid(west=west, south=south, east=east, north=north, rows=max(2, mosaic.shape[0]), cols=max(2, mosaic.shape[1]))
    # Crop by resampling onto a grid of the requested bbox at the tile resolution.
    cropped = sample_grid(mosaic, grid, window).astype(np.float32)
    return cropped, window


def fetch_osm(west: float, south: float, east: float, north: float, cache_dir: Path) -> VectorSet:
    area = (east - west) * (north - south)
    if area > 0.25:
        raise ValidationError(
            "That bbox is too large for the OpenStreetMap map API (0.25 square degrees). "
            "Pass vectors_path with a GeoJSON extract instead."
        )
    digest = hashlib.sha1(f"{west:.5f},{south:.5f},{east:.5f},{north:.5f}".encode()).hexdigest()[:16]
    cache = cache_dir / "osm" / f"{digest}.osm"
    if not cache.is_file():
        url = f"https://api.openstreetmap.org/api/0.6/map?bbox={west},{south},{east},{north}"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(_http_get(url, timeout=90))
    return parse_osm_xml(cache.read_bytes())


def _read_dem_file(path: Path) -> tuple[np.ndarray, GeoGrid, dict]:
    if not path.is_file():
        raise DataUnavailable(f"DEM file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        heights, grid = load_dem_npz(path)
        source = dict(TERRARIUM)
        source["note"] = f"Loaded from {path.name}. Confirm the attribution matches that file."
        return heights, grid, source
    if suffix in {".tif", ".tiff"}:
        return _read_geotiff(path)
    raise ValidationError(f"dem_path must be a .npz or a GeoTIFF. Got {path.name}.")


def _read_geotiff(path: Path) -> tuple[np.ndarray, GeoGrid, dict]:
    try:
        import rasterio
    except ImportError as exc:
        raise DataUnavailable(
            "Reading a GeoTIFF needs the optional rasterio extra: pip install rasterio. "
            "You can also convert the raster to an npz with keys elevation, west, south, east, north."
        ) from exc
    with rasterio.open(path) as src:
        crs = src.crs
        if crs is not None and not crs.is_geographic:
            raise ValidationError(
                f"{path.name} is not in a geographic CRS ({crs}). Reproject to EPSG:4326 before dem_path."
            )
        band = src.read(1).astype(np.float32)
        if src.nodata is not None:
            band = np.where(band == src.nodata, np.nanmedian(band), band)
        bounds = src.bounds
    grid = GeoGrid(west=bounds.left, south=bounds.bottom, east=bounds.right, north=bounds.top, rows=band.shape[0], cols=band.shape[1])
    name = path.name.lower()
    if "copernicus" in name or "cop30" in name or "glo-30" in name:
        source = dict(COPERNICUS)
    elif "srtm" in name:
        source = dict(SRTM)
    else:
        source = {
            "name": path.name,
            "role": "elevation",
            "url": "",
            "attribution": f"Elevation loaded from {path.name}. Add the dataset's own attribution beside this scene.",
        }
    return band, grid, source


def _optional_landcover(path: str, fallback: GeoGrid) -> tuple[GeoGrid | None, np.ndarray | None, dict | None]:
    del fallback
    if not path:
        return None, None, None
    file = Path(path)
    if file.suffix.lower() == ".npz":
        blob = np.load(file)
        classes = np.asarray(blob["classes"])
        grid = GeoGrid(
            west=float(blob["west"]),
            south=float(blob["south"]),
            east=float(blob["east"]),
            north=float(blob["north"]),
            rows=classes.shape[0],
            cols=classes.shape[1],
        )
        return grid, classes, dict(WORLDCOVER_SOURCE)
    if file.suffix.lower() in {".tif", ".tiff"}:
        band, grid, _source = _read_geotiff(file)
        return grid, band.astype(np.int32), dict(WORLDCOVER_SOURCE)
    raise ValidationError("landcover_path must be an npz (classes, west, south, east, north) or a WorldCover GeoTIFF.")


def _optional_hydro(path: str) -> tuple[np.ndarray, GeoGrid] | None:
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        raise DataUnavailable(f"hydro_path does not exist: {file}")
    if file.suffix.lower() != ".npz":
        raise ValidationError("hydro_path must be an npz with accumulation, west, south, east, north.")
    blob = np.load(file)
    accumulation = np.asarray(blob["accumulation"], dtype=np.float32)
    grid = GeoGrid(
        west=float(blob["west"]),
        south=float(blob["south"]),
        east=float(blob["east"]),
        north=float(blob["north"]),
        rows=accumulation.shape[0],
        cols=accumulation.shape[1],
    )
    return accumulation, grid


def _frame(
    grid: GeoGrid,
    heights: np.ndarray,
    vectors: VectorSet,
    play_extent_m: float,
    focus_lat: float | None,
    focus_lon: float | None,
) -> tuple[tuple[float, float, float, float], str]:
    del heights
    if grid.extent_east_m <= play_extent_m * 1.08 and grid.extent_north_m <= play_extent_m * 1.08:
        return (grid.west, grid.south, grid.east, grid.north), "full"
    if focus_lat is not None and focus_lon is not None:
        window = center_window(
            grid.west, grid.south, grid.east, grid.north, focus_lat=focus_lat, focus_lon=focus_lon, extent_m=play_extent_m
        )
        return window, "focus"
    rough = grid_for_bounds(grid.west, grid.south, grid.east, grid.north, 96)
    cover = np.zeros((rough.rows, rough.cols), dtype=np.uint8)
    from roblox_terrain.fields import paint_cover

    cover = paint_cover(cover, vectors, rough, PRIORITY)
    stream = np.zeros(cover.shape, dtype=bool)
    for line in vectors.lines:
        stream |= rasterize_polyline(rough, line.coords, width=1)
    agri = np.isin(cover, (BY_NAME["cropland"], BY_NAME["pasture"], BY_NAME["orchard"])).astype(np.float64)
    forest = (cover == BY_NAME["forest"]).astype(np.float64)
    score = agri + 0.2 * forest + 0.35 * stream.astype(np.float64)
    win_rows = max(4, min(rough.rows, int(round(play_extent_m / rough.cell_height_m))))
    win_cols = max(4, min(rough.cols, int(round(play_extent_m / rough.cell_width_m))))
    integral = np.pad(score, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    best: tuple[float, int, int] | None = None
    step_r = max(1, win_rows // 10)
    step_c = max(1, win_cols // 10)
    for row in range(0, rough.rows - win_rows + 1, step_r):
        for col in range(0, rough.cols - win_cols + 1, step_c):
            total = (
                integral[row + win_rows, col + win_cols]
                - integral[row, col + win_cols]
                - integral[row + win_rows, col]
                + integral[row, col]
            )
            if best is None or total > best[0]:
                best = (float(total), row, col)
    assert best is not None
    return window_bounds(rough, extent_m=play_extent_m, row=best[1], col=best[2], rows=win_rows, cols=win_cols), "agricultural window"


def _burn_water(grid: GeoGrid, vectors: VectorSet) -> tuple[np.ndarray, np.ndarray]:
    streams = np.zeros((grid.rows, grid.cols), dtype=bool)
    lakes = np.zeros((grid.rows, grid.cols), dtype=bool)
    for line in vectors.lines:
        width = 2 if line.kind == "river" else 1
        streams |= rasterize_polyline(grid, line.coords, width=width)
    from roblox_terrain.raster import rasterize_polygon

    for poly in vectors.polygons:
        if poly.cover != "water":
            continue
        lakes |= rasterize_polygon(grid, poly.rings)
    return streams, lakes


def _clip_vectors(vectors: VectorSet, grid: GeoGrid) -> VectorSet:
    """Keep features that touch the window. Full geometry is retained for rasterizing."""
    pad = 0.002
    kept = VectorSet(attribution=vectors.attribution)
    for poly in vectors.polygons:
        shell = poly.rings[0]
        if shell[:, 0].max() < grid.west - pad or shell[:, 0].min() > grid.east + pad:
            continue
        if shell[:, 1].max() < grid.south - pad or shell[:, 1].min() > grid.north + pad:
            continue
        kept.polygons.append(poly)
    for line in vectors.lines:
        coords = line.coords
        if coords[:, 0].max() < grid.west - pad or coords[:, 0].min() > grid.east + pad:
            continue
        if coords[:, 1].max() < grid.south - pad or coords[:, 1].min() > grid.north + pad:
            continue
        kept.lines.append(line)
    return kept


def _clip_dem_to_request(
    heights: np.ndarray,
    dem_grid: GeoGrid,
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[np.ndarray, GeoGrid]:
    """Keep the part of a DEM the caller actually asked for."""
    west = max(west, dem_grid.west)
    south = max(south, dem_grid.south)
    east = min(east, dem_grid.east)
    north = min(north, dem_grid.north)
    if east - west < 1e-6 or north - south < 1e-6:
        raise DataUnavailable(
            "The elevation grid does not overlap the requested bbox. "
            f"DEM is {dem_grid.west:.4f},{dem_grid.south:.4f} to {dem_grid.east:.4f},{dem_grid.north:.4f}."
        )
    full = (
        abs(west - dem_grid.west) < 1e-5
        and abs(south - dem_grid.south) < 1e-5
        and abs(east - dem_grid.east) < 1e-5
        and abs(north - dem_grid.north) < 1e-5
    )
    if full:
        return heights, dem_grid
    target = grid_for_bounds(west, south, east, north, max(dem_grid.rows, dem_grid.cols))
    sampled = sample_grid(heights, dem_grid, target).astype(np.float32)
    return sampled, target


def _zoom_for(lat: float, target_m: float) -> int:
    meters = 156543.03 * math.cos(math.radians(lat))
    zoom = int(round(math.log2(meters / target_m)))
    return max(10, min(15, zoom))


def _tile_range(west: float, south: float, east: float, north: float, zoom: int) -> tuple[int, int, int, int]:
    x0, y1 = _tile_xy(south, west, zoom)
    x1, y0 = _tile_xy(north, east, zoom)
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def _tile_xy(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)
    return x, y


def tile_bounds(zoom: int, x: int, y: int) -> tuple[float, float, float, float]:
    """West, south, east, north of one XYZ tile. Y grows south."""
    n = 2**zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = _tile_lat(y, n)
    south = _tile_lat(y + 1, n)
    return west, south, east, north


def _tile_lat(y: int, n: int) -> float:
    mercator = math.pi * (1 - 2 * y / n)
    return math.degrees(math.atan(math.sinh(mercator)))


def _stitch(tiles: dict[tuple[int, int], np.ndarray], zoom: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    xs = sorted({key[0] for key in tiles})
    ys = sorted({key[1] for key in tiles})
    rows = [np.concatenate([tiles[(x, y)] for x in xs], axis=1) for y in ys]
    mosaic = np.concatenate(rows, axis=0)
    west, _south, _east, north = tile_bounds(zoom, xs[0], ys[0])
    _west, south, east, _north = tile_bounds(zoom, xs[-1], ys[-1])
    return mosaic, (west, south, east, north)


def _cached_tile(cache_dir: Path, zoom: int, x: int, y: int) -> bytes:
    path = cache_dir / "terrarium" / str(zoom) / str(x) / f"{y}.png"
    if path.is_file():
        return path.read_bytes()
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png"
    payload = _http_get(url, timeout=40)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _http_get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise DataUnavailable(f"Download failed for {url}: {exc}") from exc


def dump_geojson(vectors: VectorSet, path: Path) -> None:
    path.write_text(json.dumps(vectors.to_geojson()), encoding="utf-8")
