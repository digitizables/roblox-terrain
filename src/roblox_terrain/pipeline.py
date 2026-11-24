"""Build a terrain scene. This module does not talk to Studio."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from roblox_terrain.acquire import resolve_real_inputs
from roblox_terrain.cover import NAMES
from roblox_terrain.errors import PipelineError, ValidationError
from roblox_terrain.fields import build_fields
from roblox_terrain.grid import METERS_PER_DEGREE_LAT, GeoGrid, grid_for_bounds
from roblox_terrain.hydrology import derive
from roblox_terrain.noise import blur
from roblox_terrain.presets import PRESETS, get_preset
from roblox_terrain.scene import TerrainRequest, TerrainScene
from roblox_terrain.sources import HYDRO_DERIVED, SYNTHETIC
from roblox_terrain.surface import apply_micro, assign_materials, carve_ditches, place_props
from roblox_terrain.synthetic import synthetic_landscape
from roblox_terrain.vectors import VectorSet
from roblox_terrain.voxels import estimate_cost

Progress = Callable[[str, float, str], None]


def generate_realistic_terrain(
    request: TerrainRequest | None = None,
    *,
    cache_dir=None,
    progress: Progress | None = None,
    **overrides: Any,
) -> TerrainScene:
    """Generate one editable landscape description.

    The return value is in memory. Pass it to ``apply_scene`` to write it
    into Studio, or save it with ``save_scene``. No Studio connection is opened.
    """
    report = progress or _quiet
    base = request or TerrainRequest()
    chosen = base.resolved(overrides)
    _validate(chosen)
    report("prepare", 0.04, chosen.preset)
    inputs, real_place, preset_name, field_style, boundary_style, field_size, crops = _inputs(chosen, cache_dir, report)
    heights = blur(inputs.heights.astype(np.float64))
    heights = (0.82 * inputs.heights + 0.18 * heights).astype(np.float32)
    threshold = chosen.stream_threshold or None
    report("hydrology", 0.28, "filling pits and tracing drainage")
    hydro = derive(
        heights,
        cell_h=inputs.grid.cell_height_m,
        cell_w=inputs.grid.cell_width_m,
        stream_mask=inputs.stream_mask,
        lake_mask=inputs.lake_mask,
        stream_threshold=threshold,
    )
    report("fields", 0.48, f"streams={hydro.stream_cells}")
    layer = build_fields(
        hydro.surface,
        inputs.cover,
        inputs.vectors,
        inputs.grid,
        field_style=field_style,
        boundary_style=boundary_style,
        field_size_m=field_size,
        crops=crops,
        seed=int(chosen.seed),
    )
    surface, water = carve_ditches(hydro.surface, hydro.water_depth, layer, inputs.grid)
    hydro.surface = surface
    hydro.water_depth = water
    east, north = inputs.grid.local_axes()
    report("detail", 0.66, f"fields={len(layer.fields)}")
    detailed = apply_micro(
        surface,
        hydro,
        layer,
        east,
        north,
        micro_detail=chosen.micro_detail,
        furrow_strength=chosen.furrow_strength,
        seed=int(chosen.seed),
    )
    materials = assign_materials(inputs.cover, hydro, layer)
    props = place_props(
        inputs.cover,
        hydro,
        layer,
        detailed,
        inputs.grid,
        vegetation_density=chosen.vegetation_density,
        tree_budget=int(chosen.tree_budget),
        rock_budget=int(chosen.rock_budget),
        hedge_budget=int(chosen.hedge_budget),
        seed=int(chosen.seed),
    )
    if not np.isfinite(detailed).all():
        raise PipelineError("The heightfield contains a value that is not finite.")
    baseline = float(np.percentile(detailed, 2))
    lifted = (detailed - baseline) * chosen.vertical_exaggeration * chosen.studs_per_meter
    shift = 8.0 - float(np.min(lifted))
    warnings = _warnings(chosen, inputs, real_place)
    stats = _stats(chosen, inputs, hydro, layer, props, detailed, materials)
    cost = estimate_cost(
        inputs.grid.extent_east_m,
        inputs.grid.extent_north_m,
        float(np.max(lifted + shift) - np.min(lifted + shift)),
        studs_per_meter=chosen.studs_per_meter,
    )
    stats.update(cost)
    if cost["voxel_chunks"] > 80:
        warnings.append(
            f"Applying this scene writes {cost['voxel_chunks']} terrain chunks. "
            "That is a long Studio edit. Lower play_extent_m or studs_per_meter to shorten it."
        )
    if cost["voxel_chunks"] > 200 or cost["voxels"] > 8_000_000:
        raise ValidationError(
            f"This request is {cost['voxel_chunks']} chunks and {cost['voxels']} voxels. "
            "Lower play_extent_m or studs_per_meter. A 400 m window at 2 studs per meter stays inside Studio's limits."
        )
    report("done", 1.0, f"fields={len(layer.fields)} props={len(props)}")
    parameters = {key: getattr(chosen, key) for key in chosen.resolved({}).__dict__ if key not in {"elevation", "cover"}}
    parameters["field_style"] = field_style
    parameters["boundary_style"] = boundary_style
    parameters["field_size_m"] = field_size
    title = PRESETS[preset_name].title if preset_name in PRESETS else preset_name
    return TerrainScene(
        name=title,
        preset=preset_name,
        seed=int(chosen.seed),
        real_place=real_place,
        grid=inputs.grid,
        parameters=parameters,
        sources=inputs.sources,
        warnings=warnings,
        heights_m=detailed,
        cover=inputs.cover,
        materials=materials,
        water_depth_m=np.asarray(hydro.water_depth, dtype=np.float32),
        labels=layer.labels,
        fields=layer.fields,
        props=props,
        stats=stats,
        vertical_baseline_m=baseline,
        vertical_shift_studs=shift,
        notes=list(inputs.notes),
    )


def plan_tiles(
    preset: str = "cotswolds-windrush",
    *,
    play_extent_m: float = 400.0,
    studs_per_meter: float = 2.0,
    west: float | None = None,
    south: float | None = None,
    east: float | None = None,
    north: float | None = None,
) -> list[dict[str, Any]]:
    """Tile origins that cover a preset bbox with abutting play windows.

    This does not generate terrain. An agent can call ``generate_realistic_terrain``
    once per tile, passing that tile's bounds and origin.
    """
    spec = get_preset(preset)
    west = spec.west if west is None else west
    south = spec.south if south is None else south
    east = spec.east if east is None else east
    north = spec.north if north is None else north
    mid_lat = (south + north) / 2
    east_m = (east - west) * METERS_PER_DEGREE_LAT * math.cos(math.radians(mid_lat))
    north_m = (north - south) * METERS_PER_DEGREE_LAT
    count_x = max(1, int(math.ceil(east_m / play_extent_m)))
    count_z = max(1, int(math.ceil(north_m / play_extent_m)))
    tile_studs = play_extent_m * studs_per_meter
    tiles = []
    for j in range(count_z):
        for i in range(count_x):
            tile_west = west + (east - west) * i / count_x
            tile_east = west + (east - west) * (i + 1) / count_x
            # j = 0 is the northern tile, which is the smaller Z.
            tile_north = north - (north - south) * j / count_z
            tile_south = north - (north - south) * (j + 1) / count_z
            origin_x = (i - (count_x - 1) / 2) * tile_studs
            origin_z = (j - (count_z - 1) / 2) * tile_studs
            tiles.append(
                {
                    "i": i,
                    "j": j,
                    "west": tile_west,
                    "south": tile_south,
                    "east": tile_east,
                    "north": tile_north,
                    "origin_x_studs": origin_x,
                    "origin_z_studs": origin_z,
                    "focus_lat": (tile_south + tile_north) / 2,
                    "focus_lon": (tile_west + tile_east) / 2,
                }
            )
    return tiles


def _inputs(request: TerrainRequest, cache_dir, report: Progress):
    spec = get_preset(request.preset) if request.preset in PRESETS else None
    if request.preset not in PRESETS and request.preset != "custom":
        raise ValidationError(f"Unknown preset {request.preset!r}. Known presets: {', '.join(sorted(PRESETS))}, custom.")
    field_style = request.field_style
    boundary_style = request.boundary_style
    field_size = request.field_size_m
    crops = ("wheat", "pasture", "barley", "fallow")
    real_place = True
    preset_name = request.preset
    if spec is not None:
        if field_style == "auto":
            field_style = spec.field_style
        if boundary_style == "auto":
            boundary_style = spec.boundary_style
        if field_size <= 0:
            field_size = spec.field_size_m
        crops = spec.crops
        real_place = spec.real_place
    else:
        if field_style == "auto":
            field_style = "irregular"
        if boundary_style == "auto":
            boundary_style = "hedge"
        if field_size <= 0:
            field_size = 120.0
    if request.elevation is not None:
        report("elevation", 0.16, "caller-supplied array")
        return _from_arrays(request, field_style, boundary_style, field_size, crops), False, preset_name, field_style, boundary_style, field_size, crops
    if spec is not None and not spec.real_place:
        report("elevation", 0.16, "synthetic valley")
        return _synthetic(request), False, preset_name, field_style, boundary_style, field_size, crops
    west, south, east, north = _bounds(request, spec)
    report("elevation", 0.12, "loading elevation and land cover")
    inputs = resolve_real_inputs(
        preset_id=preset_name,
        west=west,
        south=south,
        east=east,
        north=north,
        play_extent_m=request.play_extent_m,
        grid_size=int(request.grid_size),
        focus_lat=request.focus_lat,
        focus_lon=request.focus_lon,
        allow_network=bool(request.allow_network),
        cache_dir=cache_dir,
        dem_path=request.dem_path,
        landcover_path=request.landcover_path,
        vectors_path=request.vectors_path,
        hydro_path=request.hydro_path,
        bundled=bool(spec and spec.bundled and spec.real_place and not request.dem_path),
    )
    return inputs, real_place, preset_name, field_style, boundary_style, field_size, crops


def _bounds(request: TerrainRequest, spec) -> tuple[float, float, float, float]:
    if None not in (request.west, request.south, request.east, request.north):
        return float(request.west), float(request.south), float(request.east), float(request.north)
    stored = _bounds_from_dem(request.dem_path) if request.dem_path else None
    if stored is not None:
        return stored
    if spec is None:
        raise ValidationError(
            "preset custom needs west, south, east, and north. "
            "An .npz DEM can supply those four values. A GeoTIFF needs the bbox as well."
        )
    return spec.west, spec.south, spec.east, spec.north


def _bounds_from_dem(path: str) -> tuple[float, float, float, float] | None:
    file = Path(path)
    if not file.is_file() or file.suffix.lower() != ".npz":
        return None
    blob = np.load(file)
    keys = ("west", "south", "east", "north")
    if any(key not in blob.files for key in keys):
        return None
    west, south, east, north = (float(blob[key]) for key in keys)
    return west, south, east, north


def _from_arrays(request: TerrainRequest, field_style, boundary_style, field_size, crops):
    del field_style, boundary_style, field_size, crops
    elevation = np.asarray(request.elevation, dtype=np.float32)
    if elevation.ndim != 2:
        raise ValidationError("elevation must be a 2D array")
    rows, cols = elevation.shape
    cover = np.asarray(request.cover if request.cover is not None else np.full((rows, cols), 3, dtype=np.uint8))
    if cover.shape != elevation.shape:
        raise ValidationError("cover must have the same shape as elevation")
    cell = request.cell_size_m or 8.0
    north_m = rows * cell
    east_m = cols * cell
    south = 52.0
    west = 0.0
    north = south + north_m / METERS_PER_DEGREE_LAT
    east = west + east_m / (METERS_PER_DEGREE_LAT * math.cos(math.radians(south)))
    grid = GeoGrid(west=west, south=south, east=east, north=north, rows=rows, cols=cols)
    from roblox_terrain.acquire import Inputs

    return Inputs(
        grid=grid,
        heights=elevation,
        cover=cover.astype(np.uint8),
        vectors=VectorSet(),
        sources=[dict(SYNTHETIC), dict(HYDRO_DERIVED)],
        notes=["Heights and cover were passed in by the caller."],
    )


def _synthetic(request: TerrainRequest):
    spec = get_preset("synthetic-valley")
    grid = grid_for_bounds(spec.west, spec.south, spec.east, spec.north, int(request.grid_size))
    heights, cover = synthetic_landscape(grid.rows, grid.cols, seed=int(request.seed))
    from roblox_terrain.acquire import Inputs

    return Inputs(
        grid=grid,
        heights=heights,
        cover=cover,
        vectors=VectorSet(),
        sources=[dict(SYNTHETIC), dict(HYDRO_DERIVED)],
        notes=["synthetic-valley is procedural. It is not a surveyed landscape."],
    )


def _validate(request: TerrainRequest) -> None:
    if not 24 <= int(request.grid_size) <= 320:
        raise ValidationError("grid_size must be from 24 to 320.")
    if not 80 <= float(request.play_extent_m) <= 4000:
        raise ValidationError("play_extent_m must be from 80 to 4000.")
    if not 0.5 <= float(request.studs_per_meter) <= 8:
        raise ValidationError("studs_per_meter must be from 0.5 to 8.")
    if not 0.5 <= float(request.vertical_exaggeration) <= 4:
        raise ValidationError("vertical_exaggeration must be from 0.5 to 4.")
    if request.field_style not in {"auto", "rectangular", "irregular"}:
        raise ValidationError("field_style must be auto, rectangular, or irregular.")
    if request.boundary_style not in {"auto", "hedge", "ditch", "fence", "none"}:
        raise ValidationError("boundary_style must be auto, hedge, ditch, fence, or none.")
    for name, limit in (
        ("tree_budget", 2000),
        ("rock_budget", 800),
        ("hedge_budget", 2000),
        ("surface_patch_budget", 4000),
    ):
        value = int(getattr(request, name))
        if value < 0 or value > limit:
            raise ValidationError(f"{name} must be from 0 to {limit}.")
    if not 0 <= float(request.vegetation_density) <= 2:
        raise ValidationError("vegetation_density must be from 0 to 2.")
    if not 0 <= float(request.furrow_strength) <= 1.5:
        raise ValidationError("furrow_strength must be from 0 to 1.5.")
    if not 0 <= float(request.micro_detail) <= 2:
        raise ValidationError("micro_detail must be from 0 to 2.")
    if float(request.detail_radius_studs) < 0:
        raise ValidationError("detail_radius_studs cannot be negative.")
    bounds = (request.west, request.south, request.east, request.north)
    if any(item is not None for item in bounds) and any(item is None for item in bounds):
        raise ValidationError("Pass all of west, south, east, and north, or none of them.")
    if all(item is not None for item in bounds):
        if request.east <= request.west or request.north <= request.south:
            raise ValidationError("The bbox is empty.")


def _warnings(request: TerrainRequest, inputs, real_place: bool) -> list[str]:
    warnings = []
    if abs(request.vertical_exaggeration - 1) > 0.05:
        warnings.append(
            f"Vertical exaggeration is {request.vertical_exaggeration}. Heights are not true scale. "
            "Set vertical_exaggeration to 1 for survey scale."
        )
    if not real_place:
        warnings.append("This scene is not a real-world extract.")
    warnings.extend(inputs.notes)
    return warnings


def _stats(request, inputs, hydro, layer, props, heights, materials) -> dict[str, Any]:
    cover_counts = Counter(NAMES.get(int(code), str(int(code))) for code in inputs.cover.ravel())
    total = max(1, inputs.cover.size)
    crops = Counter(parcel.crop for parcel in layer.fields)
    kinds = Counter(prop.kind for prop in props)
    return {
        "extent_m": [round(inputs.grid.extent_east_m, 1), round(inputs.grid.extent_north_m, 1)],
        "cell_m": [round(inputs.grid.cell_width_m, 2), round(inputs.grid.cell_height_m, 2)],
        "grid": [inputs.grid.rows, inputs.grid.cols],
        "elevation_min_m": round(float(heights.min()), 2),
        "elevation_max_m": round(float(heights.max()), 2),
        "relief_m": round(float(heights.max() - heights.min()), 2),
        "field_count": len(layer.fields),
        "mapped_fields": layer.mapped_fields,
        "subdivided_fields": layer.subdivided_fields,
        "crops": dict(crops),
        "stream_cells": hydro.stream_cells,
        "lake_cells": hydro.lake_cells,
        "cover_fraction": {name: round(count / total, 3) for name, count in sorted(cover_counts.items())},
        "props": dict(kinds),
        "material_ids": sorted({int(item) for item in np.unique(materials)}),
        "studs_per_meter": request.studs_per_meter,
    }


def _quiet(stage: str, fraction: float, detail: str = "") -> None:
    del stage, fraction, detail
