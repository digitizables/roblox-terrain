"""Materials, micro-relief, and the props that voxels cannot carry.

Furrows are a few decimeters tall. At Roblox's 4-stud voxels they barely move
the collision surface. The part layer inside the detail radius repeats the
same wave so a player standing in a field can see the cultivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from roblox_terrain.cover import BARE, CROPLAND, FOREST, ORCHARD, PASTURE, SCRUB, SNOW, UNKNOWN, URBAN, WATER, WETLAND
from roblox_terrain.fields import FieldLayer, FieldParcel
from roblox_terrain.grid import GeoGrid
from roblox_terrain.hydrology import Hydrology
from roblox_terrain.noise import fbm

MATERIALS = (
    "Grass",
    "LeafyGrass",
    "Ground",
    "Mud",
    "Rock",
    "Sand",
    "Sandstone",
    "Slate",
    "Limestone",
    "Snow",
    "Water",
    "Cobblestone",
    "Asphalt",
    "Basalt",
)
MATERIAL_INDEX = {name: index for index, name in enumerate(MATERIALS)}

COVER_MATERIAL = {
    UNKNOWN: "Grass",
    WATER: "Water",
    CROPLAND: "Ground",
    PASTURE: "Grass",
    ORCHARD: "Grass",
    FOREST: "LeafyGrass",
    SCRUB: "Grass",
    WETLAND: "Mud",
    BARE: "Sandstone",
    URBAN: "Cobblestone",
    SNOW: "Snow",
}


@dataclass
class Prop:
    kind: str
    species: str
    east_m: float
    north_m: float
    elevation_m: float
    yaw_deg: float
    scale: float
    height_m: float
    color: tuple[float, float, float]


def apply_micro(
    surface: np.ndarray,
    hydro: Hydrology,
    fields: FieldLayer,
    east: np.ndarray,
    north: np.ndarray,
    *,
    micro_detail: float,
    furrow_strength: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    rows, cols = surface.shape
    detail = (fbm(rows, cols, rng) - 0.5) * (0.85 * micro_detail)
    wet = hydro.water_depth >= 0.3
    detail = np.where(wet, detail * 0.1, detail)
    shaped = np.asarray(surface, dtype=np.float64) + detail
    for parcel in fields.fields:
        if not parcel.furrow or furrow_strength <= 0:
            continue
        theta = math.radians(parcel.orientation_deg)
        across = east * math.cos(theta) - north * math.sin(theta)
        wave = np.sin(across * (2 * math.pi) / parcel.spacing_m)
        mask = fields.labels == parcel.label
        shaped += mask * wave * furrow_strength * 0.18
    return shaped.astype(np.float32)


def assign_materials(cover: np.ndarray, hydro: Hydrology, fields: FieldLayer) -> np.ndarray:
    materials = np.full(cover.shape, MATERIAL_INDEX["Grass"], dtype=np.uint8)
    for code, name in COVER_MATERIAL.items():
        materials[cover == code] = MATERIAL_INDEX[name]
    steep = hydro.slope_deg >= 34
    moderate = (hydro.slope_deg >= 22) & ~steep & ~np.isin(cover, (CROPLAND, PASTURE, ORCHARD, URBAN))
    materials[moderate] = MATERIAL_INDEX["Slate"]
    materials[steep] = MATERIAL_INDEX["Rock"]
    for parcel in fields.fields:
        mask = (fields.labels == parcel.label) & (hydro.slope_deg < 20) & (hydro.water_depth < 0.3)
        materials[mask] = MATERIAL_INDEX[parcel.material]
    bank = _near(hydro.water_depth >= 0.3) & (hydro.water_depth < 0.3) & (hydro.slope_deg < 18)
    materials[bank] = MATERIAL_INDEX["Mud"]
    water = hydro.water_depth >= 0.3
    materials[water] = MATERIAL_INDEX["Water"]
    return materials


def place_props(
    cover: np.ndarray,
    hydro: Hydrology,
    fields: FieldLayer,
    heights: np.ndarray,
    grid: GeoGrid,
    *,
    vegetation_density: float,
    tree_budget: int,
    rock_budget: int,
    hedge_budget: int,
    seed: int,
) -> list[Prop]:
    rng = np.random.default_rng(seed + 91)
    east, north = grid.local_axes()
    props: list[Prop] = []
    props.extend(_trees(cover, hydro, fields, heights, east, north, grid, vegetation_density, tree_budget, rng))
    props.extend(_hedges(fields, heights, grid, hedge_budget, rng))
    props.extend(_rocks(hydro, heights, east, north, rock_budget, rng))
    return props


def _trees(
    cover: np.ndarray,
    hydro: Hydrology,
    fields: FieldLayer,
    heights: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    grid: GeoGrid,
    density: float,
    budget: int,
    rng: np.random.Generator,
) -> list[Prop]:
    if budget <= 0 or density <= 0:
        return []
    water = hydro.water_depth >= 0.3
    slope = hydro.slope_deg
    north_facing = (hydro.down_north > 0) & (slope > 5)
    aspect = np.where(north_facing, 1.15, np.where(hydro.down_north < 0, 0.88, 1.0))
    slope_factor = np.clip(1.1 - slope / 45.0, 0.05, 1.1)
    weights = {
        FOREST: 0.72,
        SCRUB: 0.22,
        ORCHARD: 0.0,
        PASTURE: 0.035,
        CROPLAND: 0.004,
        WETLAND: 0.05,
        URBAN: 0.01,
        UNKNOWN: 0.015,
    }
    chance = np.zeros(cover.shape, dtype=np.float64)
    for code, weight in weights.items():
        chance[cover == code] = weight
    # Inside a plowed field, keep the interior open. Trees belong on the hedge.
    for parcel in fields.fields:
        if parcel.crop in {"wheat", "barley", "roots", "maize", "fallow", "rapeseed"}:
            chance[fields.labels == parcel.label] *= 0.15
    chance *= aspect * slope_factor * density
    chance[water] = 0
    chance[slope > 40] = 0
    order = _weighted_cells(chance, rng)
    props: list[Prop] = []
    forest_share = int(budget * 0.62)
    other_share = budget - forest_share
    forest_used = 0
    other_used = 0
    for row, col in order:
        is_forest = int(cover[row, col]) in {FOREST, SCRUB}
        if is_forest and forest_used >= forest_share:
            continue
        if not is_forest and other_used >= other_share:
            continue
        if len(props) >= budget:
            break
        if rng.random() > min(1.0, float(chance[row, col]) * 4.0):
            continue
        species = _species(int(cover[row, col]), hydro, row, col, rng)
        height = float(rng.uniform(*_HEIGHTS[species]))
        jitter_e = float(rng.uniform(-0.35, 0.35) * grid.cell_width_m)
        jitter_n = float(rng.uniform(-0.35, 0.35) * grid.cell_height_m)
        props.append(
            Prop(
                kind="tree",
                species=species,
                east_m=float(east[row, col]) + jitter_e,
                north_m=float(north[row, col]) + jitter_n,
                elevation_m=float(heights[row, col]),
                yaw_deg=float(rng.uniform(0, 360)),
                scale=float(rng.uniform(0.85, 1.15)),
                height_m=height,
                color=_LEAF[species],
            )
        )
        if is_forest:
            forest_used += 1
        else:
            other_used += 1
    props.extend(_orchard_rows(fields, heights, east, north, grid, budget - len(props), rng))
    return props


def _orchard_rows(
    fields: FieldLayer,
    heights: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    grid: GeoGrid,
    budget: int,
    rng: np.random.Generator,
) -> list[Prop]:
    props: list[Prop] = []
    if budget <= 0:
        return props
    for parcel in fields.fields:
        if parcel.crop != "orchard":
            continue
        theta = math.radians(parcel.orientation_deg)
        mask = fields.labels == parcel.label
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        along = east * math.sin(theta) + north * math.cos(theta)
        across = east * math.cos(theta) - north * math.sin(theta)
        a0, a1 = float(along[mask].min()), float(along[mask].max())
        c0, c1 = float(across[mask].min()), float(across[mask].max())
        spacing = 8.0
        across_step = c0 + spacing * 0.5
        while across_step < c1 and len(props) < budget:
            along_step = a0 + spacing * 0.5
            while along_step < a1 and len(props) < budget:
                point_east = along_step * math.sin(theta) + across_step * math.cos(theta)
                point_north = along_step * math.cos(theta) - across_step * math.sin(theta)
                row, col = _nearest_cell(east, north, point_east, point_north)
                if mask[row, col]:
                    props.append(
                        Prop(
                            kind="tree",
                            species="apple",
                            east_m=point_east,
                            north_m=point_north,
                            elevation_m=float(heights[row, col]),
                            yaw_deg=parcel.orientation_deg,
                            scale=float(rng.uniform(0.9, 1.1)),
                            height_m=float(rng.uniform(3.2, 4.4)),
                            color=_LEAF["apple"],
                        )
                    )
                along_step += spacing
            across_step += spacing
    return props


def _hedges(fields: FieldLayer, heights: np.ndarray, grid: GeoGrid, budget: int, rng: np.random.Generator) -> list[Prop]:
    """One prop per hedge or fence segment. The apply step turns it into a part."""
    props: list[Prop] = []
    if budget <= 0:
        return props
    east, north = grid.local_axes()
    for line in fields.boundaries:
        if line.kind not in {"hedge", "fence"}:
            continue
        for start, end in zip(line.points, line.points[1:]):
            if len(props) >= budget:
                return props
            length = math.hypot(end[0] - start[0], end[1] - start[1])
            if length < 2.5:
                continue
            # Split long runs so a hedge can follow a curve without one huge part.
            pieces = max(1, int(math.ceil(length / 16.0)))
            for piece in range(pieces):
                if len(props) >= budget:
                    return props
                t0 = piece / pieces
                t1 = (piece + 1) / pieces
                a = (start[0] + (end[0] - start[0]) * t0, start[1] + (end[1] - start[1]) * t0)
                b = (start[0] + (end[0] - start[0]) * t1, start[1] + (end[1] - start[1]) * t1)
                mid_e = (a[0] + b[0]) / 2
                mid_n = (a[1] + b[1]) / 2
                row, col = _nearest_cell(east, north, mid_e, mid_n)
                yaw = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))
                span = math.hypot(b[0] - a[0], b[1] - a[1])
                props.append(
                    Prop(
                        kind=line.kind,
                        species="hawthorn" if line.kind == "hedge" else "post",
                        east_m=mid_e,
                        north_m=mid_n,
                        elevation_m=float(heights[row, col]),
                        yaw_deg=yaw,
                        scale=span,
                        height_m=float(rng.uniform(1.6, 2.6) if line.kind == "hedge" else 1.1),
                        color=(0.22, 0.38, 0.16) if line.kind == "hedge" else (0.45, 0.36, 0.26),
                    )
                )
    return props


def _rocks(
    hydro: Hydrology,
    heights: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> list[Prop]:
    if budget <= 0:
        return []
    score = np.clip((hydro.slope_deg - 18) / 25.0, 0, 1)
    score = np.where(hydro.water_depth >= 0.3, 0, score)
    chosen = _weighted_cells(score, rng)
    props: list[Prop] = []
    for row, col in chosen:
        if len(props) >= budget:
            break
        if float(score[row, col]) <= 0:
            continue
        if rng.random() > min(1.0, float(score[row, col])):
            continue
        props.append(
            Prop(
                kind="rock",
                species="limestone" if rng.random() < 0.7 else "boulder",
                east_m=float(east[row, col]),
                north_m=float(north[row, col]),
                elevation_m=float(heights[row, col]),
                yaw_deg=float(rng.uniform(0, 360)),
                scale=float(rng.uniform(0.7, 2.4)),
                height_m=float(rng.uniform(0.4, 1.3)),
                color=(0.55, 0.52, 0.46),
            )
        )
    return props


def carve_ditches(surface: np.ndarray, water: np.ndarray, fields: FieldLayer, grid: GeoGrid) -> tuple[np.ndarray, np.ndarray]:
    """Lower a narrow bed along ditch boundaries and give it a little water."""
    if not any(line.kind == "ditch" for line in fields.boundaries):
        return surface, water
    shaped = np.array(surface, dtype=np.float64, copy=True)
    wet = np.array(water, dtype=np.float64, copy=True)
    east, north = grid.local_axes()
    for line in fields.boundaries:
        if line.kind != "ditch":
            continue
        for point in _resample(line.points, step=max(grid.cell_width_m, grid.cell_height_m)):
            row, col = _nearest_cell(east, north, point[0], point[1])
            shaped[row, col] -= 0.55
            wet[row, col] = max(float(wet[row, col]), 0.35)
    return shaped.astype(np.float32), wet.astype(np.float32)


_HEIGHTS = {
    "oak": (9.0, 14.0),
    "ash": (8.0, 13.0),
    "beech": (8.0, 12.0),
    "willow": (6.0, 10.0),
    "hawthorn": (3.0, 5.0),
    "apple": (3.2, 4.4),
    "pine": (8.0, 13.0),
}
_LEAF = {
    "oak": (0.18, 0.36, 0.14),
    "ash": (0.24, 0.42, 0.18),
    "beech": (0.20, 0.40, 0.16),
    "willow": (0.28, 0.46, 0.22),
    "hawthorn": (0.22, 0.38, 0.16),
    "apple": (0.22, 0.40, 0.18),
    "pine": (0.12, 0.28, 0.16),
}


def _species(cover: int, hydro: Hydrology, row: int, col: int, rng: np.random.Generator) -> str:
    if hydro.water_depth[max(0, row - 2) : row + 3, max(0, col - 2) : col + 3].max() > 0.2 and cover != FOREST:
        if rng.random() < 0.55:
            return "willow"
    if cover == SCRUB:
        return "hawthorn" if rng.random() < 0.6 else "ash"
    if cover == PASTURE:
        return rng.choice(["oak", "ash", "hawthorn"])
    roll = rng.random()
    if roll < 0.4:
        return "oak"
    if roll < 0.7:
        return "beech"
    if roll < 0.9:
        return "ash"
    return "pine"


def _weighted_cells(weights: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int]]:
    flat = weights.ravel()
    order = np.argsort(rng.random(flat.size) - np.clip(flat, 0, 1))
    rows, cols = weights.shape
    return [(int(index // cols), int(index % cols)) for index in order if flat[index] > 0]


def _nearest_cell(east: np.ndarray, north: np.ndarray, point_e: float, point_n: float) -> tuple[int, int]:
    distance = (east - point_e) ** 2 + (north - point_n) ** 2
    index = int(np.argmin(distance))
    cols = east.shape[1]
    return index // cols, index % cols


def _resample(points: list[tuple[float, float]], step: float) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    sampled = [points[0]]
    for start, end in zip(points, points[1:]):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        count = max(1, int(length / max(step, 0.5)))
        for step_index in range(1, count + 1):
            t = step_index / count
            sampled.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
    return sampled


def _near(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out
