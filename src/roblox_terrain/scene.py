"""The request an agent fills in, and the scene the pipeline returns."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roblox_terrain.fields import FieldParcel
from roblox_terrain.grid import GeoGrid
from roblox_terrain.surface import Prop


@dataclass
class TerrainRequest:
    """Parameters for one generation.

    Arrays are a test hook. Leave them empty and pass a preset, a bbox, or
    file paths. ``apply`` is recorded here so the plugin and the CLI share one
    object; the pipeline itself never opens Studio.
    """

    preset: str = "cotswolds-windrush"
    west: float | None = None
    south: float | None = None
    east: float | None = None
    north: float | None = None
    focus_lat: float | None = None
    focus_lon: float | None = None
    grid_size: int = 96
    play_extent_m: float = 400.0
    studs_per_meter: float = 2.0
    vertical_exaggeration: float = 1.2
    seed: int = 1
    field_style: str = "auto"
    boundary_style: str = "auto"
    field_size_m: float = 0.0
    vegetation_density: float = 0.8
    tree_budget: int = 180
    rock_budget: int = 40
    hedge_budget: int = 160
    detail_radius_studs: float = 220.0
    surface_patch_budget: int = 420
    furrow_strength: float = 0.7
    micro_detail: float = 0.35
    stream_threshold: int = 0
    origin_x_studs: float = 0.0
    origin_z_studs: float = 0.0
    hide_baseplate: bool = True
    place_marker: bool = True
    allow_network: bool = False
    dem_path: str = ""
    landcover_path: str = ""
    vectors_path: str = ""
    hydro_path: str = ""
    apply: bool = False
    cell_size_m: float = 0.0
    elevation: np.ndarray | None = None
    cover: np.ndarray | None = None

    def resolved(self, overrides: dict[str, Any] | None = None) -> TerrainRequest:
        """Copy, then apply a flat dict of CLI or config values."""
        payload = {key: getattr(self, key) for key in _PUBLIC}
        for key, value in (overrides or {}).items():
            if key not in _PUBLIC:
                continue
            if value is None or value == "":
                continue
            payload[key] = value
        return TerrainRequest(**payload)


_PUBLIC = (
    "preset",
    "west",
    "south",
    "east",
    "north",
    "focus_lat",
    "focus_lon",
    "grid_size",
    "play_extent_m",
    "studs_per_meter",
    "vertical_exaggeration",
    "seed",
    "field_style",
    "boundary_style",
    "field_size_m",
    "vegetation_density",
    "tree_budget",
    "rock_budget",
    "hedge_budget",
    "detail_radius_studs",
    "surface_patch_budget",
    "furrow_strength",
    "micro_detail",
    "stream_threshold",
    "origin_x_studs",
    "origin_z_studs",
    "hide_baseplate",
    "place_marker",
    "allow_network",
    "dem_path",
    "landcover_path",
    "vectors_path",
    "hydro_path",
    "apply",
    "cell_size_m",
)


@dataclass
class TerrainScene:
    name: str
    preset: str
    seed: int
    real_place: bool
    grid: GeoGrid
    parameters: dict[str, Any]
    sources: list[dict[str, Any]]
    warnings: list[str]
    heights_m: np.ndarray
    cover: np.ndarray
    materials: np.ndarray
    water_depth_m: np.ndarray
    labels: np.ndarray
    fields: list[FieldParcel]
    props: list[Prop]
    stats: dict[str, Any]
    vertical_baseline_m: float
    vertical_shift_studs: float
    notes: list[str] = field(default_factory=list)

    def surface_studs(self, elevation_m: np.ndarray | float) -> np.ndarray | float:
        studs = self.parameters["studs_per_meter"]
        exaggeration = self.parameters["vertical_exaggeration"]
        lifted = (np.asarray(elevation_m, dtype=np.float64) - self.vertical_baseline_m) * exaggeration * studs
        return lifted + self.vertical_shift_studs

    def summary(self) -> str:
        relief = self.stats.get("relief_m", 0)
        place = "surveyed extract" if self.real_place else "procedural stand-in"
        return (
            f"{self.name}: {self.stats.get('extent_m', ['?', '?'])[0]:.0f} m by "
            f"{self.stats.get('extent_m', ['?', '?'])[1]:.0f} m, "
            f"{self.stats.get('field_count', 0)} fields, "
            f"relief {relief:.0f} m, {place}."
        )

    def manifest(self) -> dict[str, Any]:
        studs = float(self.parameters["studs_per_meter"])
        origin_x = float(self.parameters["origin_x_studs"])
        origin_z = float(self.parameters["origin_z_studs"])
        fields = []
        for parcel in self.fields:
            polygon = []
            for east, north in parcel.polygon:
                polygon.append(
                    {
                        "x": round(east * studs + origin_x, 2),
                        "z": round(-north * studs + origin_z, 2),
                    }
                )
            fields.append(
                {
                    "id": parcel.id,
                    "crop": parcel.crop,
                    "material": parcel.material,
                    "color": [round(channel, 3) for channel in parcel.color],
                    "orientation_deg": round(parcel.orientation_deg, 1),
                    "boundary": parcel.boundary,
                    "area_m2": round(parcel.area_m2, 1),
                    "polygon_studs": polygon,
                }
            )
        return {
            "schema": 1,
            "plugin": "terrain",
            "name": self.name,
            "preset": self.preset,
            "seed": self.seed,
            "real_place": self.real_place,
            "summary": self.summary(),
            "parameters": _jsonable(self.parameters),
            "bounds": {
                "west": self.grid.west,
                "south": self.grid.south,
                "east": self.grid.east,
                "north": self.grid.north,
            },
            "sources": self.sources,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "stats": _jsonable(self.stats),
            "fields": fields,
            "props": [
                {
                    "kind": prop.kind,
                    "species": prop.species,
                    "east_m": round(prop.east_m, 2),
                    "north_m": round(prop.north_m, 2),
                    "height_m": round(prop.height_m, 2),
                }
                for prop in self.props[:800]
            ],
            "props_truncated": len(self.props) > 800,
        }


def save_scene(scene: TerrainScene, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(scene.manifest(), indent=2)
    (directory / "manifest.json").write_text(text + "\n", encoding="utf-8")
    np.savez_compressed(
        directory / "arrays.npz",
        heights=scene.heights_m,
        cover=scene.cover,
        materials=scene.materials,
        water=scene.water_depth_m,
        labels=scene.labels,
        west=scene.grid.west,
        south=scene.grid.south,
        east=scene.grid.east,
        north=scene.grid.north,
    )
    return directory


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
