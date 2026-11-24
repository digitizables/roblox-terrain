"""Register terrain commands with the plugin host.

Generation is a normal Python call. This module is the adapter-facing
contract: status, generate, inspect, plan, and clear. ``apply`` is a command
argument, not a saved setting, so a config file cannot turn a dry run into a
place edit.
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import numpy as np

from roblox_studio_plugin_host.contract import CommandSpec, InvokeContext, ParameterSpec, PluginManifest
from roblox_studio_plugin_host.discovery import coerce_params
from roblox_studio_plugin_host.errors import UnknownCommand

from roblox_terrain import __version__
from roblox_terrain.acquire import bundled_available
from roblox_terrain.apply import apply_scene, clear_generated_terrain, impact
from roblox_terrain.errors import ConfirmationRequired, DataUnavailable, TerrainError
from roblox_terrain.pipeline import generate_realistic_terrain, plan_tiles
from roblox_terrain.presets import PRESETS
from roblox_terrain.scene import TerrainRequest, _jsonable, save_scene

_PRESET_CHOICES = tuple(PRESETS) + ("custom",)
_FIELD_STYLES = ("auto", "rectangular", "irregular")
_BOUNDARIES = ("auto", "hedge", "ditch", "fence", "none")


def _string(name: str, summary: str, default: Any = None, choices: tuple[str, ...] = ()) -> ParameterSpec:
    return ParameterSpec(name, "string", summary, default=default, choices=choices)


def _integer(name: str, summary: str, default: int) -> ParameterSpec:
    return ParameterSpec(name, "integer", summary, default=default)


def _number(name: str, summary: str, default: float | None = None) -> ParameterSpec:
    return ParameterSpec(name, "number", summary, default=default)


def _boolean(name: str, summary: str, default: bool) -> ParameterSpec:
    return ParameterSpec(name, "boolean", summary, default=default)


_CONFIG: tuple[ParameterSpec, ...] = (
    _string(
        "preset",
        "Named region. cotswolds-windrush ships with the package. synthetic-valley is procedural.",
        default="cotswolds-windrush",
        choices=_PRESET_CHOICES,
    ),
    _integer("grid_size", "Cells along the longer side of the analysis grid (24 to 320).", 96),
    _number("play_extent_m", "Meters on a side of the playable window framed inside the source.", 400),
    _number("studs_per_meter", "Horizontal scale. 2 studs per meter makes a 4-stud voxel about 2 meters.", 2),
    _number("vertical_exaggeration", "Height scale. 1 is survey scale. The default of 1.2 reads better in play.", 1.2),
    _integer("seed", "Seed for crops, furrows, and vegetation placement.", 1),
    _string("field_style", "auto uses the preset. rectangular or irregular overrides it.", "auto", _FIELD_STYLES),
    _string("boundary_style", "auto uses the preset. hedge, ditch, fence, or none overrides it.", "auto", _BOUNDARIES),
    _number("field_size_m", "Target field width in meters. 0 keeps the preset size.", 0),
    _number("vegetation_density", "Multiplier on tree placement when tree_budget is above 0.", 0.8),
    _integer("tree_budget", "Trees to place. 0 writes terrain only.", 0),
    _integer("rock_budget", "Rock parts. 0 writes none.", 0),
    _integer("hedge_budget", "Hedge and fence segments. 0 writes none.", 0),
    _number("detail_radius_studs", "Close-up field patches are placed only inside this radius, and only when surface_patch_budget is above 0.", 220),
    _integer("surface_patch_budget", "Close-up crop patches. 0 writes none.", 0),
    _number("furrow_strength", "Height of cultivation ripples, from 0 to 1.5. Visible on the voxel grid only as a few decimeters.", 0.7),
    _number("micro_detail", "Amplitude in meters of the procedural roughness added to the DEM.", 0.35),
    _integer("stream_threshold", "Ignored. The bed is the mapped river, or one drainage stem when the map has none.", 0),
    _integer("edge_skirt_voxels", "Voxels over which the rim slopes down to a low contour. 0 leaves a vertical cut.", 20),
    _boolean("dress_materials", "Tint terrain materials and water for this place. clear_generated_terrain restores the previous colors.", True),
    _boolean("hide_baseplate", "Hide Workspace.Baseplate after a confirmed apply. clear_generated_terrain restores it.", True),
    _boolean("place_marker", "Drop a marker part at the driest cell nearest the origin.", False),
    _boolean("allow_network", "Download AWS Terrain Tiles and OpenStreetMap when no local file covers the bbox.", False),
)

_GENERATE: tuple[ParameterSpec, ...] = _CONFIG + (
    _number("west", "West longitude. Pass all four edges, or none."),
    _number("south", "South latitude."),
    _number("east", "East longitude."),
    _number("north", "North latitude."),
    _number("focus_lat", "Latitude to center when the source is larger than play_extent_m."),
    _number("focus_lon", "Longitude to center when the source is larger than play_extent_m."),
    _number("origin_x_studs", "East offset of this window in the place. plan_tiles fills this in."),
    _number("origin_z_studs", "South offset of this window. North is negative Z."),
    _number("cell_size_m", "Meters per cell when elevation is passed as an array. Ignored by the CLI."),
    _string("dem_path", "Local .npz or GeoTIFF elevation. Empty uses the preset bundle or the network.", ""),
    _string("landcover_path", "Optional ESA WorldCover GeoTIFF, or an npz of class ids.", ""),
    _string("vectors_path", "Optional GeoJSON or OSM XML of fields and waterways.", ""),
    _string("hydro_path", "Optional npz with an accumulation array (HydroSHEDS or similar).", ""),
    _boolean("apply", "Write the scene into the open place. Requires confirmation. Not saved in config.", False),
)

_PLAN: tuple[ParameterSpec, ...] = (
    _string("preset", "Region to tile.", "cotswolds-windrush", _PRESET_CHOICES),
    _number("play_extent_m", "Meters on a side of each tile.", 400),
    _number("studs_per_meter", "Used only to convert tile origins into studs.", 2),
    _number("west", "Optional west edge. Pass all four, or none."),
    _number("south", "Optional south edge."),
    _number("east", "Optional east edge."),
    _number("north", "Optional north edge."),
)


class TerrainPlugin:
    """The object named by plugin.json and by the roblox_studio.plugins entry point."""

    def manifest(self) -> PluginManifest:
        return PluginManifest(
            id="terrain",
            name="Realistic terrain",
            version=__version__,
            summary=(
                "Build a playable landscape from real elevation and land cover, "
                "with fields, drainage, and close-up detail."
            ),
            capabilities=("terrain.generate", "terrain.inspect", "terrain.plan", "terrain.apply"),
            commands=(
                CommandSpec(
                    "status",
                    "Report the bundled sample, optional GeoTIFF support, and the last generated scene.",
                    mutates_place=False,
                ),
                CommandSpec(
                    "generate_realistic_terrain",
                    "Build one landscape and cache it. Pass apply=true to write it into the open place.",
                    mutates_place=False,
                    parameters=_GENERATE,
                ),
                CommandSpec(
                    "inspect_scene",
                    "Read the manifest of the last scene cached in this plugin home.",
                    mutates_place=False,
                ),
                CommandSpec(
                    "plan_tiles",
                    "List abutting windows that cover a bbox. Does not generate terrain.",
                    mutates_place=False,
                    parameters=_PLAN,
                ),
                CommandSpec(
                    "clear_generated_terrain",
                    "Air the recorded voxel box, restore the baseplate, and delete Workspace.RealisticTerrain.",
                    mutates_place=True,
                ),
            ),
            config=_CONFIG,
        )

    def invoke(self, command: str, params: dict[str, Any], *, context: InvokeContext) -> dict[str, Any]:
        spec = self.manifest().command(command)
        if spec is None:
            known = [item.name for item in self.manifest().commands]
            raise UnknownCommand("terrain", command, known)
        clean = coerce_params(spec.parameters, params)
        if command == "status":
            return _status(context)
        if command == "generate_realistic_terrain":
            return _generate(clean, context)
        if command == "inspect_scene":
            return _inspect(context)
        if command == "plan_tiles":
            return _plan(clean)
        if command == "clear_generated_terrain":
            return _clear(context)
        raise UnknownCommand("terrain", command, [item.name for item in self.manifest().commands])


PLUGIN = TerrainPlugin()


def _scene_dir(context: InvokeContext):
    return context.home / "cache" / "terrain" / "last-scene"


def _request(context: InvokeContext, params: dict[str, Any]) -> TerrainRequest:
    merged: dict[str, Any] = {}
    allowed = {item.name for item in _GENERATE}
    for key, value in context.config.items():
        if key in allowed and key != "apply":
            merged[key] = value
    merged.update(params)
    return TerrainRequest().resolved(merged)


def _generate(params: dict[str, Any], context: InvokeContext) -> dict[str, Any]:
    request = _request(context, params)
    cache = context.home / "cache" / "terrain"
    scene = generate_realistic_terrain(request, cache_dir=cache / "downloads", progress=context.progress)
    directory = _scene_dir(context)
    save_scene(scene, directory)
    result = _scene_payload(scene, directory)
    if not request.apply:
        return result
    if not context.confirmed:
        raise ConfirmationRequired(
            impact(scene) + f"\nThe scene was saved at {directory}. inspect_scene reads it back."
        )
    if context.gateway is None:
        raise TerrainError(
            "apply is true and the edit is confirmed, but no Studio connection is open. "
            "Run the command through roblox-plugins so the host can open one."
        )
    applied = apply_scene(context.gateway, scene, confirm=True, progress=context.progress)
    result.update(_jsonable(applied))
    result["applied"] = True
    return result


def _scene_payload(scene, directory) -> dict[str, Any]:
    manifest = scene.manifest()
    fields = [
        {
            "id": parcel["id"],
            "crop": parcel["crop"],
            "material": parcel["material"],
            "area_m2": parcel["area_m2"],
            "boundary": parcel["boundary"],
            "orientation_deg": parcel["orientation_deg"],
        }
        for parcel in manifest["fields"]
    ]
    stats = manifest["stats"]
    return {
        "applied": False,
        "preset": scene.preset,
        "seed": scene.seed,
        "real_place": scene.real_place,
        "scene_dir": str(directory),
        "summary": scene.summary(),
        "stats": stats,
        "warnings": list(scene.warnings),
        "notes": list(scene.notes),
        "fields": fields,
        "sources": scene.sources,
        "performance": {
            "voxels": stats.get("voxels"),
            "voxel_chunks": stats.get("voxel_chunks"),
            "voxels_xy": stats.get("voxels_xy"),
            "voxels_y": stats.get("voxels_y"),
        },
        "apply_hint": (
            "Re-run generate_realistic_terrain with apply=true and confirm "
            "(--yes or ROBLOX_STUDIO_CONFIRM=1) to write this scene into Studio. "
            "Nothing has been written."
        ),
    }


def _inspect(context: InvokeContext) -> dict[str, Any]:
    path = _scene_dir(context) / "manifest.json"
    if not path.is_file():
        raise DataUnavailable(
            "No generated scene is cached in this plugin home. Run generate_realistic_terrain first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scene_dir"] = str(path.parent)
    return payload


def _plan(params: dict[str, Any]) -> dict[str, Any]:
    tiles = plan_tiles(
        str(params.get("preset") or "cotswolds-windrush"),
        play_extent_m=float(params.get("play_extent_m") or 400),
        studs_per_meter=float(params.get("studs_per_meter") or 2),
        west=params.get("west"),
        south=params.get("south"),
        east=params.get("east"),
        north=params.get("north"),
    )
    return {"count": len(tiles), "tiles": tiles}


def _clear(context: InvokeContext) -> dict[str, Any]:
    if not context.confirmed:
        return clear_generated_terrain(context.gateway, confirm=False)
    if context.gateway is None:
        raise TerrainError(
            "clear_generated_terrain is confirmed, but no Studio connection is open. "
            "Run it through roblox-plugins so the host can open one."
        )
    return clear_generated_terrain(context.gateway, confirm=True)


def _status(context: InvokeContext) -> dict[str, Any]:
    manifest_path = _scene_dir(context) / "manifest.json"
    last: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            last = {
                "path": str(manifest_path.parent),
                "summary": payload.get("summary"),
                "preset": payload.get("preset"),
                "real_place": payload.get("real_place"),
            }
        except (OSError, json.JSONDecodeError) as exc:
            last = {"path": str(manifest_path.parent), "error": str(exc)}
    return {
        "version": __version__,
        "numpy": np.__version__,
        "bundle": bundled_available(),
        "rasterio": importlib.util.find_spec("rasterio") is not None,
        "presets": sorted(PRESETS),
        "last_scene": last,
    }
