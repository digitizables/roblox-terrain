"""Write a scene into the open place through a Studio port.

The port is the adapter (or a fake in tests). This module picks folder names,
voxel chunks, and part budgets. It does not open the MCP process itself.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable

import numpy as np

from roblox_terrain.errors import ApplyError, ConfirmationRequired, TerrainError
from roblox_terrain.scene import TerrainScene
from roblox_terrain.voxels import VoxelPlan, chunk_script, plan_voxels

ROOT = "Workspace.RealisticTerrain"
FEATURES = f"{ROOT}.Features"
Progress = Callable[[str, float, str], None]


def impact(scene: TerrainScene) -> str:
    chunks = scene.stats.get("voxel_chunks", "?")
    extent = scene.stats.get("voxels_xy", [0, 0])
    width = int(extent[0]) * 4 if extent else 0
    depth = int(extent[1]) * 4 if extent else 0
    baseplate = ""
    if scene.parameters.get("hide_baseplate", True):
        baseplate = (
            "- Hide Workspace.Baseplate when one exists (transparent, no collision, moved down). "
            "clear_generated_terrain puts it back.\n"
        )
    return (
        "This will edit the open Studio place. It does not save or publish.\n"
        f"- Replace terrain voxels in a box about {width} by {depth} studs ({chunks} chunks) "
        f"centered near ({scene.parameters.get('origin_x_studs', 0)}, {scene.parameters.get('origin_z_studs', 0)}).\n"
        f"- Replace Workspace.RealisticTerrain. The default write is voxels only "
        f"({len(scene.props)} props when a budget is set above 0).\n"
        "- Slope the outer voxels down so the window is not a vertical-sided block.\n"
        f"{baseplate}"
        "- Stop a running solo playtest so the write runs in the Edit datamodel.\n"
        "Re-run with confirm=True or ROBLOX_STUDIO_CONFIRM=1. Nothing was written."
    )


def apply_scene(port: Any, scene: TerrainScene, *, confirm: bool, progress: Progress | None = None) -> dict[str, Any]:
    """Apply ``scene``. Refuses to run until ``confirm`` is true."""
    report = progress or (lambda stage, fraction, detail: None)
    if not confirm:
        raise ConfirmationRequired(impact(scene))
    port = _as_port(port, report)
    plan = plan_voxels(scene)
    report("studio", 0.02, "edit mode")
    try:
        port.ensure_edit()
    except Exception as exc:
        raise ApplyError(f"Could not put Studio into Edit mode: {exc}") from exc
    _clear_existing(port)
    _create_root(port, scene, plan)
    written = 0
    try:
        for chunk in plan.chunks:
            payload = port.execute(chunk_script(chunk))
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                raise ApplyError(
                    f"Terrain chunk {chunk.index} failed after {written} chunks were written. "
                    f"The place is partial. Run clear_generated_terrain before trying again. Studio said: {payload}"
                )
            written += 1
            report("voxels", 0.05 + 0.7 * written / max(1, len(plan.chunks)), f"chunk {written}/{len(plan.chunks)}")
        if scene.parameters.get("dress_materials", True):
            port.execute(_dress_materials_script())
            report("materials", 0.8, "terrain colors")
    except ApplyError:
        raise
    except Exception as exc:
        raise ApplyError(
            f"Terrain write failed after {written} of {len(plan.chunks)} chunks. "
            "The place is partial. Run clear_generated_terrain before trying again. "
            f"Cause: {exc}"
        ) from exc
    specs = feature_specs(scene, plan)
    try:
        paths = port.apply_instances(specs) if specs else []
    except Exception as exc:
        raise ApplyError(
            f"Voxels were written ({written} chunks) but instance creation failed. "
            f"Workspace.RealisticTerrain may be incomplete. Cause: {exc}"
        ) from exc
    report("features", 0.9, f"instances={len(paths)}")
    baseplate = {"found": False}
    if scene.parameters.get("hide_baseplate", True):
        try:
            baseplate = port.execute(_hide_baseplate_script())
        except Exception as exc:
            raise ApplyError(
                f"Terrain and props were written, but hiding the baseplate failed: {exc}. "
                "The baseplate may still be intersecting the ground."
            ) from exc
        if isinstance(baseplate, dict) and baseplate.get("found"):
            port.apply_instances(
                [
                    {
                        "parent": ROOT,
                        "name": "BaseplateBackup",
                        "class_name": "StringValue",
                        "properties": {"Value": json.dumps(_json_backup(baseplate))},
                    }
                ]
            )
    report("done", 1.0, "applied")
    return {
        "applied": True,
        "chunks": written,
        "voxels": plan.voxels,
        "instances": len(paths),
        "root": ROOT,
        "region": {"min": list(plan.min_corner), "max": list(plan.max_corner)},
        "baseplate_hidden": bool(isinstance(baseplate, dict) and baseplate.get("found")),
        "summary": scene.summary(),
    }


def clear_generated_terrain(port: Any, *, confirm: bool) -> dict[str, Any]:
    """Remove the plugin's folder and the voxel box recorded on it."""
    if not confirm:
        raise ConfirmationRequired(
            "This will delete Workspace.RealisticTerrain, fill its recorded voxel box with air, "
            "and restore Workspace.Baseplate when a backup was stored. It does not save or publish. "
            "Re-run with confirm=True or ROBLOX_STUDIO_CONFIRM=1. Nothing was written."
        )
    port = _as_port(port, lambda *args: None)
    port.ensure_edit()
    payload = port.execute(_clear_script())
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise ApplyError(f"clear_generated_terrain failed: {payload}")
    return {"applied": True, "cleared": bool(payload.get("cleared")), "baseplate_restored": bool(payload.get("baseplate_restored"))}


def feature_specs(scene: TerrainScene, plan: VoxelPlan | None = None) -> list[dict[str, Any]]:
    """Instance specs in parent-before-child order. The adapter chunks and yields."""
    plan = plan or plan_voxels(scene)
    specs: list[dict[str, Any]] = []
    manifest = scene.manifest()
    text = json.dumps(manifest)
    if len(text) > 120_000:
        for parcel in manifest["fields"]:
            parcel["polygon_studs"] = parcel["polygon_studs"][:12]
        manifest["props"] = manifest["props"][:200]
        manifest["truncated"] = True
        text = json.dumps(manifest)
    min_c = plan.min_corner
    max_c = plan.max_corner
    specs.append(_folder("Workspace", "RealisticTerrain", _root_attributes(scene, min_c, max_c)))
    specs.append(
        {
            "parent": ROOT,
            "name": "SceneJson",
            "class_name": "StringValue",
            "properties": {"Value": text},
        }
    )
    specs.append(_folder(ROOT, "Features", {}))
    for name in ("Fields", "Vegetation", "Boundaries", "Rocks"):
        specs.append(_folder(FEATURES, name, {}))
    for parcel in scene.fields:
        specs.append(
            _folder(
                f"{FEATURES}.Fields",
                parcel.id,
                {
                    "Crop": parcel.crop,
                    "AreaM2": round(parcel.area_m2, 1),
                    "OrientationDeg": round(parcel.orientation_deg, 1),
                    "Boundary": parcel.boundary,
                    "FieldId": parcel.id,
                },
            )
        )
    specs.extend(_patches(scene))
    trunks = []
    canopies = []
    hedges = []
    rocks = []
    for index, prop in enumerate(scene.props, start=1):
        if prop.kind == "tree":
            trunk, canopy = _tree_specs(scene, prop, index)
            trunks.append(trunk)
            canopies.append(canopy)
        elif prop.kind in {"hedge", "fence"}:
            hedges.append(_boundary_spec(scene, prop, index))
        elif prop.kind == "rock":
            rocks.append(_rock_spec(scene, prop, index))
    specs.extend(trunks)
    specs.extend(canopies)
    specs.extend(hedges)
    specs.extend(rocks)
    if scene.parameters.get("place_marker", True):
        marker = _marker(scene)
        if marker is not None:
            specs.append(marker)
    return specs


def _patches(scene: TerrainScene) -> list[dict[str, Any]]:
    budget = int(scene.parameters.get("surface_patch_budget", 0))
    radius = float(scene.parameters.get("detail_radius_studs", 0))
    if budget <= 0 or radius <= 0 or not scene.fields:
        return []
    studs = float(scene.parameters["studs_per_meter"])
    origin_x = float(scene.parameters["origin_x_studs"])
    origin_z = float(scene.parameters["origin_z_studs"])
    east, north = scene.grid.local_axes()
    surface = np.asarray(scene.surface_studs(scene.heights_m), dtype=np.float64)
    candidates: list[tuple[float, float, float, int, int, Any]] = []
    for parcel in scene.fields:
        mask = scene.labels == parcel.label
        rows, cols = np.nonzero(mask)
        for row, col in zip(rows.tolist(), cols.tolist()):
            x = float(east[row, col]) * studs + origin_x
            z = -float(north[row, col]) * studs + origin_z
            if (x - origin_x) ** 2 + (z - origin_z) ** 2 > radius ** 2:
                continue
            candidates.append((x, z, float(surface[row, col]), row, col, parcel))
    if not candidates:
        return []
    if len(candidates) > budget:
        stride = math.ceil(len(candidates) / budget)
        candidates = candidates[::stride][:budget]
    cell_x = max(2.0, scene.grid.cell_width_m * studs * 0.92)
    cell_z = max(2.0, scene.grid.cell_height_m * studs * 0.92)
    specs = []
    for index, (x, z, y, _row, _col, parcel) in enumerate(candidates, start=1):
        shade = 1.0
        if parcel.furrow and parcel.spacing_m > 0:
            theta = math.radians(parcel.orientation_deg)
            across = ((x - origin_x) / studs) * math.cos(theta) - (-(z - origin_z) / studs) * math.sin(theta)
            shade = 0.84 + 0.16 * (0.5 + 0.5 * math.sin(across * 2 * math.pi / parcel.spacing_m))
        color = tuple(max(0.0, min(1.0, channel * shade)) for channel in parcel.color)
        specs.append(
            {
                "parent": f"{FEATURES}.Fields.{parcel.id}",
                "name": f"patch_{index:04d}",
                "class_name": "Part",
                "properties": {
                    "Anchored": True,
                    "CanCollide": False,
                    "Size": {"type": "Vector3", "x": round(cell_x, 2), "y": 0.35, "z": round(cell_z, 2)},
                    "CFrame": _position(x, y + 0.2, z),
                    "Color": {"type": "Color3", "r": color[0], "g": color[1], "b": color[2], "scale": "unit"},
                    "Material": {"type": "Enum", "value": f"Enum.Material.{parcel.material}"},
                },
                "attributes": {"Crop": parcel.crop, "FieldId": parcel.id},
            }
        )
    return specs


def _tree_specs(scene: TerrainScene, prop, index: int) -> tuple[dict, dict]:
    studs = float(scene.parameters["studs_per_meter"])
    x, y, z = _prop_studs(scene, prop.east_m, prop.north_m, prop.elevation_m)
    height = prop.height_m * prop.scale * studs
    trunk_h = max(2.0, height * 0.42)
    diameter = max(0.6, height * 0.08)
    canopy = max(3.0, height * 0.55)
    name = f"tree_{index:04d}"
    trunk = {
        "parent": f"{FEATURES}.Vegetation",
        "name": name,
        "class_name": "Part",
        "properties": {
            "Anchored": True,
            "CanCollide": True,
            "Shape": {"type": "Enum", "value": "Enum.PartType.Cylinder"},
            "Size": {"type": "Vector3", "x": round(trunk_h, 2), "y": round(diameter, 2), "z": round(diameter, 2)},
            "CFrame": _upright_cylinder(x, y + trunk_h / 2, z),
            "Color": {"type": "Color3", "r": 0.32, "g": 0.24, "b": 0.16, "scale": "unit"},
            "Material": {"type": "Enum", "value": "Enum.Material.Wood"},
        },
        "attributes": {"Species": prop.species, "Kind": "trunk"},
    }
    leaf = {
        "parent": f"{FEATURES}.Vegetation.{name}",
        "name": "Canopy",
        "class_name": "Part",
        "properties": {
            "Anchored": True,
            "CanCollide": False,
            "Shape": {"type": "Enum", "value": "Enum.PartType.Ball"},
            "Size": {"type": "Vector3", "x": round(canopy, 2), "y": round(canopy * 0.85, 2), "z": round(canopy, 2)},
            "CFrame": _position(x, y + trunk_h + canopy * 0.28, z),
            "Color": {"type": "Color3", "r": prop.color[0], "g": prop.color[1], "b": prop.color[2], "scale": "unit"},
            "Material": {"type": "Enum", "value": "Enum.Material.Grass"},
        },
        "attributes": {"Species": prop.species, "Kind": "canopy"},
    }
    return trunk, leaf


def _boundary_spec(scene: TerrainScene, prop, index: int) -> dict:
    studs = float(scene.parameters["studs_per_meter"])
    x, y, z = _prop_studs(scene, prop.east_m, prop.north_m, prop.elevation_m)
    length = max(2.0, prop.scale * studs)
    height = max(1.2, prop.height_m * studs)
    thickness = 1.6 if prop.kind == "hedge" else 0.35
    yaw = math.radians(prop.yaw_deg)
    local_x = (math.sin(yaw), 0.0, -math.cos(yaw))
    return {
        "parent": f"{FEATURES}.Boundaries",
        "name": f"{prop.kind}_{index:04d}",
        "class_name": "Part",
        "properties": {
            "Anchored": True,
            "CanCollide": prop.kind == "hedge",
            "Size": {"type": "Vector3", "x": round(length, 2), "y": round(height, 2), "z": thickness},
            "CFrame": _yawed(x, y + height / 2, z, local_x),
            "Color": {"type": "Color3", "r": prop.color[0], "g": prop.color[1], "b": prop.color[2], "scale": "unit"},
            "Material": {
                "type": "Enum",
                "value": "Enum.Material.Grass" if prop.kind == "hedge" else "Enum.Material.Wood",
            },
        },
        "attributes": {"Kind": prop.kind},
    }


def _rock_spec(scene: TerrainScene, prop, index: int) -> dict:
    studs = float(scene.parameters["studs_per_meter"])
    x, y, z = _prop_studs(scene, prop.east_m, prop.north_m, prop.elevation_m)
    size = max(1.0, prop.scale * studs)
    height = max(0.8, prop.height_m * studs)
    return {
        "parent": f"{FEATURES}.Rocks",
        "name": f"rock_{index:04d}",
        "class_name": "Part",
        "properties": {
            "Anchored": True,
            "CanCollide": True,
            "Size": {"type": "Vector3", "x": round(size, 2), "y": round(height, 2), "z": round(size * 0.8, 2)},
            "CFrame": _position(x, y + height * 0.25, z),
            "Color": {"type": "Color3", "r": prop.color[0], "g": prop.color[1], "b": prop.color[2], "scale": "unit"},
            "Material": {"type": "Enum", "value": "Enum.Material.Rock" if prop.species == "boulder" else "Enum.Material.Limestone"},
        },
        "attributes": {"Kind": "rock"},
    }


def _marker(scene: TerrainScene) -> dict | None:
    dry = scene.water_depth_m < 0.3
    if not np.any(dry):
        return None
    east, north = scene.grid.local_axes()
    distance = np.where(dry, east**2 + north**2, np.inf)
    index = int(np.argmin(distance))
    if not np.isfinite(distance.ravel()[index]):
        return None
    cols = east.shape[1]
    row, col = divmod(index, cols)
    x, y, z = _prop_studs(scene, float(east[row, col]), float(north[row, col]), float(scene.heights_m[row, col]))
    return {
        "parent": FEATURES,
        "name": "TerrainMarker",
        "class_name": "Part",
        "properties": {
            "Anchored": True,
            "CanCollide": False,
            "Size": {"type": "Vector3", "x": 1.5, "y": 8, "z": 1.5},
            "CFrame": _position(x, y + 6, z),
            "Color": {"type": "Color3", "r": 0.93, "g": 0.93, "b": 0.9, "scale": "unit"},
            "Material": {"type": "Enum", "value": "Enum.Material.SmoothPlastic"},
        },
        "attributes": {"Role": "marker", "Preset": scene.preset},
    }


def _prop_studs(scene: TerrainScene, east_m: float, north_m: float, elevation_m: float) -> tuple[float, float, float]:
    studs = float(scene.parameters["studs_per_meter"])
    origin_x = float(scene.parameters["origin_x_studs"])
    origin_z = float(scene.parameters["origin_z_studs"])
    x = east_m * studs + origin_x
    z = -north_m * studs + origin_z
    y = float(np.asarray(scene.surface_studs(elevation_m)))
    return x, y, z


def _folder(parent: str, name: str, attributes: dict) -> dict:
    return {
        "parent": parent,
        "name": name,
        "class_name": "Folder",
        "attributes": attributes,
    }


def _root_attributes(scene: TerrainScene, min_c: tuple[float, float, float], max_c: tuple[float, float, float]) -> dict:
    return {
        "PluginId": "terrain",
        "Schema": 1,
        "Preset": scene.preset,
        "Seed": int(scene.seed),
        "FieldCount": len(scene.fields),
        "RealPlace": bool(scene.real_place),
        "VoxelMinX": min_c[0],
        "VoxelMinY": min_c[1],
        "VoxelMinZ": min_c[2],
        "VoxelMaxX": max_c[0],
        "VoxelMaxY": max_c[1],
        "VoxelMaxZ": max_c[2],
    }


def _position(x: float, y: float, z: float) -> dict:
    return {"type": "CFrame", "x": round(x, 3), "y": round(y, 3), "z": round(z, 3)}


def _upright_cylinder(x: float, y: float, z: float) -> dict:
    # Cylinder parts are long along local X. Point local X up.
    return _yawed(x, y, z, (0.0, 1.0, 0.0), up=(1.0, 0.0, 0.0), forward=(0.0, 0.0, -1.0))


def _yawed(
    x: float,
    y: float,
    z: float,
    local_x: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    forward: tuple[float, float, float] | None = None,
) -> dict:
    axis = _unit(local_x)
    if forward is None:
        # local_x × local_y, with local_y = up, for a horizontal hedge.
        forward = _unit((axis[1] * up[2] - axis[2] * up[1], axis[2] * up[0] - axis[0] * up[2], axis[0] * up[1] - axis[1] * up[0]))
        if abs(forward[0]) + abs(forward[1]) + abs(forward[2]) < 1e-6:
            forward = (0.0, 0.0, -1.0)
    else:
        forward = _unit(forward)
    components = [
        round(x, 3),
        round(y, 3),
        round(z, 3),
        axis[0],
        up[0],
        forward[0],
        axis[1],
        up[1],
        forward[1],
        axis[2],
        up[2],
        forward[2],
    ]
    return {"type": "CFrame", "components": components}


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) or 1.0
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _create_root(port: Any, scene: TerrainScene, plan: VoxelPlan) -> None:
    spec = _folder("Workspace", "RealisticTerrain", _root_attributes(scene, plan.min_corner, plan.max_corner))
    port.apply_instances([spec])


def _clear_existing(port: Any) -> None:
    payload = port.execute(_clear_script())
    if isinstance(payload, dict) and payload.get("ok") is False:
        raise ApplyError(f"Could not clear a previous RealisticTerrain: {payload}")


def _as_port(port: Any, progress: Progress) -> Any:
    if hasattr(port, "execute") and hasattr(port, "apply_instances") and hasattr(port, "ensure_edit"):
        return port
    if hasattr(port, "luau") and hasattr(port, "datamodel"):
        from roblox_studio_plugin_host.gateway import AdapterPort

        return AdapterPort(port, progress)
    raise TerrainError(
        "apply_scene needs a Studio port (execute, apply_instances, ensure_edit) "
        "or a connected RobloxStudio agent."
    )


def _json_backup(payload: dict) -> dict:
    backup = payload.get("backup") or {}
    components = backup.get("components") or []
    if isinstance(components, dict):
        components = [components[str(index)] for index in range(1, 13) if str(index) in components]
    return {
        "found": True,
        "transparency": backup.get("transparency", 0),
        "can_collide": backup.get("can_collide", True),
        "components": components,
    }


def _dress_materials_script() -> str:
    """Tint this place's terrain toward meadow, soil, limestone, and river water.

    The previous colors are stored on RealisticTerrain so clear can put them back.
    SetMaterialColor is place-wide. That is the trade for a ground that is not
    the engine's default plastic green.
    """
    return """
local HttpService = game:GetService("HttpService")
local terrain = workspace.Terrain
local folder = workspace:FindFirstChild("RealisticTerrain")
local targets = {
    Grass = {106, 122, 68},
    LeafyGrass = {78, 104, 52},
    Ground = {128, 108, 74},
    Mud = {90, 76, 58},
    Rock = {132, 128, 120},
    Sandstone = {174, 156, 122},
    Limestone = {186, 176, 152},
    Slate = {112, 114, 110},
    Sand = {190, 174, 136},
    Water = {58, 104, 114},
}
local previous = {}
if folder and not folder:FindFirstChild("MaterialColorBackup") then
    for name in pairs(targets) do
        local matOk, mat = pcall(function()
            return Enum.Material[name]
        end)
        if matOk then
            local colorOk, color = pcall(function()
                return terrain:GetMaterialColor(mat)
            end)
            if colorOk then
                previous[name] = { color.R, color.G, color.B }
            end
        end
    end
    local water = {
        color = { terrain.WaterColor.R, terrain.WaterColor.G, terrain.WaterColor.B },
        transparency = terrain.WaterTransparency,
        reflectance = terrain.WaterReflectance,
        waveSize = terrain.WaterWaveSize,
        waveSpeed = terrain.WaterWaveSpeed,
    }
    local record = Instance.new("StringValue")
    record.Name = "MaterialColorBackup"
    record.Value = HttpService:JSONEncode({ materials = previous, water = water })
    record.Parent = folder
end
for name, rgb in pairs(targets) do
    local matOk, mat = pcall(function()
        return Enum.Material[name]
    end)
    if matOk then
        pcall(function()
            terrain:SetMaterialColor(mat, Color3.fromRGB(rgb[1], rgb[2], rgb[3]))
        end)
    end
end
terrain.WaterColor = Color3.fromRGB(58, 104, 114)
terrain.WaterTransparency = 0.2
terrain.WaterReflectance = 0.3
terrain.WaterWaveSize = 0.06
terrain.WaterWaveSpeed = 6
task.wait()
return HttpService:JSONEncode({ ok = true, dressed = true })
""".strip()


def _hide_baseplate_script() -> str:
    return """
local HttpService = game:GetService("HttpService")
local plate = workspace:FindFirstChild("Baseplate")
if plate == nil or not plate:IsA("BasePart") then
    return HttpService:JSONEncode({ ok = true, found = false })
end
local components = { plate.CFrame:GetComponents() }
local backup = {
    transparency = plate.Transparency,
    can_collide = plate.CanCollide,
    components = components,
}
plate.Transparency = 1
plate.CanCollide = false
plate.CFrame = CFrame.new(components[1], components[2] - 400, components[3])
return HttpService:JSONEncode({ ok = true, found = true, backup = backup })
""".strip()


def _clear_script() -> str:
    return """
local HttpService = game:GetService("HttpService")
local folder = workspace:FindFirstChild("RealisticTerrain")
if folder == nil then
    return HttpService:JSONEncode({ ok = true, cleared = false, baseplate_restored = false })
end
local function num(name)
    local value = folder:GetAttribute(name)
    if typeof(value) == "number" then
        return value
    end
    return nil
end
local minX, minY, minZ = num("VoxelMinX"), num("VoxelMinY"), num("VoxelMinZ")
local maxX, maxY, maxZ = num("VoxelMaxX"), num("VoxelMaxY"), num("VoxelMaxZ")
if minX and minY and minZ and maxX and maxY and maxZ and maxX > minX and maxY > minY and maxZ > minZ then
    local size = Vector3.new(maxX - minX, maxY - minY, maxZ - minZ)
    local center = Vector3.new((minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2)
    workspace.Terrain:FillBlock(CFrame.new(center), size, Enum.Material.Air)
    task.wait()
end
local colors = folder:FindFirstChild("MaterialColorBackup")
if colors and colors:IsA("StringValue") and colors.Value ~= "" then
    local decoded, data = pcall(function()
        return HttpService:JSONDecode(colors.Value)
    end)
    if decoded and type(data) == "table" then
        local terrain = workspace.Terrain
        if type(data.materials) == "table" then
            for name, rgb in pairs(data.materials) do
                local matOk, mat = pcall(function()
                    return Enum.Material[name]
                end)
                if matOk and type(rgb) == "table" and rgb[1] and rgb[2] and rgb[3] then
                    pcall(function()
                        terrain:SetMaterialColor(mat, Color3.new(rgb[1], rgb[2], rgb[3]))
                    end)
                end
            end
        end
        if type(data.water) == "table" then
            local water = data.water
            pcall(function()
                if type(water.color) == "table" then
                    terrain.WaterColor = Color3.new(water.color[1], water.color[2], water.color[3])
                end
                if type(water.transparency) == "number" then
                    terrain.WaterTransparency = water.transparency
                end
                if type(water.reflectance) == "number" then
                    terrain.WaterReflectance = water.reflectance
                end
                if type(water.waveSize) == "number" then
                    terrain.WaterWaveSize = water.waveSize
                end
                if type(water.waveSpeed) == "number" then
                    terrain.WaterWaveSpeed = water.waveSpeed
                end
            end)
        end
    end
end
local restored = false
local backupValue = folder:FindFirstChild("BaseplateBackup")
if backupValue and backupValue:IsA("StringValue") and backupValue.Value ~= "" then
    local ok, data = pcall(function()
        return HttpService:JSONDecode(backupValue.Value)
    end)
    local plate = workspace:FindFirstChild("Baseplate")
    if ok and type(data) == "table" and data.found and plate and plate:IsA("BasePart") and type(data.components) == "table" then
        local c = data.components
        if #c >= 12 then
            plate.CFrame = CFrame.new(c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12])
            plate.Transparency = data.transparency or 0
            plate.CanCollide = data.can_collide ~= false
            restored = true
        end
    end
end
folder:Destroy()
task.wait()
return HttpService:JSONEncode({ ok = true, cleared = true, baseplate_restored = restored })
""".strip()
