"""Turn a heightfield into chunked Terrain:WriteVoxels programs.

Roblox voxels are 4 studs on a side, and one WriteVoxels call cannot exceed
4,194,304 voxels. Chunks stay far under that, and each program calls
``task.wait`` so Studio's main thread can breathe between columns. The macro
shape is still the DEM. Sub-voxel furrows are carried by parts, not by pretending
the voxel grid is finer than the engine allows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from roblox_terrain.errors import ValidationError
from roblox_terrain.scene import TerrainScene
from roblox_terrain.surface import MATERIAL_INDEX, MATERIALS

VOXEL_STUDS = 4
CHUNK_COLUMNS = 36
MAX_VOXELS_PER_CHUNK = 1_000_000


@dataclass(frozen=True)
class VoxelChunk:
    index: int
    min_x: float
    min_y: float
    min_z: float
    size_x: int
    size_y: int
    size_z: int
    payload: str

    @property
    def voxels(self) -> int:
        return self.size_x * self.size_y * self.size_z

    @property
    def max_corner(self) -> tuple[float, float, float]:
        return (
            self.min_x + self.size_x * VOXEL_STUDS,
            self.min_y + self.size_y * VOXEL_STUDS,
            self.min_z + self.size_z * VOXEL_STUDS,
        )


@dataclass(frozen=True)
class VoxelPlan:
    chunks: list[VoxelChunk]
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]
    voxels_x: int
    voxels_z: int

    @property
    def voxels(self) -> int:
        return sum(chunk.voxels for chunk in self.chunks)


def estimate_cost(
    extent_east_m: float,
    extent_north_m: float,
    relief_studs: float,
    *,
    studs_per_meter: float,
) -> dict[str, int | list[int]]:
    voxels_x = max(1, int(round(extent_east_m * studs_per_meter / VOXEL_STUDS)))
    voxels_z = max(1, int(round(extent_north_m * studs_per_meter / VOXEL_STUDS)))
    voxels_y = max(4, int(math.ceil((relief_studs + 24) / VOXEL_STUDS)))
    chunks = math.ceil(voxels_x / CHUNK_COLUMNS) * math.ceil(voxels_z / CHUNK_COLUMNS)
    return {
        "voxels_xy": [voxels_x, voxels_z],
        "voxels_y": voxels_y,
        "voxels": voxels_x * voxels_z * voxels_y,
        "voxel_chunks": chunks,
    }


def plan_voxels(scene: TerrainScene) -> VoxelPlan:
    """Resample the scene onto the 4-stud grid and pack column chunks."""
    studs = float(scene.parameters["studs_per_meter"])
    exaggeration = float(scene.parameters["vertical_exaggeration"])
    voxels_x = max(1, int(round(scene.grid.extent_east_m * studs / VOXEL_STUDS)))
    voxels_z = max(1, int(round(scene.grid.extent_north_m * studs / VOXEL_STUDS)))
    heights = _resample(scene.heights_m, voxels_z, voxels_x, nearest=False)
    water_m = _resample(scene.water_depth_m, voxels_z, voxels_x, nearest=False)
    materials = _resample(scene.materials, voxels_z, voxels_x, nearest=True).astype(np.int32)
    surface = np.asarray(scene.surface_studs(heights), dtype=np.float64)
    skirt = int(scene.parameters.get("edge_skirt_voxels", 20) or 0)
    surface, materials, water_m = present_columns(surface, materials, water_m, skirt_voxels=skirt)
    water_surface = surface + np.maximum(water_m, 0) * exaggeration * studs
    base_y = _align_down(float(np.min(surface)) - 12.0)
    origin_x = _align_down(float(scene.parameters["origin_x_studs"]) - (voxels_x * VOXEL_STUDS) / 2)
    origin_z = _align_down(float(scene.parameters["origin_z_studs"]) - (voxels_z * VOXEL_STUDS) / 2)
    chunks: list[VoxelChunk] = []
    index = 0
    for z0 in range(0, voxels_z, CHUNK_COLUMNS):
        for x0 in range(0, voxels_x, CHUNK_COLUMNS):
            x1 = min(voxels_x, x0 + CHUNK_COLUMNS)
            z1 = min(voxels_z, z0 + CHUNK_COLUMNS)
            local_surface = surface[z0:z1, x0:x1]
            local_water = water_surface[z0:z1, x0:x1]
            local_depth = water_m[z0:z1, x0:x1]
            local_mat = materials[z0:z1, x0:x1]
            top = max(float(local_surface.max()), float(local_water.max()))
            max_y = _align_up(top) + VOXEL_STUDS
            size_y = max(4, int(round((max_y - base_y) / VOXEL_STUDS)))
            size_x = x1 - x0
            size_z = z1 - z0
            if size_x * size_y * size_z > MAX_VOXELS_PER_CHUNK:
                raise ValidationError(
                    "One terrain chunk would exceed 1,000,000 voxels. "
                    "Lower vertical_exaggeration or studs_per_meter."
                )
            payload = _pack(local_surface, local_water, local_depth, local_mat, base_y, size_y)
            chunks.append(
                VoxelChunk(
                    index=index,
                    min_x=origin_x + x0 * VOXEL_STUDS,
                    min_y=base_y,
                    min_z=origin_z + z0 * VOXEL_STUDS,
                    size_x=size_x,
                    size_y=size_y,
                    size_z=size_z,
                    payload=payload,
                )
            )
            index += 1
    max_x = max(chunk.max_corner[0] for chunk in chunks)
    max_y = max(chunk.max_corner[1] for chunk in chunks)
    max_z = max(chunk.max_corner[2] for chunk in chunks)
    return VoxelPlan(
        chunks=chunks,
        min_corner=(min(chunk.min_x for chunk in chunks), base_y, min(chunk.min_z for chunk in chunks)),
        max_corner=(max_x, max_y, max_z),
        voxels_x=voxels_x,
        voxels_z=voxels_z,
    )


def chunk_script(chunk: VoxelChunk) -> str:
    """One yielding Luau program. It writes a single voxel box and returns JSON."""
    names = ", ".join(f"Enum.Material.{name}" for name in MATERIALS)
    return f"""
local HttpService = game:GetService("HttpService")
local terrain = workspace.Terrain
local resolution = 4
local minX = {chunk.min_x}
local minY = {chunk.min_y}
local minZ = {chunk.min_z}
local sizeX = {chunk.size_x}
local sizeY = {chunk.size_y}
local sizeZ = {chunk.size_z}
local payload = "{chunk.payload}"
local names = {{ {names} }}
local materials = {{}}
local occupancy = {{}}
for x = 1, sizeX do
    materials[x] = {{}}
    occupancy[x] = {{}}
    for y = 1, sizeY do
        materials[x][y] = table.create(sizeZ, Enum.Material.Air)
        occupancy[x][y] = table.create(sizeZ, 0)
    end
end
local function u16(at)
    return tonumber(string.sub(payload, at, at + 3), 16)
end
local function u8(at)
    return tonumber(string.sub(payload, at, at + 1), 16)
end
local column = 0
for z = 1, sizeZ do
    for x = 1, sizeX do
        local at = column * 12 + 1
        local bed = u16(at)
        local water = u16(at + 4)
        local surfaceMat = names[(u8(at + 8) or 0) + 1]
        local fillMat = names[(u8(at + 10) or 0) + 1]
        for y = 1, sizeY do
            local bottom = (y - 1) * 8
            local top = y * 8
            if top <= bed then
                materials[x][y][z] = ((bed - top) > 24) and Enum.Material.Rock or fillMat
                occupancy[x][y][z] = 1
            elseif bottom < bed then
                materials[x][y][z] = surfaceMat
                occupancy[x][y][z] = math.clamp((bed - bottom) / 8, 0.2, 1)
            elseif water > bed and top <= water then
                materials[x][y][z] = Enum.Material.Water
                occupancy[x][y][z] = 1
            elseif water > bed and bottom < water then
                materials[x][y][z] = Enum.Material.Water
                occupancy[x][y][z] = math.clamp((water - bottom) / 8, 0.25, 1)
            end
        end
        column += 1
        if column % 6 == 0 then
            task.wait()
        end
    end
end
local region = Region3.new(
    Vector3.new(minX, minY, minZ),
    Vector3.new(minX + sizeX * resolution, minY + sizeY * resolution, minZ + sizeZ * resolution)
)
terrain:WriteVoxels(region, resolution, materials, occupancy)
task.wait()
return HttpService:JSONEncode({{ ok = true, chunk = {chunk.index}, voxels = sizeX * sizeY * sizeZ }})
""".strip()


def present_columns(
    surface: np.ndarray,
    materials: np.ndarray,
    water_depth: np.ndarray,
    *,
    skirt_voxels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn the analysis grid into a voxel surface.

    The play window is a rectangle. Filling every column up to the DEM and
    stopping at that rectangle leaves a sheer dirt wall. The outer columns
    ease down to a low contour so the edge is a bank. A channel that already
    reaches the rim keeps its bed and its water. Dry neighbors are not filled
    in to fake a wider river.
    """
    shaped = np.asarray(surface, dtype=np.float64).copy()
    painted = np.asarray(materials).copy()
    water = np.asarray(water_depth, dtype=np.float64).copy()
    channel = water >= 0.15
    rows, cols = shaped.shape
    if skirt_voxels > 0 and rows > 2 and cols > 2:
        width = min(int(skirt_voxels), (min(rows, cols) - 1) // 2)
        row_dist = np.minimum(np.arange(rows), np.arange(rows)[::-1])
        col_dist = np.minimum(np.arange(cols), np.arange(cols)[::-1])
        distance = np.minimum(row_dist[:, None], col_dist[None, :])
        blend = np.clip(distance / max(width, 1), 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        floor = float(np.percentile(shaped, 12))
        eased = floor + (shaped - floor) * blend
        shaped = np.where(channel, shaped, eased)
        skirt = distance < width
        toe = distance <= max(2, width // 5)
        bank = skirt & ((np.asarray(surface) - floor) > 16.0) & (distance < width * 0.55)
        painted[toe & ~channel] = MATERIAL_INDEX["Ground"]
        painted[bank & ~channel] = MATERIAL_INDEX["Rock"]
        water[skirt & ~channel] = 0.0
    return shaped.astype(np.float64), painted, water.astype(np.float64)


def decode_payload(payload: str, size_x: int, size_z: int) -> list[tuple[int, int, int, int]]:
    """Test helper. Columns are z-major, then x, matching the Luau loops."""
    expected = size_x * size_z * 12
    if len(payload) != expected:
        raise ValueError(f"payload length {len(payload)} != {expected}")
    columns = []
    for index in range(size_x * size_z):
        at = index * 12
        columns.append(
            (
                int(payload[at : at + 4], 16),
                int(payload[at + 4 : at + 8], 16),
                int(payload[at + 8 : at + 10], 16),
                int(payload[at + 10 : at + 12], 16),
            )
        )
    return columns


def _pack(
    surface: np.ndarray,
    water_surface: np.ndarray,
    water_depth: np.ndarray,
    materials: np.ndarray,
    base_y: float,
    size_y: int,
) -> str:
    mud = MATERIAL_INDEX["Mud"]
    water_id = MATERIAL_INDEX["Water"]
    limit = size_y * 8 - 1
    parts: list[str] = []
    size_z, size_x = surface.shape
    for z in range(size_z):
        for x in range(size_x):
            bed = int(round((float(surface[z, x]) - base_y) / VOXEL_STUDS * 8))
            bed = max(8, min(limit, bed))
            if float(water_depth[z, x]) >= 0.25:
                water = int(round((float(water_surface[z, x]) - base_y) / VOXEL_STUDS * 8))
                water = max(bed + 1, min(limit, water))
            else:
                water = 0
            surface_mat = int(materials[z, x])
            if surface_mat == water_id:
                surface_mat = mud
            fill_mat = surface_mat
            parts.append(f"{bed:04x}{water:04x}{surface_mat:02x}{fill_mat:02x}")
    return "".join(parts)


def _resample(array: np.ndarray, rows: int, cols: int, *, nearest: bool) -> np.ndarray:
    source = np.asarray(array)
    if source.shape == (rows, cols):
        return source
    ys = np.linspace(0, source.shape[0] - 1, rows)
    xs = np.linspace(0, source.shape[1] - 1, cols)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    if nearest:
        r = np.clip(np.rint(yy), 0, source.shape[0] - 1).astype(np.int32)
        c = np.clip(np.rint(xx), 0, source.shape[1] - 1).astype(np.int32)
        return source[r, c]
    return _bilinear(source.astype(np.float64), yy, xx)


def _bilinear(source: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    r0 = np.clip(np.floor(row).astype(np.int32), 0, source.shape[0] - 1)
    c0 = np.clip(np.floor(col).astype(np.int32), 0, source.shape[1] - 1)
    r1 = np.clip(r0 + 1, 0, source.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, source.shape[1] - 1)
    fr = row - r0
    fc = col - c0
    return (
        source[r0, c0] * (1 - fr) * (1 - fc)
        + source[r0, c1] * (1 - fr) * fc
        + source[r1, c0] * fr * (1 - fc)
        + source[r1, c1] * fr * fc
    )


def _align_down(value: float) -> float:
    return math.floor(value / VOXEL_STUDS) * VOXEL_STUDS


def _align_up(value: float) -> float:
    return math.ceil(value / VOXEL_STUDS) * VOXEL_STUDS
