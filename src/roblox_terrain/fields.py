"""Agricultural parcels.

Mapped polygons are kept when they are already about one field. Larger
farmland and cropland that has no polygon are split into parcels whose long
axis follows the contour, so the pattern agrees with the drainage instead of
with the screen. Rectangular presets stay on a straight grid. Irregular
presets stagger and warp that grid so hedged countryside is not a chessboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from roblox_terrain.cover import AGRICULTURAL, BY_NAME, NAMES
from roblox_terrain.grid import GeoGrid
from roblox_terrain.hydrology import gradients
from roblox_terrain.raster import paint_priority, rasterize_polygon
from roblox_terrain.vectors import VectorSet

CROPS = {
    "wheat": ("Ground", (0.79, 0.68, 0.34), True, 11.0),
    "barley": ("Sandstone", (0.66, 0.58, 0.30), True, 10.0),
    "rapeseed": ("LeafyGrass", (0.86, 0.80, 0.22), False, 14.0),
    "pasture": ("Grass", (0.34, 0.52, 0.26), False, 18.0),
    "fallow": ("Mud", (0.46, 0.36, 0.26), True, 12.0),
    "roots": ("Mud", (0.40, 0.32, 0.22), True, 9.0),
    "maize": ("Grass", (0.28, 0.48, 0.16), True, 8.0),
    "orchard": ("Grass", (0.36, 0.50, 0.24), False, 16.0),
}


@dataclass
class FieldParcel:
    id: str
    label: int
    crop: str
    material: str
    color: tuple[float, float, float]
    furrow: bool
    spacing_m: float
    orientation_deg: float
    boundary: str
    area_m2: float
    centroid_east: float
    centroid_north: float
    polygon: list[tuple[float, float]] = field(default_factory=list)
    cover_hint: str = "cropland"


@dataclass
class BoundaryLine:
    kind: str
    points: list[tuple[float, float]]


@dataclass
class FieldLayer:
    labels: np.ndarray
    fields: list[FieldParcel]
    boundaries: list[BoundaryLine]
    mapped_fields: int
    subdivided_fields: int


def build_fields(
    heights: np.ndarray,
    cover: np.ndarray,
    vectors: VectorSet,
    grid: GeoGrid,
    *,
    field_style: str,
    boundary_style: str,
    field_size_m: float,
    crops: tuple[str, ...],
    seed: int,
) -> FieldLayer:
    east, north = grid.local_axes()
    labels = np.zeros(heights.shape, dtype=np.int32)
    hints: dict[int, str] = {}
    orientations: dict[int, float] = {}
    mapped = 0
    next_id = 1
    subdivide = np.zeros(heights.shape, dtype=bool)
    agricultural = [poly for poly in vectors.polygons if BY_NAME.get(poly.cover) in AGRICULTURAL]
    painted: list[tuple[float, np.ndarray, str]] = []
    for poly in agricultural:
        mask = rasterize_polygon(grid, poly.rings)
        area = float(mask.sum()) * grid.cell_width_m * grid.cell_height_m
        painted.append((area, mask, poly.cover))
    painted.sort(key=lambda item: item[0])
    single_limit = field_size_m * field_size_m * 1.8
    for area, mask, cover_name in painted:
        claim = mask & (labels == 0) & ~subdivide
        cells = int(claim.sum())
        if cells < 3:
            continue
        if area <= single_limit:
            labels[claim] = next_id
            hints[next_id] = cover_name
            orientations[next_id] = _orientation(heights, claim, east, north, grid)
            next_id += 1
            mapped += 1
        else:
            subdivide[claim] = True
    cover_agri = np.isin(cover, AGRICULTURAL)
    subdivide |= cover_agri & (labels == 0)
    subdivided = 0
    if np.any(subdivide):
        for component in _components(subdivide, min_cells=4):
            angle = _orientation(heights, component, east, north, grid)
            parcels = _bin_component(
                component,
                east,
                north,
                angle,
                field_size_m=field_size_m,
                style=field_style,
                seed=seed + next_id,
            )
            for parcel_mask in parcels:
                if int(parcel_mask.sum()) < 3:
                    continue
                labels[parcel_mask] = next_id
                majority = _majority_cover(cover, parcel_mask)
                hints[next_id] = NAMES.get(majority, "cropland")
                orientations[next_id] = angle
                next_id += 1
                subdivided += 1
    labels = _merge_slivers(labels, min_cells=max(3, int(0.15 * field_size_m * field_size_m / max(grid.cell_width_m * grid.cell_height_m, 1))))
    fields = _records(
        labels,
        hints,
        orientations,
        east,
        north,
        grid,
        boundary_style=boundary_style,
        crops=crops,
        seed=seed,
    )
    boundaries = _boundaries(labels, east, north, grid, boundary_style, fields)
    return FieldLayer(
        labels=labels,
        fields=fields,
        boundaries=boundaries,
        mapped_fields=mapped,
        subdivided_fields=subdivided,
    )


def _orientation(heights: np.ndarray, mask: np.ndarray, east: np.ndarray, north: np.ndarray, grid: GeoGrid) -> float:
    down_east, down_north, _slope = gradients(heights, grid.cell_height_m, grid.cell_width_m)
    weight = mask.astype(np.float64)
    total = float(weight.sum()) or 1.0
    mean_east = float((down_east * weight).sum() / total)
    mean_north = float((down_north * weight).sum() / total)
    if math.hypot(mean_east, mean_north) < 1e-4:
        return 0.0
    # Long axis follows the contour: perpendicular to downhill, clockwise from north.
    contour_east = -mean_north
    contour_north = mean_east
    return math.degrees(math.atan2(contour_east, contour_north)) % 180.0


def _bin_component(
    component: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    angle_deg: float,
    *,
    field_size_m: float,
    style: str,
    seed: int,
) -> list[np.ndarray]:
    theta = math.radians(angle_deg)
    along = east * math.sin(theta) + north * math.cos(theta)
    across = east * math.cos(theta) - north * math.sin(theta)
    irregular = 0.0 if style == "rectangular" else 1.0
    across = across + irregular * 12.0 * np.sin(along / 85.0 + seed * 0.01)
    along = along + irregular * 8.0 * np.sin(across / 70.0)
    along_pitch = field_size_m * (1.45 if style == "rectangular" else 1.05)
    across_pitch = field_size_m * (0.42 if style == "rectangular" else 0.78)
    row_bin = np.floor(across / across_pitch).astype(np.int32)
    stagger = (row_bin % 2) * along_pitch * (0.0 if style == "rectangular" else 0.4)
    col_bin = np.floor((along + stagger) / along_pitch).astype(np.int32)
    keys = np.zeros(component.shape, dtype=np.int32)
    rows, cols = component.shape
    lookup: dict[tuple[int, int], int] = {}
    next_key = 1
    ys, xs = np.nonzero(component)
    for row, col in zip(ys.tolist(), xs.tolist()):
        key = (int(row_bin[row, col]), int(col_bin[row, col]))
        ident = lookup.get(key)
        if ident is None:
            ident = next_key
            lookup[key] = ident
            next_key += 1
        keys[row, col] = ident
    masks = []
    for ident in range(1, next_key):
        mask = keys == ident
        # A bin can touch opposite sides of a wood. Split those.
        for piece in _components(mask, min_cells=3):
            masks.append(piece)
    del rows, cols
    return masks


def _records(
    labels: np.ndarray,
    hints: dict[int, str],
    orientations: dict[int, float],
    east: np.ndarray,
    north: np.ndarray,
    grid: GeoGrid,
    *,
    boundary_style: str,
    crops: tuple[str, ...],
    seed: int,
) -> list[FieldParcel]:
    ids = [int(item) for item in np.unique(labels) if int(item) > 0]
    drafts = []
    cell_area = grid.cell_width_m * grid.cell_height_m
    for label in ids:
        mask = labels == label
        area = float(mask.sum()) * cell_area
        centroid_east = float(east[mask].mean()) if np.any(mask) else 0.0
        centroid_north = float(north[mask].mean()) if np.any(mask) else 0.0
        drafts.append((centroid_north, centroid_east, label, area, mask))
    drafts.sort(key=lambda item: (-item[0], item[1], item[2]))
    rng = np.random.default_rng(seed)
    catalog = crops or ("wheat", "pasture")
    parcels: list[FieldParcel] = []
    for index, (_cn, _ce, label, area, mask) in enumerate(drafts, start=1):
        hint = hints.get(label, "cropland")
        crop = _crop_for(hint, catalog, rng)
        material, color, furrow, spacing = CROPS[crop]
        polygon = _outline(mask, east, north, grid)
        parcels.append(
            FieldParcel(
                id=f"f{index:02d}",
                label=label,
                crop=crop,
                material=material,
                color=color,
                furrow=furrow,
                spacing_m=spacing,
                orientation_deg=orientations.get(label, 0.0),
                boundary=boundary_style,
                area_m2=area,
                centroid_east=float(east[mask].mean()),
                centroid_north=float(north[mask].mean()),
                polygon=polygon,
                cover_hint=hint,
            )
        )
    return parcels


def _crop_for(hint: str, catalog: tuple[str, ...], rng: np.random.Generator) -> str:
    if hint == "pasture":
        return "pasture"
    if hint == "orchard":
        return "orchard"
    choice = catalog[int(rng.integers(0, len(catalog)))]
    return choice if choice in CROPS else "wheat"


def _majority_cover(cover: np.ndarray, mask: np.ndarray) -> int:
    values = cover[mask]
    if values.size == 0:
        return BY_NAME["cropland"]
    codes, counts = np.unique(values, return_counts=True)
    return int(codes[int(np.argmax(counts))])


def _components(mask: np.ndarray, min_cells: int) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    rows, cols = mask.shape
    found: list[np.ndarray] = []
    for row in range(rows):
        for col in range(cols):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    n_row, n_col = r + d_row, c + d_col
                    if n_row < 0 or n_col < 0 or n_row >= rows or n_col >= cols:
                        continue
                    if seen[n_row, n_col] or not mask[n_row, n_col]:
                        continue
                    seen[n_row, n_col] = True
                    stack.append((n_row, n_col))
            if len(cells) >= min_cells:
                piece = np.zeros(mask.shape, dtype=bool)
                for r, c in cells:
                    piece[r, c] = True
                found.append(piece)
    return found


def _merge_slivers(labels: np.ndarray, min_cells: int) -> np.ndarray:
    merged = labels.copy()
    for label in [int(item) for item in np.unique(merged) if int(item) > 0]:
        mask = merged == label
        if int(mask.sum()) >= min_cells:
            continue
        neighbor = _dominant_neighbor(merged, mask)
        if neighbor:
            merged[mask] = neighbor
    return merged


def _dominant_neighbor(labels: np.ndarray, mask: np.ndarray) -> int:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    ring = dilated & ~mask
    values = labels[ring]
    values = values[values > 0]
    if values.size == 0:
        return 0
    codes, counts = np.unique(values, return_counts=True)
    return int(codes[int(np.argmax(counts))])


def _outline(mask: np.ndarray, east: np.ndarray, north: np.ndarray, grid: GeoGrid) -> list[tuple[float, float]]:
    segments = _shared_edges(mask.astype(np.int32), east, north, grid, only_label=None, mask_mode=mask)
    loops = _chain(segments)
    if not loops:
        return []
    loop = max(loops, key=len)
    return _simplify(loop, tolerance=max(grid.cell_width_m, grid.cell_height_m) * 0.75)


def _boundaries(
    labels: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    grid: GeoGrid,
    kind: str,
    fields: list[FieldParcel],
) -> list[BoundaryLine]:
    if kind == "none" or not fields:
        return []
    segments = _shared_edges(labels, east, north, grid, only_label=None, mask_mode=None)
    lines = []
    tolerance = max(grid.cell_width_m, grid.cell_height_m) * 0.8
    for chain in _chain(segments):
        simple = _simplify(chain, tolerance=tolerance)
        if len(simple) >= 2:
            lines.append(BoundaryLine(kind=kind, points=simple))
    return lines


def _shared_edges(
    labels: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    grid: GeoGrid,
    only_label: int | None,
    mask_mode: np.ndarray | None,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Edges between a field and a different label, in east/north meters."""
    rows, cols = labels.shape
    half_h = grid.cell_height_m / 2
    half_w = grid.cell_width_m / 2
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    target = mask_mode if mask_mode is not None else labels > 0

    def add(row: int, col: int, other_row: int, other_col: int, horizontal: bool) -> None:
        if not target[row, col]:
            return
        other = 0
        if 0 <= other_row < rows and 0 <= other_col < cols:
            if mask_mode is not None:
                if target[other_row, other_col]:
                    return
            else:
                other = int(labels[other_row, other_col])
                here = int(labels[row, col])
                if other == here or other > here:
                    return
        center_e = float(east[row, col])
        center_n = float(north[row, col])
        if horizontal:
            edge_n = center_n - half_h if other_row > row else center_n + half_h
            a = (center_e - half_w, edge_n)
            b = (center_e + half_w, edge_n)
        else:
            edge_e = center_e + half_w if other_col > col else center_e - half_w
            a = (edge_e, center_n - half_h)
            b = (edge_e, center_n + half_h)
        segments.append((a, b))

    for row in range(rows):
        for col in range(cols):
            if not target[row, col]:
                continue
            add(row, col, row - 1, col, True)
            add(row, col, row + 1, col, True)
            add(row, col, row, col - 1, False)
            add(row, col, row, col + 1, False)
    del only_label
    return segments


def _chain(segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    unused = set(range(len(segments)))
    chains: list[list[tuple[float, float]]] = []

    def near(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return abs(a[0] - b[0]) < 0.2 and abs(a[1] - b[1]) < 0.2

    while unused:
        index = unused.pop()
        start, end = segments[index]
        chain = [start, end]
        progressed = True
        while progressed:
            progressed = False
            for other in list(unused):
                a, b = segments[other]
                if near(a, chain[-1]):
                    chain.append(b)
                    unused.remove(other)
                    progressed = True
                    break
                if near(b, chain[-1]):
                    chain.append(a)
                    unused.remove(other)
                    progressed = True
                    break
                if near(a, chain[0]):
                    chain.insert(0, b)
                    unused.remove(other)
                    progressed = True
                    break
                if near(b, chain[0]):
                    chain.insert(0, a)
                    unused.remove(other)
                    progressed = True
                    break
        if len(chain) >= 2:
            chains.append(chain)
    return chains


def _simplify(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    keep = _douglas(points, tolerance)
    simplified = [points[index] for index in keep]
    if simplified[0] != simplified[-1] and _closed_enough(points):
        simplified.append(simplified[0])
    return simplified


def _closed_enough(points: list[tuple[float, float]]) -> bool:
    return abs(points[0][0] - points[-1][0]) < 0.3 and abs(points[0][1] - points[-1][1]) < 0.3


def _douglas(points: list[tuple[float, float]], tolerance: float) -> list[int]:
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        ax, ay = points[start]
        bx, by = points[end]
        length = math.hypot(bx - ax, by - ay) or 1.0
        worst = 0.0
        worst_index = None
        for index in range(start + 1, end):
            px, py = points[index]
            dist = abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax) / length
            if dist > worst:
                worst = dist
                worst_index = index
        if worst_index is not None and worst > tolerance:
            keep.add(worst_index)
            stack.append((start, worst_index))
            stack.append((worst_index, end))
    return sorted(keep)


def paint_cover(cover: np.ndarray, vectors: VectorSet, grid: GeoGrid, priority: dict[int, int]) -> np.ndarray:
    """Rasterize polygon land cover. Lines are not painted here; hydrology burns them."""
    painted = cover.copy()
    order = sorted(vectors.polygons, key=lambda poly: priority.get(BY_NAME.get(poly.cover, 0), 0))
    for poly in order:
        code = BY_NAME.get(poly.cover)
        if code is None:
            continue
        mask = rasterize_polygon(grid, poly.rings)
        paint_priority(painted, mask, code, priority)
    return painted
