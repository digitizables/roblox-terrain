"""Depression filling, D8 drainage, streams, and lakes.

The display surface keeps real ridges and valleys. Only sub-meter pits are
filled, so a noisy tile does not become a field of puddles. A mapped river is
shifted into the local valley and cut as one shallow bed. With no map, only
the main drainage stem is cut. A one-cell slot reads as a crack, so the bed
has a width.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from roblox_terrain.raster import _bresenham

NEIGHBORS = (
    (-1, -1, 2**0.5),
    (-1, 0, 1.0),
    (-1, 1, 2**0.5),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (1, -1, 2**0.5),
    (1, 0, 1.0),
    (1, 1, 2**0.5),
)


@dataclass
class Hydrology:
    surface: np.ndarray
    water_depth: np.ndarray
    accumulation: np.ndarray
    streams: np.ndarray
    lakes: np.ndarray
    slope_deg: np.ndarray
    down_east: np.ndarray
    down_north: np.ndarray
    stream_cells: int
    lake_cells: int


def derive(
    dem: np.ndarray,
    *,
    cell_h: float,
    cell_w: float,
    stream_mask: np.ndarray | None = None,
    lake_mask: np.ndarray | None = None,
    stream_threshold: int | None = None,
) -> Hydrology:
    """Build the display surface.

    ``stream_threshold`` is accepted and ignored. Painting every cell above an
    accumulation cutoff drew a straight ditch across the hillside. The bed is
    the mapped river, shifted sideways into the valley, or the single main
    stem when the map has no waterway.
    """
    del stream_threshold
    dem = np.asarray(dem, dtype=np.float64)
    filled = fill_depressions(dem, epsilon=0.0)
    filled_flow = fill_depressions(dem, epsilon=0.002)
    depth = filled - dem
    flow_to, _slope = d8(filled_flow, cell_h, cell_w)
    accumulation = flow_accumulation(flow_to, dem.shape)
    mapped = np.zeros(dem.shape, dtype=bool) if stream_mask is None else np.asarray(stream_mask, dtype=bool)
    if np.any(mapped):
        # The mapped line is the river. Do not also cut the steepest-descent
        # scratch; on a smooth DEM that scratch is a straight ditch.
        center = _snap_centerline(dem, mapped, cell_h, cell_w)
    else:
        center = np.zeros(dem.shape, dtype=bool)
        cols = dem.shape[1]
        for flat in _main_stem(accumulation, flow_to):
            row, col = divmod(int(flat), cols)
            center[row, col] = True
        center &= ~_border(dem.shape)
    lakes = _lake_components(depth > 0.6, min_cells=max(6, dem.size // 800))
    if lake_mask is not None:
        lakes = lakes | lake_mask
    surface = np.where(depth < 0.6, filled, dem)
    surface = _descend_centerline(surface, center, cell_h, cell_w)
    surface, channel_water = _incise(surface, center, cell_h, cell_w)
    water = np.zeros_like(surface)
    if np.any(lakes):
        spill = _component_spill(filled, lakes)
        water = np.where(lakes, np.clip(spill - surface, 0.3, 8.0), 0.0)
    water = np.maximum(water, channel_water)
    streams = center
    down_east, down_north, slope_deg = gradients(surface, cell_h, cell_w)
    return Hydrology(
        surface=surface.astype(np.float32),
        water_depth=water.astype(np.float32),
        accumulation=accumulation.astype(np.float32),
        streams=streams,
        lakes=lakes,
        slope_deg=slope_deg.astype(np.float32),
        down_east=down_east.astype(np.float32),
        down_north=down_north.astype(np.float32),
        stream_cells=int(streams.sum()),
        lake_cells=int(lakes.sum()),
    )


def fill_depressions(dem: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Priority-flood. ``epsilon`` gives flats a tiny slope toward the border."""
    rows, cols = dem.shape
    filled = np.full(dem.shape, np.inf, dtype=np.float64)
    visited = np.zeros(dem.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for row in range(rows):
        for col in (0, cols - 1):
            _seed(heap, filled, visited, dem, row, col)
    for col in range(cols):
        for row in (0, rows - 1):
            if not visited[row, col]:
                _seed(heap, filled, visited, dem, row, col)
    while heap:
        elev, row, col = heapq.heappop(heap)
        for d_row, d_col, _dist in NEIGHBORS:
            n_row = row + d_row
            n_col = col + d_col
            if n_row < 0 or n_col < 0 or n_row >= rows or n_col >= cols or visited[n_row, n_col]:
                continue
            visited[n_row, n_col] = True
            spill = elev + epsilon
            filled[n_row, n_col] = dem[n_row, n_col] if dem[n_row, n_col] >= spill else spill
            heapq.heappush(heap, (filled[n_row, n_col], n_row, n_col))
    filled[~visited] = dem[~visited]
    return filled


def d8(filled: np.ndarray, cell_h: float, cell_w: float) -> tuple[np.ndarray, np.ndarray]:
    """Steepest-descent pointer as a flat index, or -1 at outlets and pits."""
    rows, cols = filled.shape
    flow_to = np.full(rows * cols, -1, dtype=np.int32)
    steepest = np.zeros((rows, cols), dtype=np.float64)
    for d_row, d_col, _distance in NEIGHBORS:
        n_row = slice(max(d_row, 0), rows + min(d_row, 0))
        n_col = slice(max(d_col, 0), cols + min(d_col, 0))
        s_row = slice(max(-d_row, 0), rows + min(-d_row, 0))
        s_col = slice(max(-d_col, 0), cols + min(-d_col, 0))
        drop = filled[s_row, s_col] - filled[n_row, n_col]
        run = ((abs(d_row) * cell_h) ** 2 + (abs(d_col) * cell_w) ** 2) ** 0.5
        slope = drop / run
        better = slope > steepest[s_row, s_col] + 1e-12
        steepest[s_row, s_col] = np.where(better, slope, steepest[s_row, s_col])
        src = np.arange(rows * cols).reshape(rows, cols)[s_row, s_col]
        dst_rows = np.arange(rows)[n_row]
        dst_cols = np.arange(cols)[n_col]
        dst = (dst_rows[:, None] * cols + dst_cols[None, :]).ravel()
        flat_src = src.ravel()
        flow_to[flat_src] = np.where(better.ravel(), dst, flow_to[flat_src])
    # Do not point at a higher or equal cell. The slope test already requires a drop.
    return flow_to, steepest


def flow_accumulation(flow_to: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    count = shape[0] * shape[1]
    indegree = np.zeros(count, dtype=np.int32)
    for destination in flow_to:
        if destination >= 0:
            indegree[int(destination)] += 1
    accumulation = np.ones(count, dtype=np.float64)
    queue = [index for index in range(count) if indegree[index] == 0]
    head = 0
    while head < len(queue):
        src = queue[head]
        head += 1
        dst = int(flow_to[src])
        if dst >= 0:
            accumulation[dst] += accumulation[src]
            indegree[dst] -= 1
            if indegree[dst] == 0:
                queue.append(dst)
    return accumulation.reshape(shape)


def gradients(heights: np.ndarray, cell_h: float, cell_w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downhill components in east and north, plus slope in degrees.

    Row 0 is north, so the axis-0 gradient points south. ``down_north`` is
    positive on a north-facing slope. ``down_east`` is positive where the
    ground falls toward the east.
    """
    d_south, d_east = np.gradient(np.asarray(heights, dtype=np.float64), cell_h, cell_w)
    down_east = -d_east
    down_north = d_south
    slope = np.hypot(d_south, d_east)
    slope_deg = np.degrees(np.arctan(slope))
    return down_east, down_north, slope_deg


def _seed(heap: list, filled: np.ndarray, visited: np.ndarray, dem: np.ndarray, row: int, col: int) -> None:
    visited[row, col] = True
    filled[row, col] = dem[row, col]
    heapq.heappush(heap, (float(dem[row, col]), row, col))


def _border(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _lake_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    rows, cols = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    keep = np.zeros(mask.shape, dtype=bool)
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
                for r, c in cells:
                    keep[r, c] = True
    return keep


def _component_spill(filled: np.ndarray, lakes: np.ndarray) -> np.ndarray:
    """Spill elevation per lake, broadcast back onto the lake cells."""
    spill = np.zeros_like(filled)
    rows, cols = lakes.shape
    seen = np.zeros(lakes.shape, dtype=bool)
    for row in range(rows):
        for col in range(cols):
            if not lakes[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    n_row, n_col = r + d_row, c + d_col
                    if (
                        n_row < 0
                        or n_col < 0
                        or n_row >= rows
                        or n_col >= cols
                        or seen[n_row, n_col]
                        or not lakes[n_row, n_col]
                    ):
                        continue
                    seen[n_row, n_col] = True
                    stack.append((n_row, n_col))
            level = min(float(filled[r, c]) for r, c in cells)
            for r, c in cells:
                spill[r, c] = level
    return spill


def _main_stem(accumulation: np.ndarray, flow_to: np.ndarray) -> list[int]:
    flat_acc = accumulation.ravel()
    mouth = int(np.argmax(flat_acc))
    incoming: dict[int, list[int]] = {}
    for src, dst in enumerate(flow_to):
        if int(dst) >= 0:
            incoming.setdefault(int(dst), []).append(src)
    path = [mouth]
    seen = {mouth}
    while True:
        choices = [item for item in incoming.get(path[-1], []) if item not in seen]
        if not choices:
            break
        nxt = max(choices, key=lambda item: float(flat_acc[item]))
        seen.add(nxt)
        path.append(nxt)
        if len(path) > flat_acc.size:
            break
    path.reverse()
    return path


_EIGHT = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _snap_centerline(dem: np.ndarray, mask: np.ndarray, cell_h: float, cell_w: float) -> np.ndarray:
    """Move a mapped line sideways onto the lowest ground within about 40 m.

    The search stays on the cross-section. A disk would follow the downstream
    slope and pile the whole river at the bottom of the window.
    """
    center = np.zeros(dem.shape, dtype=bool)
    for chain in _chains(mask, dem):
        snapped = [
            _lowest_across(dem, row, col, _tangent(chain, index), cell_h, cell_w)
            for index, (row, col) in enumerate(chain)
        ]
        if _on_border(chain[0], dem.shape) and snapped[0] != chain[0]:
            snapped.insert(0, chain[0])
        if len(chain) > 1 and _on_border(chain[-1], dem.shape) and snapped[-1] != chain[-1]:
            snapped.append(chain[-1])
        _paint_chain(center, snapped)
    return center


def _descend_centerline(surface: np.ndarray, center: np.ndarray, cell_h: float, cell_w: float) -> np.ndarray:
    """Take rises out of the thalweg without digging a new valley through a hill."""
    original = np.asarray(surface, dtype=np.float64)
    shaped = original.copy()
    if not np.any(center):
        return shaped
    for chain in _chains(center, shaped):
        if len(chain) >= 2 and float(shaped[chain[0]]) < float(shaped[chain[-1]]):
            chain = list(reversed(chain))
        previous: tuple[int, int, float] | None = None
        for row, col in chain:
            if previous is None:
                previous = (row, col, float(shaped[row, col]))
                continue
            p_row, p_col, p_elev = previous
            run = math.hypot((row - p_row) * cell_h, (col - p_col) * cell_w)
            ceiling = p_elev - 0.002 * run
            floor = _local_floor(original, row, col, cell_h, cell_w)
            target = max(ceiling, floor)
            if shaped[row, col] > target:
                shaped[row, col] = target
            previous = (row, col, float(shaped[row, col]))
    return shaped


def _incise(
    surface: np.ndarray,
    center: np.ndarray,
    cell_h: float,
    cell_w: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a shallow valley and put a flat water surface in the inner half of it.

    Half-width is about 20 m, so the bed is several 4-stud voxels across.
    The water is about 2 m deep in the middle and about 16 m wide, which is
    one voxel of water under the default stud scale.
    """
    shaped = np.asarray(surface, dtype=np.float64).copy()
    water = np.zeros(shaped.shape, dtype=np.float64)
    if not np.any(center):
        return shaped, water
    half = 20.0
    wet_half = 8.0
    peak = 2.0
    lip = 1.1
    rows, cols = shaped.shape
    dist = np.full(shaped.shape, np.inf)
    ref = shaped.copy()
    rad_r = int(math.ceil(half / max(cell_h, 1e-6))) + 1
    rad_c = int(math.ceil(half / max(cell_w, 1e-6))) + 1
    ys, xs = np.nonzero(center)
    for y, x in zip(ys.tolist(), xs.tolist()):
        r0 = max(0, y - rad_r)
        r1 = min(rows, y + rad_r + 1)
        c0 = max(0, x - rad_c)
        c1 = min(cols, x + rad_c + 1)
        rr = (np.arange(r0, r1) - y) * cell_h
        cc = (np.arange(c0, c1) - x) * cell_w
        local = np.hypot(rr[:, None], cc[None, :])
        window = dist[r0:r1, c0:c1]
        closer = local < window
        dist[r0:r1, c0:c1] = np.where(closer, local, window)
        ref[r0:r1, c0:c1] = np.where(closer, shaped[y, x], ref[r0:r1, c0:c1])
    near = dist <= half
    inner = dist <= wet_half
    cut = np.zeros(shaped.shape, dtype=np.float64)
    t_in = np.zeros(shaped.shape, dtype=np.float64)
    t_in[inner] = dist[inner] / wet_half
    cut[inner] = lip + peak * (1.0 - t_in[inner] ** 2)
    outer = near & ~inner
    t_out = (dist[outer] - wet_half) / (half - wet_half)
    cut[outer] = lip * (1.0 - t_out)
    shaped = np.where(near, np.minimum(shaped, ref - cut), shaped)
    water[inner] = peak * (1.0 - (np.clip(dist[inner], 0.0, wet_half) / wet_half) ** 2)
    return shaped, water


def _chains(mask: np.ndarray, dem: np.ndarray) -> list[list[tuple[int, int]]]:
    """8-connected paths. Longer paths come first. Each starts at its higher end."""
    cells = {(int(r), int(c)) for r, c in zip(*np.nonzero(mask))}
    if not cells:
        return []
    neighbors = {
        cell: [(cell[0] + dr, cell[1] + dc) for dr, dc in _EIGHT if (cell[0] + dr, cell[1] + dc) in cells]
        for cell in cells
    }
    remaining = set(cells)
    chains: list[list[tuple[int, int]]] = []
    while remaining:
        ends = [cell for cell in remaining if sum(n in remaining for n in neighbors[cell]) <= 1]
        start = max(ends or list(remaining), key=lambda cell: (float(dem[cell]), -cell[0], -cell[1]))
        chain = [start]
        remaining.remove(start)
        while True:
            choices = [n for n in neighbors[chain[-1]] if n in remaining]
            if not choices:
                bridge = [n for n in neighbors[chain[-1]] if n in cells and n not in chain]
                if bridge:
                    chain.append(min(bridge, key=lambda n: float(dem[n])))
                break
            if len(chain) >= 2:
                pr, pc = chain[-2]
                cr, cc = chain[-1]
                vr, vc = cr - pr, cc - pc
                choices.sort(key=lambda n: -((n[0] - cr) * vr + (n[1] - cc) * vc))
            nxt = choices[0]
            chain.append(nxt)
            remaining.remove(nxt)
        if len(chain) >= 2 and float(dem[chain[0]]) < float(dem[chain[-1]]):
            chain.reverse()
        chains.append(chain)
    chains.sort(key=len, reverse=True)
    return chains


def _tangent(chain: list[tuple[int, int]], index: int) -> tuple[float, float]:
    if len(chain) < 2:
        return 1.0, 0.0
    if index <= 0:
        r0, c0 = chain[0]
        r1, c1 = chain[1]
    elif index >= len(chain) - 1:
        r0, c0 = chain[-2]
        r1, c1 = chain[-1]
    else:
        r0, c0 = chain[index - 1]
        r1, c1 = chain[index + 1]
    return float(r1 - r0), float(c1 - c0)


def _lowest_across(
    dem: np.ndarray,
    row: int,
    col: int,
    tangent: tuple[float, float],
    cell_h: float,
    cell_w: float,
) -> tuple[int, int]:
    t_row, t_col = tangent
    t_east = t_col * cell_w
    t_north = -t_row * cell_h
    scale = math.hypot(t_east, t_north) or 1.0
    t_east /= scale
    t_north /= scale
    p_east = -t_north
    p_north = t_east
    rows, cols = dem.shape
    rad_r = max(1, int(math.ceil(40.0 / max(cell_h, 1e-6))))
    rad_c = max(1, int(math.ceil(40.0 / max(cell_w, 1e-6))))
    along_limit = max(cell_h, cell_w)
    best = (row, col)
    best_score = float(dem[row, col])
    for nr in range(max(0, row - rad_r), min(rows, row + rad_r + 1)):
        for nc in range(max(0, col - rad_c), min(cols, col + rad_c + 1)):
            east = (nc - col) * cell_w
            north = -(nr - row) * cell_h
            along = abs(east * t_east + north * t_north)
            lateral = abs(east * p_east + north * p_north)
            if along > along_limit or lateral > 40.0:
                continue
            score = float(dem[nr, nc]) + lateral * 0.002
            if score < best_score:
                best_score = score
                best = (nr, nc)
    return best


def _local_floor(surface: np.ndarray, row: int, col: int, cell_h: float, cell_w: float) -> float:
    rad_r = max(1, int(round(12.0 / max(cell_h, 1e-6))))
    rad_c = max(1, int(round(12.0 / max(cell_w, 1e-6))))
    r0 = max(0, row - rad_r)
    r1 = min(surface.shape[0], row + rad_r + 1)
    c0 = max(0, col - rad_c)
    c1 = min(surface.shape[1], col + rad_c + 1)
    return float(np.min(surface[r0:r1, c0:c1]))


def _on_border(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    row, col = cell
    rows, cols = shape
    return row <= 0 or col <= 0 or row >= rows - 1 or col >= cols - 1


def _paint_chain(center: np.ndarray, points: list[tuple[int, int]]) -> None:
    rows, cols = center.shape
    if not points:
        return
    previous = points[0]
    if 0 <= previous[0] < rows and 0 <= previous[1] < cols:
        center[previous] = True
    for point in points[1:]:
        for row, col in _bresenham(previous[0], previous[1], point[0], point[1]):
            if 0 <= row < rows and 0 <= col < cols:
                center[row, col] = True
        previous = point
