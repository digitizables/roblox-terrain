"""Depression filling, D8 drainage, streams, and lakes.

The display surface keeps real ridges and valleys. Only sub-meter pits are
filled, so a noisy tile does not become a field of puddles. Channels are
carved where flow accumulation or a mapped waterway says water runs, and a
main stem is forced to descend so the river is continuous on the voxel grid.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

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
    dem = np.asarray(dem, dtype=np.float64)
    filled = fill_depressions(dem, epsilon=0.0)
    filled_flow = fill_depressions(dem, epsilon=0.002)
    depth = filled - dem
    flow_to, _slope = d8(filled_flow, cell_h, cell_w)
    accumulation = flow_accumulation(flow_to, dem.shape)
    # About one fiftieth of the cells is a trunk and a tributary. A lower
    # cutoff paints the whole valley floor as water on a coarse play grid.
    threshold = stream_threshold if stream_threshold is not None else max(36, dem.size // 48)
    streams = accumulation >= threshold
    if stream_mask is not None:
        streams = streams | stream_mask
    streams = streams & ~_border(dem.shape)
    lakes = _lake_components(depth > 0.6, min_cells=max(6, dem.size // 800))
    if lake_mask is not None:
        lakes = lakes | lake_mask
    surface = np.where(depth < 0.6, filled, dem)
    surface = _carve_channels(surface, streams, accumulation, cell_h, cell_w)
    surface = _enforce_main_stem(surface, accumulation, flow_to, cell_h, cell_w)
    water = np.zeros_like(surface)
    if np.any(lakes):
        spill = _component_spill(filled, lakes)
        water = np.where(lakes, np.clip(spill - surface, 0.3, 8.0), 0.0)
    channel_depth = np.clip(0.25 + 0.12 * np.log2(accumulation + 1.0), 0.35, 1.6)
    water = np.where(streams, np.maximum(water, channel_depth), water)
    # Banks should read as land. A one-cell stream is water; the carve is the bed.
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


def _carve_channels(
    surface: np.ndarray,
    streams: np.ndarray,
    accumulation: np.ndarray,
    cell_h: float,
    cell_w: float,
) -> np.ndarray:
    carved = surface.copy()
    if not np.any(streams):
        return carved
    depth = np.clip(0.45 * np.log2(accumulation + 1.0) - 1.2, 0.5, 2.4)
    carved[streams] -= depth[streams]
    # A wider bed on the trunk stream, one cell either side, shallower.
    wide = _widen(streams) & ~streams
    carved[wide] -= 0.35
    del cell_h, cell_w
    return carved


def _widen(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _enforce_main_stem(
    surface: np.ndarray,
    accumulation: np.ndarray,
    flow_to: np.ndarray,
    cell_h: float,
    cell_w: float,
) -> np.ndarray:
    """Walk the trunk from source to mouth and remove uphill steps."""
    path = _main_stem(accumulation, flow_to)
    if len(path) < 4:
        return surface
    shaped = surface.copy()
    rows, cols = surface.shape
    previous = None
    for flat in path:
        row, col = divmod(int(flat), cols)
        if previous is None:
            previous = (row, col, float(shaped[row, col]))
            continue
        p_row, p_col, p_elev = previous
        run = (((row - p_row) * cell_h) ** 2 + ((col - p_col) * cell_w) ** 2) ** 0.5
        ceiling = p_elev - max(0.02, 0.004 * run)
        if shaped[row, col] > ceiling:
            shaped[row, col] = ceiling
        previous = (row, col, float(shaped[row, col]))
    return shaped


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
