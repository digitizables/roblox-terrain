"""Terrain pipeline, voxel payload, and plugin CLI. No Studio process."""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT.parent / "roblox-studio-adapter"
for extra in (ROOT / "src", ADAPTER / "src", ADAPTER / "plugins" / "host" / "src"):
    text = str(extra)
    if extra.is_dir() and text not in sys.path:
        sys.path.insert(0, text)

import numpy as np

from roblox_studio_adapter.luau_gen import yield_risk
from roblox_terrain import TerrainRequest, generate_realistic_terrain, plan_tiles
from roblox_terrain.acquire import BUNDLE_DEM, BUNDLE_LAND, load_dem_npz
from roblox_terrain.apply import _upright_cylinder, _yawed, apply_scene, feature_specs
from roblox_terrain.cover import AGRICULTURAL, FOREST, PASTURE, WATER
from roblox_terrain.errors import ConfirmationRequired, DataUnavailable, ValidationError
from roblox_terrain.fields import FieldLayer, build_fields
from roblox_terrain.grid import GeoGrid
from roblox_terrain.hydrology import Hydrology, derive
from roblox_terrain.pngutil import read_rgb_png, terrarium_elevation
from roblox_terrain.raster import rasterize_ring
from roblox_terrain.surface import MATERIAL_INDEX, assign_materials
from roblox_terrain.vectors import VectorSet, load_vectors
from roblox_terrain.voxels import _pack, chunk_script, decode_payload, plan_voxels
from roblox_studio_plugin_host.cli import main


def _png(width: int, height: int, rows: list[bytes]) -> bytes:
    """Encode raw RGB rows with an explicit filter byte already prepended."""
    raw = b"".join(rows)
    if len(raw) != height * (1 + width * 3):
        raise ValueError("row payload has the wrong length")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    distances = (abs(estimate - left), abs(estimate - up), abs(estimate - up_left))
    return (left, up, up_left)[distances.index(min(distances))]


class PngTests(unittest.TestCase):
    def test_terrarium_formula(self) -> None:
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        rgb[0, 0] = (128, 150, 0)
        self.assertAlmostEqual(float(terrarium_elevation(rgb)[0, 0]), 150.0, places=3)

    def test_paeth_roundtrip(self) -> None:
        width, height = 4, 3
        pixels = np.array(
            [
                [[10, 20, 30], [11, 40, 50], [12, 80, 90], [200, 10, 10]],
                [[10, 20, 31], [255, 0, 1], [12, 81, 91], [199, 11, 12]],
                [[9, 19, 29], [250, 5, 6], [13, 70, 80], [198, 12, 13]],
            ],
            dtype=np.uint8,
        )
        raw_rows = []
        previous = bytearray(width * 3)
        for row in pixels:
            scan = bytearray(row.reshape(-1).tolist())
            filtered = bytearray(len(scan))
            for index, value in enumerate(scan):
                left = scan[index - 3] if index >= 3 else 0
                up = previous[index]
                up_left = previous[index - 3] if index >= 3 else 0
                filtered[index] = (value - _paeth(left, up, up_left)) & 255
            previous = scan
            raw_rows.append(bytes([4]) + bytes(filtered))
        decoded = read_rgb_png(_png(width, height, raw_rows))
        self.assertTrue(np.array_equal(decoded, pixels))

    def test_bundled_extract_is_cotswolds_relief(self) -> None:
        elevation, grid = load_dem_npz(BUNDLE_DEM)
        self.assertGreater(float(elevation.min()), 100.0)
        self.assertLess(float(elevation.max()), 400.0)
        self.assertGreater(float(elevation.max() - elevation.min()), 40.0)
        self.assertAlmostEqual(grid.west, -1.790, places=3)
        self.assertAlmostEqual(grid.north, 51.920, places=3)
        vectors = load_vectors(BUNDLE_LAND)
        self.assertGreaterEqual(len(vectors.polygons), 40)
        self.assertTrue(any(poly.cover in {"cropland", "pasture"} for poly in vectors.polygons))
        self.assertIn("OpenStreetMap", vectors.attribution)


class RasterAndWaterTests(unittest.TestCase):
    def test_ring_covers_the_interior(self) -> None:
        grid = GeoGrid(west=0, south=0, east=1, north=1, rows=20, cols=20)
        ring = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]])
        mask = rasterize_ring(grid, ring)
        self.assertTrue(bool(mask[10, 10]))
        self.assertFalse(bool(mask[0, 0]))

    def test_tilted_plane_drains_south(self) -> None:
        rows, cols = 24, 16
        dem = np.linspace(40.0, 10.0, rows)[:, None] * np.ones((1, cols))
        hydro = derive(dem, cell_h=10.0, cell_w=10.0, stream_threshold=8)
        self.assertLess(float(hydro.down_north[4:-4, 4:-4].mean()), 0.0)
        self.assertGreater(float(hydro.accumulation[-3, cols // 2]), float(hydro.accumulation[2, cols // 2]))
        self.assertTrue(bool(np.all(hydro.accumulation[1:, cols // 2] >= hydro.accumulation[:-1, cols // 2] - 1e-6)))

    def test_pit_becomes_a_lake(self) -> None:
        dem = np.full((30, 30), 20.0)
        dem[10:18, 10:18] = 15.0
        hydro = derive(dem, cell_h=8.0, cell_w=8.0)
        self.assertGreater(hydro.lake_cells, 0)
        self.assertGreater(float(hydro.water_depth[14, 14]), 0.2)

    def test_fields_stay_inside_agricultural_cover(self) -> None:
        grid = GeoGrid(west=0.0, south=51.90, east=0.01, north=51.91, rows=40, cols=40)
        cover = np.full((40, 40), PASTURE, dtype=np.uint8)
        cover[:, :8] = FOREST
        heights = np.linspace(30.0, 20.0, 40)[:, None] + np.zeros((40, 40))
        layer = build_fields(
            heights,
            cover,
            VectorSet(),
            grid,
            field_style="rectangular",
            boundary_style="hedge",
            field_size_m=40,
            crops=("wheat", "pasture"),
            seed=1,
        )
        self.assertGreater(len(layer.fields), 0)
        labeled = layer.labels > 0
        self.assertTrue(bool(np.all(np.isin(cover[labeled], AGRICULTURAL))))
        self.assertTrue(bool(np.all(layer.labels[:, :8] == 0)))

    def test_water_and_rock_materials(self) -> None:
        cover = np.full((6, 6), PASTURE, dtype=np.uint8)
        cover[1, 1] = WATER
        slope = np.zeros((6, 6), dtype=np.float32)
        slope[2, 2] = 40
        depth = np.zeros((6, 6), dtype=np.float32)
        depth[1, 1] = 1.0
        zeros = np.zeros((6, 6), dtype=np.float32)
        hydro = Hydrology(
            surface=zeros,
            water_depth=depth,
            accumulation=np.ones((6, 6), dtype=np.float32),
            streams=depth > 0,
            lakes=np.zeros((6, 6), dtype=bool),
            slope_deg=slope,
            down_east=zeros,
            down_north=zeros,
            stream_cells=1,
            lake_cells=0,
        )
        materials = assign_materials(cover, hydro, FieldLayer(np.zeros((6, 6), np.int32), [], [], 0, 0))
        self.assertEqual(int(materials[1, 1]), MATERIAL_INDEX["Water"])
        self.assertEqual(int(materials[2, 2]), MATERIAL_INDEX["Rock"])


class SceneTests(unittest.TestCase):
    def test_synthetic_scene_respects_budgets_and_part_frames(self) -> None:
        scene = generate_realistic_terrain(
            TerrainRequest(
                preset="synthetic-valley",
                grid_size=36,
                seed=3,
                tree_budget=12,
                rock_budget=4,
                hedge_budget=8,
                surface_patch_budget=24,
                detail_radius_studs=120,
            )
        )
        self.assertFalse(scene.real_place)
        self.assertGreater(scene.stats["relief_m"], 5)
        self.assertGreater(scene.stats["field_count"], 0)
        self.assertLessEqual(scene.stats["props"].get("tree", 0), 12)
        self.assertLessEqual(scene.stats["props"].get("rock", 0), 4)
        self.assertLessEqual(scene.stats["props"].get("hedge", 0), 8)
        self.assertTrue(any("not a real-world" in warning for warning in scene.warnings))
        specs = feature_specs(scene)
        self.assertTrue(all("." not in spec["name"] for spec in specs))
        trunk_at = [index for index, spec in enumerate(specs) if str(spec["name"]).startswith("tree_")]
        canopy_at = [index for index, spec in enumerate(specs) if spec["name"] == "Canopy"]
        self.assertTrue(trunk_at and canopy_at)
        self.assertLess(max(trunk_at), min(canopy_at))
        trunk = next(spec for spec in specs if str(spec["name"]).startswith("tree_"))
        components = trunk["properties"]["CFrame"]["components"]
        self.assertEqual(components[3], 0.0)
        self.assertEqual(components[6], 1.0)
        self.assertEqual(components[9], 0.0)
        hedge = next(spec for spec in specs if str(spec["name"]).startswith("hedge_"))
        self.assertTrue(hedge["properties"]["CanCollide"])
        patch = next(spec for spec in specs if str(spec["name"]).startswith("patch_"))
        self.assertFalse(patch["properties"]["CanCollide"])
        self.assertEqual(patch["properties"]["Color"]["scale"], "unit")

    def test_oversized_request_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            generate_realistic_terrain(
                TerrainRequest(preset="synthetic-valley", grid_size=24, studs_per_meter=8)
            )

    def test_voxel_script_yields_and_keeps_water_off_the_bed(self) -> None:
        surface = np.full((2, 2), 20.0)
        water_surface = np.full((2, 2), 24.0)
        depth = np.full((2, 2), 1.0)
        materials = np.full((2, 2), MATERIAL_INDEX["Water"], dtype=np.int32)
        payload = _pack(surface, water_surface, depth, materials, 0.0, 8)
        for bed, water, surface_mat, fill_mat in decode_payload(payload, 2, 2):
            self.assertEqual(surface_mat, MATERIAL_INDEX["Mud"])
            self.assertEqual(fill_mat, MATERIAL_INDEX["Mud"])
            self.assertGreater(water, bed)
        scene = generate_realistic_terrain(
            TerrainRequest(preset="synthetic-valley", grid_size=24, seed=1, tree_budget=0, hedge_budget=0, rock_budget=0)
        )
        plan = plan_voxels(scene)
        self.assertGreater(len(plan.chunks), 0)
        for chunk in plan.chunks:
            self.assertLessEqual(chunk.voxels, 1_000_000)
            script = chunk_script(chunk)
            self.assertIn("task.wait", script)
            self.assertIn("WriteVoxels", script)
            self.assertIn("Enum.Material.Rock", script)
            self.assertIsNone(yield_risk(script))
        self.assertEqual(_upright_cylinder(0, 1, 0)["components"][6], 1.0)
        yawed = _yawed(0, 0, 0, (0.0, 0.0, -1.0))
        self.assertEqual(yawed["components"][9], -1.0)

    def test_apply_requires_confirmation_and_a_port_can_fail_cleanly(self) -> None:
        scene = generate_realistic_terrain(
            TerrainRequest(
                preset="synthetic-valley",
                grid_size=24,
                seed=1,
                tree_budget=2,
                hedge_budget=2,
                rock_budget=1,
                surface_patch_budget=4,
            )
        )
        port = _FakePort()
        with self.assertRaises(ConfirmationRequired) as caught:
            apply_scene(port, scene, confirm=False)
        self.assertIn("does not save or publish", str(caught.exception))
        self.assertEqual(port.edits, 0)
        applied = apply_scene(port, scene, confirm=True)
        self.assertTrue(applied["applied"])
        self.assertGreater(applied["chunks"], 0)
        self.assertGreater(port.edits, 0)
        failing = _FakePort(fail_on_chunk=2)
        with self.assertRaises(Exception) as failed:
            apply_scene(failing, scene, confirm=True)
        self.assertIn("clear_generated_terrain", str(failed.exception))

    def test_tiles_abut_and_north_is_negative_z(self) -> None:
        tiles = plan_tiles("cotswolds-windrush", play_extent_m=400, studs_per_meter=2)
        self.assertGreater(len(tiles), 1)
        by_row: dict[int, list[dict]] = {}
        for tile in tiles:
            by_row.setdefault(tile["j"], []).append(tile)
        northern = min(by_row)
        southern = max(by_row)
        self.assertLess(by_row[northern][0]["origin_z_studs"], by_row[southern][0]["origin_z_studs"])
        row = sorted(by_row[northern], key=lambda item: item["i"])
        for left, right in zip(row, row[1:]):
            self.assertAlmostEqual(left["east"], right["west"])

    def test_bundled_window_is_a_real_field_landscape(self) -> None:
        scene = generate_realistic_terrain(
            TerrainRequest(preset="cotswolds-windrush", grid_size=48, seed=1, tree_budget=20, hedge_budget=12, rock_budget=4)
        )
        self.assertTrue(scene.real_place)
        self.assertGreater(scene.stats["relief_m"], 8)
        self.assertGreater(scene.stats["field_count"], 0)
        self.assertGreater(scene.stats["elevation_min_m"], 80)
        names = {source["name"] for source in scene.sources}
        self.assertIn("AWS Terrain Tiles (Terrarium)", names)
        self.assertIn("OpenStreetMap", names)
        with self.assertRaises(DataUnavailable):
            generate_realistic_terrain(
                TerrainRequest(
                    preset="cotswolds-windrush",
                    west=10.0,
                    south=51.90,
                    east=10.02,
                    north=51.92,
                    grid_size=24,
                )
            )


class _FakePort:
    def __init__(self, fail_on_chunk: int | None = None) -> None:
        self.fail_on_chunk = fail_on_chunk
        self.edits = 0
        self.chunks = 0
        self.specs: list[dict] = []

    def ensure_edit(self) -> None:
        self.edits += 1

    def execute(self, code: str) -> dict:
        if "WriteVoxels" in code:
            self.chunks += 1
            if self.fail_on_chunk is not None and self.chunks == self.fail_on_chunk:
                raise RuntimeError("studio hiccup")
            return {"ok": True, "chunk": self.chunks, "voxels": 1}
        if "cleared" in code:
            return {"ok": True, "cleared": False, "baseplate_restored": False}
        return {"ok": True, "found": False}

    def apply_instances(self, specs: list[dict]) -> list[str]:
        self.specs.extend(specs)
        return [str(spec.get("name", "instance")) for spec in specs]

    def progress(self, stage: str, fraction: float, detail: str = "") -> None:
        del stage, fraction, detail


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = self.tmp.name
        self._old_root = os.environ.get("ROBLOX_STUDIO_PLUGIN_ROOT")
        os.environ["ROBLOX_STUDIO_PLUGIN_ROOT"] = str(ROOT)

    def tearDown(self) -> None:
        if self._old_root is None:
            os.environ.pop("ROBLOX_STUDIO_PLUGIN_ROOT", None)
        else:
            os.environ["ROBLOX_STUDIO_PLUGIN_ROOT"] = self._old_root
        self.tmp.cleanup()

    def test_install_configure_and_dry_run(self) -> None:
        listed = self._run(["--json", "list"])
        ids = [row["id"] for row in listed["plugins"]]
        self.assertIn("terrain", ids)
        self.assertEqual(self._code(["install", "terrain"]), 0)
        self.assertEqual(
            self._code(
                [
                    "config",
                    "terrain",
                    "--set",
                    "preset=synthetic-valley",
                    "--set",
                    "grid_size=28",
                    "--set",
                    "tree_budget=6",
                    "--set",
                    "hedge_budget=4",
                    "--set",
                    "rock_budget=2",
                    "--set",
                    "surface_patch_budget=8",
                ]
            ),
            0,
        )
        self.assertEqual(self._code(["config", "terrain", "--set", "apply=true"]), 2)
        result = self._run(["--json", "run", "terrain", "generate_realistic_terrain", "--set", "seed=4"])
        self.assertTrue(result["ok"])
        self.assertFalse(result["result"]["applied"])
        self.assertFalse(result["result"]["real_place"])
        self.assertGreater(result["result"]["stats"]["field_count"], 0)
        status = self._run(["--json", "status", "terrain"])
        self.assertTrue(status["status"]["bundle"])
        self.assertIsNotNone(status["status"]["last_scene"])
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = main(
                ["--home", self.home, "run", "terrain", "generate_realistic_terrain", "--apply", "--set", "grid_size=28"]
            )
        self.assertEqual(code, 2)
        self.assertIn("Nothing was written", err.getvalue())
        self.assertTrue((Path(self.home) / "cache" / "terrain" / "last-scene" / "manifest.json").is_file())

    def _code(self, args: list[str]) -> int:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(["--home", self.home, *args])

    def _run(self, args: list[str]) -> dict:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            code = main(["--home", self.home, *args])
        self.assertEqual(code, 0, buf.getvalue())
        return json.loads(buf.getvalue())


if __name__ == "__main__":
    unittest.main()
