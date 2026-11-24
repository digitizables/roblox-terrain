# roblox-terrain

A separate plugin that builds a playable Roblox landscape from real elevation and land cover, then refines it with fields, drainage, and close-up parts. It depends on `roblox-studio-adapter` and the plugin host. The adapter does not import it.

Visual realism and a place you can walk around come first. The result is a framed window of a real region, not a survey-accurate twin of the whole map sheet.

## Install

This package is not part of [roblox-studio-adapter](https://github.com/digitizables/roblox-studio-adapter). Install the adapter and its plugin host first, then this repository. None of the three packages are on PyPI.

```powershell
git clone https://github.com/digitizables/roblox-studio-adapter.git
git clone https://github.com/digitizables/roblox-terrain.git
python -m pip install -e roblox-studio-adapter
python -m pip install -e roblox-studio-adapter/plugins/host
python -m pip install -e roblox-terrain
roblox-plugins install terrain
```

`numpy` is required. `rasterio` is optional (`pip install -e "roblox-terrain[geotiff]"`) and is only used when `dem_path` or `landcover_path` is a GeoTIFF. The bundled example does not need it, and it does not need a network connection.

Without installing, from a shell where the adapter checkout is a sibling directory:

```powershell
$env:PYTHONPATH = "src;../roblox-studio-adapter/src;../roblox-studio-adapter/plugins/host/src"
python -m roblox_studio_plugin_host --home $env:TEMP\roblox-plugins install terrain
```

`roblox-plugins` discovers this repository from the `roblox_studio.plugins` entry point after `pip install`, or from `plugin.json` when `ROBLOX_STUDIO_PLUGIN_ROOT` points here.

## Generate a scene

Python is the primary interface. This does not open Studio:

```python
from roblox_terrain import TerrainRequest, generate_realistic_terrain, save_scene

scene = generate_realistic_terrain(TerrainRequest(preset="cotswolds-windrush", seed=1))
print(scene.summary())
save_scene(scene, "last-scene")
```

The same call from the terminal:

```powershell
roblox-plugins run terrain generate_realistic_terrain
roblox-plugins status terrain
roblox-plugins run terrain inspect_scene
```

`examples/generate_cotswolds.py` is that dry run for the bundled region. On a default 400 m window it prints a surveyed extract of about 399 m by 400 m, 23 fields, and about 38 m of relief (roughly 166–204 m above sea level inside the window). The full extract underneath is about 1.5 km by 1.7 km, 139–222 m, 82 m of relief.

## What the default window contains

`cotswolds-windrush` is the Windrush valley in Gloucestershire, between Naunton and Upper Slaughter (`51.905 N–51.920 N`, `1.790 W–1.768 W`). The generator scores pasture, cropland, and woodland and frames one 400 m agricultural window. The measured default (seed 1, grid 96) is mostly pasture with a few arable parcels, hedgerows, a thin drainage channel, 23 fields, and about 36 voxel chunks (about 1.2 million voxels). That is under the point where the plugin warns that the Studio edit will be long.

`plan_tiles` lists abutting windows that cover the rest of the bbox, with `origin_x_studs` and `origin_z_studs` so an agent can lay them next to each other. North is negative Z. It does not generate terrain.

```python
from roblox_terrain import plan_tiles
for tile in plan_tiles("cotswolds-windrush"):
    print(tile["i"], tile["j"], tile["origin_x_studs"], tile["origin_z_studs"])
```

`synthetic-valley` is a procedural stand-in for tests. It is not a real place, and the scene says so.

## Apply it in Studio

Dry-run first. Applying replaces `Workspace.RealisticTerrain`, writes terrain voxels, and can hide `Workspace.Baseplate`. It stops a solo playtest so the write runs in the Edit datamodel. It does not save and it does not publish.

```powershell
$env:ROBLOX_STUDIO_CONFIRM = "1"
roblox-plugins run terrain generate_realistic_terrain --apply --yes
```

```python
from roblox_studio_adapter import RobloxStudio
from roblox_terrain import TerrainRequest, apply_scene, generate_realistic_terrain

scene = generate_realistic_terrain(TerrainRequest(preset="cotswolds-windrush"))
with RobloxStudio.connect(name_contains="Place1") as agent:
    apply_scene(agent, scene, confirm=True)
```

Without `confirm=True` (or `--yes` / `ROBLOX_STUDIO_CONFIRM=1`) the scene is still saved and the call raises `ConfirmationRequired`. Nothing is written. `clear_generated_terrain` airs the recorded voxel box, restores the baseplate from the backup stored on the folder, and deletes `Workspace.RealisticTerrain`. It uses the same confirmation.

The tests in this repository do not open Studio. A live apply has to be run against the Studio window you have open.

## Parameters

Saved config (`roblox-plugins config terrain --set key=value`) and the `generate_realistic_terrain` command accept the same knobs, except `apply`, which is command-only.

| Parameter | Default | Role |
| --- | --- | --- |
| `preset` | `cotswolds-windrush` | `cotswolds-windrush`, `whittlesey-fens`, `palouse`, `synthetic-valley`, or `custom` |
| `west`, `south`, `east`, `north` | preset bbox | All four, or none. A custom `.npz` can supply them |
| `focus_lat`, `focus_lon` | agricultural window | Center the play window instead of scoring fields |
| `play_extent_m` | `400` | Meters on a side. 80 to 4000 |
| `grid_size` | `96` | Analysis cells on the longer side. 24 to 320 |
| `studs_per_meter` | `2` | 4-stud voxels are then about 2 m |
| `vertical_exaggeration` | `1.2` | `1` is survey scale |
| `seed` | `1` | Crops, furrows, vegetation |
| `field_style` | `auto` | `rectangular` or `irregular`. Auto follows the preset |
| `boundary_style` | `auto` | `hedge`, `ditch`, `fence`, or `none` |
| `field_size_m` | preset | `0` keeps the preset size. Cotswolds uses 110 m |
| `vegetation_density` | `0.8` | 0 to 2 |
| `tree_budget`, `rock_budget`, `hedge_budget` | 180, 40, 160 | Caps on parts |
| `detail_radius_studs` | `220` | Close-up crop patches only inside this radius |
| `surface_patch_budget` | `420` | Cap on those patches |
| `furrow_strength`, `micro_detail` | 0.7, 0.35 | Cultivation ripples and DEM roughness |
| `stream_threshold` | `0` | `0` picks a catchment from the grid size |
| `origin_x_studs`, `origin_z_studs` | `0` | Where this window sits in the place |
| `hide_baseplate`, `place_marker` | true | Baseplate hide is restored by clear |
| `allow_network` | false | Download tiles only when no file covers the bbox |
| `dem_path` | empty | `.npz` or GeoTIFF |
| `landcover_path` | empty | WorldCover GeoTIFF, or an npz of class ids 0–10 |
| `vectors_path` | empty | GeoJSON FeatureCollection or OSM XML |
| `hydro_path` | empty | npz with an `accumulation` array |
| `apply` | false | Write the open place. Not stored in config |

`whittlesey-fens` and `palouse` are bboxes only. They download AWS Terrain Tiles and OpenStreetMap when `allow_network` is set, or they read `dem_path` and `vectors_path`. Downloads are cached under the plugin home.

An `.npz` elevation file stores `elevation` plus `west`, `south`, `east`, `north`. Row 0 is north. A GeoTIFF must be in geographic coordinates (longitude, latitude) or the loader refuses it.

## Pipeline

1. Load elevation from the bundle, an `.npz`, a GeoTIFF, or AWS Terrain Tiles.
2. Paint land cover from OpenStreetMap polygons, then an optional WorldCover raster.
3. Fill only sub-meter pits, trace D8 drainage, and burn mapped waterways in.
4. Keep OSM fields that are already about a field in size. Subdivide larger farmland along the contour.
5. Add furrows, hedges or ditches, and a light roughness the DEM does not carry.
6. Assign Roblox materials from cover, slope, and water. Place trees, orchard rows, hedges, fences, and rocks inside the budgets.
7. Resample onto 4-stud voxels in chunks. Close-up crop color is a part layer, because `SetMaterialColor` is global and a 4-stud voxel is too coarse for a furrow.

The folder `Workspace.RealisticTerrain` keeps the manifest on a `SceneJson` StringValue, one folder per field, and attributes for the preset, seed, and voxel box. An agent can move, recolor, or delete those instances afterwards with the adapter's data-model tools.

## Data sources

The scene records only the sources it actually used.

**Elevation, bundled example.** [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) in Terrarium encoding, zoom 14, tiles X 8110–8111 and Y 5417–5418, cropped to the preset bbox. Terrarium stores meters as `(red * 256 + green + blue / 256) - 32768`. The tiles are a [Tilezen composite](https://github.com/tilezen/joerd/blob/master/docs/attribution.md): NASA SRTM plus regional models. Over Europe that regional model is commonly EU-DEM. This is not a raw Copernicus GLO-30 download. Copernicus DEM, when you pass a file whose name contains `copernicus`, is attributed as © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA.

**Land cover and waterways, bundled example.** © OpenStreetMap contributors, [Open Database License](https://www.openstreetmap.org/copyright). The crop has 66 polygons (pasture, woodland, cropland, urban, water, one orchard, one scrub) and two waterways. ESA WorldCover is supported as a local GeoTIFF (classes 10–100) and is not fetched. WorldCover contains modified Copernicus Sentinel data: https://esa-worldcover.org/en/data-access

**Hydrology.** Computed here: priority-flood, D8, and flow accumulation, with mapped waterways burned in. A HydroSHEDS accumulation grid can be passed as `hydro_path`. HydroSHEDS (Lehner, Verdin, Jarvis, 2008) has its own license; keep that notice with the file you supply. https://www.hydrosheds.org/

The crop and this attribution also live in `src/roblox_terrain/data/SOURCES.txt`. Rebuild the crop, when you still have the tiles and the OSM extract, with `python tools/build_sample.py`.

## Limits

- Voxels are 4 studs. At 2 studs per meter that is about 2 m. Furrows (about 0.2 m) and crop color are parts inside `detail_radius_studs`. Farther away, crop variation is the voxel material (`Ground`, `Sandstone`, `LeafyGrass`, `Mud`, `Grass`), because a material color override is global.
- Default vertical exaggeration is 1.2. Set it to 1 when heights must match the DEM.
- One call frames one play window. A 1.5 km extract is several windows. Use `plan_tiles`, then generate each tile with that tile's bounds and origin.
- The plugin warns above about 80 voxel chunks and refuses a request above about 200 chunks or 8 million voxels. A 400 m window at 2 studs per meter stays inside that.
- Each voxel chunk is one `execute_luau` call with `task.wait` every six columns, and under one million voxels. Instance creates go through the adapter's chunked apply.
- Hedges and trunks collide. Field carpets, fences, and canopies do not. Trees are cylinders whose long axis is turned upright. Names have no dots, so the adapter can address them.
- OpenStreetMap in this extract is hedgerows, woods, and farm polygons, not a complete crop map. Unlabeled agricultural cover is subdivided procedurally. WorldCover is the fallback raster when you have a file.
- Terrarium is coarser than a survey DEM and is already a blend of sources. Do not treat the result as an elevation certificate.
- `synthetic-valley` must not be described as a surveyed place.
- A failed chunk leaves `Workspace.RealisticTerrain` in place so `clear_generated_terrain` can still air the recorded box. Read the place before you retry. The adapter does not retry edits.
- The test suite does not start Studio and does not claim a live apply succeeded.

## Tests

From the repository root:

```powershell
$env:PYTHONPATH = "src;../roblox-studio-adapter/src;../roblox-studio-adapter/plugins/host/src"
python -m unittest tests.test_terrain -v
```

## License

MIT, same as the adapter. Not affiliated with Roblox Corporation. Dataset licenses stay with the sources above.
