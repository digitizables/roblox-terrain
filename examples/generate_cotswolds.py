"""Generate the bundled Windrush valley window. This does not open Studio.

From the repository root:

    $env:PYTHONPATH = "src"
    python examples/generate_cotswolds.py

The scene is written next to this file as last-cotswolds/. Pass that
directory to your own apply step, or use the plugin CLI with --apply
after ROBLOX_STUDIO_CONFIRM=1.
"""

from __future__ import annotations

from pathlib import Path

from roblox_terrain import TerrainRequest, generate_realistic_terrain, save_scene


def main() -> None:
    scene = generate_realistic_terrain(TerrainRequest(preset="cotswolds-windrush", seed=1))
    out = Path(__file__).resolve().parent / "last-cotswolds"
    save_scene(scene, out)
    print(scene.summary())
    print(
        f"elevation {scene.stats['elevation_min_m']:.0f}-{scene.stats['elevation_max_m']:.0f} m, "
        f"{scene.stats['field_count']} fields, "
        f"{scene.stats['voxel_chunks']} voxel chunks"
    )
    for warning in scene.warnings:
        print(f"warning: {warning}")
    print("sources:")
    for source in scene.sources:
        print(f"  {source['name']}: {source['attribution']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
