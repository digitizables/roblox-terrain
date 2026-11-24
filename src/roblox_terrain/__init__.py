"""Realistic terrain for Roblox Studio, installed beside roblox-studio-adapter.

Generate a scene in this process, then apply it through a Studio port. The
core adapter does not import this package. The terminal UI is optional.
"""

from __future__ import annotations

from roblox_terrain.apply import apply_scene, clear_generated_terrain
from roblox_terrain.pipeline import generate_realistic_terrain, plan_tiles
from roblox_terrain.scene import TerrainRequest, TerrainScene, save_scene

__version__ = "0.1.0"

__all__ = [
    "TerrainRequest",
    "TerrainScene",
    "__version__",
    "apply_scene",
    "clear_generated_terrain",
    "generate_realistic_terrain",
    "plan_tiles",
    "save_scene",
]
