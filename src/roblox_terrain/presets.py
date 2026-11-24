"""Named regions.

``cotswolds-windrush`` ships with the package: a real elevation grid and the
OpenStreetMap land cover for that bbox. The other geographic presets are the
same pipeline with a bbox. They download only when ``allow_network`` is set,
or they read files the caller passes. ``synthetic-valley`` is a procedural
stand-in for tests and does not claim to be a surveyed place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    id: str
    title: str
    summary: str
    west: float
    south: float
    east: float
    north: float
    field_style: str
    boundary_style: str
    field_size_m: float
    crops: tuple[str, ...]
    bundled: bool = False
    real_place: bool = True


PRESETS: dict[str, Preset] = {
    "cotswolds-windrush": Preset(
        id="cotswolds-windrush",
        title="Windrush valley, Cotswolds",
        summary=(
            "About 1.6 km by 1.5 km of Gloucestershire between Naunton and Upper Slaughter. "
            "Rolling limestone, hedged fields, woodland, and a river. The generator frames "
            "an agricultural window inside this extract unless you set play_extent_m to cover it."
        ),
        west=-1.790,
        south=51.905,
        east=-1.768,
        north=51.920,
        field_style="irregular",
        boundary_style="hedge",
        field_size_m=110,
        crops=("pasture", "pasture", "wheat", "barley", "fallow", "rapeseed"),
        bundled=True,
    ),
    "whittlesey-fens": Preset(
        id="whittlesey-fens",
        title="Whittlesey Fens",
        summary=(
            "A flat arable stretch of the Cambridgeshire Fens. Not bundled. "
            "Pass allow_network or a local DEM. Fields are long rectangles and boundaries are ditches."
        ),
        west=-0.150,
        south=52.545,
        east=-0.125,
        north=52.560,
        field_style="rectangular",
        boundary_style="ditch",
        field_size_m=180,
        crops=("wheat", "wheat", "roots", "barley", "fallow", "rapeseed"),
    ),
    "palouse": Preset(
        id="palouse",
        title="Palouse hills",
        summary=(
            "Rolling wheat country near Colfax, Washington. Not bundled. "
            "Boundaries are fences. Large rectangular parcels follow the slope."
        ),
        west=-117.410,
        south=46.875,
        east=-117.380,
        north=46.895,
        field_style="rectangular",
        boundary_style="fence",
        field_size_m=220,
        crops=("wheat", "wheat", "wheat", "fallow", "barley"),
    ),
    "synthetic-valley": Preset(
        id="synthetic-valley",
        title="Synthetic valley",
        summary=(
            "A procedural valley, terrace, wood, and lake used to exercise the pipeline without data files. "
            "It is not a real place."
        ),
        west=-1.800,
        south=51.900,
        east=-1.800 + (400 / (111_320.0 * 0.622)),
        north=51.900 + (400 / 111_320.0),
        field_style="irregular",
        boundary_style="hedge",
        field_size_m=70,
        crops=("wheat", "pasture", "barley", "fallow"),
        bundled=True,
        real_place=False,
    ),
}


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown preset {name!r}. Known presets: {known}.") from exc
