"""Land-cover classes shared by the raster, the field pass, and materials.

OpenStreetMap tags and ESA WorldCover codes both land in this small set.
Water is a class here and is also refined by the hydrology pass.
"""

from __future__ import annotations

UNKNOWN = 0
WATER = 1
CROPLAND = 2
PASTURE = 3
ORCHARD = 4
FOREST = 5
SCRUB = 6
WETLAND = 7
BARE = 8
URBAN = 9
SNOW = 10

NAMES = {
    UNKNOWN: "unknown",
    WATER: "water",
    CROPLAND: "cropland",
    PASTURE: "pasture",
    ORCHARD: "orchard",
    FOREST: "forest",
    SCRUB: "scrub",
    WETLAND: "wetland",
    BARE: "bare",
    URBAN: "urban",
    SNOW: "snow",
}

BY_NAME = {name: code for code, name in NAMES.items()}

AGRICULTURAL = (CROPLAND, PASTURE, ORCHARD)

# Higher values overwrite lower ones when polygons overlap.
PRIORITY = {
    UNKNOWN: 0,
    PASTURE: 2,
    SCRUB: 3,
    WETLAND: 3,
    BARE: 3,
    CROPLAND: 4,
    ORCHARD: 5,
    FOREST: 6,
    SNOW: 7,
    URBAN: 8,
    WATER: 10,
}

# ESA WorldCover 2020/2021 class values.
WORLDCOVER = {
    10: FOREST,
    20: SCRUB,
    30: PASTURE,
    40: CROPLAND,
    50: URBAN,
    60: BARE,
    70: SNOW,
    80: WATER,
    90: WETLAND,
    95: WETLAND,
    100: BARE,
}

LANDUSE = {
    "farmland": CROPLAND,
    "farmyard": URBAN,
    "meadow": PASTURE,
    "orchard": ORCHARD,
    "vineyard": ORCHARD,
    "forest": FOREST,
    "grass": PASTURE,
    "allotments": CROPLAND,
    "residential": URBAN,
    "commercial": URBAN,
    "industrial": URBAN,
    "cemetery": PASTURE,
    "quarry": BARE,
    "village_green": PASTURE,
}

NATURAL = {
    "wood": FOREST,
    "scrub": SCRUB,
    "water": WATER,
    "wetland": WETLAND,
    "grassland": PASTURE,
    "bare_rock": BARE,
    "sand": BARE,
    "beach": BARE,
    "heath": SCRUB,
    "fell": BARE,
}

WATERWAY = {
    "river": "river",
    "stream": "stream",
    "ditch": "ditch",
    "drain": "ditch",
    "canal": "river",
    "brook": "stream",
}
