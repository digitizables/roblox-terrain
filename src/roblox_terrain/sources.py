"""Attribution for the datasets the pipeline can consume.

The scene records only the sources that were actually used.
"""

from __future__ import annotations

TERRARIUM = {
    "name": "AWS Terrain Tiles (Terrarium)",
    "role": "elevation",
    "url": "https://registry.opendata.aws/terrain-tiles/",
    "attribution": (
        "Elevation resampled from AWS Open Data Terrain Tiles in Terrarium encoding. "
        "The tiles are a Tilezen composite and include NASA SRTM plus regional models "
        "(over Europe, commonly EU-DEM). They are not a raw Copernicus GLO-30 download. "
        "See https://github.com/tilezen/joerd/blob/master/docs/attribution.md"
    ),
}

COPERNICUS = {
    "name": "Copernicus DEM",
    "role": "elevation",
    "url": "https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model",
    "attribution": (
        "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, "
        "provided under COPERNICUS by the European Union and ESA; all rights reserved."
    ),
}

SRTM = {
    "name": "NASA SRTM",
    "role": "elevation",
    "url": "https://www.earthdata.nasa.gov/data/instruments/srtm",
    "attribution": "Elevation derived from NASA Shuttle Radar Topography Mission data.",
}

OSM = {
    "name": "OpenStreetMap",
    "role": "land cover, fields, and waterways",
    "url": "https://www.openstreetmap.org/copyright",
    "attribution": "© OpenStreetMap contributors. Data is available under the Open Database License.",
}

WORLDCOVER = {
    "name": "ESA WorldCover",
    "role": "land cover",
    "url": "https://esa-worldcover.org/",
    "attribution": (
        "ESA WorldCover project 2020/2021. Contains modified Copernicus Sentinel data. "
        "https://esa-worldcover.org/en/data-access"
    ),
}

HYDRO_DERIVED = {
    "name": "DEM-derived drainage",
    "role": "hydrology",
    "url": "",
    "attribution": (
        "Rivers and lakes are computed in this plugin: priority-flood depression filling, "
        "D8 flow direction, and flow accumulation, with OpenStreetMap waterways burned in when present."
    ),
}

HYDROSHEDS = {
    "name": "HydroSHEDS",
    "role": "hydrology",
    "url": "https://www.hydrosheds.org/",
    "attribution": (
        "Flow accumulation or streams supplied by the caller from HydroSHEDS "
        "(Lehner, B., Verdin, K., Jarvis, A., 2008). HydroSHEDS has its own license; "
        "keep that attribution with the file you pass in hydro_path."
    ),
}

SYNTHETIC = {
    "name": "Procedural reference",
    "role": "elevation and land cover",
    "url": "",
    "attribution": "Generated inside roblox-terrain for tests. Not a surveyed landscape.",
}
