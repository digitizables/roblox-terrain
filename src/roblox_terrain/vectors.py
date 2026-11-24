"""Polygons and waterways, from a compact GeoJSON or from an OSM XML extract."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from roblox_terrain.cover import LANDUSE, NATURAL, WATERWAY


@dataclass
class PolyFeature:
    cover: str
    rings: list[np.ndarray]
    kind: str
    osm_id: str


@dataclass
class LineFeature:
    kind: str
    coords: np.ndarray
    osm_id: str


@dataclass
class VectorSet:
    polygons: list[PolyFeature] = field(default_factory=list)
    lines: list[LineFeature] = field(default_factory=list)
    attribution: str = "© OpenStreetMap contributors"

    def to_geojson(self) -> dict:
        features = []
        for poly in self.polygons:
            coordinates = [_ring_list(ring) for ring in poly.rings]
            features.append(
                {
                    "type": "Feature",
                    "properties": {"cover": poly.cover, "kind": poly.kind, "osm_id": poly.osm_id},
                    "geometry": {"type": "Polygon", "coordinates": coordinates},
                }
            )
        for line in self.lines:
            features.append(
                {
                    "type": "Feature",
                    "properties": {"cover": line.kind, "kind": line.kind, "osm_id": line.osm_id},
                    "geometry": {"type": "LineString", "coordinates": _ring_list(line.coords)},
                }
            )
        return {
            "type": "FeatureCollection",
            "attribution": self.attribution,
            "features": features,
        }

    @classmethod
    def from_geojson(cls, payload: dict) -> VectorSet:
        vectors = cls(attribution=str(payload.get("attribution") or cls.attribution))
        for feature in payload.get("features", []):
            props = feature.get("properties") or {}
            geometry = feature.get("geometry") or {}
            kind = str(props.get("kind") or props.get("cover") or "")
            osm_id = str(props.get("osm_id") or "")
            gtype = geometry.get("type")
            coords = geometry.get("coordinates") or []
            if gtype == "Polygon":
                rings = [_as_array(ring) for ring in coords if len(ring) >= 4]
                if rings:
                    vectors.polygons.append(
                        PolyFeature(cover=str(props.get("cover") or kind), rings=rings, kind=kind, osm_id=osm_id)
                    )
            elif gtype == "LineString":
                line = _as_array(coords)
                if len(line) >= 2:
                    vectors.lines.append(LineFeature(kind=str(props.get("cover") or kind), coords=line, osm_id=osm_id))
        return vectors


def parse_osm_xml(text: str | bytes) -> VectorSet:
    """Read an OSM 0.6 map document into agricultural polygons and waterways.

    Multipolygon relations become polygons with holes. A way that is a member
    of such a relation is not also emitted on its own, so a hole is not painted
    as a second woodland.
    """
    if isinstance(text, bytes):
        root = ET.fromstring(text)
    else:
        root = ET.fromstring(text)
    nodes: dict[str, tuple[float, float]] = {}
    ways: dict[str, dict] = {}
    for element in root:
        if element.tag == "node":
            nodes[element.get("id", "")] = (float(element.get("lon", "0")), float(element.get("lat", "0")))
        elif element.tag == "way":
            refs = [node.get("ref", "") for node in element if node.tag == "nd"]
            tags = {tag.get("k", ""): tag.get("v", "") for tag in element if tag.tag == "tag"}
            ways[element.get("id", "")] = {"refs": refs, "tags": tags}
    consumed: set[str] = set()
    vectors = VectorSet()
    for element in root:
        if element.tag != "relation":
            continue
        tags = {tag.get("k", ""): tag.get("v", "") for tag in element if tag.tag == "tag"}
        cover = _cover_from_tags(tags)
        if cover is None:
            continue
        outers: list[list[tuple[float, float]]] = []
        inners: list[list[tuple[float, float]]] = []
        for member in element:
            if member.tag != "member" or member.get("type") != "way":
                continue
            ref = member.get("ref", "")
            way = ways.get(ref)
            if way is None:
                continue
            line = _coords(way["refs"], nodes)
            if len(line) < 2:
                continue
            role = member.get("role") or "outer"
            consumed.add(ref)
            if role == "inner":
                inners.append(line)
            else:
                outers.append(line)
        for shell in _assemble(outers):
            holes = _assemble(inners)
            rings = [np.asarray(shell, dtype=np.float64)] + [np.asarray(hole, dtype=np.float64) for hole in holes]
            kind = _kind(tags)
            vectors.polygons.append(PolyFeature(cover=cover, rings=rings, kind=kind, osm_id=element.get("id", "")))
    for way_id, way in ways.items():
        if way_id in consumed:
            continue
        tags = way["tags"]
        line = _coords(way["refs"], nodes)
        if len(line) < 2:
            continue
        waterway = WATERWAY.get(tags.get("waterway", ""))
        if waterway:
            vectors.lines.append(LineFeature(kind=waterway, coords=np.asarray(line, dtype=np.float64), osm_id=way_id))
            continue
        cover = _cover_from_tags(tags)
        if cover is None or len(line) < 4:
            continue
        if not _closed(line):
            continue
        vectors.polygons.append(
            PolyFeature(
                cover=cover,
                rings=[np.asarray(line, dtype=np.float64)],
                kind=_kind(tags),
                osm_id=way_id,
            )
        )
    return vectors


def load_vectors(path: Path) -> VectorSet:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        return VectorSet.from_geojson(payload)
    raise ValueError(f"{path} is not a GeoJSON FeatureCollection")


def _cover_from_tags(tags: dict[str, str]) -> str | None:
    if "landuse" in tags and tags["landuse"] in LANDUSE:
        return _name(LANDUSE[tags["landuse"]])
    if "natural" in tags and tags["natural"] in NATURAL:
        return _name(NATURAL[tags["natural"]])
    return None


def _name(code: int) -> str:
    from roblox_terrain.cover import NAMES

    return NAMES[code]


def _kind(tags: dict[str, str]) -> str:
    for key in ("landuse", "natural", "waterway"):
        if key in tags:
            return f"{key}={tags[key]}"
    return "untagged"


def _coords(refs: list[str], nodes: dict[str, tuple[float, float]]) -> list[tuple[float, float]]:
    line = []
    for ref in refs:
        point = nodes.get(ref)
        if point is not None:
            line.append((round(point[0], 6), round(point[1], 6)))
    return line


def _closed(line: list[tuple[float, float]]) -> bool:
    return len(line) >= 4 and _same(line[0], line[-1])


def _same(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) <= 1e-6 and abs(a[1] - b[1]) <= 1e-6


def _assemble(segments: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Join way fragments that share an endpoint into rings or open lines."""
    unused = set(range(len(segments)))
    rings: list[list[tuple[float, float]]] = []
    while unused:
        index = unused.pop()
        ring = list(segments[index])
        progressed = True
        while progressed and not _same(ring[0], ring[-1]):
            progressed = False
            for other in list(unused):
                segment = segments[other]
                if _same(segment[0], ring[-1]):
                    ring.extend(segment[1:])
                    unused.remove(other)
                    progressed = True
                    break
                if _same(segment[-1], ring[-1]):
                    ring.extend(list(reversed(segment[:-1])))
                    unused.remove(other)
                    progressed = True
                    break
        if len(ring) >= 4:
            rings.append(ring)
    return rings


def _ring_list(coords: np.ndarray) -> list[list[float]]:
    return [[round(float(lon), 6), round(float(lat), 6)] for lon, lat in np.asarray(coords)]


def _as_array(coords: list) -> np.ndarray:
    return np.asarray(coords, dtype=np.float64)
