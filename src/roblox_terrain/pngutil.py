"""Stdlib PNG reader for 8-bit RGB Terrarium tiles. No Pillow dependency."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def read_rgb_png(path: Path | bytes) -> np.ndarray:
    """Return an array of shape (height, width, 3). Interlaced PNGs are rejected."""
    data = path if isinstance(path, bytes) else Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Not a PNG file")
    pos = 8
    width = height = None
    idat = b""
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit, color, _comp, _filter, interlace = struct.unpack(">IIBBBBB", chunk)
            if bit != 8 or color != 2 or interlace != 0:
                raise ValueError("Only non-interlaced 8-bit RGB PNGs are supported")
        elif kind == b"IDAT":
            idat += chunk
        elif kind == b"IEND":
            break
    if width is None or height is None:
        raise ValueError("PNG is missing a header")
    raw = zlib.decompress(idat)
    stride = width * 3
    rows = np.empty((height, width, 3), dtype=np.uint8)
    cursor = 0
    previous = np.zeros(stride, dtype=np.int16)
    for row in range(height):
        filter_kind = raw[cursor]
        cursor += 1
        scan = np.frombuffer(raw[cursor : cursor + stride], dtype=np.uint8).astype(np.int16).copy()
        cursor += stride
        if filter_kind == 0:
            pass
        elif filter_kind == 1:
            for index in range(stride):
                left = scan[index - 3] if index >= 3 else 0
                scan[index] = (scan[index] + left) & 255
        elif filter_kind == 2:
            scan = (scan + previous) & 255
        elif filter_kind == 3:
            for index in range(stride):
                left = scan[index - 3] if index >= 3 else 0
                scan[index] = (scan[index] + ((left + int(previous[index])) // 2)) & 255
        elif filter_kind == 4:
            for index in range(stride):
                left = int(scan[index - 3]) if index >= 3 else 0
                up = int(previous[index])
                up_left = int(previous[index - 3]) if index >= 3 else 0
                scan[index] = (scan[index] + _paeth(left, up, up_left)) & 255
        else:
            raise ValueError(f"Unsupported PNG filter {filter_kind}")
        previous = scan
        rows[row] = scan.reshape(width, 3).astype(np.uint8)
    return rows


def terrarium_elevation(rgb: np.ndarray) -> np.ndarray:
    """Decode Terrarium RGB to elevation in meters.

    ``elevation = (red * 256 + green + blue / 256) - 32768``.
    """
    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)
    return red * 256.0 + green + blue / 256.0 - 32768.0


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    pa = abs(estimate - left)
    pb = abs(estimate - up)
    pc = abs(estimate - up_left)
    if pa <= pb and pa <= pc:
        return left
    if pb <= pc:
        return up
    return up_left
