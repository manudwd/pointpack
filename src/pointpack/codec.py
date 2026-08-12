"""Packed RGB point records for HTTP / IPC (vs JSON float arrays).

Layout (little-endian):
  magic[8] = b'PTPACK01'
  meta_len:u32, point_count:u32, unix_ms:u64
  meta_json[meta_len]
  points[point_count * 16]  each: xyz float32 LE + rgb uint8 + 1 pad
"""

from __future__ import annotations

import gzip
import json
import struct
from typing import Any

import numpy as np

__all__ = [
    "MAGIC",
    "POINT_STRIDE",
    "compress_body",
    "etag_for_unix_ms",
    "pack",
    "pack_points",
    "unpack",
    "unpack_header",
]

MAGIC = b"PTPACK01"
POINT_STRIDE = 16  # 3*f32 + 3*u8 + 1 pad
_HEADER = struct.Struct("<8sIIQ")


def etag_for_unix_ms(unix_ms: int) -> str:
    return f'"{int(unix_ms)}"'


def pack_points(xyz: np.ndarray, rgb: np.ndarray) -> bytes:
    """Pack Nx16 point records only (CPU-overlap friendly)."""
    xyz_a = np.ascontiguousarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb_a = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if xyz_a.shape[0] != rgb_a.shape[0]:
        raise ValueError("xyz/rgb length mismatch")
    n = int(xyz_a.shape[0])
    if n == 0:
        return b""
    records = np.zeros((n, POINT_STRIDE), dtype=np.uint8)
    records[:, 0:12] = np.frombuffer(xyz_a.tobytes(), dtype=np.uint8).reshape(n, 12)
    records[:, 12:15] = rgb_a
    return records.tobytes()


def pack(
    *,
    meta: dict[str, Any],
    xyz: np.ndarray,
    rgb: np.ndarray,
    unix_ms: int,
    records: bytes | None = None,
) -> bytes:
    """Pack meta JSON + point records into one blob."""
    if records is None:
        records = pack_points(xyz, rgb)
        point_count = int(np.asarray(xyz).reshape(-1, 3).shape[0])
    else:
        point_count = len(records) // POINT_STRIDE
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    header = _HEADER.pack(MAGIC, len(meta_bytes), int(point_count), int(unix_ms))
    return header + meta_bytes + records


def compress_body(blob: bytes, *, level: int = 4) -> tuple[bytes, str | None]:
    """Gzip when the compressed payload is at least 5% smaller."""
    compressed = gzip.compress(blob, compresslevel=int(level))
    if len(compressed) < int(len(blob) * 0.95):
        return compressed, "gzip"
    return blob, None


def unpack_header(blob: bytes) -> tuple[dict[str, Any], int, int, int]:
    """Return meta, point_count, unix_ms, points_offset."""
    if len(blob) < _HEADER.size:
        raise ValueError("truncated header")
    magic, meta_len, point_count, unix_ms = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    meta_end = _HEADER.size + int(meta_len)
    if len(blob) < meta_end:
        raise ValueError("truncated meta")
    meta = json.loads(blob[_HEADER.size:meta_end].decode("utf-8"))
    return meta, int(point_count), int(unix_ms), meta_end


def unpack(blob: bytes) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int]:
    """Return meta, xyz float32 (N,3), rgb uint8 (N,3), unix_ms."""
    meta, point_count, unix_ms, offset = unpack_header(blob)
    need = offset + point_count * POINT_STRIDE
    if len(blob) < need:
        raise ValueError("truncated points")
    raw = np.frombuffer(blob, dtype=np.uint8, count=point_count * POINT_STRIDE, offset=offset)
    records = raw.reshape(point_count, POINT_STRIDE)
    xyz = np.frombuffer(records[:, 0:12].tobytes(), dtype=np.float32).reshape(point_count, 3).copy()
    rgb = records[:, 12:15].copy()
    return meta, xyz, rgb, unix_ms
