"""Binary RGB point-cloud wire format with uniform LOD and optional gzip."""

from .codec import (
    MAGIC,
    POINT_STRIDE,
    compress_body,
    etag_for_unix_ms,
    pack,
    pack_points,
    unpack,
    unpack_header,
)
from .lod import uniform_lod

__all__ = [
    "MAGIC",
    "POINT_STRIDE",
    "compress_body",
    "etag_for_unix_ms",
    "pack",
    "pack_points",
    "uniform_lod",
    "unpack",
    "unpack_header",
]

__version__ = "0.1.0"
