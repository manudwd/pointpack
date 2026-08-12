"""Round-trip and LOD coverage tests."""

from __future__ import annotations

import gzip

import numpy as np
import pointpack


def test_pack_unpack_roundtrip() -> None:
    xyz = np.array([[1.0, 2.0, 3.0], [4.5, -1.0, 0.25]], dtype=np.float32)
    rgb = np.array([[10, 20, 30], [255, 0, 128]], dtype=np.uint8)
    blob = pointpack.pack(meta={"k": 1}, xyz=xyz, rgb=rgb, unix_ms=123)
    meta, xyz2, rgb2, unix_ms = pointpack.unpack(blob)
    assert meta == {"k": 1}
    assert unix_ms == 123
    np.testing.assert_allclose(xyz2, xyz, rtol=0, atol=0)
    np.testing.assert_array_equal(rgb2, rgb)


def test_compress_when_smaller() -> None:
    xyz = np.zeros((50_000, 3), dtype=np.float32)
    rgb = np.zeros((50_000, 3), dtype=np.uint8)
    blob = pointpack.pack(meta={}, xyz=xyz, rgb=rgb, unix_ms=0)
    body, enc = pointpack.compress_body(blob)
    assert enc == "gzip"
    assert gzip.decompress(body) == blob


def test_uniform_lod_budget_and_endpoints() -> None:
    n = 1000
    xyz = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    rgb = np.zeros((n, 3), dtype=np.uint8)
    xyz_k, rgb_k = pointpack.uniform_lod(xyz, rgb, max_points=100)
    assert len(xyz_k) == 100
    assert len(rgb_k) == 100
    np.testing.assert_array_equal(xyz_k[0], xyz[0])
    np.testing.assert_array_equal(xyz_k[-1], xyz[-1])


def test_etag() -> None:
    assert pointpack.etag_for_unix_ms(42) == '"42"'
