"""Point budget reduction that preserves spatial coverage."""

from __future__ import annotations

import numpy as np

__all__ = ["uniform_lod"]


def uniform_lod(
    xyz: np.ndarray,
    rgb: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Even index subsample when ``len(xyz) > max_points``.

    ``np.linspace`` index pick covers the full cloud better than
    ``xyz[::stride]``, which undershoots the budget and biases toward the
    start of the array.

    Time ``O(K)`` copy after ``O(K)`` index build; space ``O(K)`` with
    ``K = min(N, max_points)``.
    """
    xyz_a = np.asarray(xyz)
    rgb_a = np.asarray(rgb)
    n = int(xyz_a.shape[0])
    if n == 0 or max_points <= 0 or n <= int(max_points):
        return xyz_a, rgb_a
    idx = np.linspace(0, n - 1, num=int(max_points), dtype=np.int64)
    return xyz_a[idx], rgb_a[idx]
