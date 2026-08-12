#!/usr/bin/env python3
"""JSON list encode vs pointpack binary (+ optional gzip)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pointpack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600_000)
    parser.add_argument("--lod", type=int, default=0, help="If >0, LOD to this many points")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((args.n, 3), dtype=np.float32)
    rgb = (rng.random((args.n, 3)) * 255).astype(np.uint8)
    if args.lod > 0:
        xyz, rgb = pointpack.uniform_lod(xyz, rgb, args.lod)

    t0 = time.perf_counter()
    json_bytes = json.dumps(
        np.concatenate([xyz.astype(np.float64), rgb.astype(np.float64)], axis=1).tolist(),
        separators=(",", ":"),
    ).encode("utf-8")
    json_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    blob = pointpack.pack(meta={"n": int(len(xyz))}, xyz=xyz, rgb=rgb, unix_ms=0)
    pack_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    gz, enc = pointpack.compress_body(blob)
    gz_s = time.perf_counter() - t0

    print(f"host={platform.node()}")
    print(f"platform={platform.platform()}")
    print(f"python={platform.python_version()}")
    print(f"numpy={np.__version__}")
    print(f"n={len(xyz)}")
    print(f"json_bytes={len(json_bytes)} encode_s={json_s:.4f}")
    print(f"pack_bytes={len(blob)} encode_s={pack_s:.4f}")
    print(f"gzip_bytes={len(gz)} encoding={enc} gzip_s={gz_s:.4f}")
    print(f"pack_vs_json_size={len(blob) / max(len(json_bytes), 1):.3f}")
    print(f"gzip_vs_json_size={len(gz) / max(len(json_bytes), 1):.3f}")
    print(f"pack_vs_json_time={pack_s / max(json_s, 1e-9):.3f}")

    plot = Path(__file__).resolve().parents[1] / "docs" / "bench_pack.png"
    _write_plot(
        n=int(len(xyz)),
        json_bytes=len(json_bytes),
        pack_bytes=len(blob),
        gzip_bytes=len(gz),
        json_s=json_s,
        pack_s=pack_s,
        out_path=plot,
    )


def _write_plot(
    *,
    n: int,
    json_bytes: int,
    pack_bytes: int,
    gzip_bytes: int,
    json_s: float,
    pack_s: float,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["JSON float arrays", "pointpack Nx16", "pointpack + gzip"]
    sizes_mb = [json_bytes / 1e6, pack_bytes / 1e6, gzip_bytes / 1e6]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=160)
    colors = ["#c44e52", "#4c72b0", "#55a868"]
    bars0 = ax0.bar(labels, sizes_mb, color=colors)
    ax0.set_ylabel("Payload size (MB)")
    ax0.set_title(f"Wire size @ {n:,} RGB points")
    ax0.tick_params(axis="x", rotation=15)
    for bar, v in zip(bars0, sizes_mb):
        ax0.annotate(
            f"{v:.1f} MB",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax0.set_ylim(0, max(sizes_mb) * 1.18)

    t_labels = ["JSON encode", "pointpack encode"]
    t_vals = [json_s, pack_s]
    bars1 = ax1.bar(t_labels, t_vals, color=["#c44e52", "#4c72b0"])
    ax1.set_ylabel("Encode wall time (s)")
    ax1.set_title(f"Encode time @ {n:,} RGB points")
    for bar, v in zip(bars1, t_vals):
        ax1.annotate(
            f"{v:.3f}s",
            xy=(bar.get_x() + bar.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax1.set_ylim(0, max(t_vals) * 1.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote={out_path}")


if __name__ == "__main__":
    main()
