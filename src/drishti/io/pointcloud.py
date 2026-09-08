"""Minimal, dependency-free point-cloud / mesh writers (binary PLY, OBJ).

Real geometry writers used by S7/S8/S10 so basic exports work without Open3D/trimesh. Heavier formats
(LAS, GeoTIFF, glTF, FBX) live in [exporters.py] behind their own guarded deps.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_ply_points(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write an Nx3 point cloud (optional Nx3 RGB in [0,1] or [0,255]) as binary PLY."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {pts.shape}")
    n = len(pts)
    has_color = colors is not None and len(colors) == n
    if has_color:
        c = np.asarray(colors, dtype=np.float64)
        if c.max() <= 1.0 + 1e-6:
            c = c * 255.0
        c = np.clip(c, 0, 255).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header\n"]
    with open(path, "wb") as f:
        f.write(("\n".join(header)).encode("ascii"))
        if has_color:
            for i in range(n):
                f.write(struct.pack("<fffBBB", pts[i, 0], pts[i, 1], pts[i, 2],
                                    c[i, 0], c[i, 1], c[i, 2]))
        else:
            f.write(pts.tobytes())


def write_obj_mesh(path: str | Path, vertices: np.ndarray, faces: np.ndarray,
                   normals: np.ndarray | None = None) -> None:
    """Write a triangle mesh as Wavefront OBJ (1-indexed faces)."""
    v = np.asarray(vertices, dtype=float)
    fidx = np.asarray(faces, dtype=np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        for p in v:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        if normals is not None:
            for nrm in np.asarray(normals, dtype=float):
                f.write(f"vn {nrm[0]:.6f} {nrm[1]:.6f} {nrm[2]:.6f}\n")
        for tri in fidx:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def read_ply_points(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a PLY point cloud written by :func:`write_ply_points` (or a compatible one).

    Returns (points Nx3 float32, colors Nx3 uint8 or None). Supports binary_little_endian and ascii
    with float x/y/z and optional uchar red/green/blue.
    """
    path = Path(path)
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        fmt = None
        n = 0
        props: list[tuple[str, str]] = []
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n = int(line.split()[2])
            elif line.startswith("element"):
                # another element (e.g. faces) — stop collecting vertex props
                pass
            elif line.startswith("property") and n and not line.startswith("property list"):
                _, ptype, pname = line.split()[:3]
                props.append((pname, ptype))
            elif line == "end_header":
                break
        names = [p[0] for p in props]
        has_rgb = {"red", "green", "blue"}.issubset(names)

        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n)
            data = np.atleast_2d(data)
            pts = data[:, :3].astype(np.float32)
            cols = data[:, [names.index("red"), names.index("green"), names.index("blue")]].astype(np.uint8) if has_rgb else None
            return pts, cols

        # binary_little_endian: build a struct dtype in property order
        type_map = {"float": "<f4", "float32": "<f4", "double": "<f8",
                    "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4"}
        dt = np.dtype([(nm, type_map[tp]) for nm, tp in props])
        arr = np.frombuffer(f.read(dt.itemsize * n), dtype=dt, count=n)
        pts = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
        cols = np.column_stack([arr["red"], arr["green"], arr["blue"]]).astype(np.uint8) if has_rgb else None
        return pts, cols
