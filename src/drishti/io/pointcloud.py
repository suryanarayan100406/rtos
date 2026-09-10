"""Minimal, dependency-free point-cloud / mesh writers (binary PLY, OBJ).

Real geometry writers used by S7/S8/S10 so basic exports work without Open3D/trimesh. Heavier formats
(LAS, GeoTIFF, glTF, FBX) live in [exporters.py] behind their own guarded deps.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

# Packed binary-PLY vertex record (x,y,z float32 + r,g,b uint8) — 15 bytes, no padding. numpy's default
# (align=False) layout matches struct.pack("<fffBBB", ...) byte-for-byte, so we can serialize a whole
# chunk with one vectorized ``tobytes()`` instead of a per-point Python loop (catastrophic at 100M+ pts).
_PLY_RGB_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
)


def _rgb_to_u8(colors: np.ndarray, n: int) -> np.ndarray:
    """Normalize an Nx3 color array (float [0,1] or [0,255], or uint8) to uint8, validated against n."""
    c = np.asarray(colors, dtype=np.float64)
    if c.ndim != 2 or c.shape != (n, 3):
        raise ValueError(f"colors must be {n}x3 to match points, got {c.shape}")
    if c.size and c.max() <= 1.0 + 1e-6:
        c = c * 255.0
    return np.clip(c, 0, 255).astype(np.uint8)


def _pack_rgb_chunk(pts: np.ndarray, c_u8: np.ndarray) -> bytes:
    rec = np.empty(len(pts), dtype=_PLY_RGB_DTYPE)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["r"], rec["g"], rec["b"] = c_u8[:, 0], c_u8[:, 1], c_u8[:, 2]
    return rec.tobytes()


def write_ply_points(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write an Nx3 point cloud (optional Nx3 RGB in [0,1] or [0,255]) as binary PLY."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {pts.shape}")
    n = len(pts)
    has_color = colors is not None and len(colors) == n

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
            f.write(_pack_rgb_chunk(pts, _rgb_to_u8(colors, n)))  # vectorized, no per-point loop
        else:
            f.write(pts.tobytes())


class StreamingPlyWriter:
    """Append points to a binary PLY incrementally so peak RAM tracks one chunk, not the whole cloud.

    S7 tiled fusion emits the dense cloud tile-by-tile; holding every tile's points to concatenate at the
    end costs RAM proportional to the *whole* scene (100M+ points on a large aerial capture => OOM, even
    though each tile fits easily). A binary PLY needs the total vertex count in its header up front, so this
    writer streams each chunk to a temp body file, tracks the running count and bounding box, and on
    :meth:`close` writes the real header and copies the body in (disk-to-disk, buffered — RAM-cheap). Use as
    a context manager: the final PLY is written only on clean exit; an exception discards the temp body.
    """

    def __init__(self, path: str | Path, with_color: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.with_color = with_color
        self._body_path = self.path.with_suffix(self.path.suffix + ".body.tmp")
        # Handle intentionally outlives __init__ — it spans many add() calls and is closed in
        # close()/_discard() (this class *is* the context manager), so an inline `with` can't apply.
        self._body = open(self._body_path, "wb")  # noqa: SIM115
        self.count = 0
        self.bbox_min: np.ndarray | None = None
        self.bbox_max: np.ndarray | None = None

    def add(self, points: np.ndarray, colors: np.ndarray | None = None) -> None:
        """Append a chunk (Nx3 points, optional Nx3 colors). Empty chunks are ignored."""
        pts = np.ascontiguousarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must be Nx3, got {pts.shape}")
        n = len(pts)
        if n == 0:
            return
        if self.with_color:
            self._body.write(_pack_rgb_chunk(pts, _rgb_to_u8(colors, n)))
        else:
            self._body.write(pts.tobytes())
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        self.bbox_min = mn if self.bbox_min is None else np.minimum(self.bbox_min, mn)
        self.bbox_max = mx if self.bbox_max is None else np.maximum(self.bbox_max, mx)
        self.count += n

    def close(self) -> None:
        """Finalize the PLY: write the header with the final count and stream the body in."""
        if self._body.closed:
            return
        self._body.close()
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {self.count}",
                  "property float x", "property float y", "property float z"]
        if self.with_color:
            header += ["property uchar red", "property uchar green", "property uchar blue"]
        header += ["end_header\n"]
        with open(self.path, "wb") as f:
            f.write(("\n".join(header)).encode("ascii"))
            with open(self._body_path, "rb") as b:
                shutil.copyfileobj(b, f, length=8 * 1024 * 1024)
        self._body_path.unlink(missing_ok=True)

    def _discard(self) -> None:
        if not self._body.closed:
            self._body.close()
        self._body_path.unlink(missing_ok=True)

    def __enter__(self) -> StreamingPlyWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            self._discard()  # don't leave a truncated/bogus PLY on failure
        return False


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
