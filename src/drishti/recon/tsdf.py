"""TSDF fusion via Open3D — real, guarded.

Integrates per-keyframe metric depth + color into a scalable TSDF volume and extracts a dense colored
point cloud. No Open3D installed => loud failure. The fused cloud is the metric dense product S8 meshes.

A single monolithic ``ScalableTSDFVolume`` spanning a large aerial scene at a fine voxel size hashes an
enormous number of surface blocks and can exhaust host RAM (the OS OOM-kills the run). :func:`fuse_tsdf_tiled`
bounds peak memory by partitioning frames into ground tiles (by camera-center XY), fusing each tile in its
own volume that is freed before the next, and cropping each tile's points to its core cell so overlapping
tiles do not double-count. It keeps the full ``depth_trunc`` range — memory is bounded by tile *extent*,
not by throwing away far-field depth.
"""
from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ..runtime.deps import require


def _make_volume(o3d, voxel_m: float, sdf_trunc: float):
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_m),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )


def _integrate(o3d, vol, intr, color_bgr: np.ndarray, depth: np.ndarray,
               extr: np.ndarray, depth_trunc: float) -> None:
    rgb = np.ascontiguousarray(color_bgr[..., ::-1])  # BGR->RGB
    color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False,
    )
    vol.integrate(rgbd, intr, extr.astype(np.float64))


def fuse_tsdf(
    frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    fx: float, fy: float, cx: float, cy: float, width: int, height: int,
    voxel_m: float, sdf_trunc: float, depth_trunc: float,
):
    """Fuse (color_bgr, depth_m, extrinsic_world2cam) frames into a dense point cloud.

    Single-volume path. Returns (points Nx3 float64, colors Nx3 float64 in [0,1], n_frames_used).
    For large aerial scenes prefer :func:`fuse_tsdf_tiled`, which bounds peak RAM.
    """
    o3d = require("open3d", purpose="TSDF fusion")
    intr = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)
    vol = _make_volume(o3d, voxel_m, sdf_trunc)
    used = 0
    for color_bgr, depth, extr in frames:
        if depth is None or color_bgr is None:
            continue
        _integrate(o3d, vol, intr, color_bgr, depth, extr, depth_trunc)
        used += 1
    if used == 0:
        raise RuntimeError("TSDF fusion integrated 0 frames (no valid depth).")
    pcd = vol.extract_point_cloud()
    return np.asarray(pcd.points), np.asarray(pcd.colors), used


def fuse_tsdf_tiled(
    frame_specs: Sequence[dict],
    load: Callable[[dict], tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    fx: float, fy: float, cx: float, cy: float, width: int, height: int,
    voxel_m: float, sdf_trunc: float, depth_trunc: float,
    tile_m: float, overlap_m: float,
):
    """Memory-bounded TSDF fusion by spatial tiling of the ground footprint.

    ``frame_specs`` is a re-iterable sequence of descriptors, each carrying ``center_xy`` (the camera
    position in the ground CRS, an (x, y) pair) plus whatever ``load`` needs to lazily produce that
    frame's ``(color_bgr, depth_m, extrinsic_world2cam)`` — or ``None`` to skip it. Frames are grouped
    into ``tile_m``-sized core cells over the camera-center XY extent; each cell is fused from the frames
    whose camera center falls within ``overlap_m`` of it (so a tile has enough surrounding views to close
    its core), then the extracted points are cropped to the core cell so neighbouring tiles do not
    double-count. Boundary cells extend to +/-inf so nothing is dropped at the scene edge.

    Returns (points Nx3 float64, colors Nx3 float64 in [0,1], n_frames_used) — ``n_frames_used`` is the
    count of distinct frames integrated into at least one tile.
    """
    o3d = require("open3d", purpose="TSDF fusion")
    if tile_m <= 0.0:
        raise ValueError(f"tile_m must be > 0, got {tile_m}")
    intr = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)

    centers = np.array([s["center_xy"] for s in frame_specs], dtype=float)
    if len(centers) == 0:
        raise RuntimeError("TSDF fusion received 0 frames (no valid poses/depth).")
    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)
    nx = max(1, int(np.ceil((maxs[0] - mins[0]) / tile_m)))
    ny = max(1, int(np.ceil((maxs[1] - mins[1]) / tile_m)))

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    used_idx: set[int] = set()

    for iy in range(ny):
        for ix in range(nx):
            x0 = mins[0] + ix * tile_m
            y0 = mins[1] + iy * tile_m
            # frames whose camera center lies within the core cell expanded by the overlap margin
            sel = np.where(
                (centers[:, 0] >= x0 - overlap_m) & (centers[:, 0] < x0 + tile_m + overlap_m)
                & (centers[:, 1] >= y0 - overlap_m) & (centers[:, 1] < y0 + tile_m + overlap_m)
            )[0]
            if len(sel) == 0:
                continue

            vol = _make_volume(o3d, voxel_m, sdf_trunc)
            n_here = 0
            for i in sel:
                loaded = load(frame_specs[int(i)])
                if loaded is None:
                    continue
                color_bgr, depth, extr = loaded
                if depth is None or color_bgr is None:
                    continue
                _integrate(o3d, vol, intr, color_bgr, depth, extr, depth_trunc)
                used_idx.add(int(i))
                n_here += 1
            if n_here == 0:
                del vol
                continue

            pcd = vol.extract_point_cloud()
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors)
            del vol, pcd
            gc.collect()
            if len(pts) == 0:
                continue

            # crop to the core cell so overlapping tiles don't emit duplicate points; boundary
            # cells reach to +/-inf so far-field ground beyond the camera-center span is kept.
            x_lo = -np.inf if ix == 0 else x0
            x_hi = np.inf if ix == nx - 1 else x0 + tile_m
            y_lo = -np.inf if iy == 0 else y0
            y_hi = np.inf if iy == ny - 1 else y0 + tile_m
            keep = (
                (pts[:, 0] >= x_lo) & (pts[:, 0] < x_hi)
                & (pts[:, 1] >= y_lo) & (pts[:, 1] < y_hi)
            )
            if keep.any():
                all_pts.append(pts[keep])
                all_cols.append(cols[keep])

    if not used_idx:
        raise RuntimeError("TSDF fusion integrated 0 frames (no valid depth).")
    if not all_pts:
        return (np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float), len(used_idx))
    return np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0), len(used_idx)
