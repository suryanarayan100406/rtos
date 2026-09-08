"""TSDF fusion via Open3D — real, guarded.

Integrates per-keyframe metric depth + color into a scalable TSDF volume and extracts a dense colored
point cloud. No Open3D installed => loud failure. The fused cloud is the metric dense product S8 meshes.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..runtime.deps import require


def fuse_tsdf(
    frames: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    fx: float, fy: float, cx: float, cy: float, width: int, height: int,
    voxel_m: float, sdf_trunc: float, depth_trunc: float,
):
    """Fuse (color_bgr, depth_m, extrinsic_world2cam) frames into a dense point cloud.

    Returns (points Nx3 float64, colors Nx3 float64 in [0,1], n_frames_used).
    """
    o3d = require("open3d", purpose="TSDF fusion")
    intr = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), fx, fy, cx, cy)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(voxel_m),
        sdf_trunc=float(sdf_trunc),
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    used = 0
    for color_bgr, depth, extr in frames:
        if depth is None or color_bgr is None:
            continue
        rgb = np.ascontiguousarray(color_bgr[..., ::-1])  # BGR->RGB
        color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )
        vol.integrate(rgbd, intr, extr.astype(np.float64))
        used += 1
    if used == 0:
        raise RuntimeError("TSDF fusion integrated 0 frames (no valid depth).")
    pcd = vol.extract_point_cloud()
    return np.asarray(pcd.points), np.asarray(pcd.colors), used
