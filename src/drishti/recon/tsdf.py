"""TSDF fusion via Open3D — real, guarded.

Integrates per-keyframe metric depth + color into a scalable TSDF volume and extracts a dense colored
point cloud. No Open3D installed => loud failure. The fused cloud is the metric dense product S8 meshes.

A single monolithic ``ScalableTSDFVolume`` spanning a large aerial scene at a fine voxel size hashes an
enormous number of surface blocks and can exhaust host RAM (the OS OOM-kills the run). :func:`fuse_tsdf_tiled`
bounds peak memory by partitioning the *ground* into tiles, fusing each tile in its own volume that is
freed before the next, and cropping each tile's points to its core cell so overlapping tiles do not
double-count.

The subtlety, and the fix that actually bounds memory for aerial (nadir) capture: a camera ~100 m AGL sees
a ground footprint far wider than a tile, so simply grouping frames by camera-center XY and integrating
each frame's *full* depth map still paints TSDF blocks across the whole scene — every tile then costs
almost as much as the monolithic volume. So before integrating, each depth map is clipped (via
:func:`_clip_depth_to_box`) to the tile's ground-XY window: a tile's volume only hashes blocks for its own
ground region (+overlap).

Bounding XY is necessary but not sufficient. Monocular depth (Depth Anything V2, made metric by a per-frame
scale+shift fit) is *noisy*, and with a large ``depth_trunc`` the unreliable far field scatters back-projected
points across a thick vertical Z slab — every ground column then hashes many vertical blocks and one tile
alone can exhaust RAM. So the same clip also rejects pixels whose back-projected world-Z falls outside the
scene's true surface envelope (``z_lo..z_hi``), which S7 derives from S2's metric sparse points. That drops
gross far-field flyers — pixels no real surface produced — thinning the slab. It is a quality *gain*, not a
loss: sparse points bracket the true ground and structure, and only depth the geometry never supported is
discarded. Voxel size and the ``depth_trunc`` range are untouched; ``tile_m`` and the Z band together trade
tile count and coverage against peak RAM.
"""
from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ..logging import get_logger
from ..runtime.deps import require

log = get_logger("drishti.recon.tsdf")


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


def robust_z_band(
    z_values: np.ndarray, margin_m: float, lo_pct: float = 1.0, hi_pct: float = 99.0,
) -> tuple[float, float]:
    """Robust vertical surface envelope ``[z_lo, z_hi]`` from a set of world-Z samples.

    ``z_values`` are the Z coordinates of the metric sparse points (COLMAP-triangulated true surface, in
    the same ground CRS as the poses). Trimming to the ``[lo_pct, hi_pct]`` percentiles discards sparse
    triangulation flyers; ``margin_m`` then pads both ends so legitimate structure (building tops, ground
    dips) and the depth's own noise at the true surface survive, while gross far-field monocular flyers —
    which land far outside this band — are clipped. Empty/all-invalid input returns an open band
    ``(-inf, inf)`` so clipping becomes a no-op (fail-soft: never delete geometry when we cannot bound it).
    """
    z = np.asarray(z_values, dtype=float).ravel()
    z = z[np.isfinite(z)]
    if z.size == 0:
        return (-np.inf, np.inf)
    z_lo = float(np.percentile(z, lo_pct)) - float(margin_m)
    z_hi = float(np.percentile(z, hi_pct)) + float(margin_m)
    return (z_lo, z_hi)


def _clip_depth_to_box(
    depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
    extr: np.ndarray, x_lo: float, x_hi: float, y_lo: float, y_hi: float,
    z_lo: float = -np.inf, z_hi: float = np.inf,
) -> np.ndarray:
    """Return a copy of ``depth`` with every pixel whose back-projected world point falls outside the box
    ``[x_lo, x_hi) x [y_lo, y_hi) x [z_lo, z_hi)`` set to 0 (invalid).

    The XY bounds are what let per-tile memory track tile extent for nadir aerial imagery: each camera sees
    a ground footprint far wider than a tile, so integrating a frame's *full* depth map would paint TSDF
    blocks across the whole scene regardless of which tile owns the camera. Clipping to the tile's ground
    window means a tile's volume only hashes blocks for its own ground region (+overlap).

    The Z bounds bound the *vertical* slab: noisy monocular depth over a large ``depth_trunc`` scatters
    far-field points across many vertical blocks per ground column. Rejecting pixels whose back-projected
    world-Z lands outside the scene's true surface envelope (from S2 sparse points, see
    :func:`robust_z_band`) drops those flyers. Depth range (``depth_trunc``) and voxel size are untouched —
    only pixels no real surface produced, or that another tile owns, are dropped from *this* tile.
    """
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    d = depth.astype(np.float64)
    # camera-frame ray * depth
    xc = (us - cx) / fx * d
    yc = (vs - cy) / fy * d
    pc = np.stack([xc, yc, d], axis=-1)          # (h, w, 3)
    R = extr[:3, :3]
    t = extr[:3, 3]
    # world = R^T (cam - t); for row-vectors that is (pc - t) @ R
    pw = (pc - t) @ R
    inside = (
        (d > 0) & (pw[..., 0] >= x_lo) & (pw[..., 0] < x_hi)
        & (pw[..., 1] >= y_lo) & (pw[..., 1] < y_hi)
        & (pw[..., 2] >= z_lo) & (pw[..., 2] < z_hi)
    )
    out = depth.copy()
    out[~inside] = 0.0
    return out


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
    z_lo: float = -np.inf, z_hi: float = np.inf,
    sink: Callable[[np.ndarray, np.ndarray], None] | None = None,
):
    """Memory-bounded TSDF fusion by spatial tiling of the ground footprint.

    ``frame_specs`` is a re-iterable sequence of descriptors, each carrying ``center_xy`` (the camera
    position in the ground CRS, an (x, y) pair) plus whatever ``load`` needs to lazily produce that
    frame's ``(color_bgr, depth_m, extrinsic_world2cam)`` — or ``None`` to skip it. The ground is split
    into ``tile_m``-sized core cells; each cell is fused from the frames whose camera center falls within
    ``overlap_m`` of it, and — critically for wide-footprint nadir imagery — each frame's depth map is
    clipped to the cell's ground window (core + overlap) in XY *and* to the scene's true surface envelope
    ``[z_lo, z_hi]`` in Z before integration, so the tile's volume only hashes blocks for its own ground
    region and only where a real surface exists (see :func:`_clip_depth_to_box`). The extracted points are
    then cropped to the core cell so neighbouring tiles do not double-count. Boundary cells extend to
    +/-inf in XY so nothing is dropped at the scene edge; ``z_lo``/``z_hi`` default to an open band
    (no vertical clipping) when the caller cannot supply a surface envelope.

    Memory: each tile's volume is freed before the next, so peak RAM tracks a single tile — *except* that
    accumulating every tile's extracted points to concatenate at the end costs RAM proportional to the
    whole cloud (100M+ points on a large aerial scene => OOM). Pass ``sink`` to avoid that: it is called
    ``sink(points, colors)`` once per non-empty tile with that tile's core-cropped points, which are then
    freed instead of retained (e.g. a :class:`~drishti.io.pointcloud.StreamingPlyWriter` that streams them
    to disk). With a sink the returned point/color arrays are empty (the sink owns the data); with
    ``sink=None`` the points are accumulated and returned in full (convenient for small clouds and tests).

    Returns (points Nx3 float64, colors Nx3 float64 in [0,1], n_frames_used) — ``n_frames_used`` is the
    count of distinct frames integrated into at least one tile. ``points``/``colors`` are empty when a
    ``sink`` consumed the tiles.
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
    z_desc = "open" if not (np.isfinite(z_lo) and np.isfinite(z_hi)) else f"{z_lo:.1f}..{z_hi:.1f}m ({z_hi - z_lo:.1f}m)"
    log.info(
        "TSDF tiled fusion: %d frames over %.0f x %.0f m -> %d x %d = %d tiles "
        "(tile_m=%.0f, overlap_m=%.0f, voxel_m=%.3f, depth_trunc=%.0f, z_band=%s)",
        len(centers), maxs[0] - mins[0], maxs[1] - mins[1], nx, ny, nx * ny,
        tile_m, overlap_m, voxel_m, depth_trunc, z_desc,
    )

    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    used_idx: set[int] = set()

    for iy in range(ny):
        for ix in range(nx):
            x0 = mins[0] + ix * tile_m
            y0 = mins[1] + iy * tile_m
            # Core cell to crop to (dedup). Boundary cells reach +/-inf so far-field ground beyond the
            # camera-center span is kept.
            cx_lo = -np.inf if ix == 0 else x0
            cx_hi = np.inf if ix == nx - 1 else x0 + tile_m
            cy_lo = -np.inf if iy == 0 else y0
            cy_hi = np.inf if iy == ny - 1 else y0 + tile_m
            # Depth-integration window = core + overlap on interior edges (support for a clean seam),
            # +/-inf on boundary edges (keep the scene-edge far field). Clipping each depth map to THIS
            # window is what bounds per-tile memory: a nadir camera's footprint is far wider than a tile,
            # so without clipping every tile would hash TSDF blocks across the whole scene.
            ix_lo = -np.inf if ix == 0 else x0 - overlap_m
            ix_hi = np.inf if ix == nx - 1 else x0 + tile_m + overlap_m
            iy_lo = -np.inf if iy == 0 else y0 - overlap_m
            iy_hi = np.inf if iy == ny - 1 else y0 + tile_m + overlap_m

            # Frames that could see into this window: camera center within the window (nadir cameras look
            # straight down, so the center is the best cheap proxy) expanded by the overlap margin.
            sel = np.where(
                (centers[:, 0] >= x0 - overlap_m) & (centers[:, 0] < x0 + tile_m + overlap_m)
                & (centers[:, 1] >= y0 - overlap_m) & (centers[:, 1] < y0 + tile_m + overlap_m)
            )[0]
            if len(sel) == 0:
                continue

            # Logged before integration so an OOM inside a tile is attributable to a specific tile
            # (the post-extract summary below never prints if the volume exhausts RAM first).
            log.info("  tile [%d,%d] integrating %d candidate frames...", ix, iy, len(sel))
            vol = _make_volume(o3d, voxel_m, sdf_trunc)
            n_here = 0
            for i in sel:
                loaded = load(frame_specs[int(i)])
                if loaded is None:
                    continue
                color_bgr, depth, extr = loaded
                if depth is None or color_bgr is None:
                    continue
                depth_clipped = _clip_depth_to_box(
                    depth, fx, fy, cx, cy, extr, ix_lo, ix_hi, iy_lo, iy_hi, z_lo, z_hi
                )
                if not np.any(depth_clipped > 0):
                    continue  # this frame sees no ground inside the tile window
                _integrate(o3d, vol, intr, color_bgr, depth_clipped, extr, depth_trunc)
                used_idx.add(int(i))
                n_here += 1
            if n_here == 0:
                del vol
                gc.collect()
                continue

            pcd = vol.extract_point_cloud()
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors)
            del vol, pcd
            gc.collect()
            if len(pts) == 0:
                continue

            # crop to the core cell so overlapping tiles don't emit duplicate points
            keep = (
                (pts[:, 0] >= cx_lo) & (pts[:, 0] < cx_hi)
                & (pts[:, 1] >= cy_lo) & (pts[:, 1] < cy_hi)
            )
            n_keep = int(keep.sum())
            log.info("  tile [%d,%d] fused %d frames -> %d pts (%d after core-crop)",
                     ix, iy, n_here, len(pts), n_keep)
            if keep.any():
                if sink is not None:
                    # Stream this tile's points to the sink and free them: peak RAM tracks one tile,
                    # not the whole (100M+ point) cloud. The sink owns the data from here.
                    sink(pts[keep], cols[keep])
                else:
                    all_pts.append(pts[keep])
                    all_cols.append(cols[keep])
            del pts, cols
            gc.collect()

    if not used_idx:
        raise RuntimeError("TSDF fusion integrated 0 frames (no valid depth).")
    if not all_pts:
        return (np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float), len(used_idx))
    return np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0), len(used_idx)
