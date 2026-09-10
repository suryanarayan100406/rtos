"""Tiled TSDF fusion: partition math, core-cell cropping, and dedup.

Open3D is an optional native dep and is usually absent in CI, so we inject a minimal fake ``open3d``
module. The fake volume simply remembers every point handed to it (one point per integrated frame, at
that frame's camera XY) and returns them from ``extract_point_cloud`` — enough to exercise the tiling,
overlap selection, and boundary-inclusive cropping that :func:`fuse_tsdf_tiled` owns.
"""
import sys
import types

import numpy as np
import pytest


class _FakeImage:
    def __init__(self, arr):
        self.arr = arr


class _FakeRGBD:
    def __init__(self, color, depth):
        self.color, self.depth = color, depth


class _FakePCD:
    def __init__(self, pts):
        self.points = pts
        self.colors = np.tile(np.array([0.5, 0.5, 0.5]), (len(pts), 1))


class _FakeVolume:
    """Records the world XY of each integrated camera; emits one point per camera at that XY."""

    def __init__(self, **kwargs):
        self._cams = []

    def integrate(self, rgbd, intr, extr):
        # extr is world->cam (R | -R c). Recover camera center c = -R^T t.
        R = np.asarray(extr)[:3, :3]
        t = np.asarray(extr)[:3, 3]
        c = -R.T @ t
        self._cams.append(c)

    def extract_point_cloud(self):
        return _FakePCD(np.array(self._cams, dtype=float) if self._cams else np.empty((0, 3)))


def _install_fake_open3d(monkeypatch):
    o3d = types.ModuleType("open3d")
    o3d.camera = types.SimpleNamespace(
        PinholeCameraIntrinsic=lambda *a, **k: object()
    )
    o3d.geometry = types.SimpleNamespace(
        Image=_FakeImage,
        RGBDImage=types.SimpleNamespace(
            create_from_color_and_depth=lambda color, depth, **k: _FakeRGBD(color, depth)
        ),
    )
    o3d.pipelines = types.SimpleNamespace(
        integration=types.SimpleNamespace(
            ScalableTSDFVolume=_FakeVolume,
            TSDFVolumeColorType=types.SimpleNamespace(RGB8=0),
        )
    )
    monkeypatch.setitem(sys.modules, "open3d", o3d)


def _spec(x, y):
    R = np.eye(3)
    c = np.array([x, y, 10.0])
    extr = np.eye(4)
    extr[:3, :3] = R
    extr[:3, 3] = -R @ c
    return {"center_xy": (x, y), "extr": extr, "id": (x, y)}


def _load(spec):
    color = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    return color, depth, spec["extr"]


def test_tiled_fusion_covers_all_and_no_duplicates(monkeypatch):
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    # 5x5 grid of cameras 20 m apart spanning 0..80 m; 30 m tiles => several core cells.
    specs = [_spec(20.0 * i, 20.0 * j) for i in range(5) for j in range(5)]

    pts, cols, used = fuse_tsdf_tiled(
        specs, _load, fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4,
        voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0,
        tile_m=30.0, overlap_m=8.0,
    )

    assert used == 25
    # Each camera-center point lands in exactly one core cell => exactly 25 points, no duplicates.
    assert len(pts) == 25
    assert len(cols) == 25
    xy = {(round(p[0], 3), round(p[1], 3)) for p in pts}
    assert xy == {(20.0 * i, 20.0 * j) for i in range(5) for j in range(5)}


def test_tiled_keeps_far_field_at_scene_edge(monkeypatch):
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    # Two clusters far apart on X: the boundary cells extend to +/-inf so nothing is dropped.
    specs = [_spec(0.0, 0.0), _spec(5.0, 0.0), _spec(500.0, 0.0)]
    pts, _, used = fuse_tsdf_tiled(
        specs, _load, fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4,
        voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0,
        tile_m=30.0, overlap_m=8.0,
    )
    assert used == 3
    xs = sorted(round(p[0], 1) for p in pts)
    assert xs == [0.0, 5.0, 500.0]


def test_tiled_raises_on_no_frames(monkeypatch):
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    with pytest.raises(RuntimeError, match="0 frames"):
        fuse_tsdf_tiled(
            [], _load, fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4,
            voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0, tile_m=30.0, overlap_m=8.0,
        )


def test_tiled_skips_frames_load_returns_none(monkeypatch):
    from drishti.recon.tsdf import fuse_tsdf_tiled

    _install_fake_open3d(monkeypatch)
    specs = [_spec(0.0, 0.0), _spec(10.0, 0.0), _spec(20.0, 0.0)]

    def _load_some_none(spec):
        if spec["id"] == (10.0, 0.0):
            return None
        return _load(spec)

    pts, _, used = fuse_tsdf_tiled(
        specs, _load_some_none, fx=1.0, fy=1.0, cx=2.0, cy=2.0, width=4, height=4,
        voxel_m=0.05, sdf_trunc=0.2, depth_trunc=150.0, tile_m=30.0, overlap_m=8.0,
    )
    assert used == 2
    assert len(pts) == 2


def _nadir_extr(cx_world, cy_world, height):
    """world->cam for a nadir camera at (cx_world, cy_world, height) looking straight down (-Z world).

    Camera axes: +Xc = +X world, +Yc = -Y world, +Zc = -Z world (so a positive camera-frame depth maps
    to a point *below* the camera). extr maps world->cam: p_cam = R (p_world - c).
    """
    R = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]])
    c = np.array([cx_world, cy_world, height])
    extr = np.eye(4)
    extr[:3, :3] = R
    extr[:3, 3] = -R @ c
    return extr


def test_clip_depth_to_xy_keeps_only_pixels_inside_window():
    from drishti.recon.tsdf import _clip_depth_to_xy

    # 5x5 nadir view, camera 100 m above world origin, 1 m/px ground sampling at this depth.
    fx = fy = 100.0
    cxp = cyp = 2.0  # principal point at pixel (2,2)
    h = w = 5
    depth = np.full((h, w), 100.0, dtype=np.float32)  # flat ground 100 m below
    extr = _nadir_extr(0.0, 0.0, 100.0)

    # Ground XY under each pixel: X = (u-2)/100 * 100 = u-2  in [-2..2]; Y = -(v-2) in [-2..2].
    # Keep the window [0,3) x [0,3): X in {0,1,2} -> u in {2,3,4}; Y in {0,1,2} -> v in {0,1,2}.
    out = _clip_depth_to_xy(depth, fx, fy, cxp, cyp, extr,
                            x_lo=0.0, x_hi=3.0, y_lo=0.0, y_hi=3.0)
    kept = out > 0
    expected = np.zeros((h, w), dtype=bool)
    for v in range(h):
        for u in range(w):
            X = (u - cxp)
            Y = -(v - cyp)
            expected[v, u] = (0.0 <= X < 3.0) and (0.0 <= Y < 3.0)
    assert np.array_equal(kept, expected)
    # kept pixels keep their original depth value; others are zeroed
    assert np.all(out[kept] == 100.0)
    assert np.all(out[~kept] == 0.0)


def test_clip_depth_to_xy_ignores_invalid_depth():
    from drishti.recon.tsdf import _clip_depth_to_xy

    fx = fy = 100.0
    depth = np.zeros((5, 5), dtype=np.float32)  # all invalid
    extr = _nadir_extr(0.0, 0.0, 100.0)
    out = _clip_depth_to_xy(depth, fx, fy, 2.0, 2.0, extr,
                            x_lo=-np.inf, x_hi=np.inf, y_lo=-np.inf, y_hi=np.inf)
    assert np.all(out == 0.0)  # zero-depth pixels never become valid, even in an infinite window


def test_clip_depth_to_xy_infinite_window_is_noop():
    from drishti.recon.tsdf import _clip_depth_to_xy

    fx = fy = 100.0
    depth = np.full((5, 5), 100.0, dtype=np.float32)
    extr = _nadir_extr(50.0, -30.0, 100.0)
    out = _clip_depth_to_xy(depth, fx, fy, 2.0, 2.0, extr,
                            x_lo=-np.inf, x_hi=np.inf, y_lo=-np.inf, y_hi=np.inf)
    assert np.array_equal(out, depth)  # boundary tiles (+/-inf on all sides) drop nothing
