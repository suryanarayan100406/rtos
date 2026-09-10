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
