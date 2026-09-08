"""Dependency-free PLY/OBJ writers + PLY reader round-trip."""
import numpy as np

from drishti.io.pointcloud import read_ply_points, write_obj_mesh, write_ply_points


def test_ply_roundtrip_no_color(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    p = tmp_path / "pc.ply"
    write_ply_points(p, pts)
    r, c = read_ply_points(p)
    assert c is None
    assert np.allclose(r, pts, atol=1e-5)


def test_ply_roundtrip_with_color(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cols = np.array([[255, 0, 0], [0, 128, 255]], dtype=float)
    p = tmp_path / "pcc.ply"
    write_ply_points(p, pts, cols)
    r, c = read_ply_points(p)
    assert np.allclose(r, pts, atol=1e-5)
    assert c is not None and c.dtype == np.uint8
    assert list(c[0]) == [255, 0, 0]
    assert list(c[1]) == [0, 128, 255]


def test_ply_color_normalized_0_1_scaled(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0]])
    cols = np.array([[1.0, 0.0, 0.5]])          # 0..1 -> *255
    p = tmp_path / "n.ply"
    write_ply_points(p, pts, cols)
    _, c = read_ply_points(p)
    assert list(c[0]) == [255, 0, 127]


def test_obj_mesh_is_one_indexed(tmp_path):
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    p = tmp_path / "m.obj"
    write_obj_mesh(p, v, faces)
    text = p.read_text(encoding="ascii")
    assert "v 0.000000 0.000000 0.000000" in text
    assert "f 1 2 3" in text
