"""Pure rasterization: DSM/DTM/ortho gridding + morphological ground filter."""
import numpy as np
import pytest

from drishti.geo.raster import elevation_grid, grid_shape, morphological_dtm, ortho_grid


def test_grid_shape():
    assert grid_shape(0, 0, 2.5, 2.5, 1.0) == (3, 3)
    assert grid_shape(0, 0, 10, 5, 2.0) == (3, 5)


def test_elevation_grid_max_and_north_up():
    # two points in the SW cell (0,0); one in the NE cell (2.5,2.5)
    x = np.array([0.0, 0.0, 2.5])
    y = np.array([0.0, 0.0, 2.5])
    z = np.array([1.0, 5.0, 9.0])
    grid, xmin, ymax = elevation_grid(x, y, z, res=1.0, agg="max")
    assert grid.shape == (3, 3)
    assert (xmin, ymax) == (0.0, 2.5)
    assert grid[2, 0] == 5.0        # SW cell -> bottom-left, max(1,5)
    assert grid[0, 2] == 9.0        # NE cell -> top-right (row 0 = northernmost)
    assert np.isnan(grid[1, 1])     # empty cell -> NaN


def test_elevation_grid_min_and_mean():
    x = np.array([0.0, 0.0])
    y = np.array([0.0, 0.0])
    z = np.array([2.0, 8.0])
    gmin, _, _ = elevation_grid(x, y, z, res=1.0, agg="min")
    gmean, _, _ = elevation_grid(x, y, z, res=1.0, agg="mean")
    assert np.nanmin(gmin) == 2.0
    assert np.nanmax(gmean) == 5.0


def test_elevation_grid_errors():
    with pytest.raises(ValueError):
        elevation_grid([], [], [], res=1.0)
    with pytest.raises(ValueError):
        elevation_grid([0.0], [0.0], [1.0], res=1.0, agg="bogus")


def test_ortho_grid_scales_unit_rgb():
    x = np.array([0.0])
    y = np.array([0.0])
    rgb = np.array([[1.0, 0.0, 0.5]])           # 0..1 -> scaled to 0..255
    img, _, _ = ortho_grid(x, y, rgb, res=1.0)
    assert img.dtype == np.uint8
    assert list(img[0, 0]) == [255, 0, 127]


def test_morphological_dtm_removes_structure():
    dsm = np.full((7, 7), 10.0, dtype=np.float32)
    dsm[3, 3] = 25.0                            # a building spike
    dtm = morphological_dtm(dsm, window_px=3)
    assert dtm.shape == dsm.shape
    assert dtm[3, 3] == pytest.approx(10.0)     # spike removed
    assert float(dtm.max()) <= float(dsm.max())


def test_morphological_dtm_window_one_is_identity():
    dsm = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert np.array_equal(morphological_dtm(dsm, window_px=1), dsm)
