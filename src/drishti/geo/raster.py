"""Pure rasterization: bin scattered CRS points into north-up grids (DSM/DTM/ortho).

NumPy-only so it is unit-testable without GDAL/rasterio. The S9 stage ([../stages/s9_geo.py]) wraps
these grids with a CRS + affine transform and writes GeoTIFFs. Grid row 0 is the northernmost row
(standard north-up raster orientation).
"""
from __future__ import annotations

import math

import numpy as np


def grid_shape(xmin: float, ymin: float, xmax: float, ymax: float, res: float) -> tuple[int, int]:
    w = max(1, int(math.ceil((xmax - xmin) / res)))
    h = max(1, int(math.ceil((ymax - ymin) / res)))
    return h, w


def _cells(x: np.ndarray, y: np.ndarray, xmin: float, ymax: float, res: float, h: int, w: int):
    col = np.floor((x - xmin) / res).astype(int)
    row = np.floor((ymax - y) / res).astype(int)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    return row, col, ok


def elevation_grid(x, y, z, res: float, agg: str = "max",
                   bounds: tuple[float, float, float, float] | None = None):
    """Aggregate point elevations into a grid. agg in {max,min,mean}. Empty cells => NaN.

    Returns (grid HxW float32, xmin, ymax).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    if len(x) == 0:
        raise ValueError("no points to rasterize")
    xmin, ymin, xmax, ymax = bounds or (x.min(), y.min(), x.max(), y.max())
    h, w = grid_shape(xmin, ymin, xmax, ymax, res)
    row, col, ok = _cells(x, y, xmin, ymax, res, h, w)
    row, col, z = row[ok], col[ok], z[ok]
    flat = row * w + col

    grid = np.full(h * w, np.nan, dtype=np.float64)
    if agg == "max":
        g = np.full(h * w, -np.inf)
        np.maximum.at(g, flat, z)
        grid = np.where(np.isinf(g), np.nan, g)
    elif agg == "min":
        g = np.full(h * w, np.inf)
        np.minimum.at(g, flat, z)
        grid = np.where(np.isinf(g), np.nan, g)
    elif agg == "mean":
        s = np.zeros(h * w)
        c = np.zeros(h * w)
        np.add.at(s, flat, z)
        np.add.at(c, flat, 1.0)
        grid = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    else:
        raise ValueError(f"unknown agg '{agg}'")
    return grid.reshape(h, w).astype(np.float32), float(xmin), float(ymax)


def ortho_grid(x, y, rgb, res: float, bounds=None):
    """Mean RGB per cell -> (H,W,3) uint8 + (xmin, ymax). rgb in [0,1] or [0,255]."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rgb = np.asarray(rgb, float)
    if rgb.max() <= 1.0 + 1e-6:
        rgb = rgb * 255.0
    xmin, ymin, xmax, ymax = bounds or (x.min(), y.min(), x.max(), y.max())
    h, w = grid_shape(xmin, ymin, xmax, ymax, res)
    row, col, ok = _cells(x, y, xmin, ymax, res, h, w)
    flat = (row[ok] * w + col[ok])
    out = np.zeros((h * w, 3))
    cnt = np.zeros(h * w)
    for k in range(3):
        np.add.at(out[:, k], flat, rgb[ok, k])
    np.add.at(cnt, flat, 1.0)
    nz = cnt > 0
    out[nz] /= cnt[nz, None]
    return np.clip(out.reshape(h, w, 3), 0, 255).astype(np.uint8), float(xmin), float(ymax)


def morphological_dtm(dsm: np.ndarray, window_px: int) -> np.ndarray:
    """Estimate a bare-earth DTM from a DSM by grey-erosion (local min) then dilation (opening).

    A simple, dependency-free stand-in for SMRF ground filtering: removes above-ground structure while
    preserving terrain. Real SMRF (PDAL) is used instead when available.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    k = max(1, int(window_px))
    if k == 1:
        return dsm.copy()
    filled = np.where(np.isnan(dsm), np.nanmax(dsm), dsm)
    pad = k // 2
    padded = np.pad(filled, pad, mode="edge")
    win = sliding_window_view(padded, (k, k))[: dsm.shape[0], : dsm.shape[1]]
    eroded = win.min(axis=(-1, -2))          # local min: strip structures
    padded2 = np.pad(eroded, pad, mode="edge")
    win2 = sliding_window_view(padded2, (k, k))[: dsm.shape[0], : dsm.shape[1]]
    opened = win2.max(axis=(-1, -2))         # dilate back: restore terrain level
    return opened.astype(np.float32)
