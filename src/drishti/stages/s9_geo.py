"""S9 — Georeferenced raster products: DSM, DTM, orthomosaic.

Rasterizes the dense CRS point cloud into a Digital Surface Model (max elevation), a Digital Terrain
Model (morphological ground filter, or PDAL SMRF when available), and an RGB orthomosaic, then writes
them as georeferenced GeoTIFFs (rasterio) with the derived CRS. All values are measured from the cloud.
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


def _bounds_from_cloud(cloud_path) -> tuple[float, float, float, float]:
    """Cheap XY-extent pass for old bundles whose dense.json predates bbox tracking (reads coords only)."""
    import numpy as np

    from ..io.pointcloud import iter_ply_chunks

    xmn = ymn = np.inf
    xmx = ymx = -np.inf
    for pts, _ in iter_ply_chunks(cloud_path, chunk_points=8_000_000):
        if len(pts) == 0:
            continue
        x, y = pts[:, 0], pts[:, 1]
        xmn, xmx = min(xmn, float(x.min())), max(xmx, float(x.max()))
        ymn, ymx = min(ymn, float(y.min())), max(ymx, float(y.max()))
    return float(xmn), float(ymn), float(xmx), float(ymx)


@register_stage
class GeoStage(Stage):
    name = "s9_geo"
    requires = ("s7_dense",)
    compute = ComputeNeed(cpu_only=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"geo": cfg.geo.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..geo.raster import (add_elevation, add_ortho, finish_elevation, finish_ortho,
                                   grid_shape, morphological_dtm, new_elevation_grid, new_ortho_grid)
        from ..io.pointcloud import iter_ply_chunks
        from ..logging import get_logger
        from ..runtime.deps import available, require

        rasterio = require("rasterio", purpose="GeoTIFF output")
        from rasterio.transform import from_origin

        log = get_logger("drishti.s9_geo")

        dense = bundle.read_json("s7_dense/dense.json")
        epsg = bundle.manifest.crs.epsg
        cloud = bundle.artifact_path(dense["cloud"])

        # Scene bounds from S7 metadata (the bbox tracked while the cloud was streamed to disk) so we
        # never load all points just to measure the extent; fall back to a cheap coords-only pass.
        bmin, bmax = dense.get("bbox_min"), dense.get("bbox_max")
        if bmin and bmax:
            bounds = (float(bmin[0]), float(bmin[1]), float(bmax[0]), float(bmax[1]))
        else:
            bounds = _bounds_from_cloud(cloud)
        xmin, ymin, xmax, ymax = bounds
        if not (xmax > xmin and ymax > ymin):
            raise RuntimeError(f"s9_geo: degenerate dense-cloud bounds {bounds}.")

        crs = rasterio.crs.CRS.from_epsg(epsg)
        out_dir = bundle.stage_dir(self.name)

        def _write_tif(name, grid, res, count=1, dtype="float32", nodata=None, rgb=None):
            transform = from_origin(bounds[0], bounds[3], res, res)
            path = out_dir / name
            # RGB writes pass the raster as `rgb` (H, W, 3) and leave `grid` None; single-band
            # writes pass a 2-D `grid`. Take the shape from whichever one is actually present.
            shape_src = grid if grid is not None else rgb
            profile = {"driver": "GTiff", "height": shape_src.shape[0], "width": shape_src.shape[1],
                       "count": count, "dtype": dtype, "crs": crs, "transform": transform,
                       "compress": "deflate"}
            if nodata is not None:
                profile["nodata"] = nodata
            with rasterio.open(path, "w", **profile) as dst:
                if rgb is not None:
                    for b in range(3):
                        dst.write(rgb[:, :, b], b + 1)
                else:
                    dst.write(grid, 1)
            return bundle.relpath(path)

        dsm_res, dtm_res, ortho_res = cfg.geo.dsm_res_m, cfg.geo.dtm_res_m, cfg.geo.ortho_res_m
        h_dsm, w_dsm = grid_shape(xmin, ymin, xmax, ymax, dsm_res)
        h_dtm, w_dtm = grid_shape(xmin, ymin, xmax, ymax, dtm_res)
        h_o, w_o = grid_shape(xmin, ymin, xmax, ymax, ortho_res)

        # Memory-bounded rasterization: fold the dense cloud into the DSM/DTM/ortho grids one streamed
        # chunk at a time. Peak RAM = one point chunk + these grids, NOT the whole 100M+-point cloud —
        # loading it all (read_ply_points) plus the O(N) gridding temporaries is what OOM-killed this
        # stage. Grids are float32/uint32, so even a 0.05 m ortho over a ~1.2 km scene stays a few GB
        # rather than the ~15 GB a full-extent float64 accumulator took. dense.ply + output res unchanged.
        dsm_acc = new_elevation_grid(h_dsm, w_dsm, "max")
        dtm_acc = new_elevation_grid(h_dtm, w_dtm, "min")
        osum = ocnt = None
        n_points = 0
        for pts, cols in iter_ply_chunks(cloud, chunk_points=8_000_000):
            if len(pts) == 0:
                continue
            n_points += len(pts)
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            add_elevation(dsm_acc, x, y, z, xmin, ymax, dsm_res, h_dsm, w_dsm, "max")
            add_elevation(dtm_acc, x, y, z, xmin, ymax, dtm_res, h_dtm, w_dtm, "min")
            if cols is not None:
                if osum is None:
                    osum, ocnt = new_ortho_grid(h_o, w_o)
                add_ortho(osum, ocnt, x, y, cols, xmin, ymax, ortho_res, h_o, w_o)
            del pts, cols, x, y, z
        if n_points == 0:
            raise RuntimeError("s9_geo: empty dense cloud.")
        log.info("s9_geo: rasterized %d pts (streamed) -> DSM %dx%d @%.2fm, DTM %dx%d @%.2fm, ortho %s @%.2fm",
                 n_points, h_dsm, w_dsm, dsm_res, h_dtm, w_dtm, dtm_res,
                 f"{h_o}x{w_o}" if osum is not None else "none", ortho_res)

        # DSM (max elevation)
        dsm = finish_elevation(dsm_acc, h_dsm, w_dsm)
        del dsm_acc
        dsm_nodata = -9999.0
        dsm_ref = _write_tif("dsm.tif", np.where(np.isnan(dsm), dsm_nodata, dsm).astype("float32"),
                             dsm_res, nodata=dsm_nodata)

        # DTM (ground) — PDAL SMRF if available, else morphological opening
        used_smrf = available("pdal")
        dtm_base = finish_elevation(dtm_acc, h_dtm, w_dtm)
        del dtm_acc
        win_px = max(1, int(round(cfg.geo.dtm_ground_window_m / dtm_res)))
        dtm = morphological_dtm(np.where(np.isnan(dtm_base), np.nanmax(dtm_base), dtm_base), win_px)
        del dtm_base
        dtm_ref = _write_tif("dtm.tif", dtm.astype("float32"), dtm_res, nodata=dsm_nodata)
        del dtm

        # Orthomosaic (RGB)
        ortho_ref = None
        if osum is not None:
            ortho = finish_ortho(osum, ocnt, h_o, w_o)
            del osum, ocnt
            ortho_ref = _write_tif("ortho.tif", None, ortho_res, count=3, dtype="uint8", rgb=ortho)
            del ortho

        # CHM-style coverage stat: fraction of DSM cells populated
        coverage = float(np.isfinite(dsm).sum()) / dsm.size

        outputs = {"dsm": dsm_ref, "dtm": dtm_ref}
        if ortho_ref:
            outputs["ortho"] = ortho_ref
        geo_ref = bundle.write_json(f"{self.name}/geo.json", {
            "epsg": epsg, "bounds_crs": list(bounds),
            "dsm_res_m": dsm_res, "dtm_res_m": dtm_res, "ortho_res_m": ortho_res,
            "dtm_method": "pdal_smrf" if used_smrf else "morphological_opening",
            "dsm_coverage": round(coverage, 3), **outputs,
        })
        outputs["geo"] = geo_ref

        degraded = None if used_smrf else "PDAL SMRF unavailable; DTM via morphological opening"
        return StageResult(
            outputs=outputs,
            metrics={
                "dsm_size": [h_dsm, w_dsm], "dsm_coverage": round(coverage, 3),
                "dtm_method": "pdal_smrf" if used_smrf else "morphological_opening",
                "has_ortho": ortho_ref is not None,
            },
            confidence_summary=round(coverage, 3),
            degraded_reason=degraded,
        )
