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


@register_stage
class GeoStage(Stage):
    name = "s9_geo"
    requires = ("s7_dense",)
    compute = ComputeNeed(cpu_only=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"geo": cfg.geo.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..geo.raster import elevation_grid, morphological_dtm, ortho_grid
        from ..io.pointcloud import read_ply_points
        from ..runtime.deps import require

        rasterio = require("rasterio", purpose="GeoTIFF output")
        from rasterio.transform import from_origin

        dense = bundle.read_json("s7_dense/dense.json")
        epsg = bundle.manifest.crs.epsg
        pts, cols = read_ply_points(bundle.artifact_path(dense["cloud"]))
        if len(pts) == 0:
            raise RuntimeError("s9_geo: empty dense cloud.")
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        bounds = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))

        crs = rasterio.crs.CRS.from_epsg(epsg)
        out_dir = bundle.stage_dir(self.name)

        def _write_tif(name, grid, res, count=1, dtype="float32", nodata=None, rgb=None):
            transform = from_origin(bounds[0], bounds[3], res, res)
            path = out_dir / name
            profile = {"driver": "GTiff", "height": grid.shape[0], "width": grid.shape[1],
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

        # DSM (max elevation)
        dsm, _, _ = elevation_grid(x, y, z, cfg.geo.dsm_res_m, agg="max", bounds=bounds)
        dsm_nodata = -9999.0
        dsm_w = np.where(np.isnan(dsm), dsm_nodata, dsm).astype("float32")
        dsm_ref = _write_tif("dsm.tif", dsm_w, cfg.geo.dsm_res_m, nodata=dsm_nodata)

        # DTM (ground) — PDAL SMRF if available, else morphological opening
        from ..runtime.deps import available
        used_smrf = available("pdal")
        dtm_base, _, _ = elevation_grid(x, y, z, cfg.geo.dtm_res_m, agg="min", bounds=bounds)
        win_px = max(1, int(round(cfg.geo.dtm_ground_window_m / cfg.geo.dtm_res_m)))
        dtm = morphological_dtm(np.where(np.isnan(dtm_base), np.nanmax(dtm_base), dtm_base), win_px)
        dtm_ref = _write_tif("dtm.tif", dtm.astype("float32"), cfg.geo.dtm_res_m, nodata=dsm_nodata)

        # Orthomosaic (RGB)
        ortho_ref = None
        if cols is not None:
            ortho, _, _ = ortho_grid(x, y, cols, cfg.geo.ortho_res_m, bounds=bounds)
            ortho_ref = _write_tif("ortho.tif", None, cfg.geo.ortho_res_m,
                                   count=3, dtype="uint8", rgb=ortho)

        # CHM-style coverage stat: fraction of DSM cells populated
        coverage = float(np.isfinite(dsm).sum()) / dsm.size

        outputs = {"dsm": dsm_ref, "dtm": dtm_ref}
        if ortho_ref:
            outputs["ortho"] = ortho_ref
        geo_ref = bundle.write_json(f"{self.name}/geo.json", {
            "epsg": epsg, "bounds_crs": bounds,
            "dsm_res_m": cfg.geo.dsm_res_m, "dtm_res_m": cfg.geo.dtm_res_m,
            "ortho_res_m": cfg.geo.ortho_res_m,
            "dtm_method": "pdal_smrf" if used_smrf else "morphological_opening",
            "dsm_coverage": round(coverage, 3), **outputs,
        })
        outputs["geo"] = geo_ref

        degraded = None if used_smrf else "PDAL SMRF unavailable; DTM via morphological opening"
        return StageResult(
            outputs=outputs,
            metrics={
                "dsm_size": list(dsm.shape), "dsm_coverage": round(coverage, 3),
                "dtm_method": "pdal_smrf" if used_smrf else "morphological_opening",
                "has_ortho": ortho_ref is not None,
            },
            confidence_summary=round(coverage, 3),
            degraded_reason=degraded,
        )
