"""`report` — final accuracy + confidence report (HTML + JSON).

Reads the finished manifest and emits report.json + report.html summarizing every stage's measured
metrics, confidence, timing, and any degradation. If ground check-points are supplied, elevation error
is validated against the DSM; otherwise accuracy is reported as UNVALIDATED (never fabricated).
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..report import build_report, render_html
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


def _validate_checkpoints(bundle: Bundle, cfg: DrishtiConfig) -> dict | None:
    """Validate elevation against user check-points by sampling the DSM. Returns None if not possible."""
    cp = cfg.report.check_points
    if not cp:
        return None
    from pathlib import Path

    from ..runtime.deps import available
    if not (Path(cp).is_file() and available("rasterio") and bundle.exists("s9_geo/geo.json")):
        return None
    import csv

    import numpy as np
    import rasterio

    from ..geo.project import project_lonlat

    geo = bundle.read_json("s9_geo/geo.json")
    if "dsm" not in geo:
        return None
    epsg = bundle.manifest.crs.epsg

    lats, lons, alts = [], [], []
    with open(cp, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lats.append(float(row.get("lat") or row.get("latitude")))
                lons.append(float(row.get("lon") or row.get("longitude")))
                alts.append(float(row.get("alt") or row.get("altitude") or row.get("abs_alt")))
            except (TypeError, ValueError):
                continue
    if len(lats) < 1:
        return None

    east, north = project_lonlat(lons, lats, epsg)
    errs = []
    with rasterio.open(bundle.artifact_path(geo["dsm"])) as ds:
        for e, n, a in zip(east, north, alts, strict=True):
            for val in ds.sample([(e, n)]):
                z = float(val[0])
                if z != ds.nodata and np.isfinite(z):
                    errs.append(z - a)
    if not errs:
        return None
    errs = np.asarray(errs)
    return {
        "n": int(len(errs)),
        "rmse_z_m": round(float(np.sqrt(np.mean(errs**2))), 3),
        "mae_z_m": round(float(np.mean(np.abs(errs))), 3),
        "bias_z_m": round(float(np.mean(errs)), 3),
        "note": "elevation error vs supplied check-points, sampled from the DSM",
    }


@register_stage
class ReportStage(Stage):
    name = "report"
    requires = ("s10_export",)
    compute = ComputeNeed(cpu_only=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"report": cfg.report.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        checkpoints = _validate_checkpoints(bundle, cfg)
        manifest = bundle.manifest.model_dump(mode="json")
        report = build_report(manifest, checkpoints)

        json_ref = bundle.write_json("report/report.json", report)
        html_path = bundle.path / "report" / "report.html"
        html_path.write_text(render_html(report), encoding="utf-8")
        html_ref = bundle.relpath(html_path)

        return StageResult(
            outputs={"report_json": json_ref, "report_html": html_ref},
            metrics={
                "total_wall_seconds": report["total_wall_seconds"],
                "n_failed": report["n_failed"],
                "n_degraded": report["n_degraded"],
                "accuracy_validated": report["accuracy"]["validated"],
            },
            confidence_summary=1.0 if report["n_failed"] == 0 else 0.0,
        )
