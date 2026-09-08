"""S10 — Export deliverables.

Emits the required output formats (OBJ, PLY, LAS, GeoTIFF, glTF/GLB, FBX; plus LAZ, 3D Tiles, Potree
when their tools are present) from the canonical bundle products. Each format is attempted with real
tools; a format whose tool is missing is recorded as skipped-with-reason, never faked. If a MUST format
is skipped the stage is marked degraded so the report surfaces it.
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage

_MUST = {"obj", "ply", "las", "geotiff", "gltf", "glb", "fbx"}


@register_stage
class ExportStage(Stage):
    name = "s10_export"
    requires = ("s8_mesh", "s9_geo")
    compute = ComputeNeed(cpu_only=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"export": cfg.export.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        from pathlib import Path

        from ..io import exporters as ex
        from ..io.pointcloud import read_ply_points

        mesh_doc = bundle.read_json("s8_mesh/mesh.json")
        dense_doc = bundle.read_json("s7_dense/dense.json")
        geo_doc = bundle.read_json("s9_geo/geo.json")
        epsg = bundle.manifest.crs.epsg

        mesh_ply = Path(bundle.artifact_path(mesh_doc["mesh"]))
        dense_ply = Path(bundle.artifact_path(dense_doc["cloud"]))
        out = bundle.stage_dir(self.name)

        results: dict[str, dict[str, Any]] = {}

        def record(fmt: str, ok: bool, info: str):
            results[fmt] = {"ok": ok, "path": bundle.relpath(info) if ok else None,
                            "reason": None if ok else info}

        wanted = [f.lower() for f in (cfg.export.formats + cfg.export.also)]
        # points loaded once for LAS/LAZ/potree
        pts = cols = None
        if any(f in wanted for f in ("las", "laz", "potree")):
            pts, cols = read_ply_points(dense_ply)

        for fmt in wanted:
            if fmt == "obj":
                record(fmt, *ex.export_obj(mesh_ply, out / "model.obj"))
            elif fmt == "ply":
                record(fmt, *ex.copy_product(mesh_ply, out / "model.ply"))
            elif fmt in ("gltf", "glb"):
                record(fmt, *ex.export_glb(mesh_ply, out / f"model.{fmt}"))
            elif fmt == "fbx":
                record(fmt, *ex.export_fbx_via_blender(mesh_ply, out / "model.fbx"))
            elif fmt == "las":
                record(fmt, *ex.export_las(pts, cols, out / "points.las", epsg, cfg.export.point_format))
            elif fmt == "laz":
                record(fmt, *ex.export_las(pts, cols, out / "points.laz", epsg, cfg.export.point_format, laz=True))
            elif fmt == "geotiff":
                # copy DSM/DTM/ortho
                any_ok = False
                for key in ("dsm", "dtm", "ortho"):
                    if key in geo_doc and geo_doc[key]:
                        ok, info = ex.copy_product(bundle.artifact_path(geo_doc[key]), out / f"{key}.tif")
                        any_ok = any_ok or ok
                record("geotiff", any_ok, str(out) if any_ok else "no GeoTIFFs from S9")
            elif fmt == "3dtiles":
                record(fmt, *ex.export_3dtiles(mesh_ply, out / "3dtiles"))
            elif fmt == "potree":
                record(fmt, *ex.export_potree(dense_ply, out / "potree"))
            else:
                record(fmt, False, f"unknown format '{fmt}'")

        must_wanted = [f for f in wanted if f in _MUST]
        must_ok = [f for f in must_wanted if results.get(f, {}).get("ok")]
        skipped_must = [f for f in must_wanted if not results.get(f, {}).get("ok")]

        export_ref = bundle.write_json(f"{self.name}/exports.json", {
            "epsg": epsg, "results": results,
            "must_formats": must_wanted, "must_ok": must_ok, "skipped_must": skipped_must,
        })

        degraded = None
        if skipped_must:
            degraded = "MUST formats skipped (missing tools): " + ", ".join(skipped_must)
        return StageResult(
            outputs={"exports": export_ref},
            metrics={
                "formats_ok": [f for f, r in results.items() if r["ok"]],
                "formats_skipped": {f: r["reason"] for f, r in results.items() if not r["ok"]},
                "n_ok": sum(1 for r in results.values() if r["ok"]),
            },
            confidence_summary=round(len(must_ok) / max(1, len(must_wanted)), 3),
            degraded_reason=degraded,
        )
