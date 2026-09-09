"""S8 — Surface mesh + texture.

Screened Poisson reconstruction (Open3D) over the dense cloud, with low-density (extrapolated) vertices
cropped and flagged so inferred geometry is distinguishable from measured. Colors carry through as
per-vertex color (the shipped texturing baseline); MVS-Texturing atlasing is an optional enhancement.
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


@register_stage
class MeshStage(Stage):
    name = "s8_mesh"
    requires = ("s7_dense",)
    compute = ComputeNeed(cpu_only=True)  # Open3D Poisson runs on CPU

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"mesh": cfg.mesh.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..runtime.deps import require

        if cfg.mesh.method != "poisson":
            raise RuntimeError(
                f"s8_mesh: method '{cfg.mesh.method}' not available in this build "
                "(shipped: poisson). Set mesh.method=poisson."
            )
        o3d = require("open3d", purpose="Poisson meshing")

        dense = bundle.read_json("s7_dense/dense.json")
        voxel = float(dense.get("voxel_m", 0.05))
        pcd = o3d.io.read_point_cloud(str(bundle.artifact_path(dense["cloud"])))
        if len(pcd.points) == 0:
            raise RuntimeError("s8_mesh: dense cloud is empty.")

        # The dense cloud lives in the absolute ground CRS (e.g. UTM: easting ~5.8e5, northing ~5.2e6).
        # A scene only ~100 m wide sitting on million-metre coordinates wipes out the float precision
        # that Qhull needs for normal orientation and Poisson, and it fails loudly with QH6417
        # ("wide facet due to facet merges"). Meshing is invariant to a rigid translation, so we do the
        # math in a local frame centred on the cloud, then shift the finished mesh back to the CRS.
        # Pure translation — geometry is bit-for-bit unchanged, no resolution or accuracy trade-off.
        origin = np.asarray(pcd.get_center(), dtype=float)
        pcd.translate(-origin)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=3.0 * voxel, max_nn=cfg.mesh.normal_max_nn)
        )
        pcd.orient_normals_consistent_tangent_plane(cfg.mesh.normal_max_nn)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=cfg.mesh.poisson_depth
        )
        densities = np.asarray(densities)
        n_before = len(mesh.vertices)
        if cfg.mesh.density_quantile > 0 and len(densities):
            thresh = np.quantile(densities, cfg.mesh.density_quantile)
            mesh.remove_vertices_by_mask(densities < thresh)
        mesh.compute_vertex_normals()

        n_v, n_f = len(mesh.vertices), len(mesh.triangles)
        if n_f == 0:
            raise RuntimeError("s8_mesh: Poisson produced no faces (try higher poisson_depth).")

        # Undo the local-frame shift: put the mesh back in the ground CRS the dense cloud used.
        mesh.translate(origin)

        mesh_ply = bundle.stage_dir(self.name) / "mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_ply), mesh, write_vertex_colors=True,
                                   write_vertex_normals=True)

        cropped_frac = (n_before - n_v) / max(1, n_before)
        mesh_ref = bundle.write_json(f"{self.name}/mesh.json", {
            "method": "poisson", "poisson_depth": cfg.mesh.poisson_depth,
            "n_vertices": n_v, "n_faces": n_f,
            "cropped_low_density_frac": round(cropped_frac, 4),
            "texture": "per_vertex_color",
            "mesh": bundle.relpath(mesh_ply),
        })

        degraded = None
        if cfg.mesh.texture == "mvs_texturing":
            degraded = "MVS-Texturing atlas not enabled; using per-vertex color"
        return StageResult(
            outputs={"mesh": mesh_ref, "mesh_ply": bundle.relpath(mesh_ply)},
            metrics={"n_vertices": n_v, "n_faces": n_f,
                     "cropped_low_density_frac": round(cropped_frac, 4)},
            confidence_summary=round(max(0.0, 1.0 - cropped_frac), 3),
            degraded_reason=degraded,
        )
