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

# Open3D cubes the point-cloud bbox and pads it by this factor before building the Poisson octree, so the
# finest leaf edge is extent * POISSON_BBOX_SCALE / 2**depth. Mirrored here (Open3D's create_from_point_
# cloud_poisson default scale=1.1) to size the meshing downsample to what the octree can actually resolve.
POISSON_BBOX_SCALE = 1.1


def meshing_voxel_m(extent_m: float, poisson_depth: int, dense_voxel_m: float,
                    override_m: float = 0.0) -> float:
    """Voxel size to downsample the meshing INPUT to, so Poisson isn't fed detail its octree can't hold.

    ``override_m > 0`` forces that voxel. Otherwise (auto) the target is the finest Poisson octree leaf,
    ``extent_m * POISSON_BBOX_SCALE / 2**poisson_depth`` — the smallest surface detail a depth-``poisson_depth``
    mesh over a scene ``extent_m`` across can represent. It is clamped to ``>= dense_voxel_m`` so we never
    ask for points finer than the dense cloud actually has (which would invent detail and be a no-op that
    just skips the reduction). On a large aerial scene the leaf (~0.16 m for 300 m @ depth 11) is far
    coarser than the 5 cm dense cloud, so this is what bounds S8 RAM without changing the mesh — the 5 cm
    ``dense.ply`` product is untouched.
    """
    if override_m > 0.0:
        return float(override_m)
    octree_leaf = float(extent_m) * POISSON_BBOX_SCALE / float(2 ** int(poisson_depth))
    return max(octree_leaf, float(dense_voxel_m))


@register_stage
class MeshStage(Stage):
    name = "s8_mesh"
    requires = ("s7_dense",)
    compute = ComputeNeed(cpu_only=True)  # Open3D Poisson runs on CPU

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"mesh": cfg.mesh.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..logging import get_logger
        from ..runtime.deps import require

        log = get_logger("drishti.s8_mesh")

        if cfg.mesh.method != "poisson":
            raise RuntimeError(
                f"s8_mesh: method '{cfg.mesh.method}' not available in this build "
                "(shipped: poisson). Set mesh.method=poisson."
            )
        o3d = require("open3d", purpose="Poisson meshing")

        dense = bundle.read_json("s7_dense/dense.json")
        voxel = float(dense.get("voxel_m", 0.05))
        pcd = o3d.io.read_point_cloud(str(bundle.artifact_path(dense["cloud"])))
        n_loaded = len(pcd.points)
        if n_loaded == 0:
            raise RuntimeError("s8_mesh: dense cloud is empty.")

        # The dense cloud lives in the absolute ground CRS (e.g. UTM: easting ~5.8e5, northing ~5.2e6).
        # A scene only ~100 m wide sitting on million-metre coordinates wipes out the float precision
        # that Qhull needs for normal orientation and Poisson, and it fails loudly with QH6417
        # ("wide facet due to facet merges"). Meshing is invariant to a rigid translation, so we do the
        # math in a local frame centred on the cloud, then shift the finished mesh back to the CRS.
        # Pure translation — geometry is bit-for-bit unchanged, no resolution or accuracy trade-off.
        origin = np.asarray(pcd.get_center(), dtype=float)
        pcd.translate(-origin)

        # Bound S8 memory: don't feed Poisson points finer than its octree can represent. The finest
        # octree leaf is scene_extent / 2**poisson_depth (Open3D cubes the bbox, scale ~1.1), so on a
        # large aerial scene the 5 cm dense cloud is far finer than the mesh resolution — meshing all
        # ~100 M points just OOMs for detail Poisson would smooth away anyway. Downsample the working
        # copy to that leaf size (clamped >= dense voxel, never inventing detail). dense.ply stays 5 cm.
        extent = float(np.max(pcd.get_axis_aligned_bounding_box().get_extent()))
        mesh_voxel = meshing_voxel_m(extent, cfg.mesh.poisson_depth, voxel, cfg.mesh.mesh_voxel_m)
        if mesh_voxel > voxel * 1.0001:
            pcd = pcd.voxel_down_sample(mesh_voxel)
            log.info("s8_mesh: downsampled meshing input %d -> %d pts at %.3f m voxel "
                     "(octree leaf for depth %d over %.0f m; dense.ply stays %.3f m)",
                     n_loaded, len(pcd.points), mesh_voxel, cfg.mesh.poisson_depth, extent, voxel)
        else:
            log.info("s8_mesh: meshing %d pts at native %.3f m (already at/under octree leaf)",
                     n_loaded, voxel)
        n_mesh_in = len(pcd.points)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=3.0 * mesh_voxel, max_nn=cfg.mesh.normal_max_nn)
        )
        # Normal orientation. The consistent-tangent-plane method builds an EMST/Riemannian graph over
        # every point (O(N*kNN) memory) and OOM-kills 10 M+ -point clouds — the real s8 memory bomb on
        # aerial captures. For nadir 2.5D terrain the surface is a height field seen from above, so the
        # outward normal is world-up almost everywhere; the O(N) up-prior is both memory-safe and correct.
        if n_mesh_in > cfg.mesh.consistent_normals_max_points:
            pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
            log.info("s8_mesh: oriented %d normals to world-up (aerial prior; > %d pts, "
                     "tangent-plane MST would exhaust RAM)", n_mesh_in, cfg.mesh.consistent_normals_max_points)
        else:
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
            "n_dense_points": int(n_loaded), "n_mesh_input_points": int(n_mesh_in),
            "mesh_voxel_m": round(mesh_voxel, 4),
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
                     "n_mesh_input_points": int(n_mesh_in), "mesh_voxel_m": round(mesh_voxel, 4),
                     "cropped_low_density_frac": round(cropped_frac, 4)},
            confidence_summary=round(max(0.0, 1.0 - cropped_frac), 3),
            degraded_reason=degraded,
        )
