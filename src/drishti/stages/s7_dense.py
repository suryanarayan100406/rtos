"""S7 — Dense reconstruction (TSDF fusion).

Fuses per-keyframe metric depth + color (with dynamic pixels already zeroed by S3/S4) into a scalable
TSDF volume and extracts a dense colored point cloud in the ground CRS. TSDF is the shipped dense method;
the Gaussian-splatting path requires gsplat and is selected via dense.method=gaussian.
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


@register_stage
class DenseStage(Stage):
    name = "s7_dense"
    requires = ("s4_depth", "s6_global")
    compute = ComputeNeed(vram_gb=6.0, openvino_ok=False)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"dense": cfg.dense.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..io.pointcloud import write_ply_points
        from ..io.video import read_image
        from ..recon.depthfit import intrinsics_matrix

        if cfg.dense.method != "tsdf":
            raise RuntimeError(
                f"s7_dense: method '{cfg.dense.method}' not available in this build; "
                "the shipped dense path is TSDF (set dense.method=tsdf)."
            )

        from ..recon.tsdf import fuse_tsdf

        poses_doc = bundle.read_json("s6_global/poses_global.json")
        depth_doc = bundle.read_json("s4_depth/depth.json")
        depth_by_index = {d["index"]: d for d in depth_doc["frames"]}

        intr = poses_doc["intrinsics"]
        K = intrinsics_matrix(intr.get("model", ""), intr.get("params", []))
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        width, height = int(intr["width"]), int(intr["height"])

        from pathlib import Path as _P
        kf_doc = bundle.read_json("s1_frameqa/keyframes.json")
        idx_by_name = {_P(k["path"]).name: k["index"] for k in kf_doc["keyframes"]}
        path_by_name = {_P(k["path"]).name: k["path"] for k in kf_doc["keyframes"]}

        def _frames():
            for p in poses_doc["poses"]:
                name = p["name"]
                midx = idx_by_name.get(name)
                d = depth_by_index.get(midx)
                if d is None or name not in path_by_name:
                    continue
                depth = np.load(bundle.artifact_path(d["depth"]))
                color = read_image(bundle.artifact_path(path_by_name[name]))
                R_wc = np.asarray(p["R"], dtype=float)
                center = np.asarray(p["center_crs"], dtype=float)
                extr = np.eye(4)
                extr[:3, :3] = R_wc
                extr[:3, 3] = -R_wc @ center
                yield color, depth, extr

        voxel = cfg.dense.tsdf_voxel_m
        points, colors, used = fuse_tsdf(
            _frames(), fx, fy, cx, cy, width, height,
            voxel_m=voxel, sdf_trunc=4.0 * voxel, depth_trunc=cfg.dense.depth_trunc_m,
        )
        if len(points) == 0:
            raise RuntimeError("s7_dense: TSDF produced an empty cloud.")

        ply = bundle.stage_dir(self.name) / "dense.ply"
        write_ply_points(ply, points, colors)
        dense_ref = bundle.write_json(f"{self.name}/dense.json", {
            "method": "tsdf", "voxel_m": voxel, "n_frames_fused": used,
            "n_points": int(len(points)), "cloud": bundle.relpath(ply),
            "bbox_min": points.min(axis=0).tolist(), "bbox_max": points.max(axis=0).tolist(),
        })

        return StageResult(
            outputs={"dense": dense_ref, "cloud": bundle.relpath(ply)},
            metrics={"n_points": int(len(points)), "n_frames_fused": used,
                     "voxel_m": voxel},
            confidence_summary=round(min(1.0, used / max(1, len(poses_doc["poses"]))), 3),
        )
