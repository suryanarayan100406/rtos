"""S4 — Metric depth per keyframe.

Runs Depth Anything V2 for a relative depth map, then anchors it to metric scale by fitting scale+shift
to the sparse metric points from S2 that reproject into the frame. Dynamic pixels (S3 masks) are marked
invalid. Output is real, metric, and per-frame confidence-tagged; frames with too few sparse anchors are
reported degraded rather than silently trusted (honesty policy).
"""
from __future__ import annotations

from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


@register_stage
class DepthStage(Stage):
    name = "s4_depth"
    requires = ("s2_poses", "s3_masking")
    compute = ComputeNeed(vram_gb=4.0, needs_cuda=False, openvino_ok=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"depth": cfg.depth.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        from ..io.video import read_image
        from ..models.depth import DepthAnythingV2
        from ..recon.depthfit import fit_scale_shift, intrinsics_matrix, project_points
        from ..runtime.deps import require

        cv2 = require("cv2", purpose="depth mask I/O")
        poses_doc = bundle.read_json("s2_poses/poses.json")
        sparse = np.asarray(bundle.read_json("s2_poses/sparse_points.json")["points"], dtype=float)
        masks_doc = bundle.read_json("s3_masking/masks.json")
        mask_by_index = {m["index"]: m["mask"] for m in masks_doc["masks"]}

        intr = poses_doc["intrinsics"]
        if not intr:
            raise ValueError("s4_depth: no intrinsics in poses.json.")
        K = intrinsics_matrix(intr.get("model", ""), intr.get("params", []))

        # frame index lookup: poses store basename; recover index from keyframes.json
        kf_doc = bundle.read_json("s1_frameqa/keyframes.json")
        from pathlib import Path as _P
        idx_by_name = {_P(k["path"]).name: k["index"] for k in kf_doc["keyframes"]}
        path_by_name = {_P(k["path"]).name: k["path"] for k in kf_doc["keyframes"]}

        model = DepthAnythingV2(rt, cfg.depth.model)
        depth_dir = bundle.stage_dir(self.name) / "depth"
        depth_dir.mkdir(exist_ok=True)

        index: list[dict[str, Any]] = []
        residuals: list[float] = []
        min_anchors = 8

        for p in poses_doc["poses"]:
            name = p["name"]
            if name not in path_by_name:
                continue
            img = read_image(bundle.artifact_path(path_by_name[name]))
            h, w = img.shape[:2]
            rel = model.predict(img)  # (H,W) relative depth

            R_wc = np.asarray(p["R"], dtype=float)
            center = np.asarray(p["center_crs"], dtype=float)
            uv, z = project_points(K, R_wc, center, sparse)
            u = np.round(uv[:, 0]).astype(int)
            v = np.round(uv[:, 1]).astype(int)
            vis = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            n_anchor = int(vis.sum())

            degraded_frame = None
            if n_anchor >= min_anchors:
                rel_at = rel[v[vis], u[vis]]
                a, b, inliers = fit_scale_shift(rel_at, z[vis])
                metric = (a * rel + b).astype("float32")
                resid = float(np.median(np.abs(a * rel_at[inliers] + b - z[vis][inliers])))
                residuals.append(resid)
            else:
                # not enough anchors to metric-align this frame: keep relative, flag it
                metric = rel.astype("float32")
                degraded_frame = "insufficient sparse anchors; depth left relative"
                resid = float("nan")

            metric[metric <= 0] = 0.0
            # invalidate dynamic pixels
            midx = idx_by_name.get(name)
            if midx in mask_by_index:
                m = cv2.imread(str(bundle.artifact_path(mask_by_index[midx])), cv2.IMREAD_GRAYSCALE)
                if m is not None and m.shape == metric.shape:
                    metric[m > 0] = 0.0

            dpath = depth_dir / f"{midx if midx is not None else name}.npy"
            np.save(dpath, metric)
            index.append({
                "name": name, "index": midx, "depth": bundle.relpath(dpath),
                "n_anchors": n_anchor, "resid_m": None if resid != resid else round(resid, 3),
                "degraded": degraded_frame,
            })

        n_metric = sum(1 for e in index if e["degraded"] is None)
        med_resid = float(np.median(residuals)) if residuals else float("nan")
        depth_ref = bundle.write_json(f"{self.name}/depth.json", {
            "model": cfg.depth.model, "n_frames": len(index),
            "n_metric_aligned": n_metric, "frames": index,
            "median_resid_m": None if med_resid != med_resid else round(med_resid, 3),
        })

        frac_metric = n_metric / max(1, len(index))
        return StageResult(
            outputs={"depth": depth_ref, "depth_dir": bundle.relpath(depth_dir)},
            metrics={
                "n_frames": len(index), "n_metric_aligned": n_metric,
                "frac_metric_aligned": round(frac_metric, 3),
                "median_resid_m": None if med_resid != med_resid else round(med_resid, 3),
            },
            confidence_summary=round(frac_metric, 3),
            degraded_reason=None if frac_metric > 0.7 else "many frames lacked sparse anchors for metric scale",
        )
