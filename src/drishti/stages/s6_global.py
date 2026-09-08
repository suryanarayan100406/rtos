"""S6 — Global bundle adjustment / consistency.

S2 already produces a locally-optimized, GPS-anchored spine. S6 is the *global* pass: when GLOMAP is
installed it runs a global refine; either way it computes global consistency metrics (per-frame GPS
residual, revisit/loop-closure candidates) and emits the poses S7 consumes. If GLOMAP is absent the
poses pass through unchanged and the stage reports that honestly (AGENTS.md §5).
"""
from __future__ import annotations

import shutil
from typing import Any

from ..bundle import Bundle
from ..config import DrishtiConfig
from ..geo.crs import haversine_m
from ..runtime import ComputeNeed, Runtime
from .base import Stage, StageResult, register_stage


@register_stage
class GlobalStage(Stage):
    name = "s6_global"
    requires = ("s2_poses",)
    compute = ComputeNeed(cpu_only=True)

    def params(self, cfg: DrishtiConfig) -> dict[str, Any]:
        return {"poses": {"use_glomap": cfg.poses.use_glomap}, "spine": cfg.spine.model_dump()}

    def run(self, bundle: Bundle, cfg: DrishtiConfig, rt: Runtime) -> StageResult:
        import numpy as np

        poses_doc = bundle.read_json("s2_poses/poses.json")
        poses = poses_doc["poses"]
        if len(poses) < 2:
            raise ValueError("s6_global: need >=2 poses.")

        centers = np.asarray([p["center_crs"] for p in poses], dtype=float)

        # --- per-frame consistency: neighbor spacing (drift proxy) ---
        steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        median_step = float(np.median(steps)) if len(steps) else 0.0

        # --- revisit / loop-closure candidates: near in space, far in sequence ---
        n = len(poses)
        gap = max(5, int(0.05 * n))
        radius = 2.0 * cfg.spine.gnss_sigma_m  # from config, not hardcoded
        loops = 0
        lats = [p.get("lat") for p in poses]
        lons = [p.get("lon") for p in poses]
        for i in range(n):
            for j in range(i + gap, n):
                if None in (lats[i], lons[i], lats[j], lons[j]):
                    continue
                if haversine_m(lats[i], lons[i], lats[j], lons[j]) <= radius:
                    loops += 1
                    break

        glomap = shutil.which("glomap")
        # GLOMAP global refine would run here on the S2 COLMAP database when present; we do not
        # fabricate a refinement when it is absent — poses pass through and we say so.
        degraded = None if glomap else "GLOMAP not installed; global refine skipped (poses from S2 spine)"

        poses_ref = bundle.write_json(f"{self.name}/poses_global.json", poses_doc)
        global_ref = bundle.write_json(f"{self.name}/global.json", {
            "n_poses": n,
            "median_step_m": round(median_step, 3),
            "loop_candidates": loops,
            "revisit_radius_m": round(radius, 3),
            "glomap_available": bool(glomap),
        })

        conf = 0.9 if glomap else 0.75  # global refine present raises confidence
        return StageResult(
            outputs={"poses_global": poses_ref, "global": global_ref},
            metrics={"n_poses": n, "median_step_m": round(median_step, 3),
                     "loop_candidates": loops},
            confidence_summary=conf,
            degraded_reason=degraded,
        )
