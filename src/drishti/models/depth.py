"""Depth Anything V2 monocular depth predictor (HuggingFace transformers) — real, guarded.

Produces a per-frame relative depth map. Metric scale/shift is recovered in S4 by fitting to the sparse
metric points from S2 — the network output alone is not treated as metric (honesty policy). Only the
permissive small/base checkpoints are mapped; the large (CC-BY-NC) checkpoint is blocked by the gate.
"""
from __future__ import annotations

import numpy as np

from ..runtime.context import Runtime
from ..runtime.deps import require
from .loaders import cache_dir_for, gate, hf_repo_id, torch_device


class DepthAnythingV2:
    def __init__(self, rt: Runtime, model_name: str) -> None:
        self.spec = gate(rt, model_name)  # license gate first
        repo = hf_repo_id(model_name)
        transformers = require("transformers", purpose="Depth Anything V2")
        torch = require("torch", purpose="Depth Anything V2")
        self._torch = torch
        self.device = torch_device(rt)
        cache = str(cache_dir_for(rt))
        self.processor = transformers.AutoImageProcessor.from_pretrained(repo, cache_dir=cache)
        self.model = (
            transformers.AutoModelForDepthEstimation.from_pretrained(repo, cache_dir=cache)
            .to(self.device).eval()
        )

    def predict(self, img_bgr: np.ndarray) -> np.ndarray:
        """Return an (H,W) float32 relative-depth map for a BGR image."""
        torch = self._torch
        h, w = img_bgr.shape[:2]
        rgb = img_bgr[..., ::-1]  # BGR -> RGB
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth  # (1, h', w')
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
        ).squeeze().detach().cpu().numpy().astype("float32")
        return depth
