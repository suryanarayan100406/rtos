"""RT-DETR object detector (HuggingFace transformers) — real, guarded, permissive (Apache-2.0).

Used by S3 to locate dynamic-object classes (people, vehicles, …) whose pixels must be masked out of
the static reconstruction. Detections are real model output; boxes become masks (optionally refined by
SAM2 when installed).
"""
from __future__ import annotations

import numpy as np

from ..runtime.context import Runtime
from ..runtime.deps import require
from .loaders import cache_dir_for, gate, hf_repo_id, torch_device


class RTDetrDetector:
    def __init__(self, rt: Runtime, model_name: str = "rt_detr") -> None:
        self.spec = gate(rt, model_name)  # license gate
        repo = hf_repo_id(model_name)
        transformers = require("transformers", purpose="RT-DETR detection")
        torch = require("torch", purpose="RT-DETR detection")
        self._torch = torch
        self.device = torch_device(rt)
        cache = str(cache_dir_for(rt))
        self.processor = transformers.AutoImageProcessor.from_pretrained(repo, cache_dir=cache)
        self.model = (
            transformers.AutoModelForObjectDetection.from_pretrained(repo, cache_dir=cache)
            .to(self.device).eval()
        )
        self.id2label = self.model.config.id2label

    def detect(self, img_bgr: np.ndarray, threshold: float = 0.5) -> list[dict]:
        """Return [{label, score, box=[x1,y1,x2,y2]}] for one BGR frame."""
        torch = self._torch
        h, w = img_bgr.shape[:2]
        rgb = img_bgr[..., ::-1]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target = torch.tensor([[h, w]], device=self.device)
        res = self.processor.post_process_object_detection(
            outputs, target_sizes=target, threshold=threshold
        )[0]
        dets = []
        for score, label, box in zip(res["scores"], res["labels"], res["boxes"], strict=True):
            dets.append({
                "label": self.id2label.get(int(label), str(int(label))),
                "score": float(score),
                "box": [float(v) for v in box.tolist()],
            })
        return dets
