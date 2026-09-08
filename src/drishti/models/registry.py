"""Model registry + license gate.

Every model DRISHTI can use is pinned here with its license. The shipped pipeline refuses to load a
non-permissive or reference-only model unless explicitly overridden (models.allow_reference_only).
This encodes the binding licensing posture from docs/03-TECHNOLOGY-STACK.md and AGENTS.md §5.

Licenses below reflect the shipped checkpoints/components; where a family mixes licenses (e.g. Depth
Anything V2 small/base are Apache-2.0 but large is CC-BY-NC), each variant is a separate entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Permissive licenses allowed in the shipped path.
LICENSE_ALLOWLIST = {
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC-BY-4.0",
}


class LicenseError(RuntimeError):
    """Raised when the pipeline tries to load a non-permissive / reference-only model."""


class ModelNotRegistered(KeyError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str  # depth | detect | segment | track | flow | geometry | recon | pose
    license: str
    runtimes: tuple[str, ...] = ("torch",)  # torch | openvino | native
    version: str = "unpinned"
    sha256: str | None = None
    source: str = ""
    vram_gb: float = 0.0
    reference_only: bool = False  # true => benchmark/reference only, never shipped
    notes: str = ""

    @property
    def permissive(self) -> bool:
        return self.license in LICENSE_ALLOWLIST and not self.reference_only


# --------------------------------------------------------------------------- the registry
_SPECS: dict[str, ModelSpec] = {
    # --- poses -------------------------------------------------------------
    "colmap": ModelSpec("colmap", "pose", "BSD-3-Clause", ("native",), source="colmap.github.io"),
    "glomap": ModelSpec("glomap", "pose", "BSD-3-Clause", ("native",), source="github.com/colmap/glomap"),
    # --- metric depth (shipped: permissive small/base) ---------------------
    "depth_anything_v2_small": ModelSpec(
        "depth_anything_v2_small", "depth", "Apache-2.0", ("openvino", "torch"), vram_gb=2.0,
        source="github.com/DepthAnything/Depth-Anything-V2"),
    "depth_anything_v2_base": ModelSpec(
        "depth_anything_v2_base", "depth", "Apache-2.0", ("openvino", "torch"), vram_gb=4.0,
        source="github.com/DepthAnything/Depth-Anything-V2"),
    "depth_anything_v2_large": ModelSpec(
        "depth_anything_v2_large", "depth", "CC-BY-NC-4.0", ("torch",), vram_gb=8.0,
        reference_only=True, notes="large ckpt is non-commercial; not shipped"),
    "metric3d_v2": ModelSpec(
        "metric3d_v2", "depth", "BSD-2-Clause", ("torch",), vram_gb=6.0,
        source="github.com/YvanYin/Metric3D"),
    "unidepth_v2": ModelSpec(
        "unidepth_v2", "depth", "CC-BY-NC-4.0", ("torch",), vram_gb=6.0, reference_only=True,
        notes="non-commercial; reference only"),
    # --- masking / perception ---------------------------------------------
    "rt_detr": ModelSpec("rt_detr", "detect", "Apache-2.0", ("openvino", "torch"), vram_gb=3.0,
                         source="github.com/lyuwenyu/RT-DETR", notes="permissive alternative to AGPL YOLO"),
    "sam2": ModelSpec("sam2", "segment", "Apache-2.0", ("torch",), vram_gb=5.0,
                      source="github.com/facebookresearch/sam2"),
    "bytetrack": ModelSpec("bytetrack", "track", "MIT", ("native",), source="github.com/ifzhang/ByteTrack"),
    "raft": ModelSpec("raft", "flow", "BSD-3-Clause", ("torch",), vram_gb=3.0,
                      source="github.com/princeton-vl/RAFT"),
    # --- feed-forward geometry (OPTIONAL; blocked until license verified) --
    "mapanything": ModelSpec(
        "mapanything", "geometry", "verify-before-ship", ("torch",), vram_gb=12.0,
        reference_only=True, notes="OPTIONAL enhancement; confirm license before shipping"),
    "pi3": ModelSpec(
        "pi3", "geometry", "verify-before-ship", ("torch",), vram_gb=10.0, reference_only=True,
        notes="OPTIONAL enhancement; confirm license before shipping"),
    "vggt": ModelSpec(
        "vggt", "geometry", "commercial-military-excluded", ("torch",), vram_gb=16.0,
        reference_only=True, notes="license excludes military use; reference only"),
    "dust3r": ModelSpec("dust3r", "geometry", "CC-BY-NC-4.0", ("torch",), reference_only=True),
    "mast3r": ModelSpec("mast3r", "geometry", "CC-BY-NC-4.0", ("torch",), reference_only=True),
    # --- dense / mesh ------------------------------------------------------
    "gsplat": ModelSpec("gsplat", "recon", "Apache-2.0", ("torch",), vram_gb=10.0,
                        source="github.com/nerfstudio-project/gsplat"),
    "open3d": ModelSpec("open3d", "recon", "MIT", ("native",), source="open3d.org"),
    "mvs_texturing": ModelSpec("mvs_texturing", "recon", "BSD-3-Clause", ("native",),
                              source="github.com/nmoehrle/mvs-texturing"),
}


def all_specs() -> dict[str, ModelSpec]:
    return dict(_SPECS)


class ModelRegistry:
    """Resolves model specs and enforces the license gate."""

    def __init__(self, cache_dir: str | Path = "models_cache", allow_reference_only: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.allow_reference_only = allow_reference_only

    def spec(self, name: str) -> ModelSpec:
        try:
            return _SPECS[name]
        except KeyError as exc:
            raise ModelNotRegistered(
                f"model '{name}' is not in the registry; register it (with a verified license) first"
            ) from exc

    def check_license(self, spec: ModelSpec) -> None:
        if spec.permissive:
            return
        if self.allow_reference_only:
            return  # explicit opt-in (e.g. offline benchmarking), never the default
        raise LicenseError(
            f"refusing to load '{spec.name}': license '{spec.license}'"
            f"{' (reference_only)' if spec.reference_only else ''} is not permissive. "
            f"It may be used for reference/benchmarking only. Set models.allow_reference_only=true "
            f"to override for non-shipped use."
        )

    def resolve(self, name: str) -> ModelSpec:
        """Return the spec after passing the license gate (raises otherwise)."""
        spec = self.spec(name)
        self.check_license(spec)
        return spec

    def cache_path(self, name: str) -> Path:
        return self.cache_dir / name
