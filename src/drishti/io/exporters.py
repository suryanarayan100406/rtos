"""Format exporters (real, each guarded by its own dependency).

Converts the canonical bundle products (mesh.ply, dense.ply, GeoTIFFs) into the deliverable formats.
Every exporter either produces a real file or reports precisely why it could not (missing library or
tool) — it never writes a placeholder (AGENTS.md §5). Returns (ok, path_or_reason).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..runtime.deps import available


def export_obj(mesh_ply: Path, dest: Path) -> tuple[bool, str]:
    if not available("trimesh"):
        return False, "trimesh not installed (pip install '.[recon]')"
    import trimesh
    m = trimesh.load(str(mesh_ply), process=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(dest))
    return True, str(dest)


def export_glb(mesh_ply: Path, dest: Path) -> tuple[bool, str]:
    if not available("trimesh"):
        return False, "trimesh not installed (pip install '.[recon]')"
    import trimesh
    m = trimesh.load(str(mesh_ply), process=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(dest))  # trimesh picks glTF/GLB from extension
    return True, str(dest)


def export_las(points, colors, dest: Path, epsg: int | None, point_format: str = "las_1_4",
               laz: bool = False) -> tuple[bool, str]:
    if not available("laspy"):
        return False, "laspy not installed (pip install laspy[lazrs])"
    import laspy
    import numpy as np

    header = laspy.LasHeader(point_format=7 if point_format == "las_1_4" else 3, version="1.4")
    if epsg:
        try:
            header.add_crs(__import__("pyproj").CRS.from_epsg(epsg))
        except Exception:
            pass
    las = laspy.LasData(header)
    pts = np.asarray(points, float)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    if colors is not None:
        c = np.asarray(colors)
        if c.max() <= 255:
            c = (c.astype(np.uint16) << 8)  # LAS color is 16-bit
        las.red, las.green, las.blue = c[:, 0], c[:, 1], c[:, 2]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if laz and dest.suffix != ".laz":
        dest = dest.with_suffix(".laz")
    las.write(str(dest))
    return True, str(dest)


def copy_product(src: Path, dest: Path) -> tuple[bool, str]:
    if not Path(src).exists():
        return False, f"source missing: {src}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True, str(dest)


def export_fbx_via_blender(mesh_ply: Path, dest: Path) -> tuple[bool, str]:
    """Convert PLY -> FBX out-of-process with Blender (GPL isolation). Skipped if blender absent."""
    blender = shutil.which("blender")
    if not blender:
        return False, "blender not found on PATH (FBX export needs headless Blender)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "import bpy,sys;"
        "bpy.ops.wm.read_factory_settings(use_empty=True);"
        f"bpy.ops.wm.ply_import(filepath=r'{mesh_ply}');"
        f"bpy.ops.export_scene.fbx(filepath=r'{dest}')"
    )
    proc = subprocess.run([blender, "--background", "--python-expr", script],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not dest.exists():
        return False, f"blender FBX export failed: {proc.stderr[-300:] or proc.stdout[-300:]}"
    return True, str(dest)


def export_3dtiles(mesh_or_ply: Path, dest_dir: Path) -> tuple[bool, str]:
    tool = shutil.which("py3dtiles") or shutil.which("3d-tiles-tools")
    if not tool:
        return False, "no 3D Tiles converter on PATH (py3dtiles / 3d-tiles-tools)"
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([tool, "convert", str(mesh_or_ply), "--out", str(dest_dir)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"3D Tiles conversion failed: {proc.stderr[-300:]}"
    return True, str(dest_dir)


def export_potree(ply: Path, dest_dir: Path) -> tuple[bool, str]:
    tool = shutil.which("PotreeConverter")
    if not tool:
        return False, "PotreeConverter not on PATH"
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([tool, str(ply), "-o", str(dest_dir)], capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"PotreeConverter failed: {proc.stderr[-300:]}"
    return True, str(dest_dir)
