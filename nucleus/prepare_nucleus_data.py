#!/usr/bin/env python3
"""FeARLesS nucleus-prep with OPTIONAL RIGID PRE-ALIGNMENT (centroid + PCA axes)

This script converts reconstructed nucleus surface meshes into FeARLesS-ready
artifacts and (optionally) performs a deterministic rigid pre-alignment so you
can visualize/check **aligned** meshes and reuse them directly in the SH/DH
modeling pipeline without re-aligning.

▶ Usage: edit the CONFIG block below and run the script.
No command-line arguments are required or supported in this variant.

Outputs
- Surfaces: `.vtp` (XML PolyData)
- Signed-distance volumes: `.vti` (XML ImageData) with global bounds/spacing
- `metadata.json`: global bounds, voxel spacing, per-file QA (watertight, area,
  volume, centroid) and—when alignment is enabled—the applied rigid transform
  (R, t, scale) per file.

Notes
- SDFs assume watertight inputs; non‑closed meshes are skipped unless allowed.
- Alignment is *rigid only* (no scaling) by default; optional scale
  normalization is available via CONFIG (`normalize_scale = 'area'|'volume'`).

Dependencies: vedo, numpy
"""
from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from vedo import Mesh, ProgressBar, load, volumeFromMesh


# ============================== USER CONFIG ==================================


@dataclass
class PreparationConfig:
    input_dir: Path
    surface_output: Path
    volume_output: Path
    file_suffix: str = "_nuclei.ply"  # matches e.g., 11207128_s1_nuclei.ply
    dims: Tuple[int, int, int] = (160, 160, 160)  # (nx, ny, nz)
    padding: float = 1.15
    invert_normals: bool = False  # flip SDF sign convention (negate outside)
    overwrite: bool = False
    skip_surfaces: bool = False
    skip_volumes: bool = False
    allow_nonwatertight: bool = False

    # Alignment options
    align: bool = True
    align_reference: str = "first"  # 'first' | 'origin'
    normalize_scale: str = "none"    # 'none' | 'area' | 'volume'


CONFIG = PreparationConfig(
    input_dir=Path("/path/to/nucleus_folder"),
    surface_output=Path("nucleus_data/surfaces_aligned"),
    volume_output=Path("nucleus_data/volumes_aligned"),
    file_suffix="_nuclei.ply",
    dims=(160, 160, 160),
    padding=1.15,
    invert_normals=False,
    overwrite=False,
    skip_surfaces=False,
    skip_volumes=False,
    allow_nonwatertight=False,
    align=True,
    align_reference="first",
    normalize_scale="none",
)


# ============================== IMPLEMENTATION ===============================


@dataclass
class MeshMeta:
    filename: str
    cell_id: Optional[str]
    stage: Optional[str]
    watertight: bool
    area: float
    volume_abs: float
    centroid: Tuple[float, float, float]
    transform_R: Optional[List[List[float]]] = None  # 3x3 rotation
    transform_t: Optional[List[float]] = None        # translation applied AFTER rotation
    scale: Optional[float] = None


def ensure_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Directory {path} already exists. Set overwrite=True to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def discover_input_meshes(input_dir: Path, file_suffix: str) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    pattern = f"*{file_suffix}" if file_suffix else "*.ply"
    candidates = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No mesh files matching '{pattern}' were found in {input_dir}"
        )
    return candidates


def load_mesh_files(mesh_paths: Iterable[Path]) -> List[Mesh]:
    meshes: List[Mesh] = []
    for mesh_path in mesh_paths:
        mesh = load(str(mesh_path))
        if not isinstance(mesh, Mesh):
            raise TypeError(f"File {mesh_path} did not produce a triangular mesh")
        mesh = mesh.triangulate().clean()
        mesh.compute_normals(points=True, cells=True, consistency=True)
        meshes.append(mesh)
    if not meshes:
        raise FileNotFoundError("No input meshes were found")
    return meshes


def is_watertight(mesh: Mesh) -> bool:
    try:
        return bool(mesh.is_closed())
    except Exception:  # pragma: no cover - vedo guard
        return False


def signed_volume(mesh: Mesh) -> float:
    pts = np.asarray(mesh.points())
    faces = np.asarray(mesh.faces())
    vol = 0.0
    for f in faces:
        a, b, c = pts[f]
        vol += np.dot(a, np.cross(b, c))
    return vol / 6.0


def maybe_fix_winding(mesh: Mesh, enable: bool) -> Mesh:
    if not enable:
        return mesh
    try:
        if signed_volume(mesh) < 0:
            mesh.reverse(cells=True, normals=True)
            mesh.compute_normals(points=True, cells=True, consistency=True)
    except Exception:  # pragma: no cover - vedo guard
        pass
    return mesh


def compute_bounds(meshes: Sequence[Mesh], padding: float) -> Tuple[float, ...]:
    bounds = np.array([m.bounds() for m in meshes])
    mins = bounds[:, ::2].min(axis=0)
    maxs = bounds[:, 1::2].max(axis=0)
    center = (mins + maxs) / 2.0
    half = (maxs - mins) / 2.0
    half *= padding
    mins = center - half
    maxs = center + half
    return (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )


def spacing_from_bounds(bounds: Tuple[float, ...], dims: Tuple[int, int, int]) -> Tuple[float, float, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    nx, ny, nz = dims
    sx = (xmax - xmin) / max(nx - 1, 1)
    sy = (ymax - ymin) / max(ny - 1, 1)
    sz = (zmax - zmin) / max(nz - 1, 1)
    return float(sx), float(sy), float(sz)


def filename_tokens(path: Path) -> Tuple[Optional[str], Optional[str]]:
    match = re.match(r"^(?P<cell>[^_]+)_(?P<stage>s[0-9]+).*", path.stem)
    if not match:
        return None, None
    return match.group("cell"), match.group("stage")


# ----------------------------- alignment -------------------------------------


def principal_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return centroid and PCA axes (columns) sorted by variance; right‑handed."""

    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1
    return centroid, eigvecs


def canonicalize_axes(axes: np.ndarray, ref_axes: np.ndarray) -> np.ndarray:
    """Flip column signs of ``axes`` to align with ``ref_axes`` without reflections."""

    canonical = axes.copy()
    for idx in range(3):
        if np.dot(ref_axes[:, idx], canonical[:, idx]) < 0:
            canonical[:, idx] *= -1
    if np.linalg.det(canonical) < 0:
        canonical[:, -1] *= -1
    return canonical


def rigid_align(
    points: np.ndarray,
    ref_ctr: np.ndarray,
    ref_axes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rigid transform bringing ``points`` to the reference frame."""

    centroid, axes = principal_axes(points)
    axes = canonicalize_axes(axes, ref_axes)
    rotation = ref_axes @ axes.T
    target_ctr = np.zeros(3)
    aligned = (points - centroid) @ rotation.T + target_ctr
    translation = target_ctr - centroid @ rotation.T
    return rotation, translation, aligned


def maybe_scale(points: np.ndarray, mode: str) -> Tuple[np.ndarray, float]:
    if mode == "none":
        return points, 1.0
    centroid = points.mean(axis=0)
    centered = points - centroid
    if mode == "area":
        scale = math.sqrt(np.mean(np.sum(centered**2, axis=1)))
    elif mode == "volume":
        scale = np.mean(np.sum(centered**2, axis=1)) ** (3.0 / 4.0)
    else:
        scale = 1.0
    if scale == 0:
        scale = 1.0
    return centered / scale, scale


def align_meshes(
    meshes: Sequence[Mesh],
    cfg: PreparationConfig,
) -> Tuple[List[Mesh], np.ndarray, np.ndarray, Dict[str, Dict[str, List[float]]]]:
    """Align all meshes to a shared PCA frame."""

    ref_points = np.asarray(meshes[0].points())
    ref_ctr, ref_axes = principal_axes(ref_points)
    if cfg.align_reference == "origin":
        ref_ctr = np.zeros(3)
    aligned: List[Mesh] = []
    transforms: Dict[str, Dict[str, List[float]]] = {}
    pb = ProgressBar(0, len(meshes), c=1)
    for idx in pb.range():
        mesh = meshes[idx]
        pts = np.asarray(mesh.points())
        rotation, translation, aligned_pts = rigid_align(pts, ref_ctr, ref_axes)
        scaled_pts, scale = maybe_scale(aligned_pts, cfg.normalize_scale)
        mesh_aligned = mesh.clone()
        mesh_aligned.points(scaled_pts)
        aligned.append(mesh_aligned)
        key = f"mesh_{idx:04d}"
        transforms[key] = {
            "R": rotation.tolist(),
            "t": translation.tolist(),
            "scale": float(scale),
        }
        pb.print(f"align → {key}")
    return aligned, ref_ctr, ref_axes, transforms


# ----------------------------- export stages ---------------------------------


def export_surfaces(
    meshes: Sequence[Mesh],
    paths: Sequence[Path],
    outdir: Path,
    transforms: Optional[Dict[str, Dict[str, List[float]]]] = None,
) -> Dict[str, MeshMeta]:
    pb = ProgressBar(0, len(meshes), c=1)
    metadata: Dict[str, MeshMeta] = {}
    for idx, (mesh, src) in enumerate(pb.zip(meshes, paths)):
        cell_id, stage = filename_tokens(src)
        watertight = is_watertight(mesh)
        try:
            area = float(mesh.area())
        except Exception:  # pragma: no cover - vedo guard
            area = float("nan")
        try:
            volume = abs(float(mesh.volume()))
        except Exception:  # pragma: no cover - vedo guard
            volume = float("nan")
        centroid = tuple(map(float, mesh.centerOfMass()))
        key = f"mesh_{idx:04d}"
        transform = transforms.get(key) if transforms else None
        meta = MeshMeta(
            filename=src.name,
            cell_id=cell_id,
            stage=stage,
            watertight=watertight,
            area=area,
            volume_abs=volume,
            centroid=centroid,
            transform_R=transform["R"] if transform else None,
            transform_t=transform["t"] if transform else None,
            scale=transform.get("scale") if transform else None,
        )
        metadata[src.name] = meta
        target = outdir / f"{src.stem}.vtp"
        mesh.clone().write(str(target))
        pb.print(f"surface → {target.name}")
    return metadata


def build_volumes(
    meshes: Sequence[Mesh],
    paths: Sequence[Path],
    outdir: Path,
    bounds: Tuple[float, ...],
    dims: Tuple[int, int, int],
    invert_normals: bool,
    allow_nonwatertight: bool,
) -> None:
    pb = ProgressBar(0, len(meshes), c=2)
    for mesh, src in pb.zip(meshes, paths):
        if not is_watertight(mesh) and not allow_nonwatertight:
            pb.print(f"skip volume (non-watertight): {src.name}")
            continue
        mesh = maybe_fix_winding(mesh, enable=True)
        volume = volumeFromMesh(
            mesh,
            dims=dims,
            bounds=bounds,
            signed=True,
            negate=invert_normals,
        )
        target = outdir / f"{src.stem}.vti"
        volume.write(str(target))
        pb.print(f"volume → {target.name}")


# ----------------------------- metadata IO -----------------------------------


def dump_global_metadata(
    outdir: Path,
    meshes: Sequence[Mesh],
    input_paths: Sequence[Path],
    bounds: Tuple[float, ...],
    dims: Tuple[int, int, int],
    perfile: Dict[str, MeshMeta],
    align_info: Optional[Dict[str, Dict[str, List[float]]]] = None,
) -> None:
    spacing = spacing_from_bounds(bounds, dims)
    payload = {
        "input_files": [p.name for p in input_paths],
        "dims": list(map(int, dims)),
        "bounds": list(map(float, bounds)),
        "spacing": list(map(float, spacing)),
        "count": len(input_paths),
        "notes": {
            "surfaces_ext": ".vtp",
            "volumes_ext": ".vti",
            "signed_distance": True,
        },
        "per_file": {k: asdict(v) for k, v in perfile.items()},
        "alignment": align_info or {},
    }
    with (outdir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# ----------------------------- driver ----------------------------------------


def run(cfg: PreparationConfig) -> None:
    mesh_paths = discover_input_meshes(cfg.input_dir, cfg.file_suffix)
    meshes = load_mesh_files(mesh_paths)

    transforms: Dict[str, Dict[str, List[float]]] = {}
    if cfg.align:
        meshes, _, _, transforms = align_meshes(meshes, cfg)

    bounds = compute_bounds(meshes, padding=cfg.padding)

    if not cfg.skip_surfaces:
        ensure_directory(cfg.surface_output, cfg.overwrite)
        metadata = export_surfaces(meshes, mesh_paths, cfg.surface_output, transforms=transforms)
        dump_global_metadata(
            cfg.surface_output,
            meshes,
            mesh_paths,
            bounds,
            cfg.dims,
            metadata,
            align_info=transforms,
        )

    if not cfg.skip_volumes:
        ensure_directory(cfg.volume_output, cfg.overwrite)
        build_volumes(
            meshes,
            mesh_paths,
            cfg.volume_output,
            bounds=bounds,
            dims=cfg.dims,
            invert_normals=cfg.invert_normals,
            allow_nonwatertight=cfg.allow_nonwatertight,
        )
        dump_global_metadata(
            cfg.volume_output,
            meshes,
            mesh_paths,
            bounds,
            cfg.dims,
            {},
            align_info=transforms,
        )


if __name__ == "__main__":
    run(CONFIG)
