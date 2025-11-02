#!/usr/bin/env python3
"""Utility to convert nucleus surface meshes into FeARLesS-ready datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from vedo import Mesh, ProgressBar, load, volumeFromMesh


def ensure_directory(path: Path, overwrite: bool) -> None:
    """Create ``path`` optionally clearing previous content."""
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Directory {path} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_mesh_files(mesh_paths: Iterable[Path]) -> List[Mesh]:
    meshes: List[Mesh] = []
    for mesh_path in mesh_paths:
        mesh = load(str(mesh_path))
        if not isinstance(mesh, Mesh):  # pragma: no cover - vedo guard
            raise TypeError(f"File {mesh_path} did not produce a triangular mesh")
        meshes.append(mesh)
    if not meshes:
        raise FileNotFoundError("No input meshes were found")
    return meshes


def compute_bounds(meshes: Sequence[Mesh], padding: float) -> Tuple[float, ...]:
    """Return a global bounding box padded by ``padding``."""
    bounds = np.array([m.bounds() for m in meshes])
    mins = bounds[:, ::2].min(axis=0)
    maxs = bounds[:, 1::2].max(axis=0)
    center = (mins + maxs) / 2.0
    half_size = (maxs - mins) / 2.0
    half_size *= padding
    scaled_mins = center - half_size
    scaled_maxs = center + half_size
    return (
        float(scaled_mins[0]),
        float(scaled_maxs[0]),
        float(scaled_mins[1]),
        float(scaled_maxs[1]),
        float(scaled_mins[2]),
        float(scaled_maxs[2]),
    )


def export_surfaces(meshes: Sequence[Mesh], input_paths: Sequence[Path], output_dir: Path) -> None:
    """Write meshes to ``output_dir`` preserving the original file order."""
    pb = ProgressBar(0, len(meshes), c=1)
    for mesh, source in pb.zip(meshes, input_paths):
        target = output_dir / f"{source.stem}.vtk"
        mesh.clone().write(str(target))
        pb.print(f"writing surface {target.name} ...")


def build_volumes(
    meshes: Sequence[Mesh],
    input_paths: Sequence[Path],
    output_dir: Path,
    bounds: Tuple[float, ...],
    sample_size: Tuple[int, int, int],
    invert_normals: bool,
) -> None:
    """Generate signed-distance volumes for the provided meshes."""
    pb = ProgressBar(0, len(meshes), c=2)
    for mesh, source in pb.zip(meshes, input_paths):
        volume = volumeFromMesh(
            mesh,
            dims=sample_size,
            bounds=bounds,
            signed=True,
            negate=invert_normals,
        )
        target = output_dir / f"{source.stem}.vti"
        volume.write(str(target))
        pb.print(f"writing volume {target.name} ...")


def dump_metadata(
    output_dir: Path,
    input_paths: Sequence[Path],
    bounds: Tuple[float, ...],
    sample_size: Tuple[int, int, int],
) -> None:
    metadata = {
        "input_files": [p.name for p in input_paths],
        "bounds": bounds,
        "sample_size": sample_size,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare nucleus meshes (PLY) for the FeARLesS reconstruction pipeline",
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing nucleus surface meshes in .ply format",
    )
    parser.add_argument(
        "--surface-output",
        type=Path,
        default=Path("nucleus_data/surfaces"),
        help="Directory where converted .vtk surfaces will be saved",
    )
    parser.add_argument(
        "--volume-output",
        type=Path,
        default=Path("nucleus_data/volumes"),
        help="Directory where signed-distance volumes will be stored",
    )
    parser.add_argument(
        "--sample-size",
        nargs=3,
        default=(160, 160, 160),
        metavar=("NX", "NY", "NZ"),
        help="Number of voxels along each axis for the generated volumes",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.15,
        help="Scale factor applied to the global bounding box (values > 1 enlarge the box)",
    )
    parser.add_argument(
        "--invert-normals",
        action="store_true",
        help="Invert the sign convention of the signed distance field",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output folders instead of aborting",
    )
    parser.add_argument(
        "--skip-surfaces",
        action="store_true",
        help="Do not export vtk surface meshes",
    )
    parser.add_argument(
        "--skip-volumes",
        action="store_true",
        help="Do not export voxel volumes",
    )

    args = parser.parse_args()
    sample_size = (
        int(args.sample_size[0]),
        int(args.sample_size[1]),
        int(args.sample_size[2]),
    )

    input_paths = sorted(args.input_dir.glob("*.ply"))
    if not input_paths:
        raise FileNotFoundError(f"No .ply files were found in {args.input_dir}")

    meshes = load_mesh_files(input_paths)
    bounds = compute_bounds(meshes, padding=args.padding)

    if not args.skip_surfaces:
        ensure_directory(args.surface_output, args.overwrite)
        export_surfaces(meshes, input_paths, args.surface_output)
        dump_metadata(args.surface_output, input_paths, bounds, sample_size)

    if not args.skip_volumes:
        ensure_directory(args.volume_output, args.overwrite)
        build_volumes(
            meshes,
            input_paths,
            args.volume_output,
            bounds,
            sample_size,
            invert_normals=args.invert_normals,
        )
        dump_metadata(args.volume_output, input_paths, bounds, sample_size)


if __name__ == "__main__":
    main()
