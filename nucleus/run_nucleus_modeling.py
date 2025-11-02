#!/usr/bin/env python3
"""Run the FeARLesS spherical-harmonics pipeline on nucleus meshes.

Edit the ``CONFIG`` section below to control how the reconstruction operates.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pyshtools
from vedo import Mesh, Points, ProgressBar, load, recoSurface, spher2cart


@dataclass
class ModelingConfig:
    """Parameters controlling the nucleus spherical-harmonics workflow."""

    mesh_dir: Path = Path("nucleus_data/surfaces")
    pattern: str = "*.vtk"
    times: Sequence[float] | None = None
    lmax: int = 40
    samples: int = 400
    fit_degree: int = 4
    interpolation_steps: int | None = None
    surface_resolution: int = 150
    smooth_iterations: int = 20
    overwrite: bool = False
    export_surfaces: bool = True
    export_points: bool = False
    output_dir: Path = Path("nucleus_results")


# ---------------------------------------------------------------------------
# Configure the modelling run here. Adjust the paths and numerical parameters
# to match your dataset and desired reconstruction quality.
# ---------------------------------------------------------------------------
CONFIG = ModelingConfig(
    mesh_dir=Path("nucleus_data/surfaces"),
    pattern="*.vtk",
    times=None,  # e.g. [0.0, 2.5, 5.0]
    lmax=40,
    samples=400,
    fit_degree=4,
    interpolation_steps=15,
    surface_resolution=150,
    smooth_iterations=20,
    overwrite=False,
    export_surfaces=True,
    export_points=False,
    output_dir=Path("nucleus_results"),
)


def ensure_directory(path: Path, overwrite: bool = False) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Directory {path} already exists. Set overwrite=True to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_meshes(mesh_paths: Iterable[Path]) -> List[Mesh]:
    meshes: List[Mesh] = []
    for mesh_path in mesh_paths:
        mesh = load(str(mesh_path))
        if not isinstance(mesh, Mesh):  # pragma: no cover - vedo guard
            raise TypeError(f"File {mesh_path} did not produce a triangular mesh")
        meshes.append(mesh)
    if not meshes:
        raise FileNotFoundError("No meshes were loaded")
    return meshes


def extract_time_value(path: Path) -> float:
    match = re.search(r"([-+]?[0-9]*\.?[0-9]+)", path.stem)
    if match:
        return float(match.group(1))
    raise ValueError(
        f"Could not extract a numeric time value from filename '{path.name}'"
    )


def compute_radius(bounds: Sequence[float]) -> float:
    deltas = np.array(bounds)[1::2] - np.array(bounds)[::2]
    return float(np.linalg.norm(deltas) / 2.0)


def compute_clm(mesh: Mesh, rmax: float, samples: int, origin: Sequence[float]) -> pyshtools.SHCoeffs:
    agrid: List[List[float]] = []
    for th in np.linspace(0, np.pi, samples, endpoint=False):
        lats: List[float] = []
        for ph in np.linspace(0, 2 * np.pi, samples, endpoint=False):
            direction = np.array(spher2cart(rmax, th, ph))
            intersections = mesh.intersectWithLine(origin, np.array(origin) + direction)
            if len(intersections):
                value = np.linalg.norm(np.array(intersections[0]) - np.array(origin))
                lats.append(value)
            else:
                lats.append(rmax)
        agrid.append(lats)
    agrid_arr = np.array(agrid)
    grid = pyshtools.SHGrid.from_array(agrid_arr)
    return grid.expand()


def interpolate_coefficients(
    clm_arrays: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
    degree: int,
) -> np.ndarray:
    if len(times) == 1:
        return np.repeat(clm_arrays, len(target_times), axis=0)
    interpolated = np.zeros((len(target_times),) + clm_arrays.shape[1:], dtype=float)
    degree = max(1, min(degree, len(times) - 1))
    pb = ProgressBar(0, clm_arrays.shape[1], c=3)
    for coeff_idx in pb.range():
        for l in range(clm_arrays.shape[2]):
            for m in range(clm_arrays.shape[3]):
                series = clm_arrays[:, coeff_idx, l, m]
                poly = np.poly1d(np.polyfit(times, series, degree))
                interpolated[:, coeff_idx, l, m] = poly(target_times)
        pb.print("interpolating coefficients ...")
    return interpolated


def reconstruct_surface(
    coeffs: np.ndarray,
    surface_resolution: int,
    smooth_iterations: int,
) -> Tuple[Mesh, Points]:
    clm_coeffs = pyshtools.SHCoeffs.from_array(coeffs)
    grid_reco = clm_coeffs.expand()
    agrid_reco = grid_reco.to_array()

    pts = []
    for i, longitude in enumerate(
        np.linspace(0, 360, num=agrid_reco.shape[1], endpoint=False)
    ):
        for j, latitude in enumerate(
            np.linspace(90, -90, num=agrid_reco.shape[0], endpoint=True)
        ):
            theta = np.deg2rad(90 - latitude)
            phi = np.deg2rad(longitude)
            radius = agrid_reco[j][i]
            pts.append(spher2cart(radius, theta, phi))

    cloud = Points(pts, r=8, c="r", alpha=1)
    cloud.clean(0.002)

    surface = recoSurface(cloud, dims=surface_resolution)
    largest = surface.extractLargestRegion().clone()
    if smooth_iterations > 0:
        largest.smooth(niter=smooth_iterations)
    return largest, cloud


def run(config: ModelingConfig) -> None:
    if config.interpolation_steps is not None and config.interpolation_steps <= 0:
        raise ValueError("interpolation_steps must be a positive integer")

    mesh_paths = sorted(config.mesh_dir.glob(config.pattern))
    if not mesh_paths:
        raise FileNotFoundError(
            f"No meshes matching pattern '{config.pattern}' were found in {config.mesh_dir}"
        )

    meshes = load_meshes(mesh_paths)

    if config.times is not None:
        if len(config.times) != len(meshes):
            raise ValueError("Number of provided times does not match number of meshes")
        times = np.array(config.times, dtype=float)
    else:
        times = np.array([extract_time_value(path) for path in mesh_paths], dtype=float)

    origins = [mesh.centerOfMass() for mesh in meshes]
    radii = [compute_radius(mesh.bounds()) for mesh in meshes]
    rmax = max(radii) * 1.05

    clm_list: List[Tuple[float, np.ndarray]] = []
    pb = ProgressBar(0, len(meshes), c=2)
    for mesh, time_point, origin in pb.zip(meshes, times, origins):
        coeffs = compute_clm(mesh, rmax=rmax, samples=config.samples, origin=origin)
        clm_list.append((time_point, coeffs.to_array(lmax=config.lmax)))
        pb.print("computing coefficients ...")

    clm_list.sort(key=lambda item: item[0])
    sorted_times = np.array([item[0] for item in clm_list])
    clm_arrays = np.array([item[1] for item in clm_list])

    if config.interpolation_steps is None:
        target_times = sorted_times
    else:
        target_times = np.linspace(
            sorted_times.min(), sorted_times.max(), config.interpolation_steps
        )

    interpolated = interpolate_coefficients(
        clm_arrays,
        sorted_times,
        target_times,
        degree=config.fit_degree,
    )

    ensure_directory(config.output_dir, overwrite=config.overwrite)
    np.save(config.output_dir / "coefficients.npy", interpolated)

    metadata = {
        "input_meshes": [path.name for path in mesh_paths],
        "times": sorted_times.tolist(),
        "target_times": target_times.tolist(),
        "lmax": config.lmax,
        "samples": config.samples,
        "fit_degree": config.fit_degree,
        "surface_resolution": config.surface_resolution,
        "smooth_iterations": config.smooth_iterations,
        "rmax": rmax,
    }
    with (config.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if config.export_surfaces or config.export_points:
        surfaces_dir = config.output_dir / "surfaces"
        points_dir = config.output_dir / "point_clouds"
        if config.export_surfaces:
            ensure_directory(surfaces_dir, overwrite=config.overwrite)
        if config.export_points:
            ensure_directory(points_dir, overwrite=config.overwrite)

        pb = ProgressBar(0, interpolated.shape[0], c=4)
        for idx in pb.range():
            coeffs = interpolated[idx]
            surface, cloud = reconstruct_surface(
                coeffs,
                surface_resolution=config.surface_resolution,
                smooth_iterations=config.smooth_iterations,
            )
            if config.export_surfaces:
                surface.write(str(surfaces_dir / f"nucleus_{idx:03d}.vtk"))
            if config.export_points:
                cloud.write(str(points_dir / f"nucleus_points_{idx:03d}.vtp"))
            pb.print("writing reconstructions ...")


if __name__ == "__main__":
    run(CONFIG)
