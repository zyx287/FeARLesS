"""Run the FeARLesS nucleus pipeline with configurable stages.

This module mirrors the four canonical FeARLesS scripts (`pureSPharm.py`,
`makeVoxel.py`, `computeAllIntesities.py`, `morphing.py`) while adapting them to
nucleus datasets converted to VTK meshes.  The exposed configuration block lets
researchers tune parameters inline instead of relying on command-line flags.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pyshtools
from vedo import Mesh, Points, ProgressBar, load, recoSurface, spher2cart, volumeFromMesh

# ---------------------------------------------------------------------------
# Configuration --------------------------------------------------------------
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """User editable configuration for the nucleus pipeline."""

    mesh_dir: Path = Path("nucleus_data/surfaces")
    pattern: str | Sequence[str] = ("*.vtp", "*.vtk")
    times: Sequence[float] | None = None

    # Stage toggles ---------------------------------------------------------
    run_make_voxel: bool = True
    run_pure_spharm: bool = True
    run_compute_all_intensities: bool = True
    run_morphing: bool = True

    # Stage parameters ------------------------------------------------------
    volume_dims: Tuple[int, int, int] = (160, 160, 160)
    volume_padding: float = 1.15
    invert_normals: bool = False

    lmax: int = 40
    sphere_samples: int = 400
    radius_samples: int = 120
    fit_degree: int = 4
    interpolation_steps: int | None = 15

    surface_resolution: int = 150
    smooth_iterations: int = 20

    export_surfaces: bool = True
    export_points: bool = False

    overwrite: bool = False
    output_root: Path = Path("nucleus_results")


CONFIG = PipelineConfig()


# ---------------------------------------------------------------------------
# Helper utilities ----------------------------------------------------------
# ---------------------------------------------------------------------------


def ensure_directory(path: Path, overwrite: bool = False) -> None:
    """Create ``path`` removing existing content when ``overwrite`` is true."""

    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Directory {path} already exists. Set overwrite=True to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_meshes(mesh_paths: Iterable[Path]) -> List[Mesh]:
    """Load meshes and ensure at least one surface exists."""

    meshes: List[Mesh] = []
    for mesh_path in mesh_paths:
        mesh = load(str(mesh_path))
        if not isinstance(mesh, Mesh):  # pragma: no cover - vedo guard
            raise TypeError(f"File {mesh_path} did not produce a triangular mesh")
        meshes.append(mesh)
    if not meshes:
        raise FileNotFoundError("No meshes were loaded")
    return meshes


def compute_bounds(meshes: Sequence[Mesh], padding: float) -> Tuple[float, ...]:
    """Return a global bounding box padded by ``padding``."""

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


def compute_radius(bounds: Sequence[float]) -> float:
    """Compute an enclosing radius for the supplied bounds."""

    deltas = np.array(bounds)[1::2] - np.array(bounds)[::2]
    return float(np.linalg.norm(deltas) / 2.0)


# ---------------------------------------------------------------------------
# Stage: pureSPharm ---------------------------------------------------------
# ---------------------------------------------------------------------------


def compute_clm(
    mesh: Mesh,
    rmax: float,
    samples: int,
    origin: Sequence[float],
) -> Tuple[pyshtools.SHCoeffs, np.ndarray]:
    """Cast spherical rays, gather radial distances, and expand in SH coefficients."""

    agrid: List[List[float]] = []
    for theta in np.linspace(0, np.pi, samples, endpoint=False):
        latitudes: List[float] = []
        for phi in np.linspace(0, 2 * np.pi, samples, endpoint=False):
            direction = np.array(spher2cart(rmax, theta, phi))
            intersections = mesh.intersectWithLine(origin, np.array(origin) + direction)
            if intersections:
                value = np.linalg.norm(np.array(intersections[0]) - np.array(origin))
            else:
                value = rmax
            latitudes.append(value)
        agrid.append(latitudes)
    grid_array = np.array(agrid)
    grid = pyshtools.SHGrid.from_array(grid_array)
    return grid.expand(), grid_array


def run_pure_spharm_stage(
    meshes: Sequence[Mesh],
    mesh_paths: Sequence[Path],
    config: PipelineConfig,
    times: np.ndarray,
    rmax: float,
) -> Dict[str, Path]:
    """Replicate ``pureSPharm.py`` producing CLM coefficients and metadata."""

    output_dir = config.output_root / (
        f"pure_spharm-lmax{config.lmax}-N{config.sphere_samples}-deg_fit{config.fit_degree}"
    )
    ensure_directory(output_dir, overwrite=config.overwrite)

    coeffs_list: List[np.ndarray] = []
    grids: List[np.ndarray] = []
    pb = ProgressBar(0, len(meshes), c=2)
    for mesh, source, time_point in pb.zip(meshes, mesh_paths, times):
        coeffs, grid = compute_clm(
            mesh,
            rmax=rmax,
            samples=config.sphere_samples,
            origin=mesh.centerOfMass(),
        )
        coeffs_list.append(coeffs.to_array(lmax=config.lmax))
        grids.append(grid)
        pb.print(f"pureSPharm: {source.name} @ t={time_point}")

    coeff_array = np.array(coeffs_list)
    grid_array = np.array(grids)

    np.save(output_dir / "coefficients.npy", coeff_array)
    np.save(output_dir / "radial_grids.npy", grid_array)
    np.save(output_dir / "times.npy", times)

    metadata = {
        "input_meshes": [p.name for p in mesh_paths],
        "times": times.tolist(),
        "lmax": config.lmax,
        "samples": config.sphere_samples,
        "fit_degree": config.fit_degree,
        "rmax": rmax,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "pure_spharm": output_dir,
        "coefficients": output_dir / "coefficients.npy",
        "radial_grids": output_dir / "radial_grids.npy",
    }


# ---------------------------------------------------------------------------
# Stage: makeVoxel ----------------------------------------------------------
# ---------------------------------------------------------------------------


def run_make_voxel_stage(
    meshes: Sequence[Mesh],
    mesh_paths: Sequence[Path],
    config: PipelineConfig,
    bounds: Tuple[float, ...],
) -> Path:
    """Mirror ``makeVoxel.py`` by exporting signed-distance volumes."""

    volume_dir = config.output_root / f"TIF-signedDist_sampleSize{config.volume_dims[0]}"
    ensure_directory(volume_dir, overwrite=config.overwrite)

    pb = ProgressBar(0, len(meshes), c=1)
    for mesh, source in pb.zip(meshes, mesh_paths):
        volume = volumeFromMesh(
            mesh,
            dims=config.volume_dims,
            bounds=bounds,
            signed=True,
            negate=config.invert_normals,
        )
        volume.write(str(volume_dir / f"ReferenceShape_{source.stem}.vti"))
        pb.print(f"makeVoxel: {source.name}")

    metadata = {
        "input_meshes": [p.name for p in mesh_paths],
        "volume_dims": config.volume_dims,
        "bounds": bounds,
        "invert_normals": config.invert_normals,
    }
    with (volume_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return volume_dir


# ---------------------------------------------------------------------------
# Stage: computeAllIntesities -----------------------------------------------
# ---------------------------------------------------------------------------


def build_intensity_stack(
    radial_grids: np.ndarray,
    radius_steps: int,
) -> np.ndarray:
    """Synthesize radial intensity profiles from the measured surface distances."""

    if radius_steps <= 0:
        raise ValueError("radius_samples must be positive")
    radii = np.linspace(0.0, 1.0, radius_steps, endpoint=True)
    stacks = []
    for grid in radial_grids:
        stack = np.empty((radius_steps, 2, grid.shape[0], grid.shape[1]), dtype=float)
        for idx, scale in enumerate(radii):
            stack[idx, 0] = scale * grid
            stack[idx, 1] = grid
        stacks.append(stack)
    return np.array(stacks)


def run_compute_all_intensities_stage(
    radial_grids: np.ndarray,
    config: PipelineConfig,
) -> Path:
    """Generate data compatible with ``computeAllIntesities.py`` outputs."""

    intensities_dir = (
        config.output_root
        / f"allIntensities-sampleSize{config.volume_dims[0]}-radiusDiscretisation-"
        f"{config.radius_samples}-N-{config.sphere_samples}"
    )
    ensure_directory(intensities_dir, overwrite=config.overwrite)

    stacks = build_intensity_stack(radial_grids, config.radius_samples)
    np.save(intensities_dir / "allIntensities.npy", stacks)

    metadata = {
        "radius_samples": config.radius_samples,
        "sphere_samples": config.sphere_samples,
        "volume_sample_size": config.volume_dims,
    }
    with (intensities_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return intensities_dir


# ---------------------------------------------------------------------------
# Stage: morphing -----------------------------------------------------------
# ---------------------------------------------------------------------------


def interpolate_coefficients(
    clm_arrays: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Temporal polynomial interpolation of spherical-harmonic coefficients."""

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
        pb.print("morphing: interpolating coefficients ...")
    return interpolated


def reconstruct_surface(
    coeffs: np.ndarray,
    surface_resolution: int,
    smooth_iterations: int,
) -> Tuple[Mesh, Points]:
    """Reconstruct a surface mesh and its supporting point cloud."""

    clm_coeffs = pyshtools.SHCoeffs.from_array(coeffs)
    grid_reco = clm_coeffs.expand()
    agrid_reco = grid_reco.to_array()

    pts = []
    for i, longitude in enumerate(np.linspace(0, 360, num=agrid_reco.shape[1], endpoint=False)):
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


def run_morphing_stage(
    coeff_array: np.ndarray,
    times: np.ndarray,
    config: PipelineConfig,
    mesh_paths: Sequence[Path],
) -> Dict[str, Path]:
    """Replicate ``morphing.py`` by interpolating CLMs and exporting reconstructions."""

    target_times = (
        np.linspace(times.min(), times.max(), config.interpolation_steps)
        if config.interpolation_steps
        else times
    )

    interpolated = interpolate_coefficients(
        coeff_array,
        times,
        target_times,
        degree=config.fit_degree,
    )

    clm_dir = config.output_root / "CLM" / (
        "morphing_sampleSize"
        f"{config.volume_dims[0]}-radDisc{config.radius_samples}-N{config.sphere_samples}-"
        f"degFit{config.fit_degree}-lmax{config.lmax}"
    )
    morph_dir = config.output_root / (
        "morphing_sampleSize"
        f"{config.volume_dims[0]}-radDisc{config.radius_samples}-N{config.sphere_samples}-"
        f"degFit{config.fit_degree}-lmax{config.lmax}"
    )

    ensure_directory(clm_dir, overwrite=config.overwrite)
    ensure_directory(morph_dir, overwrite=config.overwrite)

    np.save(clm_dir / "allClmMatrix.npy", coeff_array)
    np.save(clm_dir / "allClmSpline.npy", interpolated)
    np.save(clm_dir / "times.npy", times)
    np.save(clm_dir / "target_times.npy", target_times)

    metadata = {
        "input_meshes": [p.name for p in mesh_paths],
        "times": times.tolist(),
        "target_times": target_times.tolist(),
        "lmax": config.lmax,
        "sphere_samples": config.sphere_samples,
        "radius_samples": config.radius_samples,
        "fit_degree": config.fit_degree,
        "surface_resolution": config.surface_resolution,
        "smooth_iterations": config.smooth_iterations,
    }
    with (morph_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if config.export_surfaces or config.export_points:
        surfaces_dir = morph_dir / "surfaces"
        points_dir = morph_dir / "point_clouds"
        if config.export_surfaces:
            ensure_directory(surfaces_dir, overwrite=config.overwrite)
        if config.export_points:
            ensure_directory(points_dir, overwrite=config.overwrite)

        pb = ProgressBar(0, interpolated.shape[0], c=4)
        for idx in pb.range():
            surface, cloud = reconstruct_surface(
                interpolated[idx],
                surface_resolution=config.surface_resolution,
                smooth_iterations=config.smooth_iterations,
            )
            if config.export_surfaces:
                surface.write(str(surfaces_dir / f"nucleus_{idx:03d}.vtk"))
            if config.export_points:
                cloud.write(str(points_dir / f"nucleus_points_{idx:03d}.vtp"))
            pb.print("morphing: writing reconstructions ...")

    return {"clm": clm_dir, "morphing": morph_dir}


# ---------------------------------------------------------------------------
# Pipeline driver -----------------------------------------------------------
# ---------------------------------------------------------------------------


def _collect_mesh_paths(directory: Path, pattern: str | Sequence[str]) -> List[Path]:
    """Return sorted mesh paths matching one or more glob patterns."""

    if isinstance(pattern, str):
        patterns: Sequence[str] = (pattern,)
    else:
        patterns = tuple(pattern)
    paths: set[Path] = set()
    for glob_pattern in patterns:
        paths.update(directory.glob(glob_pattern))
    mesh_paths = sorted(p for p in paths if p.is_file())
    if not mesh_paths:
        patterns_text = ", ".join(patterns)
        raise FileNotFoundError(
            f"No meshes matching pattern(s) {patterns_text!r} were found in {directory}"
        )
    return mesh_paths


def run(config: PipelineConfig) -> None:
    """Execute the selected FeARLesS stages with nucleus-specific defaults."""

    mesh_paths = _collect_mesh_paths(config.mesh_dir, config.pattern)

    meshes = load_meshes(mesh_paths)

    if config.times is not None:
        if len(config.times) != len(meshes):
            raise ValueError("Number of provided times does not match number of meshes")
        times = np.array(config.times, dtype=float)
    else:
        times = np.arange(len(meshes), dtype=float)

    bounds = compute_bounds(meshes, padding=config.volume_padding)
    rmax = max(compute_radius(mesh.bounds()) for mesh in meshes) * 1.05

    stage_outputs: Dict[str, Path] = {}

    if config.run_make_voxel:
        stage_outputs["make_voxel"] = run_make_voxel_stage(meshes, mesh_paths, config, bounds)

    if config.run_pure_spharm:
        outputs = run_pure_spharm_stage(meshes, mesh_paths, config, times, rmax)
        stage_outputs.update(outputs)
        coeff_array = np.load(outputs["coefficients"])
        radial_grids = np.load(outputs["radial_grids"])
    else:
        coeff_path = stage_outputs.get("coefficients") or (
            config.output_root
            / f"pure_spharm-lmax{config.lmax}-N{config.sphere_samples}-deg_fit{config.fit_degree}"
            / "coefficients.npy"
        )
        grid_path = coeff_path.parent / "radial_grids.npy"
        coeff_array = np.load(coeff_path)
        radial_grids = np.load(grid_path)

    if config.run_compute_all_intensities:
        intensities_dir = run_compute_all_intensities_stage(radial_grids, config)
        stage_outputs["compute_all_intensities"] = intensities_dir

    if config.run_morphing:
        morph_outputs = run_morphing_stage(coeff_array, times, config, mesh_paths)
        stage_outputs.update(morph_outputs)

    summary_path = config.output_root / "pipeline_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump({k: str(v) for k, v in stage_outputs.items()}, handle, indent=2)


if __name__ == "__main__":
    run(CONFIG)
