# Nucleus Modelling Pipeline

This folder provides helper scripts to adapt **FeARLesS** to nucleus-shaped meshes.
The utilities convert surface meshes exported as `.ply` files into the file layout
expected by the FeARLesS pipeline and offer a configurable spherical-harmonics
reconstruction tailored to the nucleus of granule cells.

## Background: spherical-harmonic modelling in FeARLesS

FeARLesS represents each mesh as a radial function sampled on a spherical
parameterisation. Rays are cast from a common origin through the surface to
measure the distance at each polar/azimuthal angle. The resulting scalar field is
expanded into real spherical-harmonic coefficients (``C_{l,m}``). These
coefficients compactly encode the nucleus shape and can be interpolated through
time using low-order polynomials. Reconstructed meshes are produced by
evaluating the interpolated radial field back into 3D space and applying surface
reconstruction and smoothing.

## 1. Preparing the input data

`prepare_nucleus_data.py` converts the original meshes to `.vtp` surfaces and,
optionally, to `.vti` signed-distance volumes that can be re-used by the original
FeARLesS scripts. Like the modelling driver, it uses an editable configuration
block so you can simply point it at your nucleus folder.

```python
# nucleus/prepare_nucleus_data.py
PREPARATION_CONFIG = PreparationConfig(
    input_dir=Path("/absolute/path/to/nuclei"),
    surface_output=Path("nucleus_data/surfaces"),
    volume_output=Path("nucleus_data/volumes"),
    file_suffix="_nuclei.ply",        # match files such as cellid_s1_nuclei.ply
    dims=(160, 160, 160),
    padding=1.15,
    invert_normals=False,
    overwrite=False,
    skip_surfaces=False,
    skip_volumes=False,
)
```

Run the conversion with:

```bash
python nucleus/prepare_nucleus_data.py
```

### Parameters to consider

- **`input_dir`** – the folder containing the original `.ply` nuclei meshes.
  The default `file_suffix` of `_nuclei.ply` matches files named like
  `cellid_s1_nuclei.ply`; adjust it if your naming scheme differs.
- **`surface_output` / `volume_output`** – locations where converted `.vtp`
  surfaces and `.vti` signed-distance volumes will be written. Each directory
  receives a `metadata.json` summary with the input file order, bounding box,
  and voxel grid size.
- **`dims`** – number of voxels `(nx, ny, nz)` used when voxelising the meshes.
  Higher numbers increase fidelity at the cost of larger files and processing
  time.
- **`padding`** – scale factor applied to the shared bounding box of all meshes.
  Increase it if volumes clip the shape; decrease it to tighten the frame.
- **`invert_normals`** – flip this to `True` when your PLY files are oriented so
  the inside of the nucleus is labelled with positive distance.
- **`overwrite`** – toggle to `True` to replace existing output folders.
- **`skip_surfaces` / `skip_volumes`** – set either to `True` if you only need
  surfaces or volumes.

## 2. Running the modelling

`run_nucleus_modeling.py` mirrors the four scripts in the original FeARLesS
pipeline.  Each stage can be toggled on/off through an editable configuration
block at the top of the file and the generated folders follow the familiar
`pureSPharm`, `makeVoxel`, `computeAllIntesities`, and `morphing` naming
convention.

Update the ``CONFIG`` definition to point at your meshes, tweak the numerical
settings, choose which stages to execute, and then run the module:

```python
# nucleus/run_nucleus_modeling.py
CONFIG = PipelineConfig(
    mesh_dir=Path("nucleus_data/surfaces"),
    pattern=("*.vtp", "*.vtk"),
    times=None,                    # list of time values matching the mesh order
    run_make_voxel=True,           # export signed-distance volumes (makeVoxel)
    run_pure_spharm=True,          # compute CLMs and radial samples (pureSPharm)
    run_compute_all_intensities=True,  # save stacked radial intensities
    run_morphing=True,             # interpolate CLMs and reconstruct meshes
    volume_dims=(160, 160, 160),   # voxel grid resolution for makeVoxel
    volume_padding=1.15,           # padding factor for the shared bounding box
    invert_normals=False,          # flip signed distance convention when needed
    lmax=40,                       # maximum spherical-harmonics degree
    sphere_samples=400,            # angular samples for the ray casting grid
    radius_samples=120,            # depth samples per ray (computeAllIntesities)
    fit_degree=4,                  # polynomial degree for temporal interpolation
    interpolation_steps=15,        # # of reconstructed time points (None keeps originals)
    surface_resolution=150,        # reconstruction grid resolution
    smooth_iterations=20,          # smoothing iterations for reconstructed meshes
    export_surfaces=True,          # write reconstructed surfaces as .vtk
    export_points=False,           # write intermediate point clouds as .vtp
    overwrite=False,               # set True to replace existing outputs
    output_root=Path("nucleus_results"),
)
```

If you used the preparation script as-is, point `mesh_dir` to the aligned
surface folder (e.g., `nucleus_data/surfaces_aligned`). The default pattern
matches the `.vtp` files created by the preparation stage while still accepting
legacy `.vtk` surfaces.

Run the reconstruction with:

```bash
python nucleus/run_nucleus_modeling.py
```

### Stage-by-stage outputs

Running the script generates folders that correspond to the original FeARLesS
programs:

1. **pureSPharm** (`pure_spharm-lmax.../`)
   - `coefficients.npy`: spherical-harmonic coefficients for each mesh.
   - `radial_grids.npy`: sampled radial distances per polar/azimuthal angle,
     used to synthesise intensity stacks for the next step.
   - `times.npy` and `metadata.json`: timing information and capture settings.

2. **makeVoxel** (`TIF-signedDist_sampleSize.../`)
   - Signed-distance volumes exported as `ReferenceShape_<mesh>.vti`.
   - Shared bounding box metadata (`metadata.json`).

3. **computeAllIntesities** (`allIntensities-sampleSize.../`)
   - `allIntensities.npy`: a depth stack derived from the radial grids above.
     Each ray stores both a scaled and absolute distance profile to remain
     compatible with routines expecting the limb-bud format.

4. **morphing** (`CLM/...` and `morphing_sampleSize.../`)
   - `allClmMatrix.npy` and `allClmSpline.npy`: raw and interpolated CLM
     tensors ready for downstream analysis.
   - `metadata.json`: reconstruction parameters and paths.
   - `surfaces/` and `point_clouds/`: reconstructed meshes and supporting point
     clouds when the respective export flags are enabled.

The root folder (`nucleus_results/` by default) also contains
`pipeline_summary.json` listing the location of every generated artefact.
