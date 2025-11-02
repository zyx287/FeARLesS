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

`prepare_nucleus_data.py` converts the original meshes to `.vtk` surfaces and,
optionally, to signed-distance volumes that can be re-used by the original
FeARLesS scripts.

```bash
python nucleus/prepare_nucleus_data.py <ply-input-folder> \
    --surface-output nucleus_data/surfaces \
    --volume-output nucleus_data/volumes \
    --sample-size 160 160 160 \
    --padding 1.15 \
    --invert-normals    # optional, flip the signed distance convention
```

### Parameters

- `input_dir`: folder containing the `.ply` meshes.
- `--surface-output`: where to store the converted `.vtk` surfaces (default:
  `nucleus_data/surfaces`).
- `--volume-output`: destination for the generated signed-distance volumes
  (default: `nucleus_data/volumes`).
- `--sample-size`: number of voxels along each axis when creating volumes.
- `--padding`: padding factor applied to the shared bounding box before voxelising.
- `--invert-normals`: invert the sign of the resulting signed distance field.
- `--overwrite`: replace existing output directories instead of aborting.
- `--skip-surfaces` / `--skip-volumes`: disable either export step.

Metadata about the processed meshes, bounding box, and grid resolution are saved
next to the exported assets (`metadata.json`).

## 2. Running the modelling

`run_nucleus_modeling.py` mirrors the FeARLesS pipeline while exposing explicit
parameters through an editable configuration block at the top of the script.
Update the ``CONFIG`` definition to point at your meshes, tweak the numerical
settings, and then execute the module:

```python
# nucleus/run_nucleus_modeling.py
CONFIG = ModelingConfig(
    mesh_dir=Path("nucleus_data/surfaces"),
    pattern="*.vtk",
    times=None,              # list of time values matching the mesh order
    lmax=40,                 # maximum spherical-harmonics degree
    samples=400,             # angular samples for the ray casting grid
    fit_degree=4,            # polynomial degree for temporal interpolation
    interpolation_steps=15,  # number of reconstructed time points (None keeps originals)
    surface_resolution=150,  # marching cubes grid resolution
    smooth_iterations=20,    # smoothing iterations for reconstructed meshes
    overwrite=False,         # set True to replace existing outputs
    export_surfaces=True,    # write reconstructed surfaces as .vtk
    export_points=False,     # write intermediate point clouds as .vtp
    output_dir=Path("nucleus_results"),
)
```

Run the reconstruction with:

```bash
python nucleus/run_nucleus_modeling.py
```

### Outputs

Running the modelling script produces the following artefacts inside the chosen
``CONFIG.output_dir``:

- `coefficients.npy`: interpolated spherical-harmonics coefficients.
- `metadata.json`: configuration, input filenames, and derived parameters
  (`rmax`, time points, interpolation targets, etc.).
- `surfaces/`: reconstructed meshes (when `CONFIG.export_surfaces` is enabled).
- `point_clouds/`: point clouds used to build each surface (when
  `CONFIG.export_points` is enabled).

These assets can be re-used with the original FeARLesS post-processing scripts
or further analysed using standard 3D tooling.
