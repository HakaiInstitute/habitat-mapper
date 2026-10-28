# Troubleshooting & FAQ

## Common errors

| Problem                                                    | Cause and fix                                                                                         |
|------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `command not found: hab` / `'hab' is not recognized`        | The environment isn't active. [Activate it](#do-i-need-to-activate-the-environment-every-time).       |
| *running scripts is disabled on this system* (Windows)      | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then activate again. |
| GDAL errors during installation (Windows)                   | The environment is on a network or secondary drive. [Recreate it on `C:`](#i-get-gdal-errors-during-installation). |
| `CUDA out of memory` or the program crashes on large images | Lower `--crop-size` (e.g. `512`) or `--batch-size`. See [Tune Processing](../guides/tune-processing.md). |
| `Permission denied` when writing the output                 | The output file is open in ArcGIS Pro or QGIS. Close it and try again.                                |
| The output is completely black                              | Expected: display it with unique values. See [View Results in GIS](../get-started/view-results.md).   |
| The output doesn't line up with the input                   | The input has no coordinate reference system. Assign one and rerun.                                   |
| Warnings when processing starts                             | Usually safe to ignore. They suggest speed-ups, such as using a [tiled GeoTIFF](../guides/prepare-imagery.md#use-a-tiled-geotiff). |

## Getting started

### Do I need to activate the environment every time?

Yes, once each time you open a new terminal window. If you followed the [install guide](../get-started/install.md),
run:

--8<-- "activate-env.md"

### What kind of computer do I need?

Any recent Windows, macOS or Linux computer will work. An NVIDIA GPU makes processing much faster, but isn't
required. See [GPU acceleration](../guides/install-advanced.md#gpu-acceleration).

### I get GDAL errors during installation

This usually happens on Windows when the `uv` environment is on a different drive from `uv` itself, such as a network
drive. Create the environment on your local `C:` drive instead. Your images can stay wherever they are, because the
environment's location only matters during installation.

## Data

### Which model should I use?

The one trained on imagery like yours. See the [model overview](../models/index.md).

### What resolution should my imagery be?

For drone models, 1–10 cm per pixel, ideally about 5 cm. Mussel models were trained on finer imagery. Satellite
models have their own requirements. Each [model page](../models/index.md) lists them.

### Will it work outside western North America?

The models were trained mostly on imagery from British Columbia and California, but have worked well elsewhere,
including Alaska and Patagonia. If results are poor for your region, [get in touch](#still-stuck). We're interested
in improving coverage.

### Can I use a different band in place of red, green or blue?

No. The models depend on the specific bands they were trained on. You can [reorder
bands](../guides/prepare-imagery.md#bands-are-in-red-green-blue-then-nir-order), but not substitute them.

## Running

### How long does processing take?

From seconds to hours. It depends on image size, the model, your hardware (a GPU is much faster) and the crop size.
A progress bar shows how far along it is.

### Can I process many images at once?

Yes. Loop over a folder from the terminal or from Python. See [Process Many Images](../guides/batch-processing.md).

### Why did my results change between runs?

Check that you used the same model revision (`--rev`) and the same `--crop-size`. Both affect the output.

## Still stuck?

[Email us](mailto:habitat.mapper.support@hakai.org) or
[open an issue on GitHub](https://github.com/HakaiInstitute/habitat-mapper/issues). Include the command you ran,
the full error message, and the output of `hab --version`.
