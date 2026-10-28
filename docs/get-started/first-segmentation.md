# Your First Segmentation

You'll run a model on one image with the `hab segment` command. This produces a new GeoTIFF in which every pixel is
labelled with what the model found there.

!!! abstract "Before you start"
    - Habitat-Mapper is [installed](install.md), and your terminal prompt starts with `(habitat-env)`.
      If it doesn't, [activate the environment](install.md#3-activate-the-environment).
    - You have a GeoTIFF (`.tif`) image, such as a drone orthomosaic.

## 1. Pick a model

Each model is trained for one kind of imagery and one habitat. To list them, run:

```bash
hab models
```

For example, use `kelp-rgb` for kelp in an ordinary colour drone image. The [model overview](../models/index.md)
shows which model fits your imagery and what each one outputs.

## 2. Move to your data

Use `cd` to move into the folder that holds your image ([refresher](terminal.md#3-basic-navigation)):

```bash
cd path/to/your/images
```

## 3. Run the model

`hab segment` needs three things, in this order: the **model**, the **input image** and the **output file** to
create.

```bash
hab segment kelp-rgb site_01.tif site_01_kelp.tif
```

!!! tip "Use quotes if a name contains spaces"
    `hab segment kelp-rgb "my site.tif" "my site kelp.tif"`

The first run downloads the model, which can take a minute. Habitat-Mapper then processes the image in tiles and
shows a progress bar. When it finishes you'll see **Processing Complete** and the path to your output file.

??? question "It failed or ran out of memory"
    - `command not found` or `'hab' is not recognized`: the environment isn't active.
      [Activate it](install.md#3-activate-the-environment).
    - Out-of-memory errors: add `--crop-size 512` to process smaller tiles.
    - `Permission denied`: close the output file in ArcGIS Pro or QGIS, then try again.

    More fixes are in [Troubleshooting & FAQ](../help/faq.md).

## What next?

Your output looks black at first. That's expected. The next step shows how to display it.

[:material-arrow-right: Next: View Results in GIS](view-results.md){ .md-button .md-button--primary }

Once that works, you can:

- Handle imagery with bands in a different order. See [Prepare Your Imagery](../guides/prepare-imagery.md).
- Get cleaner results on large images. See [Tune Processing](../guides/tune-processing.md).
- Process a whole folder at once. See [Process Many Images](../guides/batch-processing.md).
- See every option. See the [CLI reference](../reference/cli.md).
