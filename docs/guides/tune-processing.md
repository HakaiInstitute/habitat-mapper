# Tune Processing

The defaults work for most images. This page covers the options worth changing and what they do. Every option is
listed in the [CLI reference](../reference/cli.md#hab-segment). The Python equivalents are noted below.

## Crop size: quality vs. memory

Habitat-Mapper splits large images into overlapping square tiles, runs the model on each one and stitches the results
together. `--crop-size` (`-z`) sets the tile width in pixels. It must be an even number. The default is 1024, or
whatever size the model requires.

**Larger tiles give fewer stitching artifacts**, but need more memory. Start high and step down if you run out of
memory or the program crashes:

```bash
hab segment kelp-rgb input.tif output.tif --crop-size 3200
```

Try `3200` → `2048` → `1024` → `512`.

<div style="display:flex;" markdown>
  <figure markdown>
  ![Output with a small crop size](../images/hab-small-window.png)
  <figcaption>Small crop size: visible tile edges</figcaption>
  </figure>
  <figure markdown>
  ![Output with a large crop size](../images/hab-big-window.png)
  <figcaption>Large crop size: smoother result</figcaption>
  </figure>
</div>

!!! note "Reproducibility"
    The crop size changes the output slightly. To reproduce an earlier result, use the same crop size and the same
    model revision (`--rev`).

Python: `crop_size=3200`.

## Batch size and GPUs

`--batch-size` sets how many tiles are processed at once. The default is 1. With a GPU, raising it to 4 or more can
speed things up considerably. Lower it again if you get out-of-memory errors. GPU support is an install option. See
[GPU acceleration](install-advanced.md#gpu-acceleration), and use `--device` to choose the hardware.

```bash
hab segment kelp-rgb input.tif output.tif --batch-size 4
```

Python: `batch_size=4`.

## Smoothing the output

Two optional filters clean up speckle in the output. Both are off by default (`0`) and take an **odd** kernel size
in pixels.

| Option    | Effect                                                                     | Python              |
|-----------|----------------------------------------------------------------------------|---------------------|
| `--blur`  | Median blur. Removes isolated pixels and smooths edges.                    | `blur_kernel_size`  |
| `--morph` | Morphological opening and closing. Removes small specks and fills small holes. | `morph_kernel_size` |

```bash
hab segment kelp-rgb input.tif output.tif --blur 5 --morph 3
```

## Model revisions

Models are updated over time. By default `hab segment` uses the latest revision. To list the revisions of a model,
run:

```bash
hab revisions kelp-rgb
```

To use a specific revision, for example to match earlier results, pass `--rev`:

```bash
hab segment kelp-rgb input.tif output.tif --rev 20240722
```

Python: `model_registry["kelp-rgb", "20240722"]`.
