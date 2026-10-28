# Use from Python

Everything the `hab` command does is available from Python. Use the Python API to process images inside a script,
notebook or pipeline, or to build on Habitat-Mapper in your own tools. For a single image, the
[command line](../get-started/first-segmentation.md) is simpler.

Install with `pip install habitat-mapper` (see [Install for Python Users](install-advanced.md)).

## Load a model

All models are available through `model_registry`:

```python
from habitat_mapper import model_registry

model_registry.list_model_names()       # ['kelp-rgb', 'kelp-rgbi', 'mussel-rgb', ...]
model_registry.list_models()            # [(name, revision), ...]
model_registry.get_latest_revision("kelp-rgb")

model = model_registry["kelp-rgb"]              # latest revision
model = model_registry["kelp-rgb", "20240722"]  # a specific revision
```

Model weights download on first use and are cached for later runs.

## Process an image

```python
model.process(
    img_path="site_01.tif",
    output_path="site_01_kelp.tif",
)
```

`process()` accepts the same options as `hab segment`:

| Python argument     | CLI flag       | Default           | Guide                                                    |
|---------------------|----------------|-------------------|----------------------------------------------------------|
| `crop_size`         | `--crop-size`  | model's preferred | [Crop size](tune-processing.md#crop-size-quality-vs-memory) |
| `batch_size`        | `--batch-size` | `1`               | [Batch size](tune-processing.md#batch-size-and-gpus)     |
| `blur_kernel_size`  | `--blur`       | `0` (off)         | [Smoothing](tune-processing.md#smoothing-the-output)     |
| `morph_kernel_size` | `--morph`      | `0` (off)         | [Smoothing](tune-processing.md#smoothing-the-output)     |
| `band_order`        | `-b`           | `None`            | [Band order](prepare-imagery.md#bands-are-in-red-green-blue-then-nir-order) |
| `quiet`             | `--quiet`      | `False`           | [Process many images](batch-processing.md)               |

For example, an RGB+NIR image stored as Blue, Green, Red, NIR, processed in large tiles:

```python
model = model_registry["kelp-rgbi"]
model.process(
    img_path="multispectral.tif",
    output_path="kelp.tif",
    crop_size=2048,
    band_order=[3, 2, 1, 4],
)
```

## Quiet mode and errors

`quiet=True` hides progress bars for segmentation, post-processing and downloads. Errors are never silenced: a
failure raises an exception either way.

## Going further

- [Python API reference](../reference/python-api.md) documents every class and parameter, including
  `ModelConfig` and `ONNXModel` for running your own ONNX models.
- [Process Many Images](batch-processing.md) shows a loop over a folder.
- The [example notebooks](https://github.com/HakaiInstitute/habitat-mapper/tree/main/examples) show end-to-end
  workflows.
