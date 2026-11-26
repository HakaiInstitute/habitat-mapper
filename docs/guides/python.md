# Use from Python

Everything the `hab` command does is available from Python. Use the Python API to process images inside a script,
notebook or pipeline, or to build on Habitat-Mapper in your own tools. For a single image, the
[command line](../get-started/first-segmentation.md) is simpler.

Install with `pip install habitat-mapper` (see [Install for Python Users](install-advanced.md)).

## Load a model

All models are available through `model_registry`:

```python
from habitat_mapper import model_registry

model_registry.list_model_names()  # ['eelgrass-rgb', 'kelp-ps8b', 'kelp-rgb', ...]
model_registry.list_models()  # [(name, revision), ...]
model_registry.revisions("kelp-rgb")  # newest first, e.g. ['20240722']
model_registry.get_latest_revision("kelp-rgb")
model_registry.config("kelp-rgb")  # the latest revision's ModelConfig, without making a model

model = model_registry["kelp-rgb"]  # latest revision
model = model_registry["kelp-rgb", "20240722"]  # a specific revision
```

An unknown model or revision raises `ModelNotFoundError`, a `KeyError` whose message lists the revisions there are.

Model weights download on first use and are cached for later runs. To download ahead of time, e.g. before working
offline, or to check what's there, use `model_cache`:

```python
from habitat_mapper import model_cache

cfg = model_registry.config("kelp-rgb")
model_cache.is_cached(cfg)  # True once its files are downloaded
model_cache.fetch(cfg)  # downloads anything missing, and returns the model file's path
```

Each lookup makes a new model, so keep the one you got to process several images: it loads the model onto your
hardware once, the first time it runs, and reuses it for every later image.

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
| `device`            | `--device`     | `$HAB_DEVICE` or `"auto"` | [GPU acceleration](install-advanced.md#gpu-acceleration) |

The options are checked before anything runs, with the same rules as `hab segment`. An invalid one, such as an odd
`crop_size`, raises a `ValueError`. To check options on their own, e.g. in a form, build a `SegmentOptions`:

```python
from habitat_mapper import SegmentOptions

options = SegmentOptions(crop_size=2048, blur_kernel_size=5)  # raises ValueError if invalid
model.process("site_01.tif", "site_01_kelp.tif", **options.model_dump())
```

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

## Logging

Habitat-Mapper logs with [loguru](https://loguru.readthedocs.io/). Importing it doesn't change your program's logging:
messages go wherever loguru sends them, which is stderr by default. To hide its debug messages, for example:

```python
import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")
```

## Show progress, cancel, or preview in your own app

`process()` takes three more arguments for building on Habitat-Mapper:

- `progress`: a function that returns a [`ProgressReporter`](../reference/python-api.md#progress-reporting), called
  once for each stage (segmenting, then smoothing). Use it to show progress in your own interface.
- `cancel`: a function called before each batch of tiles. Return `True` to stop: the partial output is deleted and
  `ProcessingCancelled` is raised.
- `observer`: a [`ProcessingObserver`](../reference/python-api.md#processing), told about each batch and each part of
  the result as it is written, e.g. to draw a live preview.

The desktop app is built this way.

## Run your own model

Describe your ONNX model with a `ModelConfig` and make an `ONNXModel` from it:

```python
from habitat_mapper import ModelConfig, ONNXModel

config = ModelConfig.model_validate_json(open("my_model.json").read())
model = ONNXModel(config)
model.process("site_01.tif", "site_01_mine.tif")
```

A `ModelConfig` can't be changed once it is made, because every model made from it shares it. Make a changed copy
instead, e.g. `config.with_changes(max_pixel_value=10000.0)`. If your model's outputs need different handling
to turn them into classes, subclass `ONNXModel` and override `postprocess`.

## Going further

- [Python API reference](../reference/python-api.md) documents every class and parameter, including
  `ModelConfig` and `ONNXModel` for running your own ONNX models.
- [Process Many Images](batch-processing.md) shows a loop over a folder.
- The [example notebooks](https://github.com/HakaiInstitute/habitat-mapper/tree/main/examples) show end-to-end
  workflows.
