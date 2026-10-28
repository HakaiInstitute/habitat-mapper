# Cheat Sheet

Common commands on one page. For every option, see the [CLI reference](cli.md). Every command also accepts `--help`.

## Commands

| Task                              | Command                                                      |
|-----------------------------------|--------------------------------------------------------------|
| Install                           | `pip install habitat-mapper`                                 |
| Upgrade                           | `pip install --upgrade habitat-mapper`                       |
| Check version                     | `hab --version`                                              |
| List models                       | `hab models`                                                 |
| List a model's revisions          | `hab revisions kelp-rgb`                                     |
| Segment an image                  | `hab segment kelp-rgb input.tif output.tif`                  |
| …with a specific revision         | `hab segment kelp-rgb input.tif output.tif --rev 20240722`   |
| …with bands stored B, G, R, NIR   | `hab segment kelp-rgbi input.tif output.tif -b 3 -b 2 -b 1 -b 4` |
| …in larger tiles (fewer seams)    | `hab segment kelp-rgb input.tif output.tif --crop-size 3200` |
| …faster on a GPU                  | `hab segment kelp-rgb input.tif output.tif --batch-size 4`   |
| …with smoothing                   | `hab segment kelp-rgb input.tif output.tif --blur 5 --morph 3` |
| …on a specific device             | `hab segment kelp-rgb input.tif output.tif --device cpu`     |
| …without progress bars            | `hab segment kelp-rgb input.tif output.tif --quiet`          |
| List the devices models can run on | `hab devices`                                               |
| Clear the model cache             | `hab clean`                                                  |

Flags can also be written in long form: `hab segment -m kelp-rgb -i input.tif -o output.tif`.

## Python

```python
from habitat_mapper import model_registry

model = model_registry["kelp-rgb"]  # or model_registry["kelp-rgb", "20240722"]
model.process(img_path="input.tif", output_path="output.tif", crop_size=3200)
```

See [Use from Python](../guides/python.md) and the [Python API reference](python-api.md).

## Where to look next

| Need                                 | Page                                                        |
|--------------------------------------|-------------------------------------------------------------|
| Which model to use, output classes   | [Models](../models/index.md)                                |
| Input data type, band order, nodata  | [Prepare Your Imagery](../guides/prepare-imagery.md)        |
| What each option does                | [Tune Processing](../guides/tune-processing.md)             |
| Folders of images                    | [Process Many Images](../guides/batch-processing.md)        |
| Errors                               | [Troubleshooting & FAQ](../help/faq.md)                     |
