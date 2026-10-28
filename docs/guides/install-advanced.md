# Install for Python Users

Habitat-Mapper is published on [PyPI](https://pypi.org/project/habitat-mapper) and supports Python 3.11 to 3.14.
Install it into a virtual environment so its dependencies don't conflict with system packages.

=== "uv project"

    ```bash
    uv add habitat-mapper
    ```

=== "pip"

    ```bash
    pip install habitat-mapper
    ```

This installs the `habitat_mapper` Python package and the `hab` command line tool into the same environment.
Model weights are not bundled. Each model downloads to a local cache the first time you use it. To clear that
cache, run `hab clean`.

## GPU acceleration

A GPU is optional. On Windows and Linux, Habitat-Mapper installs the GPU build of ONNX Runtime, including CUDA
libraries, and automatically uses a supported NVIDIA GPU when one is available. Otherwise it falls back to the
CPU. On macOS it always runs on the CPU. With a GPU you can usually raise `--batch-size` (see
[Tune Processing](tune-processing.md#batch-size-and-gpus)).

## Updating

We follow [semantic versioning](https://semver.org/).

=== "uv project"

    ```bash
    uv lock --upgrade-package habitat-mapper
    ```

=== "pip"

    ```bash
    pip install --upgrade habitat-mapper
    ```

## Next steps

- [Use from Python](python.md) shows how to call models from your own code.
- [CLI reference](../reference/cli.md) lists every command and option.
- [Cheat sheet](../reference/cheat-sheet.md) collects common commands on one page.
