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

A GPU is optional, and Habitat-Mapper runs on the CPU without one. GPU support is an install option, because the
libraries are large and depend on your hardware:

| Hardware                                   | Install                              | `--device` |
|--------------------------------------------|--------------------------------------|------------|
| NVIDIA GPU (Windows, Linux)                | `pip install "habitat-mapper[cuda]"`   | `cuda`     |
| AMD, Intel or Apple GPU                    | `pip install "habitat-mapper[webgpu]"` | `webgpu`   |
| Apple Neural Engine (macOS)                | included                             | `coreml`   |

`[cuda]` needs a recent NVIDIA driver and downloads about 2–3 GB of CUDA libraries. `[webgpu]` uses DirectX 12 on
Windows, Vulkan on Linux and Metal on macOS, so it only needs your normal graphics driver. You can install both.

By default (`--device auto`), Habitat-Mapper uses the fastest device it finds: CUDA, then WebGPU, then the CPU. If a
device can't run a model, it logs a warning and falls back to the next one. To see what it found, run:

```bash
hab devices
```

To choose a device yourself, pass `--device` to `hab segment`, `device=` to `model.process()`, or set the
`HAB_DEVICE` environment variable. CoreML is only used when you ask for it. It compiles models slowly and wasn't
faster than the CPU on the models we tested. With a GPU you can usually raise `--batch-size` (see
[Tune Processing](tune-processing.md#batch-size-and-gpus)).

!!! note "Upgrading from 0.14 or earlier"
    The CUDA libraries used to install by default on Windows and Linux. If you use an NVIDIA GPU, reinstall with
    `pip install "habitat-mapper[cuda]"` to keep using it.

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
