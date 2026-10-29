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

A GPU is optional, and Habitat-Mapper runs on the CPU without one. GPU support for most graphics cards is
included through [WebGPU](https://www.w3.org/TR/webgpu/), which uses DirectX 12 on Windows, Vulkan on Linux and Metal
on macOS. It works with your normal graphics driver, and was 2–14× faster than the CPU in our tests, depending on
the GPU.

On NVIDIA GPUs, CUDA is usually faster still, up to about twice as fast as WebGPU in our tests. Its libraries are
large (about 2 GB), so they are an optional extra on Windows and Linux. There are two versions; which one you need
depends on your GPU and driver. Run `nvidia-smi` and read `CUDA Version` in the top right corner.

| Hardware                                                        | Install                                 | `--device` |
|-----------------------------------------------------------------|-----------------------------------------|------------|
| NVIDIA RTX 20 series or newer, `CUDA Version: 13.x` or newer    | `pip install "habitat-mapper[cuda]"`    | `cuda`     |
| NVIDIA GPU with `CUDA Version: 12.x`                            | `pip install "habitat-mapper[cuda12]"`  | `cuda`     |
| Older NVIDIA GPU: GTX 10 or 900 series, Quadro P or M series    | `pip install "habitat-mapper[cuda12]"`  | `cuda`     |
| AMD, Intel or Apple GPU                                         | `pip install habitat-mapper` (included) | `webgpu`   |
| Apple Neural Engine (macOS)                                     | `pip install habitat-mapper` (included) | `coreml`   |

`[cuda]` installs CUDA 13, which needs NVIDIA driver R580 or newer and a GPU from the RTX 20 series (Turing) or
newer. CUDA 13 dropped support for older GPUs, even though their drivers report CUDA 13. `[cuda12]` installs CUDA 12
for older drivers and GPUs. If you pick the wrong one, Habitat-Mapper says which to use instead.
Install only one of them. To switch from one to the other, create a new environment: the two install some of the same
files, so uninstalling one also breaks the other. If you use a uv project, add the extra with
`uv add "habitat-mapper[cuda]"`.

On Linux, WebGPU needs your GPU's Vulkan driver. NVIDIA's standard driver includes it; for AMD and Intel GPUs it is
usually the `mesa-vulkan-drivers` package.

By default (`--device auto`), Habitat-Mapper uses the fastest device it finds: CUDA, then WebGPU, then the CPU. If a
device can't run a model, it logs a warning and falls back to the next one. If your GPU isn't being used, a message
says why and what to install. To see what it found, run:

```bash
hab devices
```

To choose a device yourself, pass `--device` to `hab segment`, `device=` to `model.process()`, or set the
`HAB_DEVICE` environment variable. CoreML is only used when you ask for it. It compiles models slowly and wasn't
faster than the CPU on the models we tested. See [Tune Processing](tune-processing.md#batch-size-and-gpus) for
choosing a batch size on a GPU.

!!! note "Upgrading from a version before 0.17"
    GPU support changed in 0.17: the CUDA libraries are no longer installed by default. See
    [Upgrade to 0.17](upgrade-to-0-17.md).

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

If you installed an extra such as `[cuda]`, include it when you upgrade with pip:
`pip install --upgrade "habitat-mapper[cuda]"`.

## Next steps

- [Use from Python](python.md) shows how to call models from your own code.
- [CLI reference](../reference/cli.md) lists every command and option.
- [Cheat sheet](../reference/cheat-sheet.md) collects common commands on one page.
