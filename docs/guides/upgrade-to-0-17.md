# Upgrade to 0.17

Habitat-Mapper 0.17 changes how GPUs are supported:

- **GPU support for AMD, Intel and Apple GPUs is now included.** Before 0.17, only NVIDIA GPUs could be used. Models
  now run on most graphics cards through WebGPU, 2–14× faster than on the CPU in our tests.
- **The NVIDIA CUDA libraries are no longer installed by default.** They are about 2 GB and only useful on NVIDIA
  GPUs, so they are now an optional extra on Windows and Linux. CUDA is still the fastest option on NVIDIA GPUs.
- **Habitat-Mapper chooses the device from your hardware.** Run `hab devices` to see what it found, and use
  `--device` to choose one yourself.

The commands below use `uv pip`, as in our [install guide](../get-started/install.md). If you use plain `pip`, drop
the `uv`. Activate your environment first.

## Without an NVIDIA GPU

Upgrade as usual:

```bash
uv pip install --upgrade habitat-mapper
```

Older versions installed the CUDA libraries on every Windows and Linux computer. Upgrading leaves them in place,
where they take up about 2 GB without being used. To remove them, delete the environment folder and
[create a new environment](../get-started/install.md).

## With an NVIDIA GPU (Windows or Linux)

1. Check which CUDA version your driver supports. Run `nvidia-smi` and read `CUDA Version` in the top right corner.
2. Upgrade with the matching extra:

    === "RTX 20 series or newer, CUDA 13 or newer"

        ```bash
        uv pip install --upgrade "habitat-mapper[cuda]"
        ```

    === "CUDA 12, or an older GPU"

        Use this for any GPU older than the RTX 20 series (e.g. GTX 10 series, Quadro P series), even if
        `nvidia-smi` shows CUDA 13: CUDA 13 no longer supports those GPUs.

        ```bash
        uv pip install --upgrade "habitat-mapper[cuda12]"
        ```

        On a newer GPU, you can instead update your NVIDIA driver to R580 or newer and use `[cuda]`.

3. Run a model on one of your images with `--device cuda`:

    ```bash
    hab segment kelp-rgb input.tif output.tif --device cuda
    ```

    It should log `Running on cuda` and finish. `hab devices` alone isn't enough: it shows that CUDA is installed, but
    not that a model can run on it.

Include the extra every time you upgrade, otherwise a fresh environment won't have the CUDA libraries.

!!! warning "Don't uninstall the old CUDA packages one by one"
    After upgrading, the CUDA libraries from the old version stay installed next to the new ones. Don't remove them
    with `pip uninstall nvidia-...`: the CUDA 12 and CUDA 13 cuDNN packages install the same files, so uninstalling
    one deletes the other's files too. To reclaim the space, delete the environment folder and
    [create a new environment](../get-started/install.md) with the extra you need. Do the same to switch between
    `[cuda]` and `[cuda12]` later.

!!! warning "CUDA stopped working after an earlier update?"
    Habitat-Mapper 0.16.3 started installing CUDA 13, which needs NVIDIA driver R580 or newer and an RTX 20 series
    or newer GPU, so CUDA stopped working on older drivers and GPUs. Processing then ran on the CPU after an error like this:

    ```
    EP Error ... CUDA failure 801 ...
    Falling back to ['CPUExecutionProvider'] and retrying.
    ```

    In 0.17, install `[cuda12]` as above. On an RTX 20 series or newer GPU, you can update the driver instead.

If you skip the extra, Habitat-Mapper still uses your NVIDIA GPU through WebGPU, which is usually slower than CUDA,
and logs a message saying which extra to install.

## In a uv project

Update the dependency in `pyproject.toml`, adding the extra if you use an NVIDIA GPU:

```bash
uv add "habitat-mapper[cuda]>=0.17"
```

`uv sync` removes the CUDA libraries from environments that no longer need them.

## Check it worked

```bash
hab devices
```

The device marked `1` in the **Auto** column is the one models run on. If it's `cpu` and you have a GPU, look for a
message above the table explaining why. The [FAQ](../help/faq.md) lists the common ones.
