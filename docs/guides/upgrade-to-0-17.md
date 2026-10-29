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

Install 0.17 into a **new environment** rather than upgrading in place. Upgrading in place leaves the old CUDA
libraries next to the new ones, and the mix can stop CUDA from working.

1. Check which CUDA version your driver supports. Run `nvidia-smi` and read `CUDA Version` in the top right corner.
2. Delete your environment folder and [create a new environment](../get-started/install.md) (steps 2 and 3 of the
   install guide). Then, with the new environment active, install with the matching extra:

    === "RTX 20 series or newer, CUDA 13 or newer"

        ```bash
        uv pip install "habitat-mapper[cuda]"
        ```

    === "CUDA 12, or an older GPU"

        Use this for any GPU older than the RTX 20 series (e.g. GTX 10 series, Quadro P series), even if
        `nvidia-smi` shows CUDA 13: CUDA 13 no longer supports those GPUs.

        ```bash
        uv pip install "habitat-mapper[cuda12]"
        ```

        On a newer GPU, you can instead update your NVIDIA driver to R580 or newer and use `[cuda]`.

3. Run a model on one of your images with `--device cuda`:

    ```bash
    hab segment kelp-rgb input.tif output.tif --device cuda
    ```

    The last `Running on` line should say `cuda`, with no `The model failed to run on cuda` warning. If CUDA can't
    run the model, Habitat-Mapper moves on to another device and the run still finishes, so check the messages
    rather than just that it finished. `hab devices` alone isn't enough either: it shows that CUDA is installed, but
    not that a model can run on it.

Include the extra every time you upgrade later, otherwise a fresh environment won't have the CUDA libraries. To switch
between `[cuda]` and `[cuda12]`, create a new environment again. Don't remove CUDA packages with
`pip uninstall nvidia-...`: the CUDA 12 and CUDA 13 cuDNN packages install the same files, so uninstalling one deletes
the other's files too.

!!! warning "CUDA stopped working after an earlier update?"
    Habitat-Mapper 0.16.3 started installing CUDA 13, which needs NVIDIA driver R580 or newer and an RTX 20 series
    or newer GPU, so CUDA stopped working on older drivers and GPUs. Processing then ran on the CPU after an error
    like this:

    ```
    EP Error ... CUDA failure 801 ...
    Falling back to ['CPUExecutionProvider'] and retrying.
    ```

    In 0.17, install `[cuda12]` as above. On an RTX 20 series or newer GPU, you can update the driver instead.

If you skip the extra, Habitat-Mapper still uses your NVIDIA GPU through WebGPU, which is usually slower than CUDA,
and logs a message saying which extra to install.

## In a uv project

Update the dependency in `pyproject.toml`, adding the extra that matches your NVIDIA GPU, if you have one:

```bash
uv add "habitat-mapper[cuda]>=0.17"     # RTX 20 series or newer, CUDA 13
uv add "habitat-mapper[cuda12]>=0.17"   # CUDA 12, or an older GPU
```

`uv sync` removes CUDA libraries that are no longer needed. If you switch between CUDA 12 and CUDA 13 (including
from 0.16, which installed CUDA 13), delete the project's `.venv` folder and run `uv sync`, for the same reason as
above.

## Check it worked

Run a model with `--device cuda` (NVIDIA) or without `--device` (other GPUs), as in step 3 above, and check which
device the last `Running on` line names. For an overview of what Habitat-Mapper found, run:

```bash
hab devices
```

The device marked `1` in the **Auto** column is the one `--device auto` tries first. If it's `cpu` and you have a GPU,
look for a message above the table explaining why. The [FAQ](../help/faq.md) lists the common ones.
