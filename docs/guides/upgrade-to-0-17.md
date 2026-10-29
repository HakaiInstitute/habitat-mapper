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

CUDA needs an RTX 20 series or newer GPU and NVIDIA driver R580 or newer. Older GPUs, such as the GTX 10 or 900 series
and Quadro P or M series, aren't supported by CUDA 13. Habitat-Mapper uses them through WebGPU instead, so follow
[Without an NVIDIA GPU](#without-an-nvidia-gpu).

On a supported GPU, install 0.17 into a **new environment** rather than upgrading in place. Upgrading in place leaves
the old CUDA libraries next to the new ones, and the mix can stop CUDA from working.

1. Check your driver. Run `nvidia-smi` and read `CUDA Version` in the top right corner. If it's below 13.0, update
   the NVIDIA driver to R580 or newer, or skip CUDA and follow [Without an NVIDIA GPU](#without-an-nvidia-gpu).
2. Delete your environment folder and [create a new environment](../get-started/install.md) (steps 2 and 3 of the
   install guide). Then, with the new environment active, install with the CUDA extra:

    ```bash
    uv pip install "habitat-mapper[cuda]"
    ```

3. Run a model on one of your images with `--device cuda`:

    ```bash
    hab segment kelp-rgb input.tif output.tif --device cuda
    ```

    The last `Running on` line should say `cuda`, with no `The model failed to run on cuda` warning. If CUDA can't
    run the model, Habitat-Mapper moves on to another device and the run still finishes, so check the messages
    rather than just that it finished. `hab devices` alone isn't enough either: it shows that CUDA is installed, but
    not that a model can run on it. If CUDA fails, run the command again with the `DEBUG` environment variable set
    to `1` to see why, and include that output if you [report the problem](https://github.com/HakaiInstitute/habitat-mapper/issues).

Include the extra every time you upgrade later, otherwise a fresh environment won't have the CUDA libraries. Don't
remove CUDA packages with `pip uninstall nvidia-...`: the CUDA 12 and CUDA 13 cuDNN packages install the same files,
so uninstalling one deletes the other's files too.

!!! warning "CUDA stopped working after an earlier update?"
    Habitat-Mapper 0.16.3 started installing CUDA 13, which needs NVIDIA driver R580 or newer and an RTX 20 series
    or newer GPU, so CUDA stopped working on older drivers and GPUs. With an older driver, processing ran on the CPU
    after an error like this:

    ```
    EP Error ... CUDA failure 801 ...
    Falling back to ['CPUExecutionProvider'] and retrying.
    ```

    With an older GPU, it stopped without writing an output, after an error like this:

    ```
    Program interrupted due to Fail: ... CUBLAS failure 8: the function requires an architectural feature absent
    from the device ...
    ```

    In 0.17, older GPUs run on WebGPU instead. On an RTX 20 series or newer GPU, update the driver to R580 or newer
    and install `[cuda]` as above.

If you skip the extra, Habitat-Mapper still uses your NVIDIA GPU through WebGPU, which is usually slower than CUDA,
and logs a message saying what to install.

## In a uv project

Update the dependency in `pyproject.toml`. Add the CUDA extra only if you have an RTX 20 series or newer NVIDIA GPU:

```bash
uv add "habitat-mapper>=0.17"           # Any other computer
uv add "habitat-mapper[cuda]>=0.17"     # RTX 20 series or newer, driver R580 or newer
```

`uv sync` removes CUDA libraries that are no longer needed. If you're upgrading from 0.16.2 or older, which installed
CUDA 12, delete the project's `.venv` folder and run `uv sync`, for the same reason as above.

## Check it worked

Run a model with `--device cuda` (NVIDIA) or without `--device` (other GPUs), as in step 3 above, and check which
device the last `Running on` line names. For an overview of what Habitat-Mapper found, run:

```bash
hab devices
```

The device marked `1` in the **Auto** column is the one `--device auto` tries first. If it's `cpu` and you have a GPU,
look for a message above the table explaining why. The [FAQ](../help/faq.md) lists the common ones.
