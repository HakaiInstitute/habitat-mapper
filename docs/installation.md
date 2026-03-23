# Installation

Habitat-Mapper is available for Python versions 3.10 through 3.13.

Please select the installation method that best matches your experience level.

=== "Beginner Guide (Recommended)"

    !!! tip "New to the Command Line?"
        Don't worry! Check the **[Beginner's Guide](beginner_guide/index.md)** that walks you through setting up your
        terminal and then come back here.

    We recommend using `uv` to manage your environment. It is faster than standard Python tools and simplifies the
    setup process across Windows, macOS, and Linux.

    ### 1. Install `uv`

    If you haven't installed `uv` yet, run the command for your operating system:

    === "Windows"
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    === "MacOS/Linux"
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    *Restart your terminal after installation to ensure the `uv` command is recognized.*

    ### 2. Create a Virtual Environment

    Navigate to your project folder and create a clean environment.

    !!! warning "Storage Location (Windows Users)"
        Create your environment on a local drive (e.g., `C:\Users\YourName\habitat-mapper`), **not** on a network drive
        (drives mapped to shared folders or servers).

        **Why?** Network drives can cause GDAL library errors during installation. Don't worry if you're unsure what a
        network drive is—if you're working in your normal Documents or Desktop folder on `C:`, you're fine.

        Once installed, you can navigate to wherever your data is stored (including network drives) to process images.

    ```bash
    # 1. Go to your preferred folder
    cd path/to/your/projects

    # 2. Create the environment named 'habitat-env'
    uv venv habitat-env
    ```

    ### 3. Activate the Environment

    You must activate the environment every time you open a new terminal to work on this project.

    === "Windows"
        !!! warning "Fixing Permission Errors"
            The first time you run the activation command below, you might see an error stating:
            *`cannot be loaded because running scripts is disabled on this system.`*

            To fix this, run the following command once:
            ```powershell
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
            ```

        **Activate the environment:**
        ```powershell
        .\habitat-env\Scripts\activate
        ```

    === "MacOS/Linux"
        ```bash
        source ./habitat-env/bin/activate
        ```

    *Your terminal prompt should now show `(habitat-env)`*

    ### 4. Install Habitat-Mapper

    With the environment active, install the package:

    ```bash
    uv pip install habitat-mapper
    ```

=== "Standard Pip (Advanced)"

    We strongly recommend installing habitat-mapper in a virtual environment to prevent compatability issues with
    system Python packages. Installation of Habitat-Mapper can be installed from PyPI using `pip`.

    ```bash
    pip install habitat-mapper
    ```

---

## Verify Installation

To verify that Habitat-Mapper was installed correctly and to check the installed version, run:

```bash
hab --version
```

!!! warning "Troubleshooting: 'Command Not Found'"
    If you see an error like `command not found` or `'hab' is not recognized as an internal or external command`:

    **The `hab` command is only available when your virtual environment is activated.**

    **Steps to fix:**

    1. Check your terminal prompt. Does it show `(habitat-env)` at the beginning?
    2. If **not**, the environment isn't active. Run the **Activation command** from Step 3 above.
    3. Try `hab --version` again.

    You'll need to activate the environment every time you open a new terminal window.

## Updating

We follow [semantic versioning](https://semver.org/). To update to the latest version:

=== "Using uv"
    `uv pip install --upgrade habitat-mapper`

=== "Standard Pip"
    `pip install --upgrade habitat-mapper `

!!! note "For existing Kelp-O-Matic users"

    If you have previously installed `kelp-o-matic`, we strongly recommend a clean transition to avoid dependency
    conflicts and duplicate files.

    **1. Uninstall the legacy package**

    Remove the old package before installing `habitat-mapper`:

    ```bash
    pip uninstall kelp-o-matic
    pip install habitat-mapper
    ```

    **2. Clean the model cache**

    By default, `habitat-mapper` uses a new cache location. To avoid storing two copies of large model files (one for
    the old tool, one for the new), run the clean command:

    ```bash
    hab clean
    ```

    Select **"y"** when prompted. This will remove legacy Kelp-O-Matic models. `habitat-mapper` will automatically
    re-download the latest optimized models when you run your first `segment` command.

## Next Steps

* **Beginners:** Head to [Processing Images](beginner_guide/execution.md) to run your first segmentation.
* **Advanced:** Check the [CLI Reference](cli.md) for available command flags.
