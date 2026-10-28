# Install

This page sets up Habitat-Mapper using `uv`, a tool that installs Python and keeps Habitat-Mapper's
dependencies separate from the rest of your computer. You don't need to install Python yourself.

!!! info "Already use Python?"
    Install with `pip install habitat-mapper` or `uv add habitat-mapper` into your own environment. See
    [Install for Python Users](../guides/install-advanced.md).

## 1. Install `uv`

Open a terminal ([how?](terminal.md#1-opening-the-terminal)) and run the command for your operating system:

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

**Close and reopen your terminal** afterwards so it can find the `uv` command.

## 2. Create an environment

An *environment* is a folder that holds Habitat-Mapper and everything it needs. Move to the folder where you
want to keep it, then create it:

```bash
cd path/to/your/projects
uv venv habitat-env
```

!!! warning "Windows: use a local drive"
    Create the environment on your local `C:` drive (for example in Documents), **not** on a network drive.
    Network drives can cause GDAL errors during installation. Your *images* can stay on a network drive.
    This only applies to where the environment lives.

## 3. Activate the environment

--8<-- "activate-env.md"

## 4. Install Habitat-Mapper

With the environment active, run:

```bash
uv pip install habitat-mapper
```

## 5. Check it worked

```bash
hab --version
```

You should see a version number. If you see `command not found` or `'hab' is not recognized`, the environment
isn't active. Repeat [step 3](#3-activate-the-environment) and try again.

## Updating later

Activate the environment, then run:

```bash
uv pip install --upgrade habitat-mapper
```

Upgrading from the old `kelp-o-matic` package? See [Upgrade from Kelp-O-Matic](../guides/upgrade-from-kom.md).

---

[:material-arrow-right: Next: Your First Segmentation](first-segmentation.md){ .md-button .md-button--primary }
