# Upgrade from Kelp-O-Matic

Habitat-Mapper is the new name for Kelp-O-Matic. If you have `kelp-o-matic` installed, switch cleanly to avoid
dependency conflicts and duplicate model files.

## 1. Replace the package

With your environment activated:

```bash
pip uninstall kelp-o-matic
pip install habitat-mapper
```

(If you followed our [install guide](../get-started/install.md), use `uv pip` in place of `pip`.)

## 2. Clear the old model cache

Habitat-Mapper keeps models in a new cache location. To remove the old Kelp-O-Matic models and free up disk
space, run:

```bash
hab clean
```

Type **y** when prompted. The latest models download automatically the next time you run `hab segment`.

## 3. Update your commands

The command is now `hab` instead of `kom`, and every model is run through `hab segment`:

```bash
hab segment kelp-rgb input.tif output.tif
```

Run `hab models` to see the model names, or browse the [model overview](../models/index.md).
