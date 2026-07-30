# Quick Reference

This page provides a condensed reference for experienced users who need quick access to commands and common workflows.

## Installation

```bash
# Using uv (recommended)
uv pip install habitat-mapper

# Using standard pip
pip install habitat-mapper

# Update to latest version
uv pip install --upgrade habitat-mapper
# or
pip install --upgrade habitat-mapper
```

## Basic Commands

!!! tip "Built-in Help"
    All commands have detailed help documentation accessible via the `--help` flag:

    - `hab --help` - List all available commands
    - `hab segment --help` - View all segment command options
    - `hab models --help` - View models command options
    - `hab revisions --help` - View revisions command options

    This is the fastest way to look up command syntax and available flags.

| Task                  | Command                                                 |
|-----------------------|---------------------------------------------------------|
| List available models | `hab models`                                            |
| List model revisions  | `hab revisions <model-name>`                            |
| Segment an image      | `hab segment -m <model> -i <input.tif> -o <output.tif>` |
| Check version         | `hab --version`                                         |
| Clear model cache     | `hab clean`                                             |

## Available Models

| Model Name             | Input Type               | Output Classes                           |
|------------------------|--------------------------|------------------------------------------|
| `eelgrass-rgb`         | RGB (3-band)             | Background, Eelgrass (*Zostera marina*)  |
| `kelp-ps8b`            | PlanetScope 8-band       | Background, Kelp (presence only)         |
| `kelp-rgb`             | RGB (3-band)             | Background, *Macrocystis*, *Nereocystis* |
| `kelp-rgbi`            | RGB+NIR (4-band)         | Background, *Macrocystis*, *Nereocystis* |
| `kelp-skema` (beta)    | Sentinel-2 L2A (`.SAFE`) | Background, Kelp (presence only)         |
| `mussel-gooseneck-rgb` | RGB (3-band)             | Background, Mussels, Gooseneck Barnacles |
| `mussel-rgb`           | RGB (3-band)             | Background, Mussels                      |

See [Model Output Reference](model_outputs.md) for detailed class definitions.

## Common Workflows

### Standard RGB Kelp Detection

```bash
hab segment -m kelp-rgb -i drone_image.tif -o kelp_mask.tif
```

### RGB+NIR Kelp Detection with Band Reordering

If your image has bands in Blue, Green, Red, NIR order (instead of RGB+NIR):

```bash
hab segment -m kelp-rgbi -i multispectral.tif -o kelp_mask.tif -b 3 -b 2 -b 1 -b 4
```

### High-Resolution Processing (Reduce Artifacts)

Use the largest crop size your system can handle:

```bash
hab segment -m kelp-rgb -i input.tif -o output.tif --crop-size 3200
```

If you get memory errors, reduce progressively: `3200` → `2048` → `1024` → `512`

### GPU Acceleration

Increase batch size if you have a CUDA-compatible GPU:

```bash
hab segment -m kelp-rgb -i input.tif -o output.tif --batch-size 4
```

### Post-Processing Adjustments

Apply median blur and morphological operations:

```bash
hab segment -m kelp-rgb -i input.tif -o output.tif --blur 7 --morph 3
```

## Python API Quick Start

```python
from habitat_mapper import model_registry
from pathlib import Path

# Load model
model = model_registry['kelp-rgb']

# Process image
model.process(
    img_path=Path("input.tif"),
    output_path=Path("output.tif"),
    crop_size=1024,
    batch_size=1
)

# With band reordering
model.process(
    img_path=Path("input.tif"),
    output_path=Path("output.tif"),
    band_order=[3, 2, 1, 4]  # BGR+NIR → RGB+NIR
)
```

See [Python API](python_lib.md) for complete documentation.

## Input Requirements Summary

- **Data type:** `uint8` (0-255) or `uint16` (0-65535)
- **Band order:** Red, Green, Blue, [NIR] (use `-b` flags to reorder if needed)
- **Format:** GeoTIFF (`.tif`) with coordinate reference system
- **Recommended:** Tiled GeoTIFF with defined nodata value

See [Input Requirements](expectations.md) for details.

## Common Flags

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--model` | `-m` | Model name (required) | - |
| `--input` | `-i` | Input raster path (required) | - |
| `--output` | `-o` | Output raster path (required) | - |
| `--crop-size` | `-z` | Tile size in pixels (must be even) | `1024` |
| `--batch-size` | | Number of tiles processed at once | `1` |
| `--band-order` | `-b` | Reorder input bands (repeat for each band) | `None` |
| `--blur-kernel` | `--blur` | Median blur kernel size (must be odd) | `5` |
| `--morph-kernel` | `--morph` | Morphological operation kernel size | `0` |
| `--revision` | `--rev` | Specific model revision | `latest` |

See [CLI Reference](cli.md) for complete flag documentation.

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| `command not found: hab` | Activate virtual environment: `source habitat-env/bin/activate` (Mac/Linux) or `.\habitat-env\Scripts\activate` (Windows) |
| `CUDA out of memory` | Reduce `--crop-size` or `--batch-size` |
| Output is all black | Use "unique values" classification in GIS, not continuous color ramp |
| Permission denied | Close the file in GIS software before processing |
| GDAL errors (Windows) | Ensure virtual environment is on local `C:` drive, not network drive |

See [FAQs](faqs.md) for more troubleshooting help.

## Additional Resources

- **[Beginner's Guide](beginner_guide/index.md)** - Step-by-step tutorial for new users
- **[CLI Reference](cli.md)** - Complete command documentation
- **[Python API](python_lib.md)** - Library usage for scripting
- **[Model Architecture](training_details.md)** - Technical details about models
- **[Glossary](glossary.md)** - Definitions of geospatial terms
