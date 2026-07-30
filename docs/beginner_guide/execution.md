# Processing Images

Run segmentation inference on your geospatial imagery using the `hab segment` command.

!!! abstract "Prerequisites"
    1. **Activate Environment:** Ensure `(habitat-env)` is active in your terminal.
    2. **Input Data:** Have your GeoTIFF orthomosaics or raster data ready.

---

## 1. Select a Model

Identify the appropriate model for your sensor capabilities and target species.

```bash
hab models
```

| Model Name           | Revision | Description                                                             | Status    |
|----------------------|----------|-------------------------------------------------------------------------|-----------|
| eelgrass-rgb         | 20260402 | Eelgrass segmentation model for RGB drone imagery.                      | Available |
| kelp-ps8b            | 20260604 | Kelp segmentation model for 8-band PlanetScope imagery.                 | Available |
| kelp-rgb             | 20240722 | Kelp segmentation model for RGB drone imagery.                          | Available |
| kelp-rgbi            | 20231214 | Kelp segmentation model for 4-band RGB+NIR drone imagery.               | Available |
| kelp-skema (beta)    | 20260414 | Kelp segmentation model for Sentinel 2 L2A data.                        | Available |
| mussel-gooseneck-rgb | 20250725 | Mussel and gooseneck barnacle segmentation model for RGB drone imagery. | Available |
| mussel-rgb           | 20250711 | Mussel segmentation model for RGB drone imagery.                        | Available |

---

## 2. Command Interface

The `hab segment` command requires the model name, input path, and output destination.

### Syntax

```bash
hab segment --model <NAME> --input <PATH> --output <PATH> [OPTIONS]
```

!!! tip "For Beginners"
    You only need three things to get started:

    - `--model` or `-m` - Which model to use (e.g., `kelp-rgb`)
    - `--input` or `-i` - Path to your image file
    - `--output` or `-o` - Path where results will be saved

    Advanced options below are for optimization and special cases. You can safely ignore them for your first run.

### Parameters

| Flag | Short | Description |
| :--- | :--- | :--- |
| `--model` | `-m` | **(Required)** Model identifier (e.g., `kelp-rgb`). |
| `--input` | `-i` | **(Required)** Path to source raster (`.tif`). |
| `--output` | `-o` | **(Required)** Path for output classification raster (`.tif`). |
| `--crop-size` | `-z` | Tile size for inference window (pixels). Default: `1024`. |
| `--batch-size` | | Inference batch size. Increase for GPU acceleration. Default: `1`. |
| `--band` | `-b` | Band mapping for non-standard sensor ordering (see below). |
| `--blur` | | Kernel size for median blur post-processing. Default: `5`. |
| `--morph` | | Kernel size for morphological opening/closing. Default: `0`. |

!!! note "Full Command Reference"
    For complete documentation of all available flags and options, see the [CLI Reference](../cli.md).

---

## 3. Execution Examples

### Standard RGB Inference

Process a standard RGB orthomosaic for kelp detection.

=== "Windows (PowerShell)"

    ```powershell
    hab segment -m kelp-rgb -i "site_01.tif" -o "site_01_class.tif"
    ```

=== "MacOS / Linux"

    ```bash
    hab segment -m kelp-rgb -i "site_01.tif" -o "site_01_class.tif"
    ```

!!! success "Process"
    The CLI will tile the input raster, perform inference, stitch the results, and apply geospatial metadata to the output file.

---

## 4. Advanced Configuration

### Memory Management (OOM Errors)
For high-resolution rasters on machines with limited VRAM/RAM, reduce the tile size.

```bash
hab segment -m kelp-rgb -z 512 -i input.tif -o output.tif
```
*(Note: `crop-size` must be an even integer).*

### Band Mapping
Models expect specific input band orders (e.g., `kelp-rgbi` expects `Red, Green, Blue, NIR`). If your sensor outputs a different order (e.g., `Blue, Green, Red, NIR`), use `-b` flags to map source indices (1-based) to the expected model input slots.

**Example:**

* **Source:** Blue (1), Green (2), Red (3), NIR (4)
* **Model Expectation:** Red, Green, Blue, NIR

**Command:**
```bash
# Map Source Band 3 -> Slot 1 (Red)
# Map Source Band 2 -> Slot 2 (Green)
# Map Source Band 1 -> Slot 3 (Blue)
# Map Source Band 4 -> Slot 4 (NIR)
hab segment -m kelp-rgbi -b 3 -b 2 -b 1 -b 4 -i input.tif -o output.tif
```

---

## 5. Troubleshooting

| Error | Cause | Solution |
| :--- | :--- | :--- |
| `command not found` | Environment inactive | Activate virtual env: `source habitat-env/bin/activate` or `.\Scripts\activate`. |
| `CUDA out of memory` | VRAM limit exceeded | Reduce `--crop-size` to `512`. |
| `Permission denied` | File lock | Close the raster in GIS software (ArcGIS/QGIS) before processing. |

Something else? [Check out the FAQs](../faqs.md)

---

## Next Steps

[:material-arrow-right: Post-Processing Methodology](post_processing.md){ .md-button .md-button--primary }
