# Prepare Your Imagery

Unless a [model's page](../models/index.md) says otherwise, Habitat-Mapper expects input imagery to meet the
requirements below. Imagery that doesn't meet them usually still runs, but gives poor or strange results.

## Requirements

### Pixels are 8- or 16-bit unsigned integers

GIS software calls this `uint8` / `Byte` (values 0–255) or `uint16` (values 0–65535). You can check this in the layer
properties in ArcGIS Pro or QGIS, or by running `gdalinfo your_image.tif`. Floating-point or signed imagery will give
unreliable results.

### Bands are in Red, Green, Blue (then NIR) order

RGB models read the first three bands as red, green and blue. RGB+NIR models also read the fourth band as
near-infrared. Any extra bands are ignored.

If your bands are in a different order, tell Habitat-Mapper which band goes where with one `-b` flag per band,
numbered from 1. For example, for an image stored as Blue, Green, Red, NIR:

```bash
hab segment kelp-rgbi input.tif output.tif -b 3 -b 2 -b 1 -b 4
```

That reads band 3 as red, band 2 as green, band 1 as blue and band 4 as NIR. In Python, pass
`band_order=[3, 2, 1, 4]` instead.

!!! warning "Don't substitute bands"
    The models learned the relationships between these specific bands. Feeding in a different band, such as
    red-edge in place of red, will not work.

### Resolution matches what the model was trained on

Drone models work best on imagery with a resolution of **1–10 cm per pixel**. About 5 cm is ideal. Satellite models
have their own requirements, listed on each [model page](../models/kelp-satellite.md).

### The image has a coordinate reference system

Without a CRS the output can't be placed on a map.

## Recommendations

### Set a *nodata* value

Pixels marked as *nodata* are skipped, which speeds up processing. Black borders around drone orthomosaics usually
use `0`; white borders use `255`. If no nodata value is set, any area that is completely black or completely white
is labelled as background.

### Use a tiled GeoTIFF

Habitat-Mapper reads the image in small square windows. A *tiled* GeoTIFF stores the image in matching chunks, which
makes reading much faster. If tiling would help your image, Habitat-Mapper prints a warning when it runs. To
convert an image:

=== "ArcGIS Pro"

    1. Open the **Copy Raster** tool.
    2. Set **Compression Type**, e.g. LZW.
    3. In **Environments**, enable **Tiled** under *Raster Storage*.

=== "QGIS"

    1. Go to **Raster → Conversion → Translate (Convert Format)**.
    2. Under **Advanced Parameters**, set the profile to **High compression**, or add the creation option
       `TILED=YES`.

=== "GDAL"

    ```bash
    gdal_translate -co TILED=YES -co COMPRESS=LZW input.tif output_tiled.tif
    ```
