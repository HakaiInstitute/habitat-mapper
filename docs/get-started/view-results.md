# View Results in GIS

The output is a GeoTIFF in which each pixel holds a small whole number, or **class value**: `0` for background,
plus `1`, `2`, … for each habitat the model detects. Because these numbers are so small, GIS software draws the
whole image almost black until you tell it to colour each value separately.

## Show each class in its own colour

=== "ArcGIS Pro"

    1. Add the output `.tif` to your map.
    2. Right-click the layer and choose **Symbology**.
    3. Set **Primary symbology** to **Unique Values** with **Field** set to `Value`.
    4. Optionally, make `0` (background) transparent so your original image shows through.

=== "QGIS"

    1. Drag the output `.tif` into the **Layers** panel.
    2. Right-click the layer and choose **Properties → Symbology**.
    3. Set **Render type** to **Paletted/Unique values** and click **Classify**.
    4. Optionally, untick or make transparent value `0` (background).

!!! warning "Don't use a stretched or continuous colour ramp"
    A continuous ramp blends the class values together, so the map stays black or grey. Always use unique
    values (ArcGIS) or paletted values (QGIS).

## What the values mean

Class values depend on the model you ran. For example, `kelp-rgb` uses `1` for giant kelp (*Macrocystis*) and
`2` for bull kelp (*Nereocystis*). The output classes for each model are listed on its page in
[Models](../models/index.md).

## The output doesn't line up with my image

Check that your input image has a coordinate reference system (CRS) assigned. Habitat-Mapper copies the input's
location information to the output, so an input without a CRS produces an output that can't be placed on the map.

## Next steps

You've completed the Get Started guide. From here:

- **Turn results into clean polygons.** See [Post-Process Drone Kelp](../guides/post-process-kelp-drone.md).
- **Using satellite imagery?** See [Satellite Kelp with SKeMa](../guides/skema/download.md).
- **Using this in research?** Please [cite Habitat-Mapper](../cite.md).
