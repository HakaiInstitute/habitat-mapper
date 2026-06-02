# Post-Processing SKeMa Output

SKeMa produces a binary GeoTIFF where `1` = kelp and `0` = background. This guide covers
converting that raster to polygons and optionally removing false positives caused by eelgrass
beds.

## Step 1: Vectorize the Raster

=== "ArcGIS Pro"

    1. Load the output `.tif` in ArcGIS Pro
    2. Open the **Geoprocessing** panel and run **Raster to Polygon**:
        - **Input raster**: your kelp output `.tif`
        - **Field**: `Value`
        - Uncheck **Simplify polygons** for maximum accuracy
    3. Delete polygons where `gridcode = 0` (background class) from the attribute table

=== "QGIS"

    1. Load the output `.tif` in QGIS
    2. From the menu, go to **Raster → Conversion → Polygonize (Raster to Vector)**:
        - **Input layer**: your kelp output `.tif`
        - **Name of the field to create**: `value`
    3. Delete features where `value = 0` (background class) using **Select by Expression**,
       then **Delete Selected Features**

## Step 2: Remove Small Polygons

Isolated single-pixel predictions are almost always noise. Delete polygons with an area below
your minimum mapping unit (a threshold of **100 m²** works well for 10 m Sentinel-2 data).

=== "ArcGIS Pro"

    Use **Select by Attributes** to select features where `Shape_Area < 100`, then delete them.

=== "QGIS"

    Use **Select by Expression** with `$area < 100`, then **Delete Selected Features**.

## Step 3: Optional — Remove Eelgrass False Positives

Kelp predictions overlapping eelgrass (*Zostera marina*) beds are typically false positives.
Eelgrass occupies similar shallow depths to kelp, and the model can confuse dense eelgrass
patches when viewed from satellite.

The [BC Marine Conservation Analysis (BCMCA)](https://bcmca.ca/) provides an eelgrass habitat
polygon dataset for British Columbia. Download the eelgrass layer from
[DataBC](https://catalogue.data.gov.bc.ca/) (search "BCMCA eelgrass") and use it to erase
overlapping kelp predictions.

=== "ArcGIS Pro"

    1. Load your kelp polygon layer and the BCMCA eelgrass polygon layer into the same map
    2. Open the **Geoprocessing** panel and run the **Erase** tool:
        - **Input Features**: your kelp polygons
        - **Erase Features**: BCMCA eelgrass polygons
        - **Output Feature Class**: `kelp_no_eelgrass`
    3. Inspect the output to confirm eelgrass areas have been removed

=== "QGIS"

    1. Load your kelp polygon layer and the BCMCA eelgrass polygon layer
    2. Go to **Vector → Geoprocessing Tools → Difference**:
        - **Input layer**: your kelp polygons
        - **Overlay layer**: BCMCA eelgrass polygons
        - **Difference**: save as `kelp_no_eelgrass.shp`
    3. Inspect the output to confirm eelgrass areas have been removed

## Step 4: Export

Export the final layer as a shapefile or GDB feature class with an appropriate name. That's it!
