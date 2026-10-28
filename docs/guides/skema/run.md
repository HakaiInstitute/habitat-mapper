# Run SKeMa

Pass the unzipped `.SAFE` folder to `hab segment` as the input, with `kelp-skema` as the model:

```bash
hab segment kelp-skema /path/to/scene.SAFE kelp_output.tif
```

Habitat-Mapper reads the Sentinel-2 bands it needs straight from the `.SAFE` folder, so you don't need to extract
or stack bands yourself. The output is a GeoTIFF on the Sentinel-2 10 m grid, in which `1` is kelp and `0` is
background.

This runs the latest SKeMa revision. To reproduce results made with an earlier revision, pass it with `--rev`, for
example `--rev 20260414`. See [revisions](../../models/kelp-satellite.md#revisions).

!!! note "Auxiliary-data models"
    Habitat-Mapper runs the Sentinel-2-only SKeMa model. The bathymetry and substrate models described in the SKeMa
    paper are available in the original [SKeMa tool](https://github.com/m5ghanba/skema). See
    [Kelp: Satellite](../../models/kelp-satellite.md#auxiliary-data-models).

!!! tip "Use quotes if the path contains spaces"
    `hab segment kelp-skema "C:\My Data\scene.SAFE" kelp_output.tif`

## Several scenes

To map several scenes, for example a time series or neighbouring tiles, loop over the `.SAFE` folders:

=== "Windows (PowerShell)"

    ```powershell
    Get-ChildItem -Directory *.SAFE | ForEach-Object {
        hab segment kelp-skema $_.FullName "$($_.BaseName)_kelp.tif" --quiet
    }
    ```

=== "macOS / Linux"

    ```bash
    for scene in ./*.SAFE; do
        hab segment kelp-skema "$scene" "$(basename "$scene" .SAFE)_kelp.tif" --quiet
    done
    ```

See [Process Many Images](../batch-processing.md) for more on batch processing.

## Next Step

Once you have your output raster, continue to [Post-Process](post-process.md) to vectorize the result and optionally
remove eelgrass false positives.
