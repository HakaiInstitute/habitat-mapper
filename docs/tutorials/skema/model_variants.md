# Selecting a Model Variant

SKeMa provides three model variants that trade off complexity against geographic applicability.
Use the `--rev` flag (short for `--revision`) to select which one to run.

## Variants at a Glance

| Revision | Input data | Applicable area | When to use |
|---|---|---|---|
| `20260414` | Sentinel-2 bands only | Anywhere | Best starting point; no auxiliary data needed |
| `20260414-full` | Bands + bathymetry + substrate | Coastal BC only | Higher accuracy in BC when auxiliary data is available |
| `20260414-ensemble` | Bands + bathymetry + substrate | Coastal BC only | Most accurate in BC; combines both approaches |

!!! warning "Geographic restriction for full and ensemble variants"
    The `full` and `ensemble` variants rely on bathymetry and substrate rasters covering
    coastal British Columbia. They are **not applicable to areas outside of BC**.
    Use the bands-only variant (`20260414`) for study areas elsewhere.

## Checking Available Revisions

To list all revisions available for a model at any time:

```bash
hab revisions kelp-skema
```

## Running a Specific Variant

Pass the revision to `--rev` along with the path to your `.SAFE` folder:

=== "Bands only (default)"

    ```bash
    hab segment \
        --model kelp-skema \
        --rev 20260414 \
        --input /path/to/scene.SAFE \
        --output kelp_output.tif
    ```

    This is also the result of omitting `--rev` entirely, since `20260414` is the latest
    revision and Habitat-Mapper defaults to `latest`.

=== "Full (BC only)"

    ```bash
    hab segment \
        --model kelp-skema \
        --rev 20260414-full \
        --input /path/to/scene.SAFE \
        --output kelp_output.tif
    ```

=== "Ensemble (BC only)"

    ```bash
    hab segment \
        --model kelp-skema \
        --rev 20260414-ensemble \
        --input /path/to/scene.SAFE \
        --output kelp_output.tif
    ```

## Next Step

Once you have your output raster, continue to [Post-Processing](post_processing.md) to
vectorize the result and optionally remove eelgrass false positives.
