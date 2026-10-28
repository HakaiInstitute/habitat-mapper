# Choose a SKeMa Variant

SKeMa provides four model variants. They differ in whether they add auxiliary environmental layers to the
Sentinel-2 bands, and so where they can be used. Use the `--rev` flag (short for `--revision`) to select one.

## Variants at a Glance

| Revision | Input data | Applicable area |
|---|---|---|
| `20260414` (default) | Sentinel-2 bands | Anywhere |
| `20260414-full` | Bands + substrate + bathymetry + slope | Coastal BC only |
| `20260414-ensemble` | Bands + substrate + bathymetry + slope | Coastal BC only |
| `20260414-full-bops` | Bands + BoPs substrate (10 m) + bathymetry + slope | Coastal BC only |

!!! tip "Start with the default"
    Use the bands-only default (`20260414`) unless you have a reason not to. In the published SKeMa evaluation
    ([Ghanbari *et al.*, 2026](https://doi.org/10.3390/rs18203604)), adding substrate and bathymetry layers **did
    not improve** kelp detection over Sentinel-2 data alone: IoU dropped from 0.60 to 0.58. The authors attribute
    this to inaccuracies in the available substrate and bathymetry data. The Habitat-Mapper revisions are not
    identical to the configurations tested in the paper, so treat the BC-only variants as alternatives to
    compare, not as upgrades.

!!! warning "Geographic restriction"
    The `full`, `ensemble` and `full-bops` variants rely on auxiliary rasters that only cover coastal British
    Columbia. They are **not applicable to areas outside of BC**. Habitat-Mapper downloads these rasters
    automatically the first time you run one of these variants.

??? info "Auxiliary data sources"
    - **Substrate:** Gregr, E.J., Haggarty, D.R., Davies, S.C., Fields, C., & Lessard, J. Comprehensive marine
      substrate classification applied to Canada's Pacific shelf. *PLoS ONE* 16, e0259156 (2021).
      [https://doi.org/10.1371/journal.pone.0259156](https://doi.org/10.1371/journal.pone.0259156)
    - **Bathymetry:** Davies, S.C., Gregr, E.J., Lessard, J., Bartier, P., & Wills, P. *Coastal Digital Elevation
      Models Integrating Ocean Bathymetry and Land Topography for Marine Ecological Analyses in Pacific Canadian
      Waters.* Fisheries and Oceans Canada (2019).

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

=== "Full BoPs (BC only)"

    ```bash
    hab segment \
        --model kelp-skema \
        --rev 20260414-full-bops \
        --input /path/to/scene.SAFE \
        --output kelp_output.tif
    ```

## Next Step

Once you have your output raster, continue to [Post-Processing](post-process.md) to
vectorize the result and optionally remove eelgrass false positives.
