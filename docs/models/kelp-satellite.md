# Kelp: Satellite

Two models map kelp canopy *presence* (not species) in satellite imagery, for regional-scale mapping.

| Model               | Use for                                          |
|---------------------|--------------------------------------------------|
| `kelp-ps8b`         | PlanetScope 8-band surface reflectance imagery   |
| `kelp-skema` (beta) | Sentinel-2 Level-2A imagery (`.SAFE` folders)    |

## Outputs

Both models output:

| Value | Class                                 |
|------:|---------------------------------------|
| **0** | Background (water and other features) |
| **1** | Kelp                                  |

---

## `kelp-ps8b`

```bash
hab segment kelp-ps8b planetscope_8band.tif kelp.tif
```

### Inputs

- **Bands:** all 8 bands of a PlanetScope 8-band surface reflectance product, in their standard order.
- **Resolution:** 3 m or finer.

### Performance

Measured on the validation split. See [Performance Metrics](metrics.md).

| Class |    IoU | Precision | Recall |     F1 |
|:------|-------:|----------:|-------:|-------:|
| Kelp  | 0.8901 |    0.9367 | 0.9472 | 0.9419 |

### Architecture

SegFormer with a MiT-B3 encoder.

### Credit

!!! quote "Who to credit"
    Trained using expert annotations of Planet imagery by **Dr. Katherine Cavanaugh _et al._**, who also contributed
    model development experience. Please cite:

    Cavanaugh, K.C., Cavanaugh, K.C., Berberian, L.A. *et al.* High-resolution planet Dove data identify local drivers
    of kelp canopy persistence. *Commun Earth Environ* (2026).
    [https://doi.org/10.1038/s43247-025-03134-y](https://doi.org/10.1038/s43247-025-03134-y)

---

## `kelp-skema` (beta)

SKeMa takes a Sentinel-2 L2A `.SAFE` folder directly. Habitat-Mapper reads the bands it needs from that folder.

```bash
hab segment kelp-skema /path/to/scene.SAFE kelp.tif
```

### Inputs

- **Data:** an unzipped Sentinel-2 Level-2A `.SAFE` folder. See
  [Download Sentinel-2 Data](../guides/skema/download.md).
- **Variants:** the default revision uses only Sentinel-2 bands and works anywhere. It's the recommended starting
  point. Other revisions add substrate, bathymetry and slope layers, and only apply to coastal British Columbia.
  See [Choose a Variant](../guides/skema/variants.md).

### Workflow

1. [Download Sentinel-2 Data](../guides/skema/download.md)
2. [Choose a Variant](../guides/skema/variants.md)
3. [Post-Process](../guides/skema/post-process.md), including removing eelgrass false positives

### Architecture

U-Net with a MaxViT-Tiny encoder.

### Performance and training data

!!! note "Coming soon"
    Performance metrics and training data details for SKeMa will be added here. In the meantime, the model's
    development and evaluation are described in
    [Ghanbari *et al.* (2026)](https://doi.org/10.3390/rs18203604).

### Credit

!!! quote "Who to credit"
    Developed by **Dr. Mohsen Ghanbari _et al._** at [Spectral Lab](https://uvicspectral.com/), University of
    Victoria, and integrated into Habitat-Mapper from the original code at
    [github.com/m5ghanba/skema](https://github.com/m5ghanba/skema/). If you use `kelp-skema` in your research,
    please cite:

    Ghanbari, M., Ernst, N., Denouden, T.A., Reshitnyk, L.Y., Steffen, P., Wachmann, A., Mora-Soto, A., Loos, E.,
    Hessing-Lewis, M., Dedeluk, N., & Costa, M. Deep Learning-Based Mapping of Floating Kelp Forests from Sentinel-2
    Imagery. *Remote Sensing* 18(20), 3604 (2026). [https://doi.org/10.3390/rs18203604](https://doi.org/10.3390/rs18203604)
