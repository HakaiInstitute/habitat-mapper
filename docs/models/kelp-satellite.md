# Kelp: Satellite

Two models map kelp canopy *presence* (not species) in satellite imagery, for regional-scale mapping.

| Model               | Use for                                          |
|---------------------|--------------------------------------------------|
| `kelp-ps8b`         | PlanetScope 8-band surface reflectance imagery   |
| `kelp-skema`         | Sentinel-2 Level-2A imagery (`.SAFE` folders)    |

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

## `kelp-skema`

SKeMa takes a Sentinel-2 L2A `.SAFE` folder directly. Habitat-Mapper reads the bands it needs from that folder.

```bash
hab segment kelp-skema /path/to/scene.SAFE kelp.tif
```

### Inputs

- **Data:** an unzipped Sentinel-2 Level-2A `.SAFE` folder. See
  [Download Sentinel-2 Data](../guides/skema/download.md).
- **Bands:** Blue (B2), Green (B3), Red (B4), NIR (B8) and Red-edge 1 (B5), read automatically from the `.SAFE`
  folder. The model computes five spectral indices from these bands itself (NDVI, NDWI, GNDVI, CIG and NDVIre), so
  no other input data is needed.

### Revisions

| Revision             | Notes                                                                                                   |
|----------------------|---------------------------------------------------------------------------------------------------------|
| `20261014` (default) | SKeMa v0.4.0: the final model evaluated in [Ghanbari *et al.* (2026)](https://doi.org/10.3390/rs18203604). |
| `20260414`           | Earlier weights. Use `--rev 20260414` to reproduce results made with this revision.                     |

### Auxiliary-data models

[Ghanbari *et al.* (2026)](https://doi.org/10.3390/rs18203604) also evaluated SKeMa models that add bathymetry and
substrate layers for coastal British Columbia. Habitat-Mapper only includes the Sentinel-2-only model, which performed
best in that evaluation. To experiment with the auxiliary-data models, use the original SKeMa tool at
[github.com/m5ghanba/skema](https://github.com/m5ghanba/skema).

### Workflow

1. [Download Sentinel-2 Data](../guides/skema/download.md)
2. [Run SKeMa](../guides/skema/run.md)
3. [Post-Process](../guides/skema/post-process.md), including removing eelgrass false positives

### Architecture

U-Net with a MaxViT-Tiny encoder.

### Performance

These results are from [Ghanbari *et al.* (2026)](https://doi.org/10.3390/rs18203604) and apply to the default
`20261014` weights. They come from *scene-level cross-validation*: the model is tested on whole Sentinel-2 scenes it
never saw in training. This is a stricter test than the validation splits reported for the other models, so the
numbers aren't directly comparable. See [Performance Metrics](metrics.md).

| Evaluation                        |    IoU | Precision | Recall |     F1 |
|:----------------------------------|-------:|----------:|-------:|-------:|
| Leave-one-scene-out (10 scenes)   | 0.6024 |    0.7261 | 0.7952 | 0.7491 |
| Three-fold (3 scenes per fold)    | 0.5974 |    0.7126 | 0.8085 | 0.7466 |

Accuracy varies with the size of the kelp beds. Across individual scenes, IoU ranged from 0.45, in a scene with small
and sparse beds, to 0.70, in a scene with large, contiguous beds. Small beds are hard to resolve at Sentinel-2's 10 m
resolution.

### Training data

The model was trained on 10 hand-annotated Sentinel-2 L2A scenes from coastal British Columbia. The annotated scenes
are available on [Zenodo](https://doi.org/10.5281/zenodo.20106069), and the pre-extracted training tiles are on
[Hugging Face](https://huggingface.co/datasets/m5ghanba/skema-kelp-tiles).

### Credit

!!! quote "Who to credit"
    Developed by **Dr. Mohsen Ghanbari _et al._** at [Spectral Lab](https://uvicspectral.com/), University of
    Victoria, and integrated into Habitat-Mapper from the original code at
    [github.com/m5ghanba/skema](https://github.com/m5ghanba/skema/). If you use `kelp-skema` in your research,
    please cite:

    Ghanbari, M., Ernst, N., Denouden, T.A., Reshitnyk, L.Y., Steffen, P., Wachmann, A., Mora-Soto, A., Loos, E.,
    Hessing-Lewis, M., Dedeluk, N., & Costa, M. Deep Learning-Based Mapping of Floating Kelp Forests from Sentinel-2
    Imagery. *Remote Sensing* 18(20), 3604 (2026). [https://doi.org/10.3390/rs18203604](https://doi.org/10.3390/rs18203604)
