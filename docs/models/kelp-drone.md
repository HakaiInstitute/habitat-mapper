# Kelp: Drone

Two models detect canopy-forming kelp in drone and aerial imagery and identify it to species: giant kelp
(*Macrocystis pyrifera*) and bull kelp (*Nereocystis luetkeana*).

| Model       | Use for                                   |
|-------------|-------------------------------------------|
| `kelp-rgb`  | Ordinary colour (RGB) imagery             |
| `kelp-rgbi` | 4-band imagery with a near-infrared band  |

```bash
hab segment kelp-rgb input.tif output.tif
```

## Inputs

- **Bands:** `kelp-rgb` reads Red, Green, Blue. `kelp-rgbi` reads Red, Green, Blue, NIR. To use a different band
  order, see [Prepare Your Imagery](../guides/prepare-imagery.md#bands-are-in-red-green-blue-then-nir-order).
- **Resolution:** 1–10 cm per pixel; about 5 cm is ideal.
- **Data type:** 8- or 16-bit unsigned integers.

## Outputs

| Value | Class                                 |
|------:|---------------------------------------|
| **0** | Background (water and other features) |
| **1** | *Macrocystis pyrifera* (giant kelp)   |
| **2** | *Nereocystis luetkeana* (bull kelp)   |

To clean these results into polygons, follow [Post-Process Drone Kelp](../guides/post-process-kelp-drone.md).

## Performance

Measured on the validation split for the latest revision. See [Performance Metrics](metrics.md).

=== "kelp-rgb"

    | Class      |    IoU | Precision | Recall |     F1 |
    |:-----------|-------:|----------:|-------:|-------:|
    | Macro      | 0.8816 |    0.9111 | 0.9645 | 0.9371 |
    | Nereo      | 0.8972 |    0.9484 | 0.9433 | 0.9458 |
    | Background | 0.9916 |    0.9964 | 0.9951 | 0.9958 |

=== "kelp-rgbi"

    | Class      |    IoU | Precision | Recall |     F1 |
    |:-----------|-------:|----------:|-------:|-------:|
    | Macro      | 0.9670 |    0.9787 | 0.9878 | 0.9832 |
    | Nereo      | 0.9328 |    0.9568 | 0.9738 | 0.9652 |
    | Background | 0.9988 |    0.9996 | 0.9992 | 0.9994 |

## Architecture

Each model is a pair of UNet++ networks: one detects kelp presence and the other classifies species. Their outputs
are combined so that each kelp pixel is labelled with a species.

- `kelp-rgb`: UNet++ with EfficientNetV2-M encoders.
- `kelp-rgbi`: UNet++ with EfficientNet-B3 encoders and SCSE decoder attention.

## Training data

The models were trained on drone orthomosaics, collected mainly with DJI Phantom drones during the summers of
2018–2021. Kelp was labelled by hand, and the images were cut into 512 × 512 pixel tiles with 50% overlap. Tiles
with no kelp were discarded. The training pipeline is described in [How the Models Work](how-it-works.md).

??? info "Dataset summary"

    **Presence/absence**

    | Split      | Scenes |  Tiles | Pixels~kelp~ | Pixels~total~ | Area (km^2^) | Res~max~ (m) | Res~min~ (m) | Res~$\mu$~ (m) | Res~$\sigma$~ (m) |
    |:-----------|-------:|-------:|-------------:|--------------:|-------------:|-------------:|-------------:|---------------:|------------------:|
    | Train      |     20 | 829404 | 216770596954 |    3738528864 |     11680.54 |       0.1040 |       0.0230 |         0.0487 |            0.0253 |
    | Validation |      4 |  31610 |   8220459464 |     295310084 |       253.79 |       0.0420 |       0.0230 |         0.0296 |            0.0085 |
    | Test       |      6 |  92608 |  24093604354 |     639694012 |       819.45 |       0.0680 |       0.0230 |         0.0385 |            0.0161 |
    | *Sum*      |     30 | 953622 | 249084660772 |    4673532960 |    12,753.78 |              |              |                |                   |

    **Species**

    | Split      | Scenes |  Tiles | Pixels~macro~ | Pixels~nereo~ | Pixels~total~ | Area (km^2^) | Res~max~ (m) | Res~min~ (m) | Res~$\mu$~ (m) | Res~$\sigma$~ (m) |
    |:-----------|-------:|-------:|--------------:|--------------:|--------------:|-------------:|-------------:|-------------:|---------------:|------------------:|
    | Train      |     17 | 336740 |     605462674 |    1158650042 |   88008624978 |      4034.11 |       0.1040 |       0.0230 |         0.0488 |            0.0266 |
    | Validation |      4 |  15805 |     127410722 |      20244320 |    4110229732 |       123.91 |       0.0420 |       0.0230 |         0.0296 |            0.0085 |
    | Test       |      6 |  46304 |     143277498 |     176569508 |   12046802177 |       409.72 |       0.0680 |       0.0230 |         0.0385 |            0.0161 |
    | *Sum*      |     27 | 398849 |     876150894 |    1355463870 |  104165656887 |      4567.75 |              |              |                |                   |

## Credit

!!! quote "Who to credit"
    Developed by the [Hakai Institute](https://hakai.org). **The Nature Conservancy of California** provided drone
    imagery used to train both models. See [Credits](../credits.md) and [How to Cite](../cite.md).
