# Mussels & Barnacles

Two models map intertidal invertebrates in high-resolution RGB drone imagery.

| Model                  | Detects                                      |
|------------------------|----------------------------------------------|
| `mussel-rgb`           | Mussels                                      |
| `mussel-gooseneck-rgb` | Mussels and gooseneck barnacles, separately  |

```bash
hab segment mussel-gooseneck-rgb input.tif output.tif
```

## Inputs

- **Bands:** Red, Green, Blue.
- **Resolution:** very high. The training imagery ranged from about 2 mm to 3 cm per pixel, most of it around 4–8 mm.
- **Data type:** 8- or 16-bit unsigned integers.

## Outputs

=== "mussel-rgb"

    | Value | Class                                            |
    |------:|--------------------------------------------------|
    | **0** | Background (water, rock, other substrates)       |
    | **1** | Mussels (typically *Mytilus* spp.)               |

=== "mussel-gooseneck-rgb"

    | Value | Class                                            |
    |------:|--------------------------------------------------|
    | **0** | Background (water, rock, other substrates)       |
    | **1** | Mussels (typically *Mytilus* spp.)               |
    | **2** | Gooseneck barnacles (*Pollicipes* spp.)          |

## Performance

Measured on the validation split. See [Performance Metrics](metrics.md).

=== "mussel-rgb"

    | Class   |    IoU | Precision | Recall |     F1 |
    |:--------|-------:|----------:|-------:|-------:|
    | Mussels | 0.8869 |    0.9343 | 0.9459 | 0.9401 |

=== "mussel-gooseneck-rgb"

    | Class        |    IoU | Precision | Recall |     F1 |
    |:-------------|-------:|----------:|-------:|-------:|
    | Mussels      | 0.8288 |    0.9119 | 0.9010 | 0.9064 |
    | Gooseneck B. | 0.7801 |    0.8652 | 0.8880 | 0.8765 |
    | Background   | 0.9831 |    0.9919 | 0.9911 | 0.9915 |

## Architecture

Both models are SegFormer networks with a MiT-B3 encoder.

## Training data

??? info "Dataset summary (mussel presence/absence)"

    | Split      | Scenes | Tiles | Pixels~mussels~ | Pixels~total~ | Area (km^2^) | Res~max~ (m) | Res~min~ (m) | Res~$\mu$~ (m) | Res~$\sigma$~ (m) |
    |:-----------|-------:|------:|----------------:|--------------:|-------------:|-------------:|-------------:|---------------:|------------------:|
    | Train      |     39 |  4834 |       933287123 |    5068816384 |      23.8147 |        0.027 |      0.00231 |      0.0082244 |        0.00492054 |
    | Validation |      8 |  1277 |       223598444 |    1339031552 |      5.74255 |   0.00518591 |   0.00330667 |     0.00421985 |       0.000545617 |
    | Test       |      8 |  1110 |       175226412 |    1163919360 |      4.64052 |   0.00578278 |     0.003671 |     0.00435929 |       0.000665025 |
    | *Sum*      |     55 |  7221 |      1332111979 |    7571767296 |    34.197799 |     0.037969 |     0.009288 |       0.016804 |          0.006131 |

## Credit

!!! quote "Who to credit"
    Developed by the [Hakai Institute](https://hakai.org). **Will McInnes** collated and labelled the training data,
    and **Alyssa Gehman** led the scientific application. See [Credits](../credits.md) and [How to Cite](../cite.md).
