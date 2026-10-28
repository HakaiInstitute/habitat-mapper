# Eelgrass

`eelgrass-rgb` maps eelgrass (*Zostera marina*) beds in RGB drone imagery.

```bash
hab segment eelgrass-rgb input.tif output.tif
```

## Inputs

- **Bands:** Red, Green, Blue.
- **Data type:** 8- or 16-bit unsigned integers.

## Outputs

| Value | Class                         |
|------:|-------------------------------|
| **0** | Background                    |
| **1** | Eelgrass (*Zostera marina*)   |

## Architecture

SegFormer with a MiT-B2 encoder.

## Performance and training data

!!! note "Coming soon"
    Performance metrics, recommended resolution and training data details for `eelgrass-rgb` will be added here.

## Credit

!!! quote "Who to credit"
    Developed by the [Hakai Institute](https://hakai.org). See [Credits](../credits.md) and
    [How to Cite](../cite.md).
