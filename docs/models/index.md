# Models

Each model is trained for one kind of imagery and one habitat. Pick the row that matches your imagery, then use the
model name with `hab segment`. Run `hab models` to see the latest revision of each.

| Model                  | Imagery                     | Input bands                           | Output classes                          | Details                                    |
|------------------------|-----------------------------|---------------------------------------|-----------------------------------------|--------------------------------------------|
| `kelp-rgb`             | Drone / aerial              | Red, Green, Blue                      | Giant kelp, bull kelp                   | [Kelp: Drone](kelp-drone.md)               |
| `kelp-rgbi`            | Drone / aerial              | Red, Green, Blue, NIR                 | Giant kelp, bull kelp                   | [Kelp: Drone](kelp-drone.md)               |
| `kelp-ps8b`            | PlanetScope satellite       | 8-band surface reflectance            | Kelp                                    | [Kelp: Satellite](kelp-satellite.md)       |
| `kelp-skema` (beta)    | Sentinel-2 satellite        | L2A `.SAFE` folder                    | Kelp                                    | [Kelp: Satellite](kelp-satellite.md)       |
| `mussel-rgb`           | Drone                       | Red, Green, Blue                      | Mussels                                 | [Mussels & Barnacles](intertidal.md)       |
| `mussel-gooseneck-rgb` | Drone                       | Red, Green, Blue                      | Mussels, gooseneck barnacles            | [Mussels & Barnacles](intertidal.md)       |
| `eelgrass-rgb`         | Drone                       | Red, Green, Blue                      | Eelgrass                                | [Eelgrass](eelgrass.md)                    |

Every model also outputs `0` for background. The exact class values are on each model's page.

!!! tip "Bands in a different order?"
    If your image's bands aren't in the order shown, you can remap them with `-b`. See
    [Prepare Your Imagery](../guides/prepare-imagery.md#bands-are-in-red-green-blue-then-nir-order).

## Revisions

Models are improved over time and each update is published as a new **revision**, named by its release date
(for example `20240722`). `hab segment` uses the latest revision unless you pass `--rev`. To list the revisions of
a model, run `hab revisions <model>`. See [Tune Processing](../guides/tune-processing.md#model-revisions).

## Performance

Each model page reports IoU, precision, recall and F1 for that model's classes. These are measured on held-out
validation data from our training datasets. They show how well the model matches careful human labelling on imagery
*similar to the training data*. Results on your imagery may differ. See [Performance Metrics](metrics.md) for
definitions.

## Using a model in research

Many of these models were built with data or code from partners. Each model page names who to credit. See
[How to Cite](../cite.md).
