# How the Models Work

Habitat-Mapper models are deep neural networks for *semantic segmentation*: they label every pixel of an image with
a class, such as kelp, mussels or background.

## Processing large images

Orthomosaics are usually far too large to pass through a neural network in one go, so Habitat-Mapper processes them
in pieces:

1. **Read in tiles.** The image is read in square windows of `--crop-size` pixels that overlap by 50%. Each window
   is reordered to the band order the model expects (see `-b`) and normalised.
2. **Predict.** Each window goes through the model, `--batch-size` windows at a time, on a GPU if one is available.
3. **Blend.** Overlapping predictions are blended with a smooth weighting window, so tile edges don't show as seams.
   Larger tiles mean fewer seams (see [Tune Processing](../guides/tune-processing.md#crop-size-quality-vs-memory)).
4. **Label and clean.** The blended scores become one class value per pixel. Then the optional `--blur` and
   `--morph` filters are applied.
5. **Write.** The output GeoTIFF gets the input's location and coordinate system, so it lines up in GIS.

Pixels marked as nodata in the input are skipped. The models are stored in the portable ONNX format and run with
ONNX Runtime, so you don't need a deep learning framework installed.

## How the models were trained

Model training code is available at [:material-github: hakai-ml-train](https://github.com/HakaiInstitute/hakai-ml-train).
Each [model page](index.md) summarises its own training data and results. The diagrams below show the general
process used for the drone kelp models.

### Preparing training data

```mermaid
flowchart LR
    subgraph Labelling["Labelling"]
        A[/"UAV Orthomosaic (RGB)"/]
        B["Manual kelp classification"]
        C[/"Presence/Absence Dataset\n(not kelp = 0, kelp = 1)"/]
        D[/"Species Dataset\n(not kelp = 0, Macrocystis = 2,\n Nereocystis = 3)"/]
    end
    subgraph Filtering["Prepare image tiles"]
        E{{"Tile image and label\n(512x512 pixel tiles, 50% overlap)"}}
        F{{"Discard image and labels tiles with no kelp\nOR where >50% is no data"}}
    end
    subgraph PADataset["P/A Dataset"]
        G[("Training ~80%")]
        H[("Validation ~10%")]
        I[("Testing ~10%")]
    end
    subgraph SpeciesDataset["Species Dataset"]
        G1[("Training ~80%")]
        H1[("Validation ~10%")]
        I1[("Testing ~10%")]
    end
    subgraph DataPrep["Data Preparation"]
        Labelling
        Filtering
        PADataset
        SpeciesDataset
    end
    A --> B
    B --> C & D
    E --> F
    C --> Filtering
    D --> Filtering
    Filtering --> PADataset & SpeciesDataset
```

### Training loop

```mermaid
flowchart LR
    subgraph Augmentation["Data Augmentation"]
        J["Randomly rotation in [0, 45] deg"]
        K["Random horizontal and/or vertical flip"]
        L["Randomly jitter brightness, contrast, and saturation"]
    end
    subgraph Validation["Validation"]
        V["Calculate average model error on all batches"]
        H2[("Validation")]
    end
    subgraph Testing["Testing"]
        Y["Calculate model performance on all batches"]
        I3[("Testing")]
        Z("Save model performance metrics")
    end
    subgraph Training["Model Training Loop"]
        P(["Start"])
        B2["Get next batch of training tiles"]
        G2[("Training")]
        Augmentation
        Q["Forward pass through model"]
        R["Calculate error of model relative to hand-labelled images"]
        S["Update model parameters with backpropagation"]
        B3{"More training tiles to process?"}
        Validation
        W{"Validation error is stabilized?\n(model converged)"}
        reset["Next training epoch"]
        X["Finished training, save model"]
        Testing
        End(["End"])
    end
    P --> B2
    G2 -.-> B2
    J --> K
    K --> L
    B2 --> Augmentation
    L --> Q
    Q --> R
    R --> S
    S --> B3
    B3 -- Yes --> B2
    H2 -.-> V
    B3 -- No --> Validation
    Validation --> W
    W -- No --> reset
    reset --> B2
    W -- Yes --> X
    I3 -.-> Y
    Y --> Z
    X --> Testing
    Testing --> End
```
