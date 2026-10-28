# CLI Examples

Worked examples of the `hab` command line tool. For every command and option, see the
[CLI Reference](reference/cli.md).

!!! tip "New to the Command Line?"
    The [Processing Images](beginner_guide/execution.md) guide is a more beginner-friendly walkthrough.

## Segment

!!! Example "Usage of segment command"

    Segment kelp in an RGB drone image

    ```bash
    hab segment kelp-rgb ./input_image.tif ./output_kelp_segmentation.tif
    ```

    Segment kelp in a 4-band RGB+NIR drone image (note that you may use flags or just positional arguments as above):

    ```bash
    hab segment -m kelp-rgbi -i ./input_image.tif -o ./output_kelp_segmentation.tif
    ```

    Same as above, but select bands RGB+NIR bands from a source image that has bands in Blue, Green, Red, RedEdge, NIR order.
    ```bash
    hab segment -m kelp-rgbi -i ./input_ms_image.tif -o ./output_kelp_segmentation.tif -b 3 -b 2 -b 1 -b 5
    ```

    Segment mussels in an RGB drone image:

    ```bash
    hab segment -m mussel-rgb -i ./input_image.tif -o ./output_mussel_segmentation.tif
    ```

    Process multiple images in a shell loop without progress noise:

    ```bash
    for f in ./images/*.tif; do
        hab segment -m kelp-rgb -i "$f" -o "./outputs/$(basename $f)" --quiet
    done
    ```

???+ tip "Tip: Reduce windowed processing artifacts"

    To reduce artifacts caused by Habitat-Mapper's moving window classification, use the largest `crop-size` that you can.
    Try starting with a crop-size around 3200 pixels and reduce it if your computer is unable to load that much data at once and the application crashes.

    We hope to improve Habitat-Mapper in the future such that the maximum crop size for your computer can be determined automatically.

    CLI Example:

    ```bash
    hab segment -m kelp-rgb --crop-size=3200 -i your-input-image.tif -o output-image.tif
    ```

    <div style="display:flex;">
        <img alt="Small window output" src="../images/hab-small-window.png" width="50%"/>
        <img alt="Big window output" src="../images/hab-big-window.png" width="50%"/>
    </div>

    Also worth noting is that the `crop-size` parameter will change the outputs.
    If you need to reproduce a result, make sure to use the same `crop-size` as the original run.

## Model Revisions

!!! Example "Usage of revisions command"

    List all available revisions for a specific model.

    ```bash
    $ hab revisions kelp-rgb
                              Revisions for kelp-rgb
    ┏━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
    ┃ Revision ┃ Latest ┃ Description                                    ┃ Status    ┃
    ┡━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
    │ 20240722 │   ✓    │ Kelp segmentation model for RGB drone imagery. │ Available │
    └──────────┴────────┴────────────────────────────────────────────────┴───────────┘
    ```

## Clean

Clear the model cache to free up space. Models will be re-downloaded as needed.
You'll be prompted to confirm before any cached models are removed from your system.

```bash
hab clean
```
