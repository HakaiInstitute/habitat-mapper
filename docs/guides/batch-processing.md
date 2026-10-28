# Process Many Images

`hab segment` processes one image per run. To process a folder, loop over the files. Use `--quiet` (`-q`) to turn
off the per-file progress bars. Errors are still shown, and a failed image still stops with an error.

## From the terminal

=== "Windows (PowerShell)"

    ```powershell
    New-Item -ItemType Directory -Force outputs | Out-Null
    Get-ChildItem .\images\*.tif | ForEach-Object {
        hab segment kelp-rgb $_.FullName ".\outputs\$($_.BaseName)_kelp.tif" --quiet
    }
    ```

=== "macOS / Linux"

    ```bash
    mkdir -p outputs
    for f in ./images/*.tif; do
        hab segment kelp-rgb "$f" "./outputs/$(basename "$f" .tif)_kelp.tif" --quiet
    done
    ```

## From Python

Loading the model once and reusing it avoids repeated setup:

```python
from pathlib import Path

from habitat_mapper import model_registry

model = model_registry["kelp-rgb"]
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

for img_path in sorted(Path("images").glob("*.tif")):
    model.process(
        img_path=img_path,
        output_path=out_dir / f"{img_path.stem}_kelp.tif",
        quiet=True,
    )
```

See [Use from Python](python.md) for more.

## Example notebooks

The [`examples/`](https://github.com/HakaiInstitute/habitat-mapper/tree/main/examples) folder in the repository has
Jupyter notebooks for bulk processing, including one that reads image paths from a Google Sheet.
