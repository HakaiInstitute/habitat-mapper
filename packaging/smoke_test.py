"""Check that a frozen build can segment an image, and that the desktop app starts.

Usage: python packaging/smoke_test.py dist/habitat-mapper/hab [dist/habitat-mapper/Habitat-Mapper]
"""

import os
import subprocess  # noqa: S404
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

SIZE = 1024


def main() -> None:
    """Segment a random RGB GeoTIFF with the frozen `hab` and check the output, then start and quit the app.

    Raises:
        SystemExit: If the executable fails or writes an unexpected output.
    """
    hab = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "input.tif"
        out_path = Path(tmp) / "output.tif"

        rng = np.random.default_rng(0)
        with rasterio.open(
            img_path,
            "w",
            driver="GTiff",
            width=SIZE,
            height=SIZE,
            count=3,
            dtype="uint8",
            crs="EPSG:32609",
            transform=from_origin(500000, 5800000, 0.05, 0.05),
        ) as dst:
            dst.write(rng.integers(0, 256, (3, SIZE, SIZE), dtype=np.uint8))

        subprocess.run([hab, "devices"], check=True)  # noqa: S603
        subprocess.run([hab, "segment", "kelp-rgb", img_path, out_path], check=True)  # noqa: S603

        with rasterio.open(out_path) as src:
            if (src.height, src.width) != (SIZE, SIZE):
                raise SystemExit(f"Output is {src.height}x{src.width}, expected {SIZE}x{SIZE}")

    if len(sys.argv) > 2:
        gui = Path(sys.argv[2]).resolve()
        subprocess.run([gui], env={**os.environ, "HAB_GUI_SMOKE_TEST": "1"}, check=True, timeout=120)  # noqa: S603
    print("Smoke test passed")  # noqa: T201


if __name__ == "__main__":
    main()
