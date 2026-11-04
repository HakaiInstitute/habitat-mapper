"""Check that a frozen `hab` executable can segment an image.

Usage: python packaging/smoke_test.py dist/habitat-mapper/hab
"""

import subprocess  # noqa: S404
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

SIZE = 1024


def main() -> None:
    """Segment a random RGB GeoTIFF with the frozen executable and check the output.

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
    print("Smoke test passed")  # noqa: T201


if __name__ == "__main__":
    main()
