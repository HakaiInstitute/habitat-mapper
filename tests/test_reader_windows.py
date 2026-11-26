"""Tests that every reader reads a window the same way: the bands asked for, padded where it leaves the image."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from rasterio.windows import Window

from habitat_mapper.reader import ImageReader, SAFEReader, TIFFReader
from tests.helpers import SAFE_SIZE, make_safe, write_raster

if TYPE_CHECKING:
    from pathlib import Path

FILL = 9


def _expected(image: np.ndarray, window: Window, band_order: list[int] | None) -> np.ndarray:
    """Read a window from an in-memory image, filling pixels outside it.

    Returns:
        The window's pixels with shape [bands, height, width].
    """
    bands, height, width = image.shape
    padded = np.full((bands, height + 2 * 100, width + 2 * 100), FILL, dtype=image.dtype)
    padded[:, 100 : 100 + height, 100 : 100 + width] = image
    rows = slice(window.row_off + 100, window.row_off + 100 + window.height)
    cols = slice(window.col_off + 100, window.col_off + 100 + window.width)
    out = padded[:, rows, cols]
    return out if band_order is None else out[[b - 1 for b in band_order]]


WINDOWS = [
    Window(0, 0, 8, 8),  # inside
    Window(SAFE_SIZE - 5, 3, 8, 8),  # past the right
    Window(4, SAFE_SIZE - 3, 8, 8),  # past the bottom
    Window(SAFE_SIZE - 3, SAFE_SIZE - 3, 8, 8),  # past the corner
    Window(-4, -2, 8, 8),  # before the top left
    Window(-2, -2, SAFE_SIZE + 4, SAFE_SIZE + 4),  # around all of it
    Window(SAFE_SIZE + 5, 0, 8, 8),  # entirely to the right
    Window(0, SAFE_SIZE + 5, 8, 6),  # entirely below
]


@pytest.fixture(params=["tiff", "safe"])
def reader_and_image(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[ImageReader, np.ndarray]:
    if request.param == "tiff":
        image = np.random.default_rng(0).integers(1, 5000, size=(4, SAFE_SIZE, SAFE_SIZE)).astype(np.uint16)
        return TIFFReader(write_raster(tmp_path / "image.tif", image)), image
    reader = SAFEReader(make_safe(tmp_path))
    return reader, reader.read_window(Window(0, 0, reader.width, reader.height))


@pytest.mark.parametrize("window", WINDOWS, ids=str)
@pytest.mark.parametrize("band_order", [None, [3, 1, 2], [2, 2]])
def test_read_window(
    reader_and_image: tuple[ImageReader, np.ndarray], window: Window, band_order: list[int] | None
) -> None:
    reader, image = reader_and_image

    data = reader.read_window(window, band_order=band_order, boundless=True, fill_value=FILL)

    np.testing.assert_array_equal(data, _expected(image, window, band_order))


@pytest.mark.parametrize("band_order", [[0], [1, 9]])
def test_read_window_rejects_missing_bands(
    reader_and_image: tuple[ImageReader, np.ndarray], band_order: list[int]
) -> None:
    reader, _ = reader_and_image
    with pytest.raises(ValueError, match="Band order"):
        reader.read_window(Window(0, 0, 4, 4), band_order=band_order)
