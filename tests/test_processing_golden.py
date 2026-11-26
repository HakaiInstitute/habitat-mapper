"""Golden tests for ImageProcessor.run, checking the output raster pixel by pixel against an answer worked out directly.

The fake model's scores for a pixel depend only on that pixel's value, so every tile that covers a pixel scores it the
same. However the overlapping tiles are weighted and blended, each pixel's class must then come out as the class its
own value picks. Any slip in where a tile's scores are placed shows up as wrong pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from habitat_mapper.config import ModelConfig, ProcessingConfig
from habitat_mapper.processing import ImageProcessor, ProcessingCancelled
from habitat_mapper.progress import NullProgressReporter
from tests.helpers import CRS, ORIGIN, write_raster

if TYPE_CHECKING:
    from pathlib import Path

CROP_SIZE = 32
NUM_CLASSES = 3
DEFAULT_OUTPUT_VALUE = 7
NODATA = 255


def class_of(pixels: np.ndarray) -> np.ndarray:
    """The class the fake model picks for each pixel, from its first band.

    Returns:
        Class labels with the pixels' shape.
    """
    return (pixels % NUM_CLASSES).astype(np.uint8)


class PixelwiseModel:
    """Segmenter whose scores for a pixel depend only on that pixel's value."""

    input_size = None

    def __init__(self) -> None:
        self.cfg = ModelConfig(
            name="pixelwise",
            revision="20250101",
            dependencies=[],
            model_filename="model.onnx",
            default_output_value=DEFAULT_OUTPUT_VALUE,
            nodata_value=NODATA,
        )

    def predict(self, batch: np.ndarray, max_pixel_value: float) -> np.ndarray:
        classes = class_of(batch[:, 0])
        # One-hot scores, transposed from [b, h, w, c] to [b, c, h, w]
        return np.eye(NUM_CLASSES, dtype=np.float32)[classes].transpose(0, 3, 1, 2)

    def postprocess(self, batch: np.ndarray) -> np.ndarray:
        return np.argmax(batch, axis=0).astype(np.uint8)


def _run(tmp_path: Path, image: np.ndarray, *, blur: int = 0, morph: int = 0) -> np.ndarray:
    write_raster(tmp_path / "input.tif", image)
    processor = ImageProcessor(
        model=PixelwiseModel(),
        config=ProcessingConfig(
            crop_size=CROP_SIZE,
            batch_size=3,
            blur_kernel_size=blur,
            morph_kernel_size=morph,
            band_order=[1, 2, 3],
        ),
        progress=NullProgressReporter,
    )
    processor.run(tmp_path / "input.tif", tmp_path / "output.tif")
    with rasterio.open(tmp_path / "output.tif") as src:
        return src.read(1)


def _textured(height: int, width: int, seed: int = 0) -> np.ndarray:
    # Starts at 1 so that no tile is all one value and skipped as blank
    rng = np.random.default_rng(seed)
    return rng.integers(1, 250, size=(3, height, width), dtype=np.uint8)


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (1, 1),
        (CROP_SIZE, CROP_SIZE),
        (CROP_SIZE - 1, CROP_SIZE + 1),
        (CROP_SIZE * 2, CROP_SIZE * 3),
        (CROP_SIZE * 3 + 5, CROP_SIZE * 2 + 17),
        (7, CROP_SIZE * 4 + 3),
        (CROP_SIZE * 4 + 3, 7),
    ],
)
def test_every_pixel_gets_its_own_class(tmp_path: Path, height: int, width: int) -> None:
    image = _textured(height, width)

    result = _run(tmp_path, image)

    np.testing.assert_array_equal(result, class_of(image[0]))


def _tiles(height: int, width: int) -> list[tuple[slice, slice]]:
    windows = ImageProcessor._generate_windows(height, width, tile_size=CROP_SIZE, stride=CROP_SIZE // 2)
    return [w.toslices() for w in windows]


@pytest.mark.parametrize(
    "blank",
    [
        # (rows, cols) of the image that are one flat value, in units of half a tile
        (slice(0, 4), slice(0, 4)),  # top-left corner
        (slice(2, 6), slice(2, 6)),  # middle
        (slice(0, 8), slice(3, 7)),  # a full-height stripe
        (slice(3, 7), slice(0, 8)),  # a full-width stripe
        (slice(4, 8), slice(4, 8)),  # bottom-right corner
    ],
)
def test_blank_tiles_dont_disturb_their_neighbours(tmp_path: Path, blank: tuple[slice, slice]) -> None:
    half = CROP_SIZE // 2
    height = width = half * 8
    image = _textured(height, width)
    rows = slice(blank[0].start * half, blank[0].stop * half)
    cols = slice(blank[1].start * half, blank[1].stop * half)
    image[:, rows, cols] = 0

    result = _run(tmp_path, image)

    # A pixel in any tile with something to segment gets its own class, unless a blank tile covers it too, in which
    # case the default is also fine. A pixel only blank tiles cover gets the default
    in_textured = np.zeros((height, width), dtype=bool)
    in_blank = np.zeros((height, width), dtype=bool)
    for tile_rows, tile_cols in _tiles(height, width):
        tile = image[:, tile_rows, tile_cols]
        if np.all(tile == tile.flat[0]):
            in_blank[tile_rows, tile_cols] = True
        else:
            in_textured[tile_rows, tile_cols] = True

    expected = class_of(image[0])
    assert in_blank.any()
    only_textured = in_textured & ~in_blank
    np.testing.assert_array_equal(result[only_textured], expected[only_textured])
    assert np.all(result[in_blank & ~in_textured] == DEFAULT_OUTPUT_VALUE)
    both = in_textured & in_blank
    assert np.all((result[both] == expected[both]) | (result[both] == DEFAULT_OUTPUT_VALUE))


def test_output_keeps_the_input_georeferencing(tmp_path: Path) -> None:
    write_raster(tmp_path / "input.tif", _textured(CROP_SIZE + 3, CROP_SIZE * 2), pixel_size=0.5)

    processor = ImageProcessor(
        model=PixelwiseModel(),
        config=ProcessingConfig(crop_size=CROP_SIZE, batch_size=2, blur_kernel_size=0, band_order=[1, 2, 3]),
    )
    processor.run(tmp_path / "input.tif", tmp_path / "output.tif")

    with rasterio.open(tmp_path / "output.tif") as src:
        assert (src.height, src.width) == (CROP_SIZE + 3, CROP_SIZE * 2)
        assert src.count == 1
        assert src.dtypes == ("uint8",)
        assert src.nodata == NODATA
        assert src.crs.to_string() == CRS
        assert src.transform == from_origin(*ORIGIN, 0.5, 0.5)
        assert src.compression.value == "LZW"


def _blocky(height: int, width: int) -> np.ndarray:
    """Classes in blocks of a few pixels, so filters change some pixels but not all.

    Returns:
        A three-band image.
    """
    rng = np.random.default_rng(1)
    blocks = rng.integers(1, 250, size=(height // 4 + 1, width // 4 + 1), dtype=np.uint8)
    noise = rng.integers(0, 2, size=(height, width), dtype=np.uint8) * NUM_CLASSES
    band = np.kron(blocks, np.ones((4, 4), dtype=np.uint8))[:height, :width] + noise
    return np.stack([band, band, band])


def _filtered(classes: np.ndarray, blur: int, morph: int) -> np.ndarray:
    if blur:
        classes = cv2.medianBlur(classes, blur)
    if morph:
        kernel = np.ones((morph, morph), np.uint8)
        classes = cv2.morphologyEx(classes, cv2.MORPH_OPEN, kernel)
        classes = cv2.morphologyEx(classes, cv2.MORPH_CLOSE, kernel)
    return classes


@pytest.mark.parametrize(("blur", "morph"), [(5, 0), (0, 3), (0, 5), (5, 3)])
def test_filters_match_filtering_the_whole_image(tmp_path: Path, blur: int, morph: int) -> None:
    # Several filter tiles across and down, so their seams are checked
    image = _blocky(CROP_SIZE * 3 + 5, CROP_SIZE * 4 + 9)

    result = _run(tmp_path, image, blur=blur, morph=morph)

    np.testing.assert_array_equal(result, _filtered(class_of(image[0]), blur, morph))


@pytest.mark.parametrize("cancel_after", [None, 1])
def test_filtering_leaves_only_the_output_behind(tmp_path: Path, cancel_after: int | None) -> None:
    write_raster(tmp_path / "input.tif", _blocky(CROP_SIZE * 2, CROP_SIZE * 2))
    processor = ImageProcessor(
        model=PixelwiseModel(),
        config=ProcessingConfig(crop_size=CROP_SIZE, blur_kernel_size=5, band_order=[1, 2, 3]),
    )
    calls = 0

    def cancel() -> bool:
        nonlocal calls
        calls += 1
        return cancel_after is not None and calls > cancel_after

    try:
        processor.run(tmp_path / "input.tif", tmp_path / "output.tif", cancel=cancel)
    except ProcessingCancelled:
        assert cancel_after is not None

    expected = {"input.tif"} if cancel_after else {"input.tif", "output.tif"}
    assert {p.name for p in tmp_path.iterdir()} == expected
