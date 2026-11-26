"""Tests for skipping constant-valued (blank) tiles in ImageProcessor.run.

Regression tests for https://github.com/HakaiInstitute/habitat-mapper/issues/198, where batches containing more than
one blank tile either raised an IndexError or dropped non-blank tiles from prediction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio

from habitat_mapper.config import ProcessingConfig
from habitat_mapper.processing import ImageProcessor
from habitat_mapper.progress import NullProgressReporter
from tests.helpers import make_stub_model, write_raster

if TYPE_CHECKING:
    from pathlib import Path

CROP_SIZE = 32
DEFAULT_OUTPUT_VALUE = 7
PREDICTED_CLASS = 1


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8])
def test_batches_with_multiple_blank_tiles(tmp_path: Path, batch_size: int) -> None:
    # Single row of tiles: mostly blank, with a textured region in the middle so that batches contain several blank
    # tiles before, after, and around non-blank ones.
    height, width = CROP_SIZE, CROP_SIZE * 8
    data = np.zeros((3, height, width), dtype=np.uint8)
    textured = slice(CROP_SIZE * 3, CROP_SIZE * 5)
    rng = np.random.default_rng(0)
    data[:, :, textured] = rng.integers(1, 255, size=(3, height, CROP_SIZE * 2), dtype=np.uint8)

    img_path = tmp_path / "input.tif"
    out_path = tmp_path / "output.tif"
    write_raster(img_path, data)

    predicted_tiles: list[np.ndarray] = []
    processor = ImageProcessor(
        model=make_stub_model(
            predicted_tiles, predicted_class=PREDICTED_CLASS, default_output_value=DEFAULT_OUTPUT_VALUE
        ),
        config=ProcessingConfig(crop_size=CROP_SIZE, batch_size=batch_size, blur_kernel_size=0, band_order=[1, 2, 3]),
        progress=NullProgressReporter,
    )
    processor.run(img_path, out_path)

    # Every non-blank tile is predicted exactly once, and no blank tile is sent to the model
    with rasterio.open(img_path) as src:
        windows = list(
            ImageProcessor._generate_windows(height, width, tile_size=CROP_SIZE, stride=processor.config.stride)
        )
        tiles = [src.read(window=w, boundless=True, fill_value=0) for w in windows]
    expected = [t for t in tiles if not np.all(t == t.flat[0])]
    assert len(predicted_tiles) == len(expected)
    assert all(not np.all(t == t.flat[0]) for t in predicted_tiles)

    with rasterio.open(out_path) as dst:
        result = dst.read(1)
    assert np.all(result[:, textured] == PREDICTED_CLASS)
    assert np.all(result[:, : CROP_SIZE * 2] == DEFAULT_OUTPUT_VALUE)
    assert np.all(result[:, CROP_SIZE * 6 :] == DEFAULT_OUTPUT_VALUE)
