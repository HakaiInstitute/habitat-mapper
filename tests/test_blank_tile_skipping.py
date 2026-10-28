"""Tests for skipping constant-valued (blank) tiles in ImageProcessor.run.

Regression tests for https://github.com/HakaiInstitute/habitat-mapper/issues/198, where batches containing more than
one blank tile either raised an IndexError or dropped non-blank tiles from prediction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from habitat_mapper.config import ModelConfig, ProcessingConfig
from habitat_mapper.processing import ImageProcessor
from habitat_mapper.progress import NullProgressReporter

if TYPE_CHECKING:
    from pathlib import Path

CROP_SIZE = 32
DEFAULT_OUTPUT_VALUE = 7
PREDICTED_CLASS = 1


def _make_stub_model(predicted_tiles: list[np.ndarray]) -> MagicMock:
    """Model stub that predicts PREDICTED_CLASS everywhere and records every tile it receives.

    Returns:
        A mock model whose predictions are appended to ``predicted_tiles``.
    """

    def _predict(batch: np.ndarray) -> np.ndarray:
        predicted_tiles.extend(batch)
        logits = np.zeros((batch.shape[0], 2, batch.shape[2], batch.shape[3]), dtype=np.float32)
        logits[:, PREDICTED_CLASS] = 1.0
        return logits

    model = MagicMock()
    model.cfg = ModelConfig(
        name="stub",
        revision="20250101",
        dependencies=[],
        model_filename="model.onnx",
        default_output_value=DEFAULT_OUTPUT_VALUE,
        nodata_value=255,
    )
    model._predict.side_effect = _predict
    model._postprocess.side_effect = lambda data: np.argmax(data, axis=0).astype(np.uint8)
    return model


def _write_raster(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:32609",
        transform=from_origin(500000, 5500000, 1, 1),
    ) as dst:
        dst.write(data)


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
    _write_raster(img_path, data)

    predicted_tiles: list[np.ndarray] = []
    processor = ImageProcessor(
        model=_make_stub_model(predicted_tiles),
        config=ProcessingConfig(crop_size=CROP_SIZE, batch_size=batch_size, blur_kernel_size=0, band_order=[1, 2, 3]),
        reporter_cls=NullProgressReporter,
    )
    processor.run(img_path, out_path)

    # Every non-blank tile is predicted exactly once, and no blank tile is sent to the model
    with rasterio.open(img_path) as src:
        windows = list(
            ImageProcessor._generate_windows(
                *ImageProcessor._calculate_extended_dimensions(height, width, processor.config),
                tile_size=CROP_SIZE,
                stride=processor.config.stride,
            )
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
