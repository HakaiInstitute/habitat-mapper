"""Tests for ImageProcessor.run behaviour that matters when one model processes several images, e.g. in the GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio

from habitat_mapper.config import ProcessingConfig
from habitat_mapper.processing import ImageProcessor, ProcessingCancelled
from habitat_mapper.progress import NullProgressReporter
from tests.test_blank_tile_skipping import CROP_SIZE, _make_stub_model, _write_raster

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from unittest.mock import MagicMock

    from rasterio.windows import Window


def _processor(model: MagicMock | None = None) -> ImageProcessor:
    return ImageProcessor(
        model=model or _make_stub_model([]),
        config=ProcessingConfig(crop_size=CROP_SIZE, batch_size=1, blur_kernel_size=0, band_order=[1, 2, 3]),
        reporter_cls=NullProgressReporter,
    )


def _random_image(path: Path, dtype: type) -> Path:
    rng = np.random.default_rng(0)
    _write_raster(path, rng.integers(1, 255, size=(3, CROP_SIZE * 2, CROP_SIZE * 2)).astype(dtype))
    return path


def test_cancel_stops_and_deletes_output(tmp_path: Path) -> None:
    model = _make_stub_model([])
    processor = _processor(model)
    out_path = tmp_path / "output.tif"
    calls = 0

    def cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    with pytest.raises(ProcessingCancelled):
        processor.run(_random_image(tmp_path / "input.tif", np.uint8), out_path, cancel=cancel)

    assert model._predict.call_count == 1
    assert not out_path.exists()


def test_auto_max_pixel_value_is_detected_per_image(tmp_path: Path) -> None:
    model = _make_stub_model([])
    processor = _processor(model)
    assert model.cfg.max_pixel_value == "auto"

    processor.run(_random_image(tmp_path / "uint8.tif", np.uint8), tmp_path / "out8.tif")
    assert model.cfg.max_pixel_value == "auto"

    # A uint16 image after a uint8 one must not be scaled with the uint8 range
    seen = []
    predict = model._predict.side_effect
    model._predict.side_effect = lambda batch: seen.append(model.cfg.max_pixel_value) or predict(batch)
    processor.run(_random_image(tmp_path / "uint16.tif", np.uint16), tmp_path / "out16.tif")
    assert set(seen) == {65535}


def test_unsupported_dtype_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported image data type int32"):
        _processor().run(_random_image(tmp_path / "input.tif", np.int32), tmp_path / "output.tif")


class _RecordingObserver:
    def __init__(self) -> None:
        self.size: tuple[int, int] | None = None
        self.batches: list[list[Window]] = []
        self.result: np.ndarray | None = None

    def image_started(self, height: int, width: int) -> None:
        self.size = (height, width)
        self.result = np.full((height, width), 99, dtype=np.uint8)

    def batch_started(self, windows: Sequence[Window]) -> None:
        self.batches.append(list(windows))

    def written(self, window: Window, data: np.ndarray) -> None:
        assert self.result is not None
        rows, cols = window.toslices()
        self.result[rows, cols] = data


def test_observer_sees_every_batch_and_the_whole_result(tmp_path: Path) -> None:
    observer = _RecordingObserver()
    out_path = tmp_path / "output.tif"
    # Blank on the left, so that the shortcut for blank tiles is covered too
    data = np.zeros((3, CROP_SIZE * 2, CROP_SIZE * 3), dtype=np.uint8)
    data[:, :, CROP_SIZE:] = np.random.default_rng(0).integers(1, 255, size=(3, CROP_SIZE * 2, CROP_SIZE * 2))
    _write_raster(tmp_path / "input.tif", data)

    _processor().run(tmp_path / "input.tif", out_path, observer=observer)

    assert observer.size == (CROP_SIZE * 2, CROP_SIZE * 3)
    assert observer.batches
    assert all(0 <= w.row_off and w.row_off + w.height <= CROP_SIZE * 2 for b in observer.batches for w in b)
    with rasterio.open(out_path) as src:
        np.testing.assert_array_equal(observer.result, src.read(1))
