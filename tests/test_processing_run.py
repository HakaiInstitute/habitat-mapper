"""Tests for ImageProcessor.run behaviour that matters when one model processes several images, e.g. in the GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio

from habitat_mapper.config import ProcessingConfig, SegmentOptions
from habitat_mapper.processing import ImageProcessor, ProcessingCancelled
from habitat_mapper.progress import NullProgressReporter
from tests.helpers import RecordingObserver, make_stub_model, write_raster

CROP_SIZE = 32

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock


def _processor(model: MagicMock | None = None) -> ImageProcessor:
    return ImageProcessor(
        model=model or make_stub_model([]),
        config=ProcessingConfig(crop_size=CROP_SIZE, batch_size=1, blur_kernel_size=0, band_order=[1, 2, 3]),
        progress=NullProgressReporter,
    )


def _random_image(path: Path, dtype: type) -> Path:
    rng = np.random.default_rng(0)
    write_raster(path, rng.integers(1, 255, size=(3, CROP_SIZE * 2, CROP_SIZE * 2)).astype(dtype))
    return path


def test_cancel_stops_and_deletes_output(tmp_path: Path) -> None:
    model = make_stub_model([])
    processor = _processor(model)
    out_path = tmp_path / "output.tif"
    calls = 0

    def cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    with pytest.raises(ProcessingCancelled):
        processor.run(_random_image(tmp_path / "input.tif", np.uint8), out_path, cancel=cancel)

    assert model.predict.call_count == 1
    assert not out_path.exists()


def test_auto_max_pixel_value_is_detected_per_image(tmp_path: Path) -> None:
    model = make_stub_model([])
    processor = _processor(model)
    assert model.cfg.max_pixel_value == "auto"

    processor.run(_random_image(tmp_path / "uint8.tif", np.uint8), tmp_path / "out8.tif")
    assert {call.args[1] for call in model.predict.call_args_list} == {255}

    # A uint16 image after a uint8 one must not be scaled with the uint8 range
    model.predict.reset_mock()
    processor.run(_random_image(tmp_path / "uint16.tif", np.uint16), tmp_path / "out16.tif")
    assert {call.args[1] for call in model.predict.call_args_list} == {65535}
    assert model.cfg.max_pixel_value == "auto"


def test_unsupported_dtype_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported image data type int32"):
        _processor().run(_random_image(tmp_path / "input.tif", np.int32), tmp_path / "output.tif")


def test_observer_sees_every_batch_and_the_whole_result(tmp_path: Path) -> None:
    observer = RecordingObserver()
    out_path = tmp_path / "output.tif"
    # Blank on the left, so that the shortcut for blank tiles is covered too
    data = np.zeros((3, CROP_SIZE * 2, CROP_SIZE * 3), dtype=np.uint8)
    data[:, :, CROP_SIZE:] = np.random.default_rng(0).integers(1, 255, size=(3, CROP_SIZE * 2, CROP_SIZE * 2))
    write_raster(tmp_path / "input.tif", data)

    _processor().run(tmp_path / "input.tif", out_path, observer=observer)

    assert observer.size == (CROP_SIZE * 2, CROP_SIZE * 3)
    assert observer.batches
    assert all(w.width == w.height == CROP_SIZE for b in observer.batches for w in b)
    with rasterio.open(out_path) as src:
        np.testing.assert_array_equal(observer.result, src.read(1))


@pytest.mark.parametrize("band_order", [[1, 2], [1, 2, 3, 4]])
def test_band_order_needs_one_band_per_model_input(band_order: list[int]) -> None:
    with pytest.raises(ValueError, match="takes 3 bands"):
        ImageProcessor.from_model(make_stub_model([]), SegmentOptions(crop_size=CROP_SIZE, band_order=band_order))


def test_band_order_can_pick_from_more_bands_than_the_model_takes() -> None:
    model = make_stub_model([])
    model.input_size = None
    processor = ImageProcessor.from_model(model, SegmentOptions(crop_size=CROP_SIZE, band_order=[6, 4, 2]))
    assert processor.config.band_order == [6, 4, 2]


def test_observer_sees_the_filtered_result(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    model = make_stub_model([])
    # Noisy predictions, so that the filters change the result
    model.predict.side_effect = lambda batch, _: rng.random((batch.shape[0], 2, *batch.shape[2:]), dtype=np.float32)
    processor = ImageProcessor(
        model=model,
        config=ProcessingConfig(crop_size=CROP_SIZE, blur_kernel_size=5, morph_kernel_size=3, band_order=[1, 2, 3]),
        progress=NullProgressReporter,
    )
    observer = RecordingObserver()
    out_path = tmp_path / "output.tif"

    processor.run(_random_image(tmp_path / "input.tif", np.uint8), out_path, observer=observer)

    with rasterio.open(out_path) as src:
        np.testing.assert_array_equal(observer.result, src.read(1))
