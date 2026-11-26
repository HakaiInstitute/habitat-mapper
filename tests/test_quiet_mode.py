"""Tests for how ONNXModel.process picks the progress output and checks its options."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from habitat_mapper.config import ModelConfig, SegmentOptions
from habitat_mapper.model import ONNXModel
from habitat_mapper.processing import ImageProcessor
from habitat_mapper.progress import NullProgressReporter, RichProgressReporter

if TYPE_CHECKING:
    from pathlib import Path


class _Recorded:
    def __init__(self) -> None:
        self.options: SegmentOptions | None = None
        self.progress: object = None
        self.fetch_progress: list[object] = []
        self.device: object = None


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Recorded:
    recorded = _Recorded()

    class _Processor:
        def run(self, *args: object, **kwargs: object) -> None:
            pass

    def from_model(model: ONNXModel, options: SegmentOptions, *, progress: object) -> _Processor:
        recorded.options, recorded.progress, recorded.device = options, progress, model.device
        return _Processor()

    monkeypatch.setattr(ImageProcessor, "from_model", staticmethod(from_model))
    return recorded


@pytest.fixture
def model(recorded: _Recorded, tmp_path: Path) -> ONNXModel:
    class _Cache:
        def fetch(self, cfg: ModelConfig, progress: object = None) -> Path:
            recorded.fetch_progress.append(progress)
            return tmp_path / "model.onnx"

    cfg = ModelConfig(name="stub", revision="20250101", dependencies=[], model_filename="model.onnx")
    return ONNXModel(cfg, cache=_Cache())  # type: ignore[arg-type]


def test_progress_bars_by_default(model: ONNXModel, recorded: _Recorded) -> None:
    model.process("in.tif", "out.tif")
    assert recorded.progress is RichProgressReporter
    [download_progress] = recorded.fetch_progress
    assert isinstance(download_progress(), RichProgressReporter)


def test_quiet_hides_progress_and_download_bars(model: ONNXModel, recorded: _Recorded) -> None:
    model.process("in.tif", "out.tif", quiet=True)
    assert recorded.progress is NullProgressReporter
    assert recorded.fetch_progress == [NullProgressReporter]


def test_given_progress_is_used(model: ONNXModel, recorded: _Recorded) -> None:
    model.process("in.tif", "out.tif", quiet=True, progress=RichProgressReporter)
    assert recorded.progress is RichProgressReporter
    assert recorded.fetch_progress == [RichProgressReporter]


def test_options_are_passed_on(model: ONNXModel, recorded: _Recorded) -> None:
    model.process("in.tif", "out.tif", crop_size=512, blur_kernel_size=5, band_order=[3, 2, 1], device="cpu")
    assert recorded.options == SegmentOptions(crop_size=512, blur_kernel_size=5, band_order=[3, 2, 1], device="cpu")
    assert recorded.device == "cpu"


def test_device_is_for_one_call_only(model: ONNXModel, recorded: _Recorded) -> None:
    model.device = "cuda"
    model.process("in.tif", "out.tif", device="cpu")
    assert recorded.device == "cpu"
    assert model.device == "cuda"
    model.process("in.tif", "out.tif")
    assert recorded.device == "cuda"


def test_wrong_band_count_is_reported_before_downloading(model: ONNXModel, recorded: _Recorded) -> None:
    with pytest.raises(ValueError, match="takes 3 bands"):
        model.process("in.tif", "out.tif", band_order=[3, 2])
    assert recorded.fetch_progress == []


def test_loaded_model_is_not_fetched_again(model: ONNXModel, recorded: _Recorded) -> None:
    model._session = object()  # type: ignore[assignment]
    model.process("in.tif", "out.tif")
    assert recorded.fetch_progress == []


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ({"crop_size": 511}, "Must be an even number"),
        ({"blur_kernel_size": 4}, "odd number"),
        ({"band_order": [0, 1, 2]}, "Band numbers start at 1"),
    ],
)
def test_invalid_options_raise_before_processing(
    model: ONNXModel, recorded: _Recorded, option: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        model.process("in.tif", "out.tif", **option)
    assert recorded.options is None
