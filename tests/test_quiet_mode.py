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
        self.fetch_quiet: list[bool] = []


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Recorded:
    recorded = _Recorded()

    class _Processor:
        def run(self, *args: object, **kwargs: object) -> None:
            pass

    def from_model(model: ONNXModel, options: SegmentOptions, *, progress: object) -> _Processor:
        recorded.options, recorded.progress = options, progress
        return _Processor()

    def get_local_model_path(self: ModelConfig, quiet: bool = False) -> Path:
        recorded.fetch_quiet.append(quiet)
        return tmp_path / "model.onnx"

    monkeypatch.setattr(ImageProcessor, "from_model", staticmethod(from_model))
    monkeypatch.setattr(ModelConfig, "get_local_model_path", get_local_model_path)
    return recorded


@pytest.fixture
def model() -> ONNXModel:
    return ONNXModel(ModelConfig(name="stub", revision="20250101", dependencies=[], model_filename="model.onnx"))


def test_progress_bars_by_default(model: ONNXModel, recorded: _Recorded) -> None:
    model.process("in.tif", "out.tif")
    assert recorded.progress is RichProgressReporter
    assert recorded.fetch_quiet == [False]


def test_quiet_hides_progress_and_download_bars(model: ONNXModel, recorded: _Recorded) -> None:
    model.process("in.tif", "out.tif", quiet=True)
    assert recorded.progress is NullProgressReporter
    assert recorded.fetch_quiet == [True]


def test_given_progress_is_used(model: ONNXModel, recorded: _Recorded) -> None:
    model.process("in.tif", "out.tif", quiet=True, progress=RichProgressReporter)
    assert recorded.progress is RichProgressReporter


def test_options_are_passed_on(model: ONNXModel, recorded: _Recorded) -> None:
    model.process("in.tif", "out.tif", crop_size=512, blur_kernel_size=5, band_order=[3, 2, 1], device="cpu")
    assert recorded.options == SegmentOptions(crop_size=512, blur_kernel_size=5, band_order=[3, 2, 1], device="cpu")
    assert model.device == "cpu"


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ({"crop_size": 511}, "Tile size must be an even number"),
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
