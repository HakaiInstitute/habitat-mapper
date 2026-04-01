"""Tests for quiet mode behavior in ImageProcessor."""

from __future__ import annotations

from habitat_mapper.config import ProcessingConfig
from habitat_mapper.processing import ImageProcessor
from habitat_mapper.progress import NullProgressReporter, RichProgressReporter


def _make_mock_model(input_channels: int = 3, input_size: int | None = None) -> object:
    from unittest.mock import MagicMock

    model = MagicMock()
    model.cfg.input_channels = input_channels
    model.input_size = input_size
    return model


class TestQuietModeFlag:
    def test_from_model_quiet_defaults_false(self) -> None:
        model = _make_mock_model()
        processor = ImageProcessor.from_model(model)
        assert processor.reporter_cls is RichProgressReporter

    def test_from_model_quiet_true(self) -> None:
        model = _make_mock_model()
        processor = ImageProcessor.from_model(model, quiet=True)
        assert processor.reporter_cls is NullProgressReporter

    def test_from_model_quiet_false_explicit(self) -> None:
        model = _make_mock_model()
        processor = ImageProcessor.from_model(model, quiet=False)
        assert processor.reporter_cls is RichProgressReporter

    def test_dataclass_reporter_cls_field(self) -> None:
        model = _make_mock_model()
        config = ProcessingConfig(crop_size=512, batch_size=1, band_order=[1, 2, 3])
        processor = ImageProcessor(model=model, config=config, reporter_cls=NullProgressReporter)
        assert processor.reporter_cls is NullProgressReporter
