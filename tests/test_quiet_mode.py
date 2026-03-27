"""Tests for quiet mode behavior in ImageProcessor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

from habitat_mapper.config import ProcessingConfig
from habitat_mapper.processing import ImageProcessor


def _make_mock_model(input_channels: int = 3, input_size: int | None = None) -> MagicMock:
    model = MagicMock()
    model.cfg.input_channels = input_channels
    model.input_size = input_size
    return model


class TestQuietModeFlag:
    def test_from_model_quiet_defaults_false(self) -> None:
        model = _make_mock_model()
        processor = ImageProcessor.from_model(model)
        assert processor.quiet is False

    def test_from_model_quiet_true(self) -> None:
        model = _make_mock_model()
        processor = ImageProcessor.from_model(model, quiet=True)
        assert processor.quiet is True

    def test_from_model_quiet_false_explicit(self) -> None:
        model = _make_mock_model()
        processor = ImageProcessor.from_model(model, quiet=False)
        assert processor.quiet is False

    def test_dataclass_quiet_field(self) -> None:
        model = _make_mock_model()
        config = ProcessingConfig(crop_size=512, batch_size=1, band_order=[1, 2, 3])
        processor = ImageProcessor(model=model, config=config, quiet=True)
        assert processor.quiet is True


def _make_mock_model_with_reader(input_channels: int = 3) -> MagicMock:
    """Return a mock model whose cfg.get_reader() yields a reader with integer dimensions."""
    model = _make_mock_model(input_channels=input_channels, input_size=None)

    reader = MagicMock()
    reader.height = 64
    reader.width = 64
    reader.dtype = "uint8"
    reader.crs = None
    reader.transform = None

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=reader)
    ctx.__exit__ = MagicMock(return_value=False)
    model.cfg.get_reader.return_value = ctx
    model.cfg.max_pixel_value = 255
    model.cfg.nodata_value = 0

    return model


class TestQuietModeConsole:
    def test_run_uses_quiet_console_when_quiet(self, tmp_path: Path) -> None:
        """When quiet=True, Progress should be constructed with Console(quiet=True)."""
        model = _make_mock_model_with_reader()
        processor = ImageProcessor.from_model(model, quiet=True)

        dst_ctx = MagicMock()
        dst_ctx.__enter__ = MagicMock(return_value=MagicMock())
        dst_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch("habitat_mapper.processing.Console") as mock_console_cls,
            patch("habitat_mapper.processing.rasterio.open", return_value=dst_ctx),
            # Empty batches → loop body never runs, Progress block still entered
            patch("habitat_mapper.processing.batched", return_value=[]),
            patch.object(processor, "_apply_final_postprocessing"),
        ):
            processor.run("input.tif", tmp_path / "output.tif")

        mock_console_cls.assert_called_once_with(quiet=True)

    def test_run_uses_no_console_when_not_quiet(self, tmp_path: Path) -> None:
        """When quiet=False, Progress should use the default console (None), not a new Console instance."""
        model = _make_mock_model_with_reader()
        processor = ImageProcessor.from_model(model, quiet=False)

        dst_ctx2 = MagicMock()
        dst_ctx2.__enter__ = MagicMock(return_value=MagicMock())
        dst_ctx2.__exit__ = MagicMock(return_value=False)

        with (
            patch("habitat_mapper.processing.Console") as mock_console_cls,
            patch("habitat_mapper.processing.rasterio.open", return_value=dst_ctx2),
            patch("habitat_mapper.processing.batched", return_value=[]),
            patch.object(processor, "_apply_final_postprocessing"),
        ):
            processor.run("input.tif", tmp_path / "output.tif")

        mock_console_cls.assert_not_called()
