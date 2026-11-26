"""Run `hab segment` end to end, with a tiny ONNX model on the CPU, and check the raster it writes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio

from habitat_mapper import main
from habitat_mapper.registry import ModelRegistry
from tests.helpers import CRS, tiny_model_config, write_raster, write_tiny_model

if TYPE_CHECKING:
    from pathlib import Path

MODEL_NAME = "tiny"


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register_model(tiny_model_config(write_tiny_model(tmp_path / "model" / "tiny.onnx"), MODEL_NAME))
    monkeypatch.setattr(main, "model_registry", registry)
    return registry


def _segment(*args: str) -> None:
    # The CLI exits when the command finishes, with 0 for success
    with pytest.raises(SystemExit) as exited:
        main.app(["segment", "--model", MODEL_NAME, "--device", "cpu", "--crop-size", "32", "--quiet", *args])
    assert exited.value.code == 0


@pytest.mark.parametrize(("band_order", "brighter_first"), [([], True), (["--band", "2", "--band", "1"], False)])
def test_segment_writes_the_models_classes(
    registry: ModelRegistry, tmp_path: Path, band_order: list[str], brighter_first: bool
) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(4, 45, 70), dtype=np.uint8)
    write_raster(tmp_path / "input.tif", image)
    if band_order:
        band_order = [*band_order, "--band", "3"]

    _segment("--input", str(tmp_path / "input.tif"), "--output", str(tmp_path / "output.tif"), *band_order)

    first, second = (image[0], image[1]) if brighter_first else (image[1], image[0])
    with rasterio.open(tmp_path / "output.tif") as src:
        np.testing.assert_array_equal(src.read(1), (first > second).astype(np.uint8))
        assert src.crs.to_string() == CRS
