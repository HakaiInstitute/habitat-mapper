"""Run `hab segment` end to end, with a tiny ONNX model on the CPU, and check the raster it writes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import pytest
import rasterio
from onnx import TensorProto, helper, numpy_helper

from habitat_mapper import main
from habitat_mapper.config import ModelConfig
from habitat_mapper.registry import ModelRegistry
from tests.helpers import CRS, write_raster

if TYPE_CHECKING:
    from pathlib import Path

MODEL_NAME = "tiny"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 1x1 convolution that scores class 1 where the first band is brighter than the second, else class 0.

    Its height and width are left unnamed, which ONNX Runtime reports as None.

    Returns:
        The model's path.
    """
    # Class 0 scores band 2 - band 1, class 1 scores band 1 - band 2
    weights = numpy_helper.from_array(np.array([[-1, 1, 0], [1, -1, 0]], dtype=np.float32).reshape(2, 3, 1, 1), "w")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "w"], ["output"])],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, None, None])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2, None, None])],
        initializer=[weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("model") / "tiny.onnx"
    onnx.save(model, path)
    return path


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch, tiny_model: Path) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register_model(
        ModelConfig(
            name=MODEL_NAME,
            revision="20250101",
            dependencies=[str(tiny_model)],
            model_filename=tiny_model.name,
            # The same for every band, so normalizing keeps whichever band is brighter
            mean=(0.5, 0.5, 0.5),
            std=(0.25, 0.25, 0.25),
            default_output_value=0,
            nodata_value=255,
        )
    )
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
