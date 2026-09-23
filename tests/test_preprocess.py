"""Tests for ONNXModel input preprocessing."""

from typing import Any

import numpy as np
import pytest

from habitat_mapper.config import ModelConfig
from habitat_mapper.model import ONNXModel

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _model(normalization: Any, max_pixel_value: Any = "auto") -> ONNXModel:  # noqa: ANN401
    config = ModelConfig(
        name="test",
        revision="20250101",
        dependencies=["./model.onnx"],
        model_filename="model.onnx",
        normalization=normalization,
        mean=MEAN,
        std=STD,
        max_pixel_value=max_pixel_value,
    )
    return ONNXModel(config)


def _reference(batch: np.ndarray, normalization: str | None, max_pixel_value: float) -> np.ndarray:
    """Float64 reference implementation of the preprocessing formulas.

    Returns:
        The normalized batch in float64.
    """
    x = batch.astype(np.float64) / max_pixel_value
    if normalization == "standard":
        return (x - np.array(MEAN)[None, :, None, None]) / np.array(STD)[None, :, None, None]
    if normalization == "min_max":
        bmin, bmax = x.min(axis=(1, 2, 3), keepdims=True), x.max(axis=(1, 2, 3), keepdims=True)
        return (x - bmin) / (bmax - bmin + 1e-8)
    if normalization == "min_max_per_channel":
        bmin, bmax = x.min(axis=(2, 3), keepdims=True), x.max(axis=(2, 3), keepdims=True)
        return (x - bmin) / (bmax - bmin + 1e-8)
    return x


@pytest.mark.parametrize("normalization", ["standard", "min_max", "min_max_per_channel", None])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_preprocess_matches_reference(normalization: str | None, dtype: type) -> None:
    rng = np.random.default_rng(0)
    batch = rng.integers(0, np.iinfo(dtype).max, size=(2, 3, 16, 16), endpoint=True, dtype=dtype)

    out = _model(normalization)._preprocess(batch)

    assert out.dtype == np.float32
    assert out.shape == batch.shape
    np.testing.assert_allclose(out, _reference(batch, normalization, np.iinfo(dtype).max), rtol=1e-5, atol=1e-5)


def test_preprocess_does_not_modify_input() -> None:
    batch = np.full((1, 3, 4, 4), 100, dtype=np.float32)
    _model("standard", max_pixel_value=255.0)._preprocess(batch)
    assert np.all(batch == 100)
