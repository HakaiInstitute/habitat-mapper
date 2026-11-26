"""Tests for how ONNXModel and the legacy kelp models turn model outputs into class labels, and read the tile size."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from habitat_mapper import model as model_module
from habitat_mapper.config import ModelConfig
from habitat_mapper.model import LegacyKelpRGBIModel, LegacyKelpRGBModel, ONNXModel
from habitat_mapper.utils import sigmoid, softmax

if TYPE_CHECKING:
    from pathlib import Path


def _config(**changes: Any) -> ModelConfig:  # noqa: ANN401
    return ModelConfig(name="test", revision="20250101", dependencies=[], model_filename="model.onnx", **changes)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_softmax_and_sigmoid_survive_huge_scores(dtype: type) -> None:
    scores = np.array([[1000.0, -1000.0], [-1000.0, 1000.0], [0.0, 0.0]], dtype=dtype).T

    probs = softmax(scores, axis=0, keepdims=True)

    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs, [[1, 0, 0.5], [0, 1, 0.5]])
    np.testing.assert_allclose(sigmoid(np.array([1000.0, -1000.0, 0.0], dtype=dtype)), [1, 0, 0.5])


def test_softmax_and_sigmoid_match_their_formulas() -> None:
    x = np.random.default_rng(0).normal(scale=5, size=(4, 6, 6))
    np.testing.assert_allclose(softmax(x, axis=0, keepdims=True), np.exp(x) / np.exp(x).sum(axis=0, keepdims=True))
    np.testing.assert_allclose(sigmoid(x), 1 / (1 + np.exp(-x)))


@pytest.mark.parametrize("activation", ["softmax", "sigmoid", None])
def test_postprocess_picks_the_highest_scoring_class(activation: str | None) -> None:
    model = ONNXModel(_config(activation=activation))
    scores = np.random.default_rng(0).normal(scale=50, size=(4, 5, 7)).astype(np.float32)
    expected = np.argmax(scores, axis=0)

    labels = model.postprocess(scores)
    np.testing.assert_array_equal(labels, expected)
    assert labels.dtype == np.uint8
    np.testing.assert_array_equal(model.postprocess(np.stack([scores, scores])), [expected, expected])


@pytest.mark.parametrize(("activation", "threshold"), [("sigmoid", 0.0), (None, 0.5)])
def test_postprocess_of_one_score_is_presence(activation: str | None, threshold: float) -> None:
    # One output channel is the chance of the class, so the label is 1 where it is more likely than not
    model = ONNXModel(_config(activation=activation))
    scores = np.array([[[-3.0, threshold - 0.01, threshold + 0.01, 3.0]]], dtype=np.float32)

    labels = model.postprocess(scores)
    np.testing.assert_array_equal(labels, [[0, 0, 1, 1]])
    assert labels.dtype == np.uint8


def test_legacy_rgb_model_labels() -> None:
    # Channel 0 is presence (a logit), channels 1-2 score macrocystis and nereocystis
    scores = np.array(
        [
            [[-2.0, -2.0], [2.0, 2.0], [2.0, -2.0]],
            [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0]],
            [[-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0]],
        ],
        dtype=np.float32,
    )

    labels = LegacyKelpRGBModel(_config()).postprocess(scores)

    # 0 background, 1 macrocystis, 2 nereocystis. 3 rows, unlike the channels, so a mix-up of axes shows
    np.testing.assert_array_equal(labels, [[0, 0], [1, 2], [2, 0]])
    assert labels.dtype == np.uint8


def test_legacy_rgbi_model_labels() -> None:
    # Channels 0-1 score background and kelp, channels 2-3 score macrocystis and nereocystis
    scores = np.array(
        [
            [[3.0, 3.0], [-3.0, -3.0], [-3.0, 3.0]],
            [[-3.0, -3.0], [3.0, 3.0], [3.0, -3.0]],
            [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0]],
            [[-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0]],
        ],
        dtype=np.float32,
    )

    labels = LegacyKelpRGBIModel(_config()).postprocess(scores)

    np.testing.assert_array_equal(labels, [[0, 0], [1, 2], [2, 0]])
    assert labels.dtype == np.uint8


def _model_with_input_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shape: list[Any]) -> ONNXModel:
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="input", shape=shape)],
        get_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(model_module, "create_session", lambda *args, **kwargs: session)
    cache = SimpleNamespace(fetch=lambda cfg, progress=None: tmp_path / "model.onnx")
    return ONNXModel(_config(), cache=cache)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("shape", "size"),
    [
        ([1, 3, 224, 224], 224),
        (["batch", 3, 224, "width"], 224),
        ([1, 3, "height", 224], 224),
        ([1, 3, "height", "width"], None),
        # ONNX Runtime reports dimensions exported without a name as None
        ([None, 3, None, None], None),
        ([1, 3, None, 512], 512),
    ],
)
def test_input_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shape: list[Any], size: int | None) -> None:
    assert _model_with_input_shape(monkeypatch, tmp_path, shape).input_size == size


@pytest.mark.parametrize("shape", [[1, 3, 224, 256], [1, 3, 224], [1, 3]])
def test_input_size_rejects_unusable_shapes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shape: list[Any]) -> None:
    with pytest.raises(ValueError, match="input shape"):
        _ = _model_with_input_shape(monkeypatch, tmp_path, shape).input_size
