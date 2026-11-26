"""Tests for falling back to the next device when a model fails on its first batch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from habitat_mapper import model as model_module
from habitat_mapper.config import ModelConfig
from habitat_mapper.model import ONNXModel

if TYPE_CHECKING:
    from collections.abc import Collection
    from pathlib import Path


class _Session:
    def __init__(self, provider: str, fail_on_run: int | None) -> None:
        self.provider = provider
        self.fail_on_run = fail_on_run
        self.runs = 0

    def get_providers(self) -> list[str]:
        return [self.provider, "CPUExecutionProvider"]

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input", shape=[1, 3, 8, 8])]

    def run(self, output_names: None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.runs += 1
        if self.runs == self.fail_on_run:
            raise RuntimeError(f"{self.provider} failed")
        return [np.zeros((1, 2, 8, 8), dtype=np.float32)]


class _Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fetches = 0

    def fetch(self, cfg: ModelConfig, progress: object = None) -> Path:
        self.fetches += 1
        return self.path


@pytest.fixture
def cache(tmp_path: Path) -> _Cache:
    return _Cache(tmp_path / "model.onnx")


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch, cache: _Cache) -> ONNXModel:
    monkeypatch.setattr(model_module, "describe_provider", lambda provider: provider)
    cfg = ModelConfig(name="stub", revision="20250101", dependencies=[], model_filename="model.onnx")
    return ONNXModel(cfg, cache=cache)  # type: ignore[arg-type]


def _use_sessions(monkeypatch: pytest.MonkeyPatch, sessions: list[_Session]) -> list[set[str]]:
    """Make create_session return the first of the sessions whose provider isn't skipped.

    Returns:
        The providers skipped on each call.
    """
    skips: list[set[str]] = []

    def fake_create_session(path: Path, device: str | None = None, *, skip: Collection[str] = ()) -> _Session:
        skips.append(set(skip))
        return next(s for s in sessions if s.provider not in skip)

    monkeypatch.setattr(model_module, "create_session", fake_create_session)
    return skips


def test_first_batch_failure_retries_on_the_next_device(model: ONNXModel, monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = _Session("CUDAExecutionProvider", fail_on_run=1)
    webgpu = _Session("WebGpuExecutionProvider", fail_on_run=None)
    skips = _use_sessions(monkeypatch, [cuda, webgpu])
    batch = np.zeros((1, 3, 8, 8), dtype=np.uint8)
    assert model.predict(batch, 255).shape == (1, 2, 8, 8)
    assert model.predict(batch, 255).shape == (1, 2, 8, 8)
    assert skips == [set(), {"CUDAExecutionProvider"}]
    assert (cuda.runs, webgpu.runs) == (1, 2)


def test_retry_does_not_load_the_model_file_again(
    model: ONNXModel, monkeypatch: pytest.MonkeyPatch, cache: _Cache
) -> None:
    _use_sessions(
        monkeypatch,
        [_Session("CUDAExecutionProvider", fail_on_run=1), _Session("WebGpuExecutionProvider", fail_on_run=None)],
    )
    model.predict(np.zeros((1, 3, 8, 8), dtype=np.uint8), 255)
    assert cache.fetches == 1


def test_changing_device_loads_a_new_session(model: ONNXModel, monkeypatch: pytest.MonkeyPatch) -> None:
    skips = _use_sessions(monkeypatch, [_Session("CPUExecutionProvider", fail_on_run=None)])
    batch = np.zeros((1, 3, 8, 8), dtype=np.uint8)
    model.predict(batch, 255)
    model.predict(batch, 255)
    model.device = "cpu"
    model.predict(batch, 255)
    assert len(skips) == 2


def test_failure_after_the_first_batch_is_raised(model: ONNXModel, monkeypatch: pytest.MonkeyPatch) -> None:
    # Later failures, e.g. running out of memory, don't mean the device can't run the model
    _use_sessions(monkeypatch, [_Session("CUDAExecutionProvider", fail_on_run=2)])
    batch = np.zeros((1, 3, 8, 8), dtype=np.uint8)
    model.predict(batch, 255)
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider failed"):
        model.predict(batch, 255)


def test_cpu_failure_is_raised(model: ONNXModel, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_sessions(monkeypatch, [_Session("CPUExecutionProvider", fail_on_run=1)])
    with pytest.raises(RuntimeError, match="CPUExecutionProvider failed"):
        model.predict(np.zeros((1, 3, 8, 8), dtype=np.uint8), 255)
