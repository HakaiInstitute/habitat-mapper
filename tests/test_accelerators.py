"""Tests for hardware accelerator discovery and execution provider selection."""

from __future__ import annotations

import subprocess  # noqa: S404
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from habitat_mapper import accelerators
from habitat_mapper.accelerators import find_accelerators, select_accelerators

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    Machine = Callable[..., list[bool]]


def _ep_device(ep_name: str, vendor: str, kind: str) -> SimpleNamespace:
    return SimpleNamespace(ep_name=ep_name, device=SimpleNamespace(vendor=vendor, type=SimpleNamespace(name=kind)))


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    """Fake the hardware and ONNX Runtime build seen by find_accelerators.

    Yields:
        A function that configures the fake machine and returns a list that records CUDA library loads.
    """
    cuda_loaded: list[bool] = []

    def configure(
        *,
        ep_devices: tuple[SimpleNamespace, ...] = (),
        available: tuple[str, ...] = ("CPUExecutionProvider",),
        nvidia: bool = False,
        webgpu: bool = False,
    ) -> list[bool]:
        devices = [_ep_device("CPUExecutionProvider", "Acme", "CPU"), *ep_devices]
        monkeypatch.setattr(accelerators.ort, "get_ep_devices", lambda: devices)
        monkeypatch.setattr(accelerators.ort, "get_available_providers", lambda: list(available))
        monkeypatch.setattr(accelerators, "has_nvidia_driver", lambda: nvidia)
        monkeypatch.setattr(accelerators, "_load_cuda_libraries", lambda: cuda_loaded.append(True))
        monkeypatch.setattr(accelerators, "_register_webgpu", lambda: "WebGpuExecutionProvider" if webgpu else None)
        return cuda_loaded

    monkeypatch.delenv(accelerators.DEVICE_ENV_VAR, raising=False)
    find_accelerators.cache_clear()
    yield configure
    find_accelerators.cache_clear()


def _devices(candidates: list[accelerators.Accelerator]) -> list[str]:
    return [a.device for a in candidates]


def test_gpu_build_without_nvidia_driver_uses_cpu_only(machine: Machine) -> None:
    # The onnxruntime-gpu wheel reports CUDA as available even on machines without an NVIDIA GPU (#155)
    cuda_loaded = machine(available=("CUDAExecutionProvider", "CPUExecutionProvider"), nvidia=False)
    assert _devices(select_accelerators()) == ["cpu"]
    assert not cuda_loaded


def test_nvidia_gpu_is_preferred(machine: Machine) -> None:
    cuda_loaded = machine(
        ep_devices=(_ep_device("CUDAExecutionProvider", "NVIDIA", "GPU"),),
        nvidia=True,
        webgpu=True,
    )
    candidates = select_accelerators()
    assert _devices(candidates) == ["cuda", "cpu"]
    assert candidates[0].hardware == "NVIDIA GPU"
    assert cuda_loaded


def test_cuda_selected_by_name_when_not_listed_as_a_device(machine: Machine) -> None:
    machine(available=("CUDAExecutionProvider", "CPUExecutionProvider"), nvidia=True)
    candidates = select_accelerators("cuda")
    assert _devices(candidates) == ["cuda", "cpu"]
    assert candidates[0].session_kwargs()["providers"] == [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]


def test_nvidia_driver_without_cuda_libraries_warns(machine: Machine, capsys: pytest.CaptureFixture[str]) -> None:
    from loguru import logger  # noqa: PLC0415

    handler = logger.add(sys.stderr, level="WARNING")
    try:
        machine(nvidia=True)
        assert _devices(select_accelerators()) == ["cpu"]
    finally:
        logger.remove(handler)
    assert "habitat-mapper[cuda]" in capsys.readouterr().err


def test_coreml_is_only_used_when_requested(machine: Machine) -> None:
    machine(ep_devices=(_ep_device("CoreMLExecutionProvider", "Apple", "GPU"),))
    assert _devices(select_accelerators("auto")) == ["cpu"]
    assert _devices(select_accelerators("coreml")) == ["coreml", "cpu"]


def test_webgpu_is_used_when_installed(machine: Machine) -> None:
    machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "AMD", "GPU"),), webgpu=True)
    assert _devices(select_accelerators()) == ["webgpu", "cpu"]


def test_webgpu_without_a_gpu_is_skipped(machine: Machine) -> None:
    machine(webgpu=True)
    assert _devices(select_accelerators()) == ["cpu"]


def test_nvidia_without_cuda_libraries_is_quiet_when_webgpu_is_used(
    machine: Machine, capsys: pytest.CaptureFixture[str]
) -> None:
    from loguru import logger  # noqa: PLC0415

    handler = logger.add(sys.stderr, level="WARNING")
    try:
        machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "NVIDIA", "GPU"),), nvidia=True, webgpu=True)
        assert _devices(select_accelerators()) == ["webgpu", "cpu"]
    finally:
        logger.remove(handler)
    assert "habitat-mapper[cuda]" not in capsys.readouterr().err


def test_cpu_request_skips_accelerators(machine: Machine) -> None:
    machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "AMD", "GPU"),), webgpu=True)
    assert _devices(select_accelerators("cpu")) == ["cpu"]


def test_device_read_from_environment(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "AMD", "GPU"),), webgpu=True)
    monkeypatch.setenv(accelerators.DEVICE_ENV_VAR, "CPU")
    assert _devices(select_accelerators()) == ["cpu"]


def test_unavailable_device_names_the_extra_to_install(machine: Machine) -> None:
    machine()
    with pytest.raises(RuntimeError, match=r"habitat-mapper\[webgpu\]"):
        select_accelerators("webgpu")


def test_unknown_device_is_rejected(machine: Machine) -> None:
    machine()
    with pytest.raises(ValueError, match="Unknown device"):
        select_accelerators("tpu")


def test_failed_accelerator_falls_back_to_cpu(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "AMD", "GPU"),), webgpu=True)
    created = []

    def fake_session(path: str, **kwargs: object) -> str:
        if "providers" not in kwargs:
            raise RuntimeError("WebGPU device lost\nmore detail")
        created.append(kwargs["providers"])
        return "session"

    monkeypatch.setattr(accelerators.ort, "InferenceSession", fake_session)
    monkeypatch.setattr(accelerators.Accelerator, "session_kwargs", _fake_session_kwargs)
    assert accelerators.create_session("model.onnx") == "session"
    assert created == [["CPUExecutionProvider"]]


def _fake_session_kwargs(self: accelerators.Accelerator) -> dict[str, object]:
    return {"providers": [self.ep_name]} if self.device == "cpu" else {"sess_options": None}


def test_importing_model_does_not_load_cuda() -> None:
    script = (
        "import onnxruntime as ort\n"
        "ort.preload_dlls = lambda *a, **k: (_ for _ in ()).throw(AssertionError('preload_dlls called'))\n"
        "import habitat_mapper.model\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
