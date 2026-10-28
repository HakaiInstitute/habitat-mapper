"""Tests for hardware accelerator discovery and execution provider selection."""

from __future__ import annotations

import os
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
        cudnn: bool = True,
        webgpu: bool = False,
        hardware_gpu: bool = True,
    ) -> list[bool]:
        devices = [_ep_device("CPUExecutionProvider", "Acme", "CPU"), *ep_devices]
        monkeypatch.setattr(accelerators.ort, "get_ep_devices", lambda: devices)
        monkeypatch.setattr(accelerators.ort, "get_available_providers", lambda: list(available))
        monkeypatch.setattr(accelerators, "has_nvidia_driver", lambda: nvidia)
        monkeypatch.setattr(accelerators, "has_cudnn", lambda: cudnn)
        monkeypatch.setattr(accelerators, "_load_cuda_libraries", lambda: cuda_loaded.append(True))
        monkeypatch.setattr(accelerators, "_register_webgpu", lambda: "WebGpuExecutionProvider" if webgpu else None)
        monkeypatch.setattr(accelerators, "has_hardware_gpu_for_webgpu", lambda: hardware_gpu)
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


def test_cuda_toolkit_without_cudnn_is_skipped(machine: Machine) -> None:
    # A system CUDA toolkit lets the CUDA provider create a session, but it fails on the first convolution without cuDNN
    machine(available=("CUDAExecutionProvider", "CPUExecutionProvider"), nvidia=True, cudnn=False, webgpu=True)
    assert _devices(select_accelerators()) == ["cpu"]
    with pytest.raises(RuntimeError, match=r"habitat-mapper\[cuda\]"):
        select_accelerators("cuda")


def test_cuda_toolkit_without_cudnn_uses_webgpu(machine: Machine) -> None:
    machine(
        ep_devices=(_ep_device("WebGpuExecutionProvider", "NVIDIA", "GPU"),),
        available=("CUDAExecutionProvider", "CPUExecutionProvider"),
        nvidia=True,
        cudnn=False,
        webgpu=True,
    )
    assert _devices(select_accelerators()) == ["webgpu", "cpu"]


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


class _FakeSession:
    def __init__(self, providers: list[str]) -> None:
        self.providers = providers

    def get_providers(self) -> list[str]:
        return self.providers


def _fake_session_kwargs(self: accelerators.Accelerator) -> dict[str, object]:
    return {"providers": [self.ep_name]}


def test_failed_accelerator_falls_back_to_cpu(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "AMD", "GPU"),), webgpu=True)

    def fake_session(path: str, providers: list[str]) -> _FakeSession:
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError("WebGPU device lost")
        return _FakeSession(providers)

    monkeypatch.setattr(accelerators.ort, "InferenceSession", fake_session)
    monkeypatch.setattr(accelerators.Accelerator, "session_kwargs", _fake_session_kwargs)
    assert accelerators.create_session("model.onnx").get_providers() == ["CPUExecutionProvider"]


def test_session_that_silently_lands_on_cpu_counts_as_a_failure(
    machine: Machine, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # E.g. WebGPU without a Vulkan driver, where ONNX Runtime used to fall back to the CPU internally
    machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "AMD", "GPU"),), webgpu=True)
    created = []

    def fake_session(path: str, providers: list[str]) -> _FakeSession:
        created.append(providers)
        return _FakeSession(["CPUExecutionProvider"])

    monkeypatch.setattr(accelerators.ort, "InferenceSession", fake_session)
    monkeypatch.setattr(accelerators.Accelerator, "session_kwargs", _fake_session_kwargs)
    from loguru import logger  # noqa: PLC0415

    handler = logger.add(sys.stderr, level="WARNING")
    try:
        accelerators.create_session("model.onnx")
    finally:
        logger.remove(handler)
    assert created == [["WebGpuExecutionProvider"], ["CPUExecutionProvider"]]
    assert "Could not run the model on webgpu" in capsys.readouterr().err


def test_accelerator_sessions_disable_ort_internal_fallback(machine: Machine) -> None:
    machine(available=("CUDAExecutionProvider", "CPUExecutionProvider"), nvidia=True)
    assert _devices(list(find_accelerators())) == ["cuda", "cpu"]
    for accelerator in find_accelerators():
        assert accelerator.session_kwargs()["enable_fallback"] == 0


def test_webgpu_on_a_software_renderer_is_skipped(machine: Machine, capsys: pytest.CaptureFixture[str]) -> None:
    from loguru import logger  # noqa: PLC0415

    handler = logger.add(sys.stderr, level="WARNING")
    try:
        machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "NVIDIA", "GPU"),), webgpu=True, hardware_gpu=False)
        assert _devices(select_accelerators()) == ["cpu"]
    finally:
        logger.remove(handler)
    assert "no Vulkan driver for your GPU" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("types", "expected"),
    [([], False), ([accelerators._VK_PHYSICAL_DEVICE_TYPE_CPU], False), ([4, 2], True), ([1], True)],
)
def test_hardware_gpu_check_ignores_software_vulkan_devices(
    monkeypatch: pytest.MonkeyPatch, types: list[int], expected: bool
) -> None:
    monkeypatch.setattr(accelerators.sys, "platform", "linux")
    monkeypatch.setattr(accelerators, "_vulkan_device_types", lambda: types)
    assert accelerators.has_hardware_gpu_for_webgpu() is expected


def test_native_stderr_is_logged_at_debug_level(capfd: pytest.CaptureFixture[str]) -> None:
    from loguru import logger  # noqa: PLC0415

    messages: list[str] = []
    handler = logger.add(messages.append, level="DEBUG", format="{level} {message}")
    try:
        with accelerators._capture_native_stderr("a test library"):
            os.write(2, b"Warning: maxDynamicUniformBuffersPerPipelineLayout artificially reduced\n")
    finally:
        logger.remove(handler)
    assert "maxDynamic" not in capfd.readouterr().err
    assert len(messages) == 1
    assert messages[0].startswith("DEBUG Output from a test library:")


def test_importing_model_does_not_load_cuda() -> None:
    script = (
        "import onnxruntime as ort\n"
        "ort.preload_dlls = lambda *a, **k: (_ for _ in ()).throw(AssertionError('preload_dlls called'))\n"
        "import habitat_mapper.model\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
