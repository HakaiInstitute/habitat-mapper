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
    from pathlib import Path

    Machine = Callable[..., list[bool]]


def _ep_device(ep_name: str, vendor: str, kind: str, vendor_id: int = 0, device_id: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        ep_name=ep_name,
        device=SimpleNamespace(
            vendor=vendor, vendor_id=vendor_id, device_id=device_id, type=SimpleNamespace(name=kind)
        ),
    )


def _log_output(level: str, action: Callable[[], object]) -> list[str]:
    from loguru import logger  # noqa: PLC0415

    messages: list[str] = []
    handler = logger.add(messages.append, level=level, format="{level} {message}")
    try:
        action()
    finally:
        logger.remove(handler)
    return [m.rstrip("\n") for m in messages]


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
        driver_cuda: int = 13,
        compute_capability: tuple[int, int] | None = (8, 9),
        ort_cuda: str = "13.0",
        cudnn: bool = True,
        cuda_wheels: bool | None = None,
        webgpu: bool = False,
        hardware_gpu: bool = True,
    ) -> list[bool]:
        devices = [_ep_device("CPUExecutionProvider", "Acme", "CPU"), *ep_devices]
        monkeypatch.setattr(accelerators.ort, "get_ep_devices", lambda: devices)
        monkeypatch.setattr(accelerators.ort, "get_available_providers", lambda: list(available))
        monkeypatch.setattr(accelerators, "nvidia_driver_cuda_version", lambda: driver_cuda if nvidia else None)
        monkeypatch.setattr(accelerators, "nvidia_compute_capability", lambda: compute_capability if nvidia else None)
        monkeypatch.setattr(accelerators.ort, "cuda_version", ort_cuda, raising=False)
        monkeypatch.setattr(accelerators, "has_cudnn", lambda: cudnn)
        installed = cudnn if cuda_wheels is None else cuda_wheels
        monkeypatch.setattr(accelerators, "_cuda_libraries_installed", lambda: installed)
        monkeypatch.setattr(accelerators, "_load_cuda_libraries", lambda: cuda_loaded.append(True))
        monkeypatch.setattr(accelerators, "_register_webgpu", lambda: "WebGpuExecutionProvider" if webgpu else None)
        monkeypatch.setattr(accelerators, "has_hardware_gpu_for_webgpu", lambda _devices: hardware_gpu)
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


def test_unavailable_device_explains_what_is_needed(machine: Machine) -> None:
    machine()
    with pytest.raises(RuntimeError, match=r"habitat-mapper\[cuda12\]"):
        select_accelerators("cuda")
    with pytest.raises(RuntimeError, match="DirectX 12"):
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
    assert "graphics driver isn't installed" in capsys.readouterr().err


def test_nvidia_without_cuda_or_vulkan_gets_one_warning(machine: Machine) -> None:
    machine(
        ep_devices=(_ep_device("WebGpuExecutionProvider", "NVIDIA", "GPU"),),
        nvidia=True,
        webgpu=True,
        hardware_gpu=False,
    )
    messages = _log_output("WARNING", select_accelerators)
    assert len(messages) == 1
    assert "habitat-mapper[cuda]" in messages[0]


def test_nvidia_on_webgpu_is_told_cuda_is_faster(machine: Machine) -> None:
    machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "NVIDIA", "GPU"),), nvidia=True, webgpu=True)
    messages = _log_output("INFO", select_accelerators)
    assert messages == [
        "INFO Install the CUDA libraries with `pip install 'habitat-mapper[cuda]'` for faster processing on "
        "NVIDIA GPUs."
    ]


def test_hardware_is_named_from_the_vendor_id_when_ort_leaves_it_blank(machine: Machine) -> None:
    machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "", "GPU", vendor_id=0x10DE),), webgpu=True)
    assert select_accelerators()[0].hardware == "NVIDIA GPU"


def test_driver_too_old_for_installed_cuda_points_at_cuda12(machine: Machine) -> None:
    # onnxruntime-gpu 1.27+ is built for CUDA 13, which needs NVIDIA driver R580 or newer
    cuda_loaded = machine(
        available=("CUDAExecutionProvider", "CPUExecutionProvider"),
        nvidia=True,
        driver_cuda=12,
        cudnn=False,
        cuda_wheels=True,
    )
    messages = _log_output("WARNING", select_accelerators)
    assert _devices(select_accelerators()) == ["cpu"]
    assert not cuda_loaded
    assert len(messages) == 1
    assert "R580" in messages[0]
    assert "habitat-mapper[cuda12]" in messages[0]


def test_cuda12_libraries_work_with_an_older_driver(machine: Machine) -> None:
    machine(available=("CUDAExecutionProvider", "CPUExecutionProvider"), nvidia=True, driver_cuda=12, ort_cuda="12.9")
    assert _devices(select_accelerators()) == ["cuda", "cpu"]


def test_older_driver_on_webgpu_is_pointed_at_cuda12(machine: Machine) -> None:
    machine(
        ep_devices=(_ep_device("WebGpuExecutionProvider", "NVIDIA", "GPU"),),
        nvidia=True,
        driver_cuda=12,
        cudnn=False,
        webgpu=True,
    )
    messages = _log_output("INFO", select_accelerators)
    assert len(messages) == 1
    assert "habitat-mapper[cuda12]" in messages[0]


def test_driver_too_old_for_any_cuda_asks_for_a_driver_update(machine: Machine) -> None:
    machine(available=("CUDAExecutionProvider", "CPUExecutionProvider"), nvidia=True, driver_cuda=11)
    messages = _log_output("WARNING", select_accelerators)
    assert _devices(select_accelerators()) == ["cpu"]
    assert messages == [
        "WARNING An NVIDIA GPU was found, but its driver is too old for CUDA. Update the driver to use CUDA."
    ]


def test_cpu_backed_webgpu_devices_are_ignored(machine: Machine) -> None:
    # ORT offers one when ORT_WEBGPU_EP_ALLOW_SOFTWARE_ADAPTER is set
    machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "Microsoft", "CPU"),), webgpu=True)
    assert _devices(select_accelerators()) == ["cpu"]


def test_missing_vulkan_driver_is_not_mentioned_when_cuda_is_used(machine: Machine) -> None:
    machine(
        ep_devices=(
            _ep_device("CUDAExecutionProvider", "NVIDIA", "GPU"),
            _ep_device("WebGpuExecutionProvider", "NVIDIA", "GPU"),
        ),
        nvidia=True,
        webgpu=True,
        hardware_gpu=False,
    )
    messages = _log_output("WARNING", select_accelerators)
    assert _devices(select_accelerators()) == ["cuda", "cpu"]
    assert messages == []


def test_cuda_request_explains_a_driver_that_is_too_old(machine: Machine) -> None:
    machine(
        available=("CUDAExecutionProvider", "CPUExecutionProvider"),
        nvidia=True,
        driver_cuda=12,
        cudnn=False,
        cuda_wheels=True,
    )
    with pytest.raises(RuntimeError, match="only supports CUDA 12"):
        select_accelerators("cuda")


def test_pascal_gpu_with_cuda13_libraries_is_pointed_at_cuda12(machine: Machine) -> None:
    # CUDA 13 dropped GPUs older than Turing (compute capability 7.5), though their R580 drivers report CUDA 13
    cuda_loaded = machine(
        available=("CUDAExecutionProvider", "CPUExecutionProvider"),
        nvidia=True,
        compute_capability=(6, 1),
        cudnn=False,
        cuda_wheels=True,
    )
    messages = _log_output("WARNING", select_accelerators)
    assert _devices(select_accelerators()) == ["cpu"]
    assert not cuda_loaded
    assert len(messages) == 1
    assert "compute capability 6.1" in messages[0]
    assert "habitat-mapper[cuda12]" in messages[0]


def test_pascal_gpu_on_webgpu_is_pointed_at_cuda12(machine: Machine) -> None:
    machine(
        ep_devices=(_ep_device("WebGpuExecutionProvider", "NVIDIA", "GPU"),),
        nvidia=True,
        compute_capability=(6, 1),
        cudnn=False,
        webgpu=True,
    )
    messages = _log_output("INFO", select_accelerators)
    assert len(messages) == 1
    assert "habitat-mapper[cuda12]" in messages[0]


def test_pascal_gpu_runs_cuda12(machine: Machine) -> None:
    machine(
        available=("CUDAExecutionProvider", "CPUExecutionProvider"),
        nvidia=True,
        compute_capability=(6, 1),
        ort_cuda="12.9",
    )
    assert _devices(select_accelerators()) == ["cuda", "cpu"]


def test_unknown_compute_capability_does_not_block_cuda(machine: Machine) -> None:
    machine(available=("CUDAExecutionProvider", "CPUExecutionProvider"), nvidia=True, compute_capability=None)
    assert _devices(select_accelerators()) == ["cuda", "cpu"]


@pytest.mark.parametrize(
    ("adapters", "expected"),
    [
        ([(0x10DE, 0x26B1, False), (0x1414, 0x8C, True)], True),
        ([(0x1414, 0x8C, True)], False),  # Only WARP: the GPU has no driver
        ([], True),  # DXGI couldn't be queried
    ],
)
def test_windows_webgpu_needs_a_matching_hardware_adapter(
    monkeypatch: pytest.MonkeyPatch, adapters: list[tuple[int, int, bool]], expected: bool
) -> None:
    monkeypatch.setattr(accelerators.sys, "platform", "win32")
    monkeypatch.setattr(accelerators, "_dxgi_adapters", lambda: adapters)
    webgpu_devices = [_ep_device("WebGpuExecutionProvider", "NVIDIA", "GPU", vendor_id=0x10DE, device_id=0x26B1)]
    assert accelerators.has_hardware_gpu_for_webgpu(webgpu_devices) is expected


def test_dxgi_query_is_skipped_off_windows() -> None:
    if sys.platform == "win32":
        pytest.skip("checks the non-Windows path")
    assert accelerators._dxgi_adapters() == []


def test_cudnn_folder_is_put_on_path_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cudnn_bin = tmp_path / "nvidia" / "cudnn" / "bin"
    cudnn_bin.mkdir(parents=True)
    monkeypatch.setattr(accelerators.sys, "platform", "win32")
    monkeypatch.setattr(
        accelerators, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(tmp_path / "nvidia")])
    )
    monkeypatch.setenv("PATH", "C:\\Windows")
    accelerators._add_cudnn_to_path()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(cudnn_bin)


def test_create_session_skips_providers_that_already_failed(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> None:
    machine(ep_devices=(_ep_device("WebGpuExecutionProvider", "AMD", "GPU"),), webgpu=True)
    created = []

    def fake_session(path: str, providers: list[str]) -> _FakeSession:
        created.append(providers)
        return _FakeSession(providers)

    monkeypatch.setattr(accelerators.ort, "InferenceSession", fake_session)
    monkeypatch.setattr(accelerators.Accelerator, "session_kwargs", _fake_session_kwargs)
    accelerators.create_session("model.onnx", skip={"WebGpuExecutionProvider"})
    assert created == [["CPUExecutionProvider"]]


@pytest.mark.parametrize(
    ("types", "expected"),
    [([], False), ([accelerators._VK_PHYSICAL_DEVICE_TYPE_CPU], False), ([4, 2], True), ([1], True)],
)
def test_hardware_gpu_check_ignores_software_vulkan_devices(
    monkeypatch: pytest.MonkeyPatch, types: list[int], expected: bool
) -> None:
    monkeypatch.setattr(accelerators.sys, "platform", "linux")
    monkeypatch.setattr(accelerators, "_vulkan_device_types", lambda: types)
    assert accelerators.has_hardware_gpu_for_webgpu([]) is expected


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
