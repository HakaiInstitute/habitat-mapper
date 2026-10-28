"""Hardware accelerator discovery and ONNX Runtime execution provider selection."""

from __future__ import annotations

import contextlib
import ctypes
import io
import os
import site
import sys
import warnings
from dataclasses import dataclass, field
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from loguru import logger

with warnings.catch_warnings():
    # onnxruntime-gpu registers its CUDA provider on import and warns when the CUDA libraries can't be loaded.
    # That's expected without an NVIDIA GPU, so CUDA support is checked explicitly in find_accelerators instead.
    warnings.filterwarnings("ignore", message="Failed to register bundled CUDA plugin EP")
    import onnxruntime as ort

from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf, NoSuchFile

if TYPE_CHECKING:
    from collections.abc import Sequence

# 4=FATAL. Errors are raised as exceptions and reported through loguru, so ORT's own error log is redundant
ort.set_default_logger_severity(4)

Device = Literal["auto", "cpu", "cuda", "coreml", "webgpu"]
DEVICES: tuple[str, ...] = get_args(Device)
DEVICE_ENV_VAR = "HAB_DEVICE"

_CPU_EP = "CPUExecutionProvider"
_COREML_EP = "CoreMLExecutionProvider"
_CUDA_EPS = ("NvTensorRtRtxExecutionProvider", "CUDAExecutionProvider")

# CoreML compiles slowly and ran slower than the CPU on our models, so it is only used when requested explicitly
_EXCLUDED_FROM_AUTO = {"coreml"}

_UNAVAILABLE_HINTS = {
    "cuda": (
        "It needs an NVIDIA GPU with a recent driver and the CUDA libraries. "
        "Install them with `pip install 'habitat-mapper[cuda]'` (Windows and Linux only)."
    ),
    "coreml": "CoreML is only available on macOS.",
    "webgpu": "Install the WebGPU plugin with `pip install 'habitat-mapper[webgpu]'`.",
}


@dataclass
class Accelerator:
    """An ONNX Runtime execution provider that can run on this machine.

    Attributes:
        device: User-facing device name, one of DEVICES other than "auto".
        ep_name: ONNX Runtime execution provider name.
        hardware: Description of the hardware the provider runs on.
        ep_devices: Devices reported by ort.get_ep_devices(). Empty for providers that are selected by name.
        options: Execution provider options.
    """

    device: str
    ep_name: str
    hardware: str
    ep_devices: tuple[Any, ...] = ()
    options: dict[str, str] = field(default_factory=dict)

    def session_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for ort.InferenceSession that run a model on this accelerator.

        Returns:
            Keyword arguments for ort.InferenceSession.
        """
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4
        if self.ep_devices:
            sess_options.add_provider_for_devices(list(self.ep_devices), self.options)
            return {"sess_options": sess_options}
        if self.ep_name == _CPU_EP:
            return {"sess_options": sess_options, "providers": [_CPU_EP]}
        return {"sess_options": sess_options, "providers": [(self.ep_name, self.options), _CPU_EP]}


def has_nvidia_driver() -> bool:
    """Check whether an NVIDIA GPU driver is installed.

    Returns:
        True if the CUDA driver library can be loaded.
    """
    library = {"win32": "nvcuda.dll", "linux": "libcuda.so.1"}.get(sys.platform)
    if library is None:
        return False
    try:
        ctypes.CDLL(library)
    except OSError:
        return False
    return True


def has_cudnn() -> bool:
    """Check whether cuDNN can be loaded, either from the [cuda] extra or a system install.

    The CUDA provider can create a session with only the CUDA toolkit installed, then fails on the first convolution
    when cuDNN is missing. That failure happens after the device has been chosen, so it can't fall back to another one.

    Returns:
        True if the cuDNN library can be loaded.
    """
    library = {"win32": "cudnn64_9.dll", "linux": "libcudnn.so.9"}.get(sys.platform)
    if library is None:
        return False
    try:
        ctypes.CDLL(library)
    except OSError:
        return False
    return True


def setup_cuda_paths() -> None:
    """Add CUDA DLL directories to PATH on Windows."""
    if sys.platform != "win32":
        return

    # Find all NVIDIA bin directories in site-packages
    dll_paths = []
    for site_dir in site.getsitepackages():
        nvidia_path = Path(site_dir) / "nvidia"
        if nvidia_path.exists():
            # Find all bin directories under nvidia/*
            dll_paths.extend(str(p) for p in nvidia_path.glob("*/bin") if p.is_dir())

    if not dll_paths:
        return

    # Add to PATH environment variable
    os.environ["PATH"] = ";".join(dll_paths) + ";" + os.environ.get("PATH", "")

    # Add as DLL directories for Python 3.8+
    if hasattr(os, "add_dll_directory"):
        for path in dll_paths:
            try:
                os.add_dll_directory(path)
            except OSError:
                pass


def _load_cuda_libraries() -> None:
    """Load the CUDA and cuDNN libraries so ONNX Runtime can register its CUDA provider.

    ONNX Runtime prints a line for every library it fails to load. That output is discarded here; whether CUDA ended up
    usable is checked through ort.get_ep_devices() afterwards.
    """
    setup_cuda_paths()  # Workaround for Windows to ensure CUDA paths are set correctly
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ort.preload_dlls(directory="")
        except (OSError, RuntimeError):
            pass


def _register_webgpu() -> str | None:
    """Register the WebGPU plugin execution provider if it is installed.

    Returns:
        The WebGPU execution provider name, or None if the plugin is not installed or failed to register.
    """
    if find_spec("onnxruntime_ep_webgpu") is None:
        return None

    import onnxruntime_ep_webgpu  # noqa: PLC0415

    ep_name: str = onnxruntime_ep_webgpu.get_ep_name()
    try:
        ort.register_execution_provider_library(ep_name, onnxruntime_ep_webgpu.get_library_path())
    except Exception as e:  # noqa: BLE001
        if "already registered" not in str(e).lower():
            logger.warning(f"The WebGPU plugin is installed but could not be loaded: {e}")
            return None
    return ep_name


def _describe(ep_devices: Sequence[Any]) -> str:
    """Describe the hardware behind a list of ONNX Runtime EP devices.

    Returns:
        A short description, e.g. "Apple GPU, Apple NPU".
    """
    names = (f"{d.device.vendor} {d.device.type.name}".strip() for d in ep_devices)
    return ", ".join(dict.fromkeys(names))


@cache
def find_accelerators() -> tuple[Accelerator, ...]:
    """Find the execution providers that can run on this machine.

    CUDA is only considered when an NVIDIA driver is present, so machines without one never try to load CUDA.

    Returns:
        Usable accelerators in order of preference. The CPU is always last.
    """
    nvidia = has_nvidia_driver()
    if nvidia:
        _load_cuda_libraries()
    cudnn = nvidia and has_cudnn()
    webgpu_ep = _register_webgpu()

    ep_devices: dict[str, list[Any]] = {}
    for ep_device in ort.get_ep_devices():
        ep_devices.setdefault(ep_device.ep_name, []).append(ep_device)
    available = ort.get_available_providers()

    found: list[Accelerator] = []
    if nvidia:
        for ep in _CUDA_EPS:
            if ep == "CUDAExecutionProvider" and not cudnn:
                continue
            if ep in ep_devices:
                found.append(Accelerator("cuda", ep, _describe(ep_devices[ep]), tuple(ep_devices[ep])))
            elif ep in available:
                found.append(Accelerator("cuda", ep, "NVIDIA GPU"))

    if webgpu_ep and webgpu_ep in ep_devices:
        found.append(Accelerator("webgpu", webgpu_ep, _describe(ep_devices[webgpu_ep]), tuple(ep_devices[webgpu_ep])))

    if nvidia and not found:
        logger.warning(f"An NVIDIA GPU was found but can't be used. {_UNAVAILABLE_HINTS['cuda']}")

    if _COREML_EP in ep_devices:
        found.append(
            Accelerator(
                "coreml",
                _COREML_EP,
                _describe(ep_devices[_COREML_EP]),
                tuple(ep_devices[_COREML_EP]),
                {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
            )
        )

    found.append(Accelerator("cpu", _CPU_EP, _describe(ep_devices.get(_CPU_EP, [])) or "CPU"))
    return tuple(found)


def select_accelerators(device: str | None = None) -> list[Accelerator]:
    """Choose which accelerators to try for a requested device.

    Args:
        device: One of DEVICES. Defaults to the HAB_DEVICE environment variable, or "auto" if it is unset.

    Returns:
        Accelerators to try in order. The CPU is always last, as a fallback.

    Raises:
        ValueError: If the device name is not recognized.
        RuntimeError: If the requested device is not available on this machine.
    """
    device = (device or os.environ.get(DEVICE_ENV_VAR) or "auto").lower()
    if device not in DEVICES:
        raise ValueError(f"Unknown device '{device}'. Choose one of: {', '.join(DEVICES)}")

    found = find_accelerators()
    cpu = found[-1]
    if device == "auto":
        return [a for a in found if a.device not in _EXCLUDED_FROM_AUTO]
    if device == "cpu":
        return [cpu]

    chosen = [a for a in found if a.device == device]
    if not chosen:
        raise RuntimeError(f"Device '{device}' is not available on this machine. {_UNAVAILABLE_HINTS[device]}")
    return [*chosen, cpu]


def create_session(model_path: str | Path, device: str | None = None) -> ort.InferenceSession:
    """Create an ONNX Runtime session on the best available accelerator.

    If an accelerator fails to load the model, a warning is logged and the next one is tried, ending with the CPU.

    Args:
        model_path: Path to the ONNX model.
        device: Device to run on. See select_accelerators.

    Returns:
        The inference session.

    Raises:
        InvalidProtobuf: If the model file is corrupt.
        NoSuchFile: If the model file does not exist.
    """
    *accelerators, cpu = select_accelerators(device)
    for accelerator in accelerators:
        try:
            session = ort.InferenceSession(str(model_path), **accelerator.session_kwargs())
        except (InvalidProtobuf, NoSuchFile):
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Could not run the model on {accelerator.device} ({accelerator.hardware}). Trying the next device."
            )
            logger.debug(f"{accelerator.ep_name} failed: {e}")
            continue

        logger.info(f"Running on {accelerator.device} ({accelerator.hardware})")
        return session

    session = ort.InferenceSession(str(model_path), **cpu.session_kwargs())
    logger.info(f"Running on cpu ({cpu.hardware})")
    return session
